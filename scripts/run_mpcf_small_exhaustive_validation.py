#!/usr/bin/env python3
"""Cross-check MPCF and capacity-threshold solvers by tiny exhaustive search."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from rmcd_f.cost_profiles import apply_attack_cost_profile
from rmcd_f.model import stable_nodes
from rmcd_f.operational_motif import build_path_closed_system, task_path_endpoints
from rmcd_f.synthetic import generate_equipment_network
from rmcd_f.task_path_fortification import (
    brute_force_mpcf,
    fortified_attack_costs,
    solve_mpcf_cut_generation,
    solve_mpcf_exact,
)
from rmcd_f.mpcf_capacity_evaluation import evaluate_role_motif_capacity_thresholds
from rmcd_f.mpcf_supplementary import brute_force_capacity_attack


@dataclass(frozen=True)
class TinyTask:
    size: int
    topology: str
    seed: int
    budgets: tuple[float, ...]
    capacity_enabled: bool
    capacity_fractions: tuple[float, ...]
    interface_density: float


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _progress(message: str) -> None:
    print(f"[{_timestamp()}] [mpcf-exhaustive] {message}", flush=True)


def _csv_values(raw: str, cast) -> tuple:
    return tuple(cast(item.strip()) for item in raw.split(",") if item.strip())


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)


def _run_task(task: TinyTask) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if task.size % 4:
        raise ValueError("Every tiny size must be divisible by four.")
    per_role = task.size // 4
    graph = generate_equipment_network(
        task.topology,
        {role: per_role for role in ("S", "C", "L", "E")},
        seed=task.seed,
        interface_density=task.interface_density,
        capacity_levels=(1, 2),
    )
    graph = apply_attack_cost_profile(
        graph, "balanced-heterogeneous", cost_seed=task.seed
    )
    system = build_path_closed_system(graph)
    task_sources, task_targets = task_path_endpoints(system)
    role_sizes_json = json.dumps(
        {role: per_role for role in ("S", "C", "L", "E")}, sort_keys=True
    )
    capacity_vector_json = json.dumps(
        {node: int(graph.nodes[node]["capacity"]) for node in stable_nodes(graph)},
        sort_keys=True,
    )
    defense_rows: list[dict[str, Any]] = []
    exact_by_budget = {}
    for budget in task.budgets:
        brute = brute_force_mpcf(graph, budget, max_nodes=task.size)
        exact = solve_mpcf_exact(graph, budget)
        cg = solve_mpcf_cut_generation(graph, budget)
        exact_by_budget[budget] = exact
        tolerance = 1e-7 * max(1.0, abs(brute.defended_margin))
        row = {
            "node_count": task.size,
            "topology": task.topology,
            "seed": task.seed,
            "budget_cost": budget,
            "brute_margin": brute.defended_margin,
            "exact_margin": exact.defended_margin,
            "cg_margin": cg.defended_margin,
            "brute_set_json": json.dumps(list(stable_nodes(brute.protection_set))),
            "exact_set_json": json.dumps(list(stable_nodes(exact.protection_set))),
            "cg_set_json": json.dumps(list(stable_nodes(cg.protection_set))),
            "brute_exact_objective_match": int(
                abs(brute.defended_margin - exact.defended_margin) <= tolerance
            ),
            "brute_cg_objective_match": int(
                abs(brute.defended_margin - cg.defended_margin) <= tolerance
            ),
            "exact_cg_objective_match": int(
                abs(exact.defended_margin - cg.defended_margin) <= tolerance
            ),
            "exact_certified": int(exact.optimal),
            "cg_certified": int(cg.optimal),
            "brute_runtime_seconds": brute.solver.runtime_seconds,
            "exact_runtime_seconds": exact.solver.runtime_seconds,
            "cg_runtime_seconds": cg.solver.runtime_seconds,
        }
        row["gate_pass"] = int(
            row["brute_exact_objective_match"]
            and row["brute_cg_objective_match"]
            and row["exact_certified"]
            and row["cg_certified"]
        )
        defense_rows.append(row)

    capacity_rows: list[dict[str, Any]] = []
    if task.capacity_enabled:
        for budget in task.budgets:
            defense = exact_by_budget[budget]
            evaluation = evaluate_role_motif_capacity_thresholds(
                graph,
                defense.protection_set,
                task.capacity_fractions,
                solver="rmcd-exact",
            )
            fortified = graph.copy()
            fortified_cost = fortified_attack_costs(graph, defense.protection_set)
            for node, value in fortified_cost.items():
                fortified.nodes[node]["attack_cost"] = value
            for point in evaluation.points:
                brute = brute_force_capacity_attack(
                    fortified, point.threshold_k, max_nodes=task.size
                )
                exact_cost = point.attack_cost
                match = bool(
                    exact_cost is not None
                    and brute.attack_cost is not None
                    and abs(float(exact_cost) - brute.attack_cost) <= 1e-7
                )
                capacity_rows.append(
                    {
                        "node_count": task.size,
                        "topology": task.topology,
                        "seed": task.seed,
                        "budget_cost": budget,
                        "role_sizes_json": role_sizes_json,
                        "interface_density": task.interface_density,
                        "capacity_levels_json": json.dumps([1, 2]),
                        "capacity_vector_json": capacity_vector_json,
                        "task_sources_json": json.dumps(list(stable_nodes(task_sources))),
                        "task_targets_json": json.dumps(list(stable_nodes(task_targets))),
                        "fixed_protection_source": "MPCF-Exact minimum-cost primary-optimal representative",
                        "fixed_protection_set_json": json.dumps(
                            list(stable_nodes(defense.protection_set))
                        ),
                        "fixed_protection_cost": defense.protection_cost,
                        "requested_remaining_fraction": point.requested_remaining_fraction,
                        "one_threshold_per_row": 1,
                        "threshold_k": point.threshold_k,
                        "initial_capacity": evaluation.initial_capacity,
                        "solver_attack_cost": exact_cost,
                        "brute_attack_cost": brute.attack_cost,
                        "solver_attack_set_json": json.dumps(
                            list(stable_nodes(point.attack_set))
                        ),
                        "brute_attack_set_json": json.dumps(
                            list(stable_nodes(brute.attack_set))
                        ),
                        "objective_match": int(match),
                        "solver_certified": int(point.global_optimum_certified),
                        "brute_force_complete": int(brute.complete),
                        "candidate_subset_count": brute.candidate_subset_count,
                        "capacity_evaluation_count": brute.capacity_evaluation_count,
                        "gate_pass": int(
                            match and point.global_optimum_certified and brute.complete
                        ),
                    }
                )
    return defense_rows, capacity_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", default="8,12,16,20")
    parser.add_argument("--capacity-sizes", default="8,12")
    parser.add_argument("--topologies", default="centralized,modular,distributed")
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--budgets", default="1,2")
    parser.add_argument("--capacity-fractions", default="0,0.5")
    parser.add_argument("--interface-density", type=float, default=0.55)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    sizes = _csv_values(args.sizes, int)
    capacity_sizes = frozenset(_csv_values(args.capacity_sizes, int))
    topologies = _csv_values(args.topologies, str)
    seeds = _csv_values(args.seeds, int)
    budgets = _csv_values(args.budgets, float)
    fractions = _csv_values(args.capacity_fractions, float)
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for marker in ("_COMPLETE", "_FAILED"):
        (output / marker).unlink(missing_ok=True)
    tasks = [
        TinyTask(
            size=size,
            topology=topology,
            seed=seed,
            budgets=budgets,
            capacity_enabled=size in capacity_sizes,
            capacity_fractions=fractions,
            interface_density=args.interface_density,
        )
        for size in sizes
        for topology in topologies
        for seed in seeds
    ]
    defenses: list[dict[str, Any]] = []
    capacities: list[dict[str, Any]] = []
    _progress(f"starting {len(tasks)} tiny graphs with workers={args.workers}")
    if args.workers == 1:
        batches = (_run_task(task) for task in tasks)
        for index, batch in enumerate(batches, start=1):
            defenses.extend(batch[0])
            capacities.extend(batch[1])
            _progress(f"completed {index}/{len(tasks)} graphs")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(_run_task, task) for task in tasks]
            for index, future in enumerate(as_completed(futures), start=1):
                defense, capacity = future.result()
                defenses.extend(defense)
                capacities.extend(capacity)
                _progress(f"completed {index}/{len(tasks)} graphs")
    defenses.sort(key=lambda row: (row["node_count"], row["topology"], row["seed"], row["budget_cost"]))
    capacities.sort(key=lambda row: (row["node_count"], row["topology"], row["seed"], row["budget_cost"], row["requested_remaining_fraction"]))
    _write_csv(output / "small_defense_exhaustive_validation.csv", defenses)
    _write_csv(output / "small_capacity_exhaustive_validation.csv", capacities)
    gates = [
        {
            "gate": "brute_equals_mpcf_exact_equals_mpcf_cg",
            "status": "PASS" if defenses and all(row["gate_pass"] for row in defenses) else "FAIL",
            "row_count": len(defenses),
        },
        {
            "gate": "capacity_solver_equals_attack_enumeration",
            "status": "PASS" if capacities and all(row["gate_pass"] for row in capacities) else "FAIL",
            "row_count": len(capacities),
        },
    ]
    _write_csv(output / "small_exhaustive_gate.csv", gates)
    (output / "study_manifest.json").write_text(
        json.dumps(
            {
                "sizes": sizes,
                "capacity_sizes": sorted(capacity_sizes),
                "topologies": topologies,
                "seeds": seeds,
                "budgets": budgets,
                "capacity_fractions": fractions,
                "defense_semantics": "all budget-feasible protection subsets",
                "capacity_semantics": "all removable attack subsets on the fortified graph",
                "capacity_fixed_defense": "MPCF-Exact minimum-cost primary-optimal representative at the listed budget",
                "capacity_rows_per_graph": len(budgets) * len(fractions),
                "role_sizes": "equal four-way split; node_count/4 per S,C,L,E role",
                "capacity_levels": [1, 2],
                "one_requested_fraction_per_capacity_row": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    passed = all(row["status"] == "PASS" for row in gates)
    (output / ("_COMPLETE" if passed else "_FAILED")).write_text("", encoding="utf-8")
    _progress(f"{'PASS' if passed else 'FAIL'}: exhaustive validation written to {output}")
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
