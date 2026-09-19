"""Generic executor: one frozen instance in, unified run records out.

Every method family (MPCF exact solvers, local protection baselines, transferred
upstream dismantling rankings, heterogeneous-cost conversions) is driven through
this single path so that the instance, the budget, the unified time limit and
the exact adaptive min-cut evaluator are provably identical across methods.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable, Mapping, Sequence

import networkx as nx

from ..cli import load_graph
from ..model import protection_cost, stable_nodes
from ..official_review import graph_fingerprint
from ..operational_motif import build_path_closed_system, solve_path_closed_motif_cut
from ..protection_baselines import (
    LOCAL_PROTECTION_METHODS,
    local_protection_sequence,
    protection_method_name,
    select_budget_feasible_ranking,
)
from ..task_path_fortification import (
    evaluate_task_path_fortification,
    solve_mpcf_cut_generation,
    solve_mpcf_exact,
    solve_mpcf_greedy,
)
from ..mpcf_supplementary import protection_scores, select_score_policy
from .ids import code_commit as resolve_commit
from .panels import InstanceSpec, uplift_for_spec
from .schema import NA, RunRecord

MPCF_METHODS: tuple[str, ...] = ("MPCF-Exact", "MPCF-CG", "MPCF-Greedy")

STATUS_LABELS: Mapping[str, str] = {
    "OPTIMAL": "optimal",
    "FEASIBLE": "feasible",
    "TIME_LIMIT": "time_limit",
    "LIMIT_REACHED": "limit_reached",
    "DUAL_CLOSED_GAP": "dual_closed_gap",
    "INFEASIBLE": "infeasible",
    "UNBOUNDED": "unbounded",
    "ERROR": "error",
}

#: Methods whose protection set is produced by a fixed structural ranking.
RANKING_METHODS: tuple[str, ...] = LOCAL_PROTECTION_METHODS

#: Heterogeneous-cost conversion variants, named ``<Score>-<Policy>``.
CONVERSION_SUFFIXES: tuple[str, ...] = (
    "Raw",
    "PerCost",
    "PrefixKnapsack",
    "FullKnapsack",
)


def _policy_for(name: str) -> tuple[str, str, int | None] | None:
    """Split ``"Degree-PerCost"`` into ``(score, policy, prefix_multiplier)``."""

    if "-" not in name:
        return None
    score_label, _, suffix = name.partition("-")
    lookup = {"Degree": "degree", "Betweenness": "betweenness"}
    score = lookup.get(score_label)
    if score is None:
        return None
    multiplier: int | None = None
    if suffix.startswith("PrefixKnapsack"):
        tail = suffix[len("PrefixKnapsack") :]
        multiplier = int(tail) if tail.isdigit() else 3
        return score, "prefix_knapsack", multiplier
    mapping = {
        "Raw": "raw",
        "PerCost": "per_cost",
        "FullKnapsack": "full_knapsack",
    }
    policy = mapping.get(suffix)
    if policy is None:
        return None
    return score, policy, multiplier


@dataclass
class SolverOutcome:
    """Method-agnostic result of one selection step."""

    protection_set: frozenset[Any]
    sequence: tuple[Any, ...]
    selection_runtime: float
    status: str
    provenance: str
    source_method: str
    variant: str
    solver_optimal: bool = False
    incumbent: float | None = None
    best_bound: float | None = None
    final_gap: float | None = None
    bb_nodes: int | None = None
    certificate_mode: str = "open"
    cg_L: float | None = None
    cg_U: float | None = None
    cg_iterations: int | None = None
    cg_cuts: int | None = None
    uplift_multiplier: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BudgetCell:
    """One ``(instance, budget)`` cell with its cross-solver certified optimum."""

    kappa_opt: float | None = None
    certifying_methods: tuple[str, ...] = ()
    certificate_conflict: int = 0
    undefended: float | None = None
    budget_abs: float = 0.0
    budget_abs_floor: int = 0
    zero_effective_budget: int = 0


def _budget_abs(total: float, ratio: float) -> float:
    return float(ratio) * float(total)


def _zero_effective(budget: float, min_cost: float) -> int:
    """A budget is effective only if at least one node is affordable."""

    return int(float(budget) + 1e-9 < float(min_cost))


# ---------------------------------------------------------------------------
# per-method selection
# ---------------------------------------------------------------------------


def select_mpcf(
    method: str,
    system: Any,
    budget: float,
    *,
    fortification_multiplier: float,
    uplifts: Mapping[Any, float] | None,
    exact_time_limit: float | None,
    cg_time_limit: float | None,
    greedy_time_limit: float | None,
    cg_max_iterations: int,
) -> SolverOutcome:
    if method == "MPCF-Exact":
        result = solve_mpcf_exact(
            system,
            budget,
            fortification_multiplier=fortification_multiplier,
            uplifts=uplifts,
            time_limit=exact_time_limit,
        )
        log = result.solver_log
        return SolverOutcome(
            protection_set=result.protection_set,
            sequence=tuple(stable_nodes(result.protection_set)),
            selection_runtime=result.solver.runtime_seconds,
            status=STATUS_LABELS.get(result.status.value, result.status.value.lower()),
            provenance="compact node-split max-flow MILP",
            source_method="MPCF-Exact",
            variant="exact-compact-milp",
            solver_optimal=result.optimal,
            incumbent=log.incumbent,
            best_bound=log.best_bound,
            final_gap=log.final_gap,
            bb_nodes=log.bb_nodes,
            certificate_mode=log.certificate_mode,
        )
    if method == "MPCF-CG":
        result = solve_mpcf_cut_generation(
            system,
            budget,
            fortification_multiplier=fortification_multiplier,
            uplifts=uplifts,
            time_limit_per_solve=cg_time_limit,
            overall_time_limit=cg_time_limit,
            max_iterations=cg_max_iterations,
        )
        log = result.solver_log
        return SolverOutcome(
            protection_set=result.protection_set,
            sequence=tuple(stable_nodes(result.protection_set)),
            selection_runtime=result.solver.runtime_seconds,
            status=STATUS_LABELS.get(result.status.value, result.status.value.lower()),
            provenance="cut-generation master plus exact PathCut separation",
            source_method="MPCF-CG",
            variant="exact-cut-generation",
            solver_optimal=result.optimal,
            incumbent=log.incumbent,
            best_bound=log.best_bound,
            final_gap=log.final_gap,
            bb_nodes=log.bb_nodes,
            certificate_mode=log.certificate_mode,
            cg_L=log.cg_L,
            cg_U=log.cg_U,
            cg_iterations=log.cg_iterations,
            cg_cuts=log.cg_cuts,
        )
    if method == "MPCF-Greedy":
        result = solve_mpcf_greedy(
            system,
            budget,
            fortification_multiplier=fortification_multiplier,
            uplifts=uplifts,
            time_limit=greedy_time_limit,
        )
        log = result.solver_log
        return SolverOutcome(
            protection_set=result.protection_set,
            sequence=tuple(
                row.selected_node for row in result.trace if row.selected_node is not None
            ),
            selection_runtime=result.solver.runtime_seconds,
            status=STATUS_LABELS.get(result.status.value, result.status.value.lower()),
            provenance="exact adaptive-margin greedy",
            source_method="MPCF-Greedy",
            variant="heuristic-greedy",
            solver_optimal=False,
            incumbent=log.incumbent,
            certificate_mode="open",
            cg_iterations=log.cg_iterations,
        )
    raise ValueError(f"Unknown MPCF method {method!r}.")


def select_local(
    method: str,
    system: Any,
    budget: float,
    *,
    seed: int,
) -> SolverOutcome:
    ranking, runtime, provenance = local_protection_sequence(system, method, seed=seed)
    selected = select_budget_feasible_ranking(
        system,
        ranking,
        budget,
        method=method,
        source_method=method,
        provenance=provenance,
        runtime_seconds=runtime,
    )
    return SolverOutcome(
        protection_set=selected.protection_set,
        sequence=selected.protection_sequence,
        selection_runtime=selected.runtime_seconds,
        status="feasible",
        provenance=selected.provenance,
        source_method=method,
        variant=f"local-{method.lower()}",
        extra={
            "ranking_length": selected.ranking_length,
            "skipped_unaffordable": selected.skipped_unaffordable,
            "skipped_unknown": selected.skipped_unknown,
        },
    )


def select_official(
    method: str,
    system: Any,
    budget: float,
    *,
    sequence: Sequence[Any],
    runtime: float,
    source: str,
) -> SolverOutcome:
    selected = select_budget_feasible_ranking(
        system,
        tuple(sequence),
        budget,
        method=method,
        source_method=method,
        provenance=(
            "unchanged structural dismantling ranking transferred to protection; "
            f"source={source}"
        ),
        runtime_seconds=runtime,
    )
    return SolverOutcome(
        protection_set=selected.protection_set,
        sequence=selected.protection_sequence,
        selection_runtime=selected.runtime_seconds,
        status="feasible",
        provenance=selected.provenance,
        source_method=method,
        variant=f"official-{method.lower()}-protect",
        extra={
            "ranking_length": selected.ranking_length,
            "skipped_unaffordable": selected.skipped_unaffordable,
            "skipped_unknown": selected.skipped_unknown,
        },
    )


def select_conversion(
    name: str,
    system: Any,
    budget: float,
) -> SolverOutcome:
    parsed = _policy_for(name)
    if parsed is None:
        raise ValueError(f"Unknown conversion variant {name!r}.")
    score, policy, multiplier = parsed
    started = perf_counter()
    scores = protection_scores(system, score)
    selection = select_score_policy(
        system,
        scores,
        budget,
        method=name,
        policy=policy,
        **({"prefix_multiplier": multiplier} if multiplier is not None else {}),
    )
    runtime = perf_counter() - started
    return SolverOutcome(
        protection_set=selection.protection_set,
        sequence=selection.protection_sequence,
        selection_runtime=runtime,
        status="feasible",
        provenance=f"heterogeneous-cost conversion policy={policy} score={score}",
        source_method=name,
        variant=f"conv-{score}-{policy}",
        extra={
            "cost_conversion": policy,
            "prefix_length": selection.prefix_length,
            "candidate_count": selection.candidate_count,
        },
    )


# ---------------------------------------------------------------------------
# one instance
# ---------------------------------------------------------------------------


def run_instance(
    spec: InstanceSpec,
    methods: Sequence[str],
    *,
    official_sequences: Mapping[tuple[str, str], tuple[Sequence[Any], float, str]] | None = None,
    exact_time_limit: float | None = None,
    cg_time_limit: float | None = None,
    greedy_time_limit: float | None = None,
    cg_max_iterations: int = 10_000,
    commit: str = "",
    config_hash_value: str = "",
    budget_ratios: Sequence[float] | None = None,
) -> tuple[list[RunRecord], dict[str, BudgetCell]]:
    """Solve every method on one frozen instance; return rows and cell ledger."""

    graph = load_graph(spec.instance_file)
    if graph_fingerprint(graph) != spec.graph_fingerprint:
        raise RuntimeError(
            f"Frozen graph fingerprint changed for {spec.instance_id}; refusing to run."
        )
    system = build_path_closed_system(graph)
    baseline = solve_path_closed_motif_cut(system)
    if not baseline.optimal or baseline.objective is None:
        raise RuntimeError(f"{spec.instance_id} is not a finite path-closed network.")
    undefended = float(baseline.objective)
    uplifts = uplift_for_spec(graph)
    total_cost = float(
        sum(float(graph.nodes[node]["protect_cost"]) for node in system.removable_nodes)
    )
    min_cost = min(
        (float(graph.nodes[node]["protect_cost"]) for node in system.removable_nodes),
        default=1.0,
    )
    ratios = tuple(budget_ratios if budget_ratios is not None else spec.budget_ratios)
    official_sequences = official_sequences or {}

    records: list[RunRecord] = []
    cells: dict[str, BudgetCell] = {}
    limits = {
        "MPCF-Exact": exact_time_limit,
        "MPCF-CG": cg_time_limit,
        "MPCF-Greedy": greedy_time_limit,
    }
    for ratio in ratios:
        budget = _budget_abs(total_cost, ratio)
        key = cell_key(spec.instance_id, ratio)
        cells[key] = BudgetCell(
            undefended=undefended,
            budget_abs=budget,
            budget_abs_floor=int(math.floor(budget + 1e-9)),
            zero_effective_budget=_zero_effective(budget, min_cost),
        )
        for method in methods:
            started = perf_counter()
            try:
                outcome = _dispatch(
                    method,
                    system,
                    budget,
                    spec=spec,
                    seed=spec.seed,
                    uplifts=uplifts,
                    official_sequences=official_sequences,
                    exact_time_limit=exact_time_limit,
                    cg_time_limit=cg_time_limit,
                    greedy_time_limit=greedy_time_limit,
                    cg_max_iterations=cg_max_iterations,
                )
            except Exception as exc:  # every failure stays visible
                records.append(
                    _error_record(
                        spec, method, ratio, budget, commit, config_hash_value, exc
                    )
                )
                continue
            record = _evaluate_outcome(
                spec=spec,
                system=system,
                method=method,
                ratio=ratio,
                budget=budget,
                outcome=outcome,
                undefended=undefended,
                commit=commit,
                config_hash_value=config_hash_value,
                total_runtime=perf_counter() - started,
                uplifts=uplifts,
                time_limit_s=limits.get(method),
            )
            records.append(record)
    return records, cells


def _dispatch(
    method: str,
    system: Any,
    budget: float,
    *,
    spec: InstanceSpec,
    seed: int,
    uplifts: Mapping[Any, float] | None,
    official_sequences: Mapping[tuple[str, str], tuple[Sequence[Any], float, str]],
    exact_time_limit: float | None,
    cg_time_limit: float | None,
    greedy_time_limit: float | None,
    cg_max_iterations: int,
) -> SolverOutcome:
    if method in MPCF_METHODS:
        return select_mpcf(
            method,
            system,
            budget,
            fortification_multiplier=spec.uplift_multiplier,
            uplifts=uplifts,
            exact_time_limit=exact_time_limit,
            cg_time_limit=cg_time_limit,
            greedy_time_limit=greedy_time_limit,
            cg_max_iterations=cg_max_iterations,
        )
    if method in RANKING_METHODS:
        return select_local(method, system, budget, seed=seed)
    if _policy_for(method) is not None:
        return select_conversion(method, system, budget)
    source = method[: -len("-Protect")] if method.endswith("-Protect") else method
    payload = official_sequences.get((spec.graph_fingerprint, source))
    if payload is None:
        raise KeyError(f"No frozen upstream ranking for {source} on {spec.instance_id}.")
    sequence, runtime, origin = payload
    return select_official(
        method, system, budget, sequence=sequence, runtime=runtime, source=origin
    )


def _evaluate_outcome(
    *,
    spec: InstanceSpec,
    system: Any,
    method: str,
    ratio: float,
    budget: float,
    outcome: SolverOutcome,
    undefended: float,
    commit: str,
    config_hash_value: str,
    total_runtime: float,
    uplifts: Mapping[Any, float] | None,
    time_limit_s: float | None = None,
) -> RunRecord:
    """Run the single unified exact min-cut evaluation and build the record."""

    evaluation_started = perf_counter()
    oracle = evaluate_task_path_fortification(
        system,
        outcome.protection_set,
        fortification_multiplier=spec.uplift_multiplier,
        uplifts=uplifts,
    )
    evaluation_runtime = perf_counter() - evaluation_started
    if not oracle.optimal or oracle.objective is None:
        raise RuntimeError("Unified exact adaptive PathCut evaluation failed.")
    kappa = float(oracle.objective)
    spent = protection_cost(system.graph, outcome.protection_set)
    return RunRecord(
        experiment=spec.experiment,
        instance_id=spec.instance_id,
        graph_id=spec.graph_id,
        base_id=spec.base_id,
        topology=spec.topology,
        N=int(spec.node_count),
        num_edges=int(spec.edge_count),
        seed=int(spec.seed),
        composition=spec.composition,
        method=outcome.source_method,
        variant=outcome.variant,
        budget_ratio=float(ratio),
        budget_abs=float(budget),
        kappa=kappa,
        kappa_opt=None,
        actual_cost=float(spent),
        selected_count=len(outcome.protection_set),
        selected_nodes=tuple(stable_nodes(outcome.protection_set)),
        runtime_selection_s=float(outcome.selection_runtime),
        runtime_evaluation_s=float(evaluation_runtime),
        runtime_total_s=float(total_runtime),
        status=outcome.status,
        incumbent=outcome.incumbent if outcome.incumbent is not None else kappa,
        best_bound=outcome.best_bound,
        final_gap=outcome.final_gap,
        bb_nodes=outcome.bb_nodes,
        replay_kappa=kappa,
        certificate_mode=outcome.certificate_mode,
        cg_L=outcome.cg_L,
        cg_U=outcome.cg_U,
        cg_iterations=outcome.cg_iterations,
        cg_cuts=outcome.cg_cuts,
        time_limit_s=time_limit_s,
        config_hash=config_hash_value,
        code_commit=commit,
    )


def _error_record(
    spec: InstanceSpec,
    method: str,
    ratio: float,
    budget: float,
    commit: str,
    config_hash_value: str,
    error: Exception,
) -> RunRecord:
    return RunRecord(
        experiment=spec.experiment,
        instance_id=spec.instance_id,
        graph_id=spec.graph_id,
        base_id=spec.base_id,
        topology=spec.topology,
        N=int(spec.node_count),
        num_edges=int(spec.edge_count),
        seed=int(spec.seed),
        composition=spec.composition,
        method=method,
        variant=f"error-{type(error).__name__}",
        budget_ratio=float(ratio),
        budget_abs=float(budget),
        status="error",
        config_hash=config_hash_value,
        code_commit=commit,
    )


# ---------------------------------------------------------------------------
# cross-solver optimum ledger
# ---------------------------------------------------------------------------


def certify_cells(
    records: Sequence[RunRecord],
    cells: Mapping[str, BudgetCell],
    *,
    tolerance: float = 1e-7,
) -> dict[str, BudgetCell]:
    """Fill ``kappa_opt`` per ``(instance, budget)`` from certified solvers.

    Two independently structured exact solvers contribute a feasible lower
    bound; their own upper bounds are what make the value certified.  A conflict
    between two certified values is recorded rather than silently resolved.
    """

    by_cell: dict[str, list[RunRecord]] = {}
    for record in records:
        by_cell.setdefault(
            cell_key(record.instance_id, record.budget_ratio), []
        ).append(record)
    for key, rows in by_cell.items():
        entry = cells.get(key)
        if entry is None:
            continue
        candidates = [
            (row.method, float(row.kappa))
            for row in rows
            if row.method in {"MPCF-Exact", "MPCF-CG"}
            and row.certificate_mode in {"solver_closed", "cg_closed"}
            and row.kappa is not None
        ]
        if not candidates:
            continue
        values = [value for _, value in candidates]
        scale = max(1.0, *(abs(value) for value in values))
        names = tuple(sorted({name for name, _ in candidates}))
        if max(values) - min(values) > tolerance * scale:
            entry.certificate_conflict = 1
            entry.certifying_methods = names
            entry.kappa_opt = None
            continue
        entry.kappa_opt = float(max(values))
        entry.certifying_methods = names
    return dict(cells)


def cell_key(instance: str, ratio: float) -> str:
    return f"{instance}|{float(ratio):.10g}"
