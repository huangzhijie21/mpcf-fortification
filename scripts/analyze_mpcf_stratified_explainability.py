#!/usr/bin/env python3
"""Stratify frozen MPCF results and explain one predeclared representative case."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import fmean, median
from typing import Any, Iterable, Mapping

from rmcd_f.cli import load_graph
from rmcd_f.model import stable_nodes
from rmcd_f.operational_motif import build_path_closed_system, solve_path_closed_motif_cut
from rmcd_f.mpcf_supplementary import alternative_minimum_path_cut


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields: fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(materialized)


def _resolve(raw: str, manifest: Path, frozen_root: Path | None, scale: str) -> Path:
    supplied = Path(raw)
    candidates = [supplied, manifest.parent / supplied]
    if frozen_root is not None:
        candidates.extend((frozen_root / scale / "instances" / supplied.name, frozen_root / "instances" / supplied.name))
    candidates.extend((manifest.parent / scale / "instances" / supplied.name, manifest.parent / "instances" / supplied.name))
    for candidate in candidates:
        path = candidate.resolve()
        if path.is_file(): return path
    raise FileNotFoundError(raw)


def _stratified_rows(successes: list[dict[str, str]], field: str) -> list[dict[str, Any]]:
    exact = {
        (row["graph_fingerprint"], float(row["budget_fraction"])): float(row["defended_margin"])
        for row in successes if row["method"] == "MPCF-Exact"
    }
    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    for row in successes:
        key = (row["graph_fingerprint"], float(row["budget_fraction"]))
        if key not in exact: continue
        optimum = exact[key]
        gap = (optimum - float(row["defended_margin"])) / optimum
        grouped[(row.get(field, ""), row["method"], float(row["budget_fraction"]))].append(gap)
    return [
        {
            "stratum_field": field,
            "stratum": stratum,
            "method": method,
            "budget_fraction": budget,
            "n": len(values),
            "relative_gap_to_mpcf_exact_mean": fmean(values),
            "relative_gap_to_mpcf_exact_median": median(values),
            "relative_gap_to_mpcf_exact_max": max(values),
            "exact_match_rate": sum(abs(value) <= 1e-9 for value in values) / len(values),
        }
        for (stratum, method, budget), values in sorted(grouped.items())
    ]


def _role_rows(
    protected: list[dict[str, str]], manifest_rows: list[dict[str, str]]
) -> list[dict[str, Any]]:
    scale_by_fingerprint = {
        row["graph_fingerprint"]: row.get("scale_tier", "") for row in manifest_rows
    }
    counts: dict[tuple[str, str, str, float, str], int] = defaultdict(int)
    totals: dict[tuple[str, str, str, float], int] = defaultdict(int)
    for row in protected:
        method = row.get("method", "")
        if method not in {"MPCF-Exact", "MPCF-Greedy"}: continue
        scale = row.get("scale_tier", "") or scale_by_fingerprint.get(
            row.get("graph_fingerprint", ""), ""
        )
        key = (method, row.get("topology", ""), scale, float(row.get("budget_fraction", 0.0)))
        role = row.get("role", "unspecified") or "unspecified"
        counts[(*key, role)] += 1; totals[key] += 1
    rows: list[dict[str, Any]] = []
    for method, topology, scale, budget in sorted(totals):
        total = totals[(method, topology, scale, budget)]
        observed_roles = {
            role
            for candidate_method, candidate_topology, candidate_scale, candidate_budget, role in counts
            if (
                candidate_method,
                candidate_topology,
                candidate_scale,
                candidate_budget,
            )
            == (method, topology, scale, budget)
        }
        for role in (*("S", "C", "L", "E"), *sorted(observed_roles - {"S", "C", "L", "E"})):
            count = counts[(method, topology, scale, budget, role)]
            rows.append(
                {
                    "method": method,
                    "topology": topology,
                    "scale_tier": scale,
                    "budget_fraction": budget,
                    "role": role,
                    "selected_count": count,
                    "selected_fraction": count / total,
                }
            )
    return rows


def _representative(
    successes: list[dict[str, str]],
    protected_rows: list[dict[str, str]],
    manifest_rows: list[dict[str, str]],
    manifest_path: Path,
    frozen_root: Path | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(row["graph_fingerprint"], float(row["budget_fraction"]), row["method"]): row for row in successes}
    pairs = []
    for fingerprint, budget, method in index:
        if method != "MPCF-Exact" or (fingerprint, budget, "MPCF-Greedy") not in index: continue
        exact = index[(fingerprint, budget, "MPCF-Exact")]
        greedy = index[(fingerprint, budget, "MPCF-Greedy")]
        gap = float(exact["defended_margin"]) - float(greedy["defended_margin"])
        pairs.append((gap, fingerprint, budget, exact, greedy))
    strict = sorted((item for item in pairs if item[0] > 1e-9), key=lambda item: (item[0], item[1], item[2]))
    pool = strict or sorted(pairs, key=lambda item: (item[0], item[1], item[2]))
    if not pool: return [], []
    gap, fingerprint, budget, exact, greedy = pool[(len(pool) - 1) // 2]
    manifest_by_fp = {row["graph_fingerprint"]: row for row in manifest_rows}
    source = manifest_by_fp[fingerprint]
    instance = _resolve(source["instance_file"], manifest_path, frozen_root, source.get("scale_tier", ""))
    graph = load_graph(instance); system = build_path_closed_system(graph)
    initial = solve_path_closed_motif_cut(system)
    current, alternative, cut_cost = alternative_minimum_path_cut(system, initial.selected_set)
    reconstructed: dict[tuple[str, float, str], set[str]] = defaultdict(set)
    for row in protected_rows:
        reconstructed[
            (
                row.get("graph_fingerprint", ""),
                float(row.get("budget_fraction", 0.0)),
                row.get("method", ""),
            )
        ].add(row.get("node_id", ""))

    def protection_set(row: Mapping[str, str]) -> frozenset[str]:
        raw = row.get("protection_set_json", "")
        if raw:
            decoded = frozenset(json.loads(raw))
            if decoded or int(float(row.get("protection_count", 0) or 0)) == 0:
                return decoded
        return frozenset(
            reconstructed[
                (fingerprint, budget, row.get("method", ""))
            ]
        )

    exact_set = protection_set(exact)
    greedy_set = protection_set(greedy)
    summary = [{
        "selection_rule": "median positive Exact-minus-Greedy margin gap; deterministic lower median",
        "positive_gap_case_count": len(strict),
        "graph_fingerprint": fingerprint,
        "topology": exact.get("topology", ""),
        "scale_tier": exact.get("scale_tier", ""),
        "seed": exact.get("seed", ""),
        "budget_fraction": budget,
        "undefended_minimum_cut_cost": cut_cost,
        "current_minimum_cut_json": json.dumps(list(stable_nodes(current))),
        "alternative_equal_cost_cut_json": "" if alternative is None else json.dumps(list(stable_nodes(alternative))),
        "alternative_cut_found": int(alternative is not None),
        "greedy_protection_json": json.dumps(list(stable_nodes(greedy_set))),
        "exact_protection_json": json.dumps(list(stable_nodes(exact_set))),
        "greedy_defended_margin": greedy["defended_margin"],
        "exact_defended_margin": exact["defended_margin"],
        "exact_minus_greedy_margin": gap,
    }]
    universe = current | (alternative or frozenset()) | greedy_set | exact_set
    nodes = [{
        "graph_fingerprint": fingerprint,
        "node_id": node,
        "role": graph.nodes[node].get("role", ""),
        "attack_cost": graph.nodes[node]["attack_cost"],
        "protect_cost": graph.nodes[node]["protect_cost"],
        "in_current_minimum_cut": int(node in current),
        "in_alternative_equal_cost_cut": int(alternative is not None and node in alternative),
        "selected_by_greedy": int(node in greedy_set),
        "selected_by_exact": int(node in exact_set),
    } for node in stable_nodes(universe)]
    return summary, nodes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-results", required=True)
    parser.add_argument("--protected-nodes", required=True)
    parser.add_argument("--instance-manifest", required=True)
    parser.add_argument("--frozen-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    trial_path = Path(args.trial_results).resolve(); protected_path = Path(args.protected_nodes).resolve(); manifest_path = Path(args.instance_manifest).resolve()
    frozen_root = Path(args.frozen_root).resolve() if args.frozen_root else None
    successes = [row for row in _read_csv(trial_path) if row.get("status") == "SUCCESS"]
    protected = _read_csv(protected_path); manifest = _read_csv(manifest_path)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=True)
    topology = _stratified_rows(successes, "topology")
    scale = _stratified_rows(successes, "scale_tier")
    budget = _stratified_rows(successes, "budget_fraction")
    roles = _role_rows(protected, manifest)
    representative, representative_nodes = _representative(
        successes, protected, manifest, manifest_path, frozen_root
    )
    _write_csv(output / "topology_stratified_relative_gap.csv", topology)
    _write_csv(output / "scale_stratified_relative_gap.csv", scale)
    _write_csv(output / "budget_stratified_relative_gap.csv", budget)
    _write_csv(output / "mpcf_selected_role_distribution.csv", roles)
    _write_csv(output / "representative_case_summary.csv", representative)
    _write_csv(output / "representative_case_nodes.csv", representative_nodes)
    gates = [
        {"gate": "exact_reference_available", "status": "PASS" if any(row["method"] == "MPCF-Exact" for row in successes) else "FAIL"},
        {"gate": "topology_and_scale_strata_nonempty", "status": "PASS" if topology and scale else "FAIL"},
        {"gate": "representative_rule_predeclared", "status": "PASS" if representative else "FAIL"},
    ]
    _write_csv(output / "stratified_explainability_gate.csv", gates)
    passed = all(row["status"] == "PASS" for row in gates)
    (output / ("_COMPLETE" if passed else "_FAILED")).write_text("", encoding="utf-8")
    return 0 if passed else 4


if __name__ == "__main__": raise SystemExit(main())
