"""Command-line interface for role-aware critical equipment-set analysis."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable

import networkx as nx

from .criticality import analyze_exclusion
from .flow import CutCertificate, motif_capacity
from .fortification import FortificationResult, solve_rmcd_f
from .keyset import CriticalEquipmentSetResult, identify_critical_equipment_set
from .model import GraphValidationError, NodeId, SolverInfo, stable_nodes, validate_graph
from .pcd import PCDResult, solve_rmcd_pcd
from .rmcd import FrontierResult, RMCDResult, solve_frontier, solve_rmcd_exact
from .synthetic import TOPOLOGIES, generate_equipment_network


EXIT_OK = 0
EXIT_INPUT_ERROR = 2
EXIT_SOLVER_ERROR = 3
EXIT_INCOMPLETE = 4


def _progress(message: str, *, quiet: bool) -> None:
    if not quiet:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(f"[{timestamp}] [rmcd-f] {message}", file=sys.stderr, flush=True)


def _is_json_node_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    return isinstance(value, float) and math.isfinite(value)


def load_graph(path: str | Path) -> nx.DiGraph:
    """Load and strictly validate the documented JSON equipment graph."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise GraphValidationError("Input JSON root must be an object.")
    if set(payload) - {"nodes", "edges", "metadata"}:
        unknown = sorted(set(payload) - {"nodes", "edges", "metadata"})
        raise GraphValidationError(f"Unknown top-level JSON fields: {unknown}.")
    nodes = payload.get("nodes")
    edges = payload.get("edges")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise GraphValidationError("Input JSON requires list fields 'nodes' and 'edges'.")

    graph = nx.DiGraph()
    seen: set[NodeId] = set()
    for index, record in enumerate(nodes):
        if not isinstance(record, dict):
            raise GraphValidationError(f"nodes[{index}] must be an object.")
        if "id" not in record:
            raise GraphValidationError(f"nodes[{index}] is missing 'id'.")
        node = record["id"]
        if not _is_json_node_id(node):
            raise GraphValidationError(
                f"nodes[{index}].id must be a finite number or string; got {node!r}."
            )
        if node in seen:
            raise GraphValidationError(f"Duplicate node ID: {node!r}.")
        seen.add(node)
        attributes = {key: value for key, value in record.items() if key != "id"}
        graph.add_node(node, **attributes)

    for index, edge in enumerate(edges):
        if not isinstance(edge, list) or len(edge) != 2:
            raise GraphValidationError(f"edges[{index}] must be [source, target].")
        source, target = edge
        if source not in seen or target not in seen:
            raise GraphValidationError(
                f"edges[{index}] references an unknown endpoint: {edge!r}."
            )
        graph.add_edge(source, target)
    metadata = payload.get("metadata", {})
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise GraphValidationError("metadata must be an object when provided.")
        graph.graph.update(metadata)
    if str(graph.graph.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        from .operational_motif import build_path_closed_system

        return build_path_closed_system(graph).graph
    return validate_graph(graph)


def graph_json(graph: nx.Graph) -> dict[str, Any]:
    """Serialize a validated equipment graph in the documented input format."""

    if str(graph.graph.get("input_semantics", "")).startswith(
        "mpcf-directed-task-thread-mapping-v"
    ):
        from .operational_motif import build_path_closed_system

        equipment = build_path_closed_system(graph).graph
    else:
        equipment = validate_graph(graph)
    return {
        "nodes": [
            {"id": node, **dict(equipment.nodes[node])}
            for node in stable_nodes(equipment.nodes)
        ],
        "edges": [
            [source, target]
            for source, target in sorted(
                equipment.edges,
                key=lambda edge: (repr(edge[0]), repr(edge[1])),
            )
        ],
        "metadata": {
            key: value
            for key, value in equipment.graph.items()
            if key != "rmcd_roles"
        },
    }


def save_graph(graph: nx.Graph, path: str | Path) -> Path:
    """Write a validated equipment graph as strict UTF-8 JSON.

    The write is atomic: several panels may freeze the same deterministic
    instance concurrently (one budget shard per process), and a concurrent
    reader must never observe a half-written file.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(graph_json(graph), ensure_ascii=False, indent=2, allow_nan=False)
    text = payload + "\n"
    # The temporary name carries the PID: several panels may freeze the same
    # deterministic instance at the same moment, and a shared temporary name
    # would let one process rename away another process's file.
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(10):
        try:
            os.replace(temporary, output)
            return output
        except OSError:
            # On Windows a rename cannot replace a file another process holds
            # open.  Every writer produces identical bytes, so if the
            # destination already holds this exact graph the race is already
            # resolved in our favour.
            try:
                if output.exists() and output.read_text(encoding="utf-8") == text:
                    temporary.unlink(missing_ok=True)
                    return output
            except OSError:
                pass
            time.sleep(0.05 * (attempt + 1))
    # Content is deterministic, so a direct write is still correct.
    output.write_text(text, encoding="utf-8")
    temporary.unlink(missing_ok=True)
    return output


def _number(value: float | int | None) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return "INF" if value > 0 else "-INF"
    return value


def _solver_json(solver: SolverInfo) -> dict[str, Any]:
    return {
        "status": solver.status.value,
        "optimal": solver.optimal,
        "scipy_status": solver.scipy_status,
        "message": solver.message,
        "objective": _number(solver.objective),
        "lower_bound": _number(solver.lower_bound),
        "upper_bound": _number(solver.upper_bound),
        "mip_gap": _number(solver.mip_gap),
        "runtime_seconds": solver.runtime_seconds,
    }


def _cut_json(cut: CutCertificate | None) -> dict[str, Any] | None:
    if cut is None:
        return None
    return {
        "cut_nodes": list(cut.cut_nodes),
        "cut_capacity": cut.cut_capacity,
        "crossing_arcs": [
            {
                "source": {"kind": source.kind, "node": source.original},
                "target": {"kind": target.kind, "node": target.original},
                "capacity": capacity,
            }
            for source, target, capacity in cut.crossing_arcs
        ],
    }


def _rmcd_json(result: RMCDResult) -> dict[str, Any]:
    return {
        "threshold_K": result.threshold,
        "status": result.status.value,
        "optimal": result.optimal,
        "method": result.method,
        "attack_set": list(stable_nodes(result.attack_set)),
        "attack_cost": _number(result.attack_cost),
        "residual_capacity": result.residual_capacity,
        "protected": list(stable_nodes(result.protected)),
        "excluded": list(stable_nodes(result.excluded)),
        "cut_certificate": _cut_json(result.cut_certificate),
        "solver": _solver_json(result.solver),
    }


def _fraction_json(value: Fraction | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact": str(value),
        "float": float(value),
    }


def _pcd_json(result: PCDResult) -> dict[str, Any]:
    payload = _rmcd_json(result.rmcd)
    payload["pcd"] = {
        "initial_capacity": result.initial_capacity,
        "lower_bound": _number(result.lower_bound),
        "upper_bound": _number(result.upper_bound),
        "absolute_gap": _number(result.absolute_gap),
        "relative_gap": _number(result.relative_gap),
        "dual_master_upper": _number(result.dual_master_upper),
        "dual_search_closed": result.dual_search_closed,
        "certified_by_pcd": result.certified_by_pcd,
        "certification_source": result.certification_source,
        "exact_fallback_used": result.exact_fallback_used,
        "homogeneous_closed_form": result.homogeneous_closed_form,
        "objective_cost_quantum": _fraction_json(result.objective_cost_quantum),
        "lattice_closed": result.lattice_closed,
        "termination_reason": result.termination_reason,
        "oracle_calls": len(result.trace),
        "cut_pool_size": len(result.cut_pool),
        "line_pool_size": len(result.line_pool),
        "trace": [
            {
                "oracle_call": item.oracle_call,
                "lambda": _fraction_json(item.lambda_value),
                "dual_value": _fraction_json(item.dual_value),
                "master_upper_before": _fraction_json(item.master_upper_before),
                "cut_nodes": list(item.cut_nodes),
                "candidate_attack": list(stable_nodes(item.candidate_attack)),
                "candidate_cost": _fraction_json(item.candidate_cost),
                "incumbent_lower": _fraction_json(item.incumbent_lower),
                "incumbent_upper": _fraction_json(item.incumbent_upper),
                "new_line_count": item.new_line_count,
            }
            for item in result.trace
        ],
        "line_pool": [
            {
                "intercept": _fraction_json(line.intercept),
                "slope": line.slope,
                "cut_nodes": list(line.cut_nodes),
                "attacked_on_cut": list(stable_nodes(line.attacked_on_cut)),
            }
            for line in result.line_pool
        ],
        "cut_pool": [list(nodes) for nodes in result.cut_pool],
    }
    return payload


def _critical_family_json(result: Any) -> dict[str, Any]:
    return {
        "core": list(stable_nodes(result.core)),
        "core_exact": result.core_exact,
        "discovered_union": list(stable_nodes(result.discovered_union)),
        "discovered_shell": list(stable_nodes(result.discovered_shell)),
        "union_exact": result.union_exact,
        "union_membership_exact": result.union_membership_exact,
        "shell_exact": result.shell_exact,
        "enumeration_complete": result.enumeration_complete,
        "enumerated_optimal_sets": [
            list(stable_nodes(nodes)) for nodes in result.enumerated_optimal_sets
        ],
        "exclusion_records": [
            {
                "node": record.node,
                "restricted_cost": _number(record.restricted_cost),
                "exclusion_penalty": _number(record.exclusion_penalty),
                "restricted_status": record.restricted_status.value,
                "exact": record.exact,
            }
            for record in result.exclusion_records
        ],
        "inclusion_records": [
            {
                "node": record.node,
                "forced_cost": _number(record.forced_cost),
                "forced_status": record.forced_status.value,
                "in_optimal_union": record.in_optimal_union,
                "exact": record.exact,
            }
            for record in result.inclusion_records
        ],
    }


def _keyset_json(result: CriticalEquipmentSetResult) -> dict[str, Any]:
    solve = result.solve_result
    family = result.family_analysis
    if result.pcd_result is None:
        solver_result = _rmcd_json(solve)
        certification_source = "exact_milp" if solve.optimal else "none"
    else:
        solver_result = _pcd_json(result.pcd_result)
        certification_source = result.pcd_result.certification_source
    return {
        "problem": "minimum_cost_role_aware_collective_critical_equipment_set",
        "threshold_K": result.threshold,
        "initial_capacity": result.initial_capacity,
        "critical_set": list(stable_nodes(result.critical_set)),
        "critical_set_size": len(result.critical_set),
        "critical_set_cost": _number(result.critical_cost),
        "residual_capacity": result.residual_capacity,
        "capacity_breached": result.capacity_breached,
        "status": solve.status.value,
        "optimal": result.optimal,
        "method": solve.method,
        "certification_source": certification_source,
        "lower_bound": _number(solve.solver.lower_bound),
        "upper_bound": _number(solve.solver.upper_bound),
        "relative_gap": _number(solve.solver.mip_gap),
        "critical_nodes": [
            {
                "node": record.node,
                "role": record.role,
                "capacity": record.capacity,
                "attack_cost": record.attack_cost,
            }
            for record in result.nodes
        ],
        "role_composition": {
            role: {
                "node_count": dict(result.role_counts)[role],
                "capacity_sum": dict(result.role_capacity)[role],
            }
            for role in ("S", "C", "L", "E")
        },
        "optimal_family_requested": family is not None,
        "optimal_family": _critical_family_json(family) if family is not None else None,
        "solver_result": solver_result,
    }


def _frontier_json(result: FrontierResult) -> dict[str, Any]:
    return {
        "initial_capacity": result.initial_capacity,
        "complete": result.complete,
        "points": [
            {
                "max_remaining_capacity": point.max_remaining_capacity,
                "rmcd": _rmcd_json(point.result),
            }
            for point in result.points
        ],
        "nestedness_not_assumed": True,
    }


def _fortification_json(result: FortificationResult) -> dict[str, Any]:
    return {
        "threshold_K": result.threshold,
        "budget": result.budget,
        "status": result.status.value,
        "optimal": result.optimal,
        "protection_set": list(stable_nodes(result.protection_set)),
        "protection_cost": result.protection_cost,
        "defended_attack_cost": _number(result.defended_attack_cost),
        "censored_unbreakable": result.censored_unbreakable,
        "adaptive_attack": (
            _rmcd_json(result.adaptive_attack) if result.adaptive_attack else None
        ),
        "solver": _solver_json(result.solver),
        "bar_c_a": result.bar_c_a,
        "attack_pool": [list(stable_nodes(nodes)) for nodes in result.attack_pool],
        "trace": [
            {
                **asdict(item),
                "protection_set": list(stable_nodes(item.protection_set)),
                "oracle_status": item.oracle_status.value,
                "oracle_attack": list(stable_nodes(item.oracle_attack)),
            }
            for item in result.trace
        ],
    }


def _add_common_input(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Equipment-network JSON file.")
    parser.add_argument("--quiet", action="store_true", help="Disable progress output.")


def _add_per_solve_time_limit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--per-solve-time-limit",
        "--time-limit",
        dest="per_solve_time_limit",
        type=float,
        help=(
            "Seconds allowed for each MILP solve (legacy alias: --time-limit). "
            "This is not a whole-command deadline."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rmcd-f",
        description="Role-Motif Capacity Dismantling and adaptive Fortification.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser(
        "generate", help="Generate one preregistered synthetic equipment network."
    )
    generate.add_argument("--topology", choices=TOPOLOGIES, required=True)
    generate.add_argument(
        "--role-sizes",
        default="12,6,6,12",
        help="Comma-separated S,C,L,E node counts.",
    )
    generate.add_argument("--seed", type=int, default=11)
    generate.add_argument(
        "--edge-budget",
        type=int,
        required=True,
        help="Total number of legal inter-role edges.",
    )
    generate.add_argument(
        "--capacity-levels",
        default="1,2,3",
        help="Comma-separated positive integer concurrency capacities.",
    )
    generate.add_argument("--output", required=True, help="Output JSON path.")
    generate.add_argument("--quiet", action="store_true")

    capacity = subparsers.add_parser("capacity", help="Compute Omega_R and certificates.")
    _add_common_input(capacity)

    attack = subparsers.add_parser(
        "attack", help="Solve threshold RMCD by the exact oracle or RMCD-PCD."
    )
    _add_common_input(attack)
    attack.add_argument("--threshold", type=int, required=True, metavar="K")
    attack.add_argument(
        "--method", choices=("exact", "pcd"), default="exact"
    )
    attack.add_argument("--max-oracle-calls", type=int, default=64)
    attack.add_argument(
        "--pcd-fallback", choices=("none", "exact"), default="none"
    )
    _add_per_solve_time_limit(attack)

    identify = subparsers.add_parser(
        "identify",
        help="Identify the minimum-cost collective critical equipment-node set.",
    )
    _add_common_input(identify)
    identify.add_argument("--threshold", type=int, required=True, metavar="K")
    identify.add_argument("--method", choices=("exact", "pcd"), default="pcd")
    identify.add_argument("--max-oracle-calls", type=int, default=64)
    identify.add_argument(
        "--pcd-fallback", choices=("none", "exact"), default="none"
    )
    identify.add_argument(
        "--analyze-optimal-family",
        action="store_true",
        help="Exactly analyze the mandatory core and substitutable shell.",
    )
    identify.add_argument("--enumerate-limit", type=int, default=100)
    _add_per_solve_time_limit(identify)

    frontier = subparsers.add_parser("frontier", help="Compute the complete frontier.")
    _add_common_input(frontier)
    _add_per_solve_time_limit(frontier)

    criticality = subparsers.add_parser(
        "criticality", help="Compute exact exclusion penalties and critical core."
    )
    _add_common_input(criticality)
    criticality.add_argument("--threshold", type=int, required=True, metavar="K")
    criticality.add_argument("--enumerate-limit", type=int, default=100)
    _add_per_solve_time_limit(criticality)

    fortify = subparsers.add_parser("fortify", help="Solve adaptive RMCD-F by CCG.")
    _add_common_input(fortify)
    fortify.add_argument("--threshold", type=int, required=True, metavar="K")
    fortify.add_argument("--budget", type=float, required=True, metavar="BP")
    _add_per_solve_time_limit(fortify)
    fortify.add_argument("--max-iterations", type=int, default=10_000)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "generate":
        try:
            role_values = tuple(int(item) for item in args.role_sizes.split(","))
            capacity_levels = tuple(
                int(item) for item in args.capacity_levels.split(",")
            )
        except ValueError as exc:
            raise ValueError(
                "--role-sizes and --capacity-levels must contain integers."
            ) from exc
        if len(role_values) != 4:
            raise ValueError("--role-sizes must contain exactly S,C,L,E counts.")
        role_sizes = dict(zip(("S", "C", "L", "E"), role_values))
        graph = generate_equipment_network(
            args.topology,
            role_sizes,
            seed=args.seed,
            edge_budget=args.edge_budget,
            capacity_levels=capacity_levels,
        )
        output = save_graph(graph, args.output)
        _progress(
            f"generated {args.topology} equipment network at {output}",
            quiet=args.quiet,
        )
        return {
            "output": str(output),
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "metadata": graph.graph["synthetic"],
        }

    graph = load_graph(args.input)
    _progress(
        f"loaded {graph.number_of_nodes()} equipment nodes and "
        f"{graph.number_of_edges()} legal role edges",
        quiet=args.quiet,
    )
    if args.command == "capacity":
        _progress("computing role-motif packing capacity", quiet=args.quiet)
        result = motif_capacity(graph)
        return {
            "capacity": result.value,
            "attacked": list(stable_nodes(result.attacked)),
            "paths": [
                {"nodes": list(path.nodes), "amount": path.amount}
                for path in result.paths
            ],
            "cut_certificate": _cut_json(result.cut),
            "u_inf_flow": result.u_inf_flow,
        }
    if args.command == "attack":
        _progress(
            f"solving RMCD by {args.method} for K={args.threshold}",
            quiet=args.quiet,
        )
        if args.method == "pcd":
            return _pcd_json(
                solve_rmcd_pcd(
                    graph,
                    args.threshold,
                    max_oracle_calls=args.max_oracle_calls,
                    fallback=args.pcd_fallback,
                    time_limit=args.per_solve_time_limit,
                )
            )
        return _rmcd_json(
            solve_rmcd_exact(
                graph,
                args.threshold,
                time_limit=args.per_solve_time_limit,
            )
        )
    if args.command == "identify":
        _progress(
            f"identifying collective critical equipment set by {args.method} "
            f"for K={args.threshold}",
            quiet=args.quiet,
        )
        return _keyset_json(
            identify_critical_equipment_set(
                graph,
                args.threshold,
                method=args.method,
                max_oracle_calls=args.max_oracle_calls,
                pcd_fallback=args.pcd_fallback,
                time_limit_per_solve=args.per_solve_time_limit,
                analyze_optimal_family=args.analyze_optimal_family,
                enumerate_limit=args.enumerate_limit,
                progress=lambda message: _progress(message, quiet=args.quiet),
            )
        )
    if args.command == "frontier":
        _progress("solving every frontier threshold independently", quiet=args.quiet)
        return _frontier_json(
            solve_frontier(
                graph,
                time_limit=args.per_solve_time_limit,
                progress=lambda message: _progress(message, quiet=args.quiet),
            )
        )
    if args.command == "criticality":
        _progress("computing exclusion penalties and optimal-set union", quiet=args.quiet)
        result = analyze_exclusion(
            graph,
            args.threshold,
            enumerate_limit=args.enumerate_limit,
            time_limit_per_solve=args.per_solve_time_limit,
            progress=lambda message: _progress(message, quiet=args.quiet),
        )
        return {
            "threshold_K": result.threshold,
            **_critical_family_json(result),
            "base_result": _rmcd_json(result.base_result),
        }
    if args.command == "fortify":
        _progress(
            f"solving adaptive RMCD-F for K={args.threshold}, Bp={args.budget}",
            quiet=args.quiet,
        )
        return _fortification_json(
            solve_rmcd_f(
                graph,
                args.threshold,
                args.budget,
                time_limit_per_solve=args.per_solve_time_limit,
                max_iterations=args.max_iterations,
                progress=lambda message: _progress(message, quiet=args.quiet),
            )
        )
    raise AssertionError(f"Unhandled command: {args.command}")


def _completion_status(
    command: str, payload: dict[str, Any]
) -> tuple[bool, str | None]:
    """Return whether a JSON result carries the command's full certificate.

    Exit code 4 is deliberately separate from input/runtime failures: a
    time-limited incumbent or bounded enumeration is useful output, but it is
    not a complete result for unattended academic experiments.
    """

    if command in {"capacity", "generate"}:
        return True, None
    if command == "attack":
        complete = bool(payload.get("optimal"))
        return complete, None if complete else "RMCD was not globally certified"
    if command == "identify":
        complete = bool(payload.get("optimal"))
        family = payload.get("optimal_family")
        if complete and family is not None:
            complete = bool(
                family.get("core_exact")
                and (
                    family.get("union_exact")
                    or family.get("enumeration_complete")
                )
            )
        return (
            complete,
            None
            if complete
            else "critical set or requested optimal-family analysis is incomplete",
        )
    if command == "frontier":
        complete = bool(payload.get("complete"))
        return complete, None if complete else "the RMCD frontier is incomplete"
    if command == "criticality":
        complete = bool(
            payload.get("core_exact")
            and (
                payload.get("union_exact")
                or payload.get("enumeration_complete")
            )
        )
        return (
            complete,
            None
            if complete
            else "critical core or optimal-set union is incomplete",
        )
    if command == "fortify":
        complete = bool(payload.get("optimal"))
        return complete, None if complete else "RMCD-F was not globally certified"
    raise AssertionError(f"Unhandled command: {command}")


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        payload = _run(args)
    except (
        GraphValidationError,
        ValueError,
        OverflowError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"rmcd-f: error: {exc}", file=sys.stderr)
        return EXIT_INPUT_ERROR
    except RuntimeError as exc:
        print(f"rmcd-f: solver error: {exc}", file=sys.stderr)
        return EXIT_SOLVER_ERROR
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    complete, reason = _completion_status(args.command, payload)
    if not complete:
        print(
            f"rmcd-f: incomplete: {reason}; inspect the emitted JSON status and bounds.",
            file=sys.stderr,
            flush=True,
        )
        return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
