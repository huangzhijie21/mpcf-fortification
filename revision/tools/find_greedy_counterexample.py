#!/usr/bin/env python3
"""Search for a minimal greedy-vs-exact counterexample.

The revision must ship a unit test asserting that one-step exact-marginal
greedy is strictly suboptimal on a tiny instance.  Rather than assert a
remembered number, this enumerates small directed task graphs -- with fixed
non-removable S/E boundaries and a complete C x L interface -- and reports
instances where ``greedy < exact``, preferring the target pair
``greedy = 3``, ``exact = 4``.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rmcd_f.operational_motif import build_path_closed_system  # noqa: E402
from rmcd_f.task_path_fortification import (  # noqa: E402
    solve_mpcf_exact,
    solve_mpcf_greedy,
)

TARGET = (3.0, 4.0)
ATTACK_LEVELS = (1, 2, 3)
UPLIFT_LEVELS = (1, 2, 3)


def build(
    c_nodes: tuple[tuple[int, int], ...],
    l_nodes: tuple[tuple[int, int], ...],
) -> nx.DiGraph:
    """``(attack_cost, uplift)`` per removable C / L node; boundaries fixed."""

    graph = nx.DiGraph()
    graph.add_node("s", role="S", capacity=1, attack_cost=1.0, protect_cost=1.0, removable=False)
    graph.add_node("t", role="E", capacity=1, attack_cost=1.0, protect_cost=1.0, removable=False)
    for index, (attack, _) in enumerate(c_nodes):
        graph.add_node(f"c{index}", role="C", capacity=1, attack_cost=float(attack), protect_cost=1.0)
    for index, (attack, _) in enumerate(l_nodes):
        graph.add_node(f"l{index}", role="L", capacity=1, attack_cost=float(attack), protect_cost=1.0)
    for index in range(len(c_nodes)):
        graph.add_edge("s", f"c{index}")
    for c in range(len(c_nodes)):
        for l in range(len(l_nodes)):
            graph.add_edge(f"c{c}", f"l{l}")
    for index in range(len(l_nodes)):
        graph.add_edge(f"l{index}", "t")
    return graph


def main() -> int:
    hits: list[tuple] = []
    shapes = (
        (2, 1),  # 2 command nodes + 1 relay node  = 3 removable
        (1, 2),  # 1 command node + 2 relay nodes  = 3 removable
    )
    for n_c, n_l in shapes:
        for c_spec in itertools.product(
            itertools.product(ATTACK_LEVELS, UPLIFT_LEVELS), repeat=n_c
        ):
            for l_spec in itertools.product(
                itertools.product(ATTACK_LEVELS, UPLIFT_LEVELS), repeat=n_l
            ):
                graph = build(c_spec, l_spec)
                try:
                    system = build_path_closed_system(graph)
                except Exception:
                    continue
                uplifts = {
                    **{f"c{i}": float(spec[1]) for i, spec in enumerate(c_spec)},
                    **{f"l{i}": float(spec[1]) for i, spec in enumerate(l_spec)},
                }
                for budget in (1.0, 2.0, 3.0):
                    try:
                        greedy = solve_mpcf_greedy(graph, budget, uplifts=uplifts)
                        exact = solve_mpcf_exact(graph, budget, uplifts=uplifts)
                    except Exception:
                        continue
                    if greedy.defended_margin is None:
                        continue
                    if greedy.defended_margin < exact.defended_margin - 1e-9:
                        hits.append(
                            (
                                (float(exact.defended_margin), float(greedy.defended_margin)),
                                budget,
                                c_spec,
                                l_spec,
                                tuple(sorted(greedy.protection_set)),
                                tuple(sorted(exact.protection_set)),
                            )
                        )
    hits.sort(key=lambda item: (0 if item[0] == TARGET else 1, item[0][0], item[1], item[2], item[3]))
    print(f"found {len(hits)} counterexamples")
    seen: set = set()
    shown = 0
    for hit in hits:
        key = (hit[0], hit[1])
        if key in seen:
            continue
        seen.add(key)
        print(
            f"  exact={hit[0][0]:.0f} greedy={hit[0][1]:.0f} B={hit[1]:.0f} "
            f"C={hit[2]} L={hit[3]} greedy_set={hit[4]} exact_set={hit[5]}"
        )
        shown += 1
        if shown >= 20:
            break
    print(f"exact target {TARGET}: {len([h for h in hits if h[0] == TARGET])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
