"""Prespecified seed-extension robustness panel.

Kept in its own module, outside the numerical-core fingerprint, because it only
*adds* a panel: it does not change the main, role, scaling or heterogeneity
panel definitions, so the code revision of an already-running experiment stays
valid.  The builder reuses the main panel's construction path unchanged, so the
new graphs are identical to the originals in topology, size, role composition,
capacity, cost definition and budget rule -- only the generator seed differs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .panels import (
    DEFAULT_BUDGET_RATIOS,
    DEFAULT_CAPACITY_LEVELS,
    DEFAULT_COST_LEVELS,
    MAIN_TIERS,
    TOPOLOGIES,
    InstanceSpec,
    ScaleTier,
    _finalize,
)
from ..cost_profiles import apply_attack_cost_profile
from ..synthetic import generate_equipment_network

import json

#: Additional seeds for the seed-extension robustness panel.
#:
#: These were fixed **before** any of their results were examined, and are
#: recorded as a code constant so the archive itself is the evidence: the set is
#: not a post-hoc selection.  Together with ``panels.DEFAULT_SEEDS`` they give
#: ten random realizations per topology-size cell (90 graphs), while the original
#: 45 graphs remain the frozen full-method benchmark.
SEED_EXTENSION_SEEDS: tuple[int, ...] = (66, 77, 88, 99, 110)

#: Methods selected for the extension.  The purpose is to test whether the
#: principal conclusions survive additional random realizations, not to rebuild
#: the complete benchmark, so the panel carries the three MPCF solvers, the
#: mission-aligned heuristics and one representative structural dismantling
#: transfer instead of all comparison variants.
SEED_EXTENSION_METHODS: tuple[str, ...] = (
    "MPCF-Exact",
    "MPCF-CG",
    "MPCF-Greedy",
    "NoProtection",
    "BetweennessProtect",
    "PathFrequencyProtect",
    "InitialPathCutProtect",
    "BPDReference-Protect",
    "OfficialGND-Protect",
)


def build_seed_extension_panel(
    output: Path,
    *,
    seeds: Sequence[int] = SEED_EXTENSION_SEEDS,
    tiers: Sequence[ScaleTier] = MAIN_TIERS,
    topologies: Sequence[str] = TOPOLOGIES,
    budget_ratios: Sequence[float] = DEFAULT_BUDGET_RATIOS,
    capacity_levels: Sequence[int] = DEFAULT_CAPACITY_LEVELS,
    cost_levels: Sequence[int] = DEFAULT_COST_LEVELS,
    fortification_multiplier: float = 1.0,
) -> list[InstanceSpec]:
    """Prespecified additional random realizations of the main panel."""

    root = Path(output) / "instances" / "seedext"
    root.mkdir(parents=True, exist_ok=True)
    specs: list[InstanceSpec] = []
    for tier in tiers:
        for topology in topologies:
            for seed in seeds:
                graph = generate_equipment_network(
                    topology,
                    dict(zip(("S", "C", "L", "E"), tier.role_sizes)),
                    seed=int(seed),
                    edge_budget=int(tier.edge_budget),
                    capacity_levels=tuple(int(value) for value in capacity_levels),
                )
                graph = apply_attack_cost_profile(
                    graph,
                    "balanced-heterogeneous",
                    cost_levels=tuple(int(value) for value in cost_levels),
                )
                specs.append(
                    _finalize(
                        experiment="seedext",
                        graph=graph,
                        instance_file=root / f"{topology}_seed{seed}_{tier.name}.json",
                        topology=topology,
                        seed=int(seed),
                        composition=tier.composition,
                        scale_tier=tier.name,
                        budget_ratios=budget_ratios,
                        attack_cost_profile="balanced-heterogeneous",
                        attack_cost_levels=cost_levels,
                        protection_cost_profile="unit",
                        protection_cost_levels=(1,),
                        uplift_multiplier=fortification_multiplier,
                        extra={
                            "role_sizes": json.dumps(list(tier.role_sizes)),
                            "panel_role": "seed_extension_robustness",
                        },
                    )
                )
    return specs
