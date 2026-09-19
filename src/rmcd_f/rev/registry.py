"""Explicit baseline variant registry.

The manuscript claims 36 comparison variants while its main table listed 22
method names, because several names silently stood for multiple execution
variants (static / dynamic / reinsertion / cost-conversion).  This module makes
every variant a first-class row with its own ``variant_id`` and writes the
audit table directly usable as a supplementary table.

The comparison set is defined constructively and asserted to be exactly 36:

``45 declared methods``
    minus the two certified MPCF solvers (``MPCF-Exact``, ``MPCF-CG``), which
    are the reference rather than a comparison, and
    minus the seven variants that are not applicable on the frozen main panel
    (disconnected projection, eigenvector periodicity, upstream brute-force
    node limit, and the two reinsertion variants that never reached full
    coverage) --
equals 36 comparison variants: ``MPCF-Greedy`` + 9 local + 26 official.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from ..nature_review_panel import family_for_method
from ..official_review import OFFICIAL_METHOD_SPECS
from ..protection_baselines import LOCAL_PROTECTION_METHODS, protection_method_name

REGISTRY_VERSION = "1.0"

REVIEW_DOI = "10.1038/s42254-023-00676-y"
REVIEW_SOURCE = "NetworkDismantling/review (Artime et al., Nat. Rev. Phys. 6, 114-131, 2024)"

DIRECTED_TASK_GRAPH = "directed typed task graph (full task-path family)"
UNDIRECTED_PROJECTION = "simple undirected physical-equipment projection"

#: Variants excluded from the frozen 45-graph main panel, with the reason that
#: must appear in the supplementary audit table.
MAIN_PANEL_EXCLUDED: Mapping[str, str] = {
    "OfficialEGND": "requires a connected projection; not applicable on sparse frozen panels",
    "OfficialEI_S2": "requires a connected projection (sigma=2)",
    "OfficialEigenvectorStatic": "bipartite power-iteration periodicity; no stable ranking",
    "OfficialEigenvectorDynamic": "bipartite power-iteration periodicity; no stable ranking",
    "OfficialNetworkEntanglementSmallR": "incomplete coverage on the frozen panel; conditional only",
    "OfficialNetworkEntanglementMidR": "incomplete coverage on the frozen panel; conditional only",
    "ReviewBruteForceFrequencyRank": "upstream exhaustive enumeration restricted to n<=18",
}

#: The two certified solvers are the comparison reference, not a variant.
REFERENCE_METHODS: tuple[str, ...] = ("MPCF-Exact", "MPCF-CG")

EXPECTED_COMPARISON_VARIANTS = 36
EXPECTED_DECLARED_METHODS = 45

#: Cost-conversion policies applied to structural score rankings under
#: heterogeneous protection cost (``rmcd_f.mpcf_supplementary``).
COST_CONVERSION_POLICIES: tuple[str, ...] = (
    "raw",
    "per_cost",
    "prefix_knapsack",
    "full_knapsack",
)
COST_CONVERSION_SCORES: tuple[str, ...] = ("degree", "betweenness")
POLICY_LABELS: Mapping[str, str] = {
    "raw": "Raw",
    "per_cost": "PerCost",
    "prefix_knapsack": "PrefixKnapsack",
    "full_knapsack": "FullKnapsack",
}
SCORE_LABELS: Mapping[str, str] = {"degree": "Degree", "betweenness": "Betweenness"}


@dataclass(frozen=True)
class VariantRecord:
    """One auditable baseline variant (one row of Supplementary Table)."""

    variant_id: str
    variant_name: str
    base_method: str
    method_class: str
    reference: str
    code_source: str
    native_graph: str
    directed_handling: str
    node_weight_handling: str
    execution_mode: str
    cost_conversion: str
    parameters: str
    runtime_definition: str
    family: str = ""
    dependency_environment: str = "control"
    experimental_upstream: int = 0
    derived_output: int = 0
    predeclared_panel: str = "main"
    in_comparison_set: int = 1
    declared_universe_size: int = EXPECTED_DECLARED_METHODS
    coverage_n: int = 0
    registry_version: str = REGISTRY_VERSION


def _reference_for(source_method: str) -> str:
    family = family_for_method(source_method)
    if family is not None and family.original_doi:
        return f"{family.display_name}: doi:{family.original_doi}"
    if source_method == "BPDReference":
        return "BPD reference: doi:10.1103/PhysRevE.94.012305"
    return f"review catalog: doi:{REVIEW_DOI}"


def _execution_mode(spec: Any) -> str:
    if spec is None:
        return "in_package_ranking"
    if spec.kind == "heuristic_static":
        return "static"
    if spec.kind == "heuristic_dynamic":
        return "dynamic"
    if spec.kind == "bruteforce_target":
        return "derived_frequency"
    if spec.dependency:
        return "reinsertion"
    return "native_sequence"


def _parameters(spec: Any) -> str:
    payload: dict[str, Any] = {}
    if spec is not None:
        payload["dependency_method"] = spec.dependency
        payload["dependency_argument"] = spec.dependency_argument
        payload["required_modules"] = list(spec.required_modules)
        payload["preparation"] = spec.preparation
        payload["kind"] = spec.kind
    return _json(payload)


def _json(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _projection_fields() -> tuple[str, str, str]:
    return (
        UNDIRECTED_PROJECTION,
        "edge direction dropped; node sequence only, replayed as a protection ranking",
        "unit node weights; concurrency capacities ignored by the upstream ranking",
    )


def declared_variants() -> list[VariantRecord]:
    """All 45 declared methods, each as an explicit variant row."""

    rows: list[VariantRecord] = []

    # --- the three MPCF solvers -------------------------------------------
    rows.append(
        VariantRecord(
            variant_id="mpcf-exact",
            variant_name="MPCF-Exact",
            base_method="MPCF",
            method_class="proposed",
            reference="this work: compact node-split max-flow MILP",
            code_source="rmcd_f.task_path_fortification.solve_mpcf_exact",
            native_graph=DIRECTED_TASK_GRAPH,
            directed_handling="directions preserved; complete directed task-path family",
            node_weight_handling="finite attack cost per removable node; finite uplift delta_v",
            execution_mode="exact",
            cost_conversion="none",
            parameters=_json({"time_limit_s": "unified", "lexicographic_tie_break": True}),
            runtime_definition="solver wall time incl. modelling and lexicographic tie-break",
            dependency_environment="control",
            in_comparison_set=0,
        )
    )
    rows.append(
        VariantRecord(
            variant_id="mpcf-cg",
            variant_name="MPCF-CG",
            base_method="MPCF",
            method_class="proposed",
            reference="this work: cut-generation master with exact PathCut separation",
            code_source="rmcd_f.task_path_fortification.solve_mpcf_cut_generation",
            native_graph=DIRECTED_TASK_GRAPH,
            directed_handling="directions preserved; complete directed task-path family",
            node_weight_handling="finite attack cost per removable node; finite uplift delta_v",
            execution_mode="exact_cut_generation",
            cost_conversion="none",
            parameters=_json({"time_limit_per_solve": "unified", "max_iterations": 10000}),
            runtime_definition="sum of master solves plus separation oracle calls",
            dependency_environment="control",
            in_comparison_set=0,
        )
    )
    rows.append(
        VariantRecord(
            variant_id="mpcf-greedy",
            variant_name="MPCF-Greedy",
            base_method="MPCF",
            method_class="proposed",
            reference="this work: one-step exact-marginal greedy ablation",
            code_source="rmcd_f.task_path_fortification.solve_mpcf_greedy",
            native_graph=DIRECTED_TASK_GRAPH,
            directed_handling="directions preserved; complete directed task-path family",
            node_weight_handling="finite attack cost per removable node; finite uplift delta_v",
            execution_mode="heuristic",
            cost_conversion="none",
            parameters=_json({"score": "exact marginal kappa gain / protect_cost"}),
            runtime_definition="sum of all candidate PathCut evaluations plus bookkeeping",
            dependency_environment="control",
        )
    )

    # --- nine local protection baselines ----------------------------------
    rows.extend(_local_variants())

    # --- 33 official / derived upstream methods ---------------------------
    rows.extend(_official_variants())

    expected = EXPECTED_DECLARED_METHODS
    if len(rows) != expected:
        raise AssertionError(
            f"Declared variant universe changed: expected {expected}, built {len(rows)}."
        )
    return rows


def _local_variants() -> list[VariantRecord]:
    descriptions: Mapping[str, tuple[str, str, str]] = {
        "NoProtection": ("empty protection set reference", "reference", "none"),
        "RandomProtect": ("uniform random ranking with a fixed seed", "heuristic", "none"),
        "DegreeProtect": ("static degree ranking on the undirected projection", "static", "none"),
        "BetweennessProtect": ("static betweenness ranking (normalized)", "static", "none"),
        "KCoreProtect": ("k-core number ranking", "static", "none"),
        "PageRankProtect": ("PageRank ranking", "static", "none"),
        "PathFrequencyProtect": (
            "task-path occurrence frequency ranking",
            "task_aware_static",
            "none",
        ),
        "InitialPathCutProtect": (
            "undefended optimal PathCut node ranking",
            "task_aware_static",
            "none",
        ),
        "BPDReference-Protect": (
            "transparent in-package belief-propagation decimation reference",
            "in_package_sequence",
            "none",
        ),
    }
    rows: list[VariantRecord] = []
    for method in LOCAL_PROTECTION_METHODS:
        description, mode, conversion = descriptions[method]
        native = (
            DIRECTED_TASK_GRAPH
            if method in {"NoProtection", "PathFrequencyProtect", "InitialPathCutProtect", "BPDReference-Protect"}
            else UNDIRECTED_PROJECTION
        )
        directed = (
            "directions preserved for task-path counting"
            if native == DIRECTED_TASK_GRAPH
            else "edge direction dropped before ranking"
        )
        rows.append(
            VariantRecord(
                variant_id="local-" + method.lower().replace("_", "-"),
                variant_name=method,
                base_method=method.replace("-Protect", ""),
                method_class="local_baseline",
                reference="in-package transparent baseline",
                code_source="rmcd_f.protection_baselines.local_protection_sequence",
                native_graph=native,
                directed_handling=directed,
                node_weight_handling="ranking ignores attack cost; budget applied by cost-aware scan",
                execution_mode=mode,
                cost_conversion=conversion,
                parameters=_json({"description": description}),
                runtime_definition="ranking construction time only; shared evaluation excluded",
                family="BPD" if method == "BPDReference-Protect" else "",
            )
        )
    return rows


def _official_variants() -> list[VariantRecord]:
    native, directed, weights = _projection_fields()
    rows: list[VariantRecord] = []
    for source, spec in OFFICIAL_METHOD_SPECS.items():
        family = family_for_method(source)
        excluded = source in MAIN_PANEL_EXCLUDED
        row = VariantRecord(
            variant_id="official-" + source.lower().replace("_", "-"),
            variant_name=protection_method_name(source),
            base_method=source,
            method_class="official_ranking_transfer",
            reference=_reference_for(source),
            code_source=f"{spec.module}.{spec.function}",
            native_graph=native,
            directed_handling=directed,
            node_weight_handling=weights,
            execution_mode=_execution_mode(spec),
            cost_conversion="none",
            parameters=_parameters(spec),
            runtime_definition=(
                "upstream sequence export time; protection-set selection and the shared "
                "exact min-cut evaluation are timed separately in the control environment"
            ),
            family=family.family if family else "",
            dependency_environment=spec.environment,
            experimental_upstream=int(spec.experimental_upstream),
            derived_output=int(spec.derived_output),
            predeclared_panel="conditional" if excluded else "main",
            in_comparison_set=0 if excluded else 1,
        )
        rows.append(row)
    return rows


def cost_conversion_variants(
    *,
    scores: Sequence[str] = COST_CONVERSION_SCORES,
    policies: Sequence[str] = COST_CONVERSION_POLICIES,
    prefix_multipliers: Sequence[int] = (3,),
) -> list[VariantRecord]:
    """Explicit rows for the heterogeneous-cost conversion family.

    These variants share a structural score and differ only in how the score is
    converted into a budget-feasible set; the conversion is a recorded field
    rather than a hidden code branch.
    """

    rows: list[VariantRecord] = []
    for score in scores:
        for policy in policies:
            if policy == "prefix_knapsack":
                for multiplier in prefix_multipliers:
                    rows.append(
                        _conversion_row(score, policy, multiplier=multiplier)
                    )
            else:
                rows.append(_conversion_row(score, policy, multiplier=None))
    return rows


def _conversion_row(score: str, policy: str, *, multiplier: int | None) -> VariantRecord:
    label = SCORE_LABELS.get(score, score.title())
    policy_label = POLICY_LABELS[policy]
    suffix = f"{multiplier}" if policy == "prefix_knapsack" and multiplier != 3 else ""
    name = f"{label}-{policy_label}{suffix}"
    parameters: dict[str, Any] = {"score": score, "policy": policy}
    if multiplier is not None:
        parameters["prefix_multiplier"] = multiplier
    return VariantRecord(
        variant_id=f"conv-{score}-{policy}" + (f"-x{multiplier}" if multiplier else ""),
        variant_name=name,
        base_method=label,
        method_class="heterogeneous_cost_conversion",
        reference="this work: cost-conversion control under heterogeneous protection cost",
        code_source="rmcd_f.mpcf_supplementary.select_score_policy",
        native_graph=UNDIRECTED_PROJECTION,
        directed_handling="edge direction dropped before scoring",
        node_weight_handling="per-node protection cost b_v drives conversion",
        execution_mode="conversion",
        cost_conversion=policy,
        parameters=_json(parameters),
        runtime_definition="ranking, conversion and knapsack DP time; shared evaluation excluded",
        predeclared_panel="supplementary",
        in_comparison_set=0,
    )


def comparison_variants() -> list[VariantRecord]:
    """Exactly the 36 pre-specified comparison variants of the main panel."""

    rows = [row for row in declared_variants() if row.in_comparison_set]
    if len(rows) != EXPECTED_COMPARISON_VARIANTS:
        raise AssertionError(
            "Comparison variant set must contain exactly "
            f"{EXPECTED_COMPARISON_VARIANTS} variants; built {len(rows)}."
        )
    return rows


def comparison_variant_names() -> tuple[str, ...]:
    return tuple(row.variant_name for row in comparison_variants())


def registry_rows(
    coverage: Mapping[str, int] | None = None,
    *,
    include_conversions: bool = True,
    prefix_multipliers: Sequence[int] = (3,),
) -> list[dict[str, Any]]:
    """Full registry table, optionally annotated with observed coverage."""

    rows = declared_variants()
    if include_conversions:
        rows = rows + cost_conversion_variants(prefix_multipliers=prefix_multipliers)
    coverage = dict(coverage or {})
    output: list[dict[str, Any]] = []
    for row in rows:
        payload = asdict(row)
        payload["coverage_n"] = int(coverage.get(row.variant_name, 0))
        payload["exclusion_reason"] = MAIN_PANEL_EXCLUDED.get(row.base_method, "")
        output.append(payload)
    return output


REGISTRY_COLUMNS: tuple[str, ...] = (
    "variant_id",
    "variant_name",
    "base_method",
    "method_class",
    "reference",
    "code_source",
    "native_graph",
    "directed_handling",
    "node_weight_handling",
    "execution_mode",
    "cost_conversion",
    "parameters",
    "coverage_n",
    "runtime_definition",
    "family",
    "dependency_environment",
    "experimental_upstream",
    "derived_output",
    "predeclared_panel",
    "in_comparison_set",
    "declared_universe_size",
    "exclusion_reason",
    "registry_version",
)


def validate_against_paper_claims() -> dict[str, int]:
    """Machine-readable reconciliation of the 22 / 36 / 45 numbers."""

    declared = declared_variants()
    comparison = comparison_variants()
    conversions = cost_conversion_variants()
    return {
        "declared_methods": len(declared),
        "certified_reference_methods": len(REFERENCE_METHODS),
        "main_panel_variants": sum(1 for row in declared if row.in_comparison_set) + len(REFERENCE_METHODS),
        "comparison_variants": len(comparison),
        "excluded_from_main_panel": len(MAIN_PANEL_EXCLUDED),
        "heterogeneous_cost_conversion_variants": len(conversions),
        "registry_rows_total": len(declared) + len(conversions),
    }
