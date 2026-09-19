"""Preregistered dismantling panels from Artime et al. (2024).

The Nature Reviews Physics Table 1 names ten algorithm families.  The
accompanying review repository exposes several reinsertion and parameter
variants but omits BPD source code.  These constants keep three concepts
separate: a canonical ten-family panel, every locally available Table-1
variant, and the wider repository inventory used only as supplementary
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass


NATURE_REVIEW_DOI = "10.1038/s42254-023-00676-y"
NATURE_REVIEW_REPOSITORY = "https://github.com/NetworkDismantling/review"


@dataclass(frozen=True)
class NatureReviewFamily:
    family: str
    display_name: str
    algorithm_type: str
    canonical_method: str
    variants: tuple[str, ...]
    original_doi: str
    implementation_source: str
    conditional_reason: str = ""


NATURE_REVIEW_TABLE1_FAMILIES: tuple[NatureReviewFamily, ...] = (
    NatureReviewFamily(
        "CI",
        "Collective Influence",
        "influence maximization",
        "OfficialCI_L2",
        ("OfficialCI_L1", "OfficialCI_L2", "OfficialCI_L3"),
        "10.1038/srep30062",
        "NetworkDismantling/review",
    ),
    NatureReviewFamily(
        "BPD",
        "Belief Propagation-guided Decimation",
        "message-passing decycling",
        "BPDReference",
        ("BPDReference",),
        "10.1103/PhysRevE.94.012305",
        "transparent in-package reference; source absent from review repository",
    ),
    NatureReviewFamily(
        "MinSum",
        "Min-Sum",
        "message-passing decycling",
        "OfficialMinSumR",
        ("OfficialMinSum", "OfficialMinSumR"),
        "10.1073/pnas.1605374113",
        "NetworkDismantling/review",
    ),
    NatureReviewFamily(
        "GND",
        "Generalized Network Dismantling",
        "spectral partitioning",
        "OfficialGNDR",
        ("OfficialGND", "OfficialGNDR"),
        "10.1073/pnas.1806108116",
        "NetworkDismantling/review",
    ),
    NatureReviewFamily(
        "EGND",
        "Ensemble GND",
        "spectral partitioning ensemble",
        "OfficialEGND",
        ("OfficialEGND",),
        "10.1007/978-3-030-36687-2_65",
        "NetworkDismantling/review",
        "requires a connected projection and may be resource intensive",
    ),
    NatureReviewFamily(
        "CoreHD",
        "CoreHD",
        "degree-based decycling",
        "OfficialCoreHD",
        ("OfficialCoreHD",),
        "10.1038/srep37954",
        "NetworkDismantling/review",
    ),
    NatureReviewFamily(
        "EI",
        "Explosive Immunization",
        "explosive percolation",
        "OfficialEI_S1",
        ("OfficialEI_S1", "OfficialEI_S2"),
        "10.1103/PhysRevLett.117.208301",
        "NetworkDismantling/review",
        "sigma=2 requires a connected projection in the bundled implementation",
    ),
    NatureReviewFamily(
        "GDM",
        "Graph Dismantling Machine",
        "geometric machine learning",
        "OfficialGDMR",
        ("OfficialGDM", "OfficialGDMR"),
        "10.1038/s41467-021-25485-8",
        "NetworkDismantling/review",
        "requires the frozen GDM model and graph-learning environment",
    ),
    NatureReviewFamily(
        "CoreGDM",
        "CoreGDM",
        "geometric machine learning on the 2-core",
        "OfficialCoreGDM",
        ("OfficialCoreGDM",),
        "10.1007/978-3-031-28276-8_8",
        "NetworkDismantling/review experimental upstream adapter",
        "requires a non-empty 2-core and remains experimental upstream",
    ),
    NatureReviewFamily(
        "FINDER",
        "FINDER",
        "deep reinforcement learning",
        "OfficialFINDER_R",
        ("OfficialFINDER_R",),
        "10.1038/s42256-020-0177-2",
        "NetworkDismantling/review plus complete FINDER source/model bundle",
        "requires the isolated legacy FINDER environment",
    ),
)


NATURE_REVIEW_TABLE1_CANONICAL_METHODS: tuple[str, ...] = tuple(
    family.canonical_method for family in NATURE_REVIEW_TABLE1_FAMILIES
)
NATURE_REVIEW_TABLE1_CANONICAL_OFFICIAL_METHODS: tuple[str, ...] = tuple(
    method
    for method in NATURE_REVIEW_TABLE1_CANONICAL_METHODS
    if method != "BPDReference"
)
NATURE_REVIEW_TABLE1_ALL_VARIANTS: tuple[str, ...] = tuple(
    dict.fromkeys(
        method
        for family in NATURE_REVIEW_TABLE1_FAMILIES
        for method in family.variants
    )
)
NATURE_REVIEW_TABLE1_ALL_OFFICIAL_VARIANTS: tuple[str, ...] = tuple(
    method for method in NATURE_REVIEW_TABLE1_ALL_VARIANTS
    if method != "BPDReference"
)


def family_for_method(method: str) -> NatureReviewFamily | None:
    for family in NATURE_REVIEW_TABLE1_FAMILIES:
        if method in family.variants:
            return family
    return None


def table1_inventory_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in NATURE_REVIEW_TABLE1_FAMILIES:
        for method in family.variants:
            rows.append(
                {
                    "family": family.family,
                    "display_name": family.display_name,
                    "algorithm_type": family.algorithm_type,
                    "method": method,
                    "canonical_representative": int(method == family.canonical_method),
                    "original_doi": family.original_doi,
                    "implementation_source": family.implementation_source,
                    "conditional_reason": family.conditional_reason,
                    "review_doi": NATURE_REVIEW_DOI,
                    "review_repository": NATURE_REVIEW_REPOSITORY,
                }
            )
    return rows
