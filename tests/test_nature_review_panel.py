from __future__ import annotations

from rmcd_f.nature_review_panel import (
    NATURE_REVIEW_TABLE1_ALL_OFFICIAL_VARIANTS,
    NATURE_REVIEW_TABLE1_CANONICAL_METHODS,
    NATURE_REVIEW_TABLE1_FAMILIES,
)
from rmcd_f.official_review import OFFICIAL_METHOD_GROUPS, OFFICIAL_METHOD_SPECS


def test_nature_review_table1_has_exactly_ten_families() -> None:
    assert len(NATURE_REVIEW_TABLE1_FAMILIES) == 10
    assert {family.family for family in NATURE_REVIEW_TABLE1_FAMILIES} == {
        "CI",
        "BPD",
        "MinSum",
        "GND",
        "EGND",
        "CoreHD",
        "EI",
        "GDM",
        "CoreGDM",
        "FINDER",
    }


def test_bpd_reference_is_transparent_local_code_not_an_official_adapter() -> None:
    assert "BPDReference" in NATURE_REVIEW_TABLE1_CANONICAL_METHODS
    assert "BPDReference" not in OFFICIAL_METHOD_SPECS
    assert "BPDReference" not in NATURE_REVIEW_TABLE1_ALL_OFFICIAL_VARIANTS


def test_official_aliases_match_the_preregistered_table1_panel() -> None:
    assert OFFICIAL_METHOD_GROUPS["nature_table1_variants"] == (
        NATURE_REVIEW_TABLE1_ALL_OFFICIAL_VARIANTS
    )
    assert set(OFFICIAL_METHOD_GROUPS["nature_table1"]) == (
        set(NATURE_REVIEW_TABLE1_CANONICAL_METHODS) - {"BPDReference"}
    )
