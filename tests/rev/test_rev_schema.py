"""Schema, identifier and variant-registry tests for the revision layer."""

from __future__ import annotations

import json

import pytest

from rmcd_f.rev import registry as R
from rmcd_f.rev.ids import (
    SEPARATOR,
    base_id,
    code_commit,
    composition_label,
    config_hash,
    graph_id,
    instance_id,
)
from rmcd_f.rev.schema import (
    CERTIFICATE_MODES,
    NA,
    RUN_COLUMNS,
    RunRecord,
    coerce_row,
    read_runs_long,
    write_runs_long,
)

#: The unified run-level format agreed for the revision.
EXPECTED_COLUMNS = (
    "experiment", "instance_id", "graph_id", "base_id", "topology", "N",
    "num_edges", "seed", "composition", "budget_ratio", "budget_abs", "method",
    "variant", "kappa", "kappa_opt", "relative_gap", "actual_cost",
    "selected_count", "selected_nodes", "runtime_selection_s",
    "runtime_evaluation_s", "runtime_total_s", "status", "incumbent",
    "best_bound", "final_gap", "bb_nodes", "replay_kappa", "certificate_mode",
    "cg_L", "cg_U", "cg_iterations", "cg_cuts", "time_limit_s", "config_hash",
    "code_commit",
)


def test_run_schema_matches_the_agreed_column_list() -> None:
    assert RUN_COLUMNS == EXPECTED_COLUMNS
    assert len(RUN_COLUMNS) == 36


def test_certificate_modes_are_the_documented_four() -> None:
    assert set(CERTIFICATE_MODES) == {
        "solver_closed",
        "grid_closed",
        "cg_closed",
        "open",
    }


def _record(**overrides) -> RunRecord:
    payload = {
        "experiment": "main",
        "instance_id": "main__centralized-N42-s11",
        "graph_id": "centralized-N42-s11",
        "base_id": "centralized-N42-s11",
        "topology": "centralized",
        "N": 42,
        "num_edges": 126,
        "seed": 11,
        "method": "MPCF-Exact",
        "variant": "exact-compact-milp",
        "budget_ratio": 0.05,
        "budget_abs": 2.1,
        "kappa": 17.0,
        "kappa_opt": 17.0,
    }
    payload.update(overrides)
    return RunRecord(**payload)


def test_relative_gap_matches_its_definition() -> None:
    assert _record().relative_gap() == pytest.approx(0.0)
    assert _record(kappa=15.0, kappa_opt=20.0).relative_gap() == pytest.approx(0.25)
    assert _record(kappa_opt=None).relative_gap() is None
    assert _record(kappa_opt=0.0).relative_gap() is None


def test_missing_fields_are_written_as_na_not_dropped() -> None:
    row = _record().to_row()
    assert list(row) == list(RUN_COLUMNS)
    assert row["cg_L"] == NA
    assert row["bb_nodes"] == NA
    assert row["incumbent"] == NA


def test_runs_long_round_trips_with_a_fixed_header() -> None:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
        path = Path(directory) / "runs_long.csv"
        records = [
            _record(),
            _record(method="MPCF-Greedy", variant="heuristic-greedy", kappa=16.0,
                    certificate_mode="open", cg_iterations=4,
                    bb_nodes=None, selected_nodes=("c0000", "l0001")),
        ]
        assert write_runs_long(path, records) == 2
        header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert header == list(RUN_COLUMNS)
        rows = read_runs_long(path)
        assert len(rows) == 2
        assert rows[0]["kappa_opt"] == 17
        assert rows[1]["selected_nodes"] == ["c0000", "l0001"]
        assert rows[0]["cg_L"] is None


def test_mapping_rows_get_relative_gap_filled_from_the_optimum() -> None:
    row = coerce_row(
        {
            "kappa": "15",
            "kappa_opt": "20",
            "relative_gap": NA,
            "selected_nodes": NA,
        }
    )
    assert row["kappa"] == 15
    path_row = {
        "experiment": "main",
        "instance_id": "x",
        "graph_id": "x",
        "base_id": "x",
        "topology": "centralized",
        "N": "42",
        "num_edges": "126",
        "seed": "11",
        "method": "MPCF-Greedy",
        "variant": "heuristic-greedy",
        "budget_ratio": "0.05",
        "budget_abs": "2.1",
        "kappa": 15.0,
        "kappa_opt": 20.0,
        "relative_gap": "NA",
    }
    from rmcd_f.rev.schema import _mapping_row

    assert _mapping_row(path_row)["relative_gap"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# identifiers
# ---------------------------------------------------------------------------


def test_graph_and_base_ids_follow_topology_n_seed() -> None:
    assert graph_id("centralized", 42, 11) == "centralized-N42-s11"
    assert base_id("modular", 112, 55) == "modular-N112-s55"


def test_instance_id_is_unique_per_panel_modifier() -> None:
    base = graph_id("centralized", 42, 11)
    main = instance_id("main", base)
    role_a = instance_id("role", base, "4-3-3-4")
    role_b = instance_id("role", base, "3-4-4-3")
    assert main == f"main{SEPARATOR}{base}"
    assert role_a != role_b
    assert role_a.startswith(f"role{SEPARATOR}{base}{SEPARATOR}")
    # Empty modifiers are skipped rather than producing a double separator.
    assert instance_id("main", base, None, "") == main


def test_composition_label_formats_role_sizes() -> None:
    assert composition_label((12, 9, 9, 12)) == "12-9-9-12"
    assert composition_label({"S": 4, "C": 3, "L": 3, "E": 4}) == "4-3-3-4"


def test_config_hash_is_stable_and_order_independent() -> None:
    left = config_hash({"a": 1, "b": [1, 2]})
    right = config_hash({"b": [1, 2], "a": 1})
    assert left == right
    assert left != config_hash({"a": 1, "b": [2, 1]})


def test_code_commit_is_reported() -> None:
    commit = code_commit()
    assert isinstance(commit, str) and len(commit) >= 12


# ---------------------------------------------------------------------------
# variant registry
# ---------------------------------------------------------------------------


def test_registry_reconciles_22_36_and_45() -> None:
    claims = R.validate_against_paper_claims()
    assert claims["declared_methods"] == 45
    assert claims["certified_reference_methods"] == 2
    assert claims["main_panel_variants"] == 38
    assert claims["comparison_variants"] == 36
    assert claims["excluded_from_main_panel"] == 7


def test_comparison_set_is_exactly_the_thirty_six_variants() -> None:
    names = R.comparison_variant_names()
    assert len(names) == 36
    assert "MPCF-Greedy" in names
    assert "MPCF-Exact" not in names
    assert "MPCF-CG" not in names
    for excluded in R.MAIN_PANEL_EXCLUDED:
        assert f"{excluded}-Protect" not in names


def test_every_variant_carries_the_audit_fields() -> None:
    required = {
        "variant_id",
        "variant_name",
        "base_method",
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
    }
    rows = R.registry_rows({"MPCF-Exact": 135})
    assert rows
    for row in rows:
        assert required <= set(row)
        assert row["variant_id"]
        assert row["runtime_definition"]
        assert json.loads(row["parameters"]) is not None
    coverage = {row["variant_name"]: row["coverage_n"] for row in rows}
    assert coverage["MPCF-Exact"] == 135


def test_variant_ids_are_unique() -> None:
    rows = R.registry_rows()
    identifiers = [row["variant_id"] for row in rows]
    assert len(identifiers) == len(set(identifiers))


def test_cost_conversion_variants_are_explicit() -> None:
    rows = R.cost_conversion_variants()
    policies = {row.cost_conversion for row in rows}
    assert policies == {"raw", "per_cost", "prefix_knapsack", "full_knapsack"}
    for row in rows:
        assert row.method_class == "heterogeneous_cost_conversion"
    names = {row.variant_name for row in rows}
    assert {"Degree-Raw", "Degree-PerCost", "Degree-FullKnapsack"} <= names
