import pickle
from pathlib import Path

from experiments.run_e4_replication_v8 import (
    CONFIGS,
    FORMAL_ROWS,
    FORMAL_SEEDS,
    ORIGINAL_E4,
    SELECTED_INSTANCES,
    formal_design,
    validate_design,
)
from scripts.build_e4_replication_reference_snapshot_v8 import (
    OUT as SNAPSHOT,
    SELECTED_INSTANCES as SNAPSHOT_INSTANCES,
    validate_selection,
)


def test_v8_selection_is_exact_and_disjoint_from_original_e4():
    validate_design()
    validate_selection()
    assert sum(map(len, SELECTED_INSTANCES.values())) == 30
    assert {
        family: tuple(names) for family, names in SELECTED_INSTANCES.items()
    } == SNAPSHOT_INSTANCES
    for family, names in SELECTED_INSTANCES.items():
        assert not (set(names) & ORIGINAL_E4[family])


def test_v8_formal_grid_is_exactly_1350_rows():
    expected = sum(
        len(SELECTED_INSTANCES[family]) * len(CONFIGS[budget]) * len(FORMAL_SEEDS)
        for family in SELECTED_INSTANCES
        for budget in CONFIGS
    )
    assert expected == FORMAL_ROWS == 1350
    design = formal_design()
    assert design["expected_rows"] == 1350
    assert design["seeds"] == [52, 53, 54, 55, 56]
    assert set(design["configs"]) == {"100", "200"}


def test_v8_reference_snapshot_is_frozen_for_60_blocks():
    path = Path(SNAPSHOT)
    assert path.is_file()
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)
    assert snapshot["snapshot_protocol"] == "saos_v8_fixed_v5_reference_20260722"
    assert len(snapshot["normalization"]) == 60
    assert len(snapshot["reference_sets"]) == 60
    assert {
        family: tuple(names)
        for family, names in snapshot["selected_instances"].items()
    } == SELECTED_INSTANCES
