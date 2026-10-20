"""Paper towel protocol: two folds and released stability; rotation is telemetry."""
from __future__ import annotations

from dataclasses import replace
import json
import math

import numpy as np
import pytest

from handson.sim.isaaclab.flexible_towel_double_fold_contract import (
    FLAT_STABLE_FRAMES_REQUIRED,
    MATERIAL_COLUMNS,
    MATERIAL_NODE_COUNT,
    MATERIAL_ROWS,
    RELEASE_STABLE_FRAMES_REQUIRED,
    TASK_ID,
    TASK_NAME,
    TOWEL_SIZE_X_M,
    TOWEL_SIZE_Y_M,
    FlexibleTowelFoldSequence,
    FoldMetrics,
    FoldStage,
    compute_fold_metrics,
)


TABLE_TOP_M = 0.89
TOWEL_CENTER_XY = np.asarray([-0.05, 0.0], dtype=np.float64)


def _flat_nodes(*, yaw_rad: float = 0.0) -> np.ndarray:
    x = np.linspace(
        -0.5 * TOWEL_SIZE_X_M,
        0.5 * TOWEL_SIZE_X_M,
        MATERIAL_COLUMNS,
    )
    y = np.linspace(
        -0.5 * TOWEL_SIZE_Y_M,
        0.5 * TOWEL_SIZE_Y_M,
        MATERIAL_ROWS,
    )
    xx, yy = np.meshgrid(x, y)
    grid = np.zeros((MATERIAL_ROWS, MATERIAL_COLUMNS, 3), dtype=np.float64)
    grid[:, :, 0] = xx
    grid[:, :, 1] = yy
    grid[:, :, 2] = TABLE_TOP_M + 0.004
    cosine, sine = math.cos(yaw_rad), math.sin(yaw_rad)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    grid[:, :, :2] = grid[:, :, :2] @ rotation.T + TOWEL_CENTER_XY
    return grid.reshape(MATERIAL_NODE_COUNT, 3)


def _fold_rows(nodes: np.ndarray) -> np.ndarray:
    grid = np.asarray(nodes, dtype=np.float64).reshape(
        MATERIAL_ROWS, MATERIAL_COLUMNS, 3
    ).copy()
    middle = MATERIAL_ROWS // 2
    for row in range(middle + 1, MATERIAL_ROWS):
        grid[row] = grid[MATERIAL_ROWS - 1 - row]
    return grid.reshape(MATERIAL_NODE_COUNT, 3)


def _fold_columns(nodes: np.ndarray) -> np.ndarray:
    grid = np.asarray(nodes, dtype=np.float64).reshape(
        MATERIAL_ROWS, MATERIAL_COLUMNS, 3
    ).copy()
    middle = MATERIAL_COLUMNS // 2
    for column in range(middle + 1, MATERIAL_COLUMNS):
        grid[:, column] = grid[:, MATERIAL_COLUMNS - 1 - column]
    return grid.reshape(MATERIAL_NODE_COUNT, 3)


def _rotate(nodes: np.ndarray, angle_rad: float) -> np.ndarray:
    result = np.asarray(nodes, dtype=np.float64).copy()
    center = np.mean(result[:, :2], axis=0)
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.asarray(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    result[:, :2] = (result[:, :2] - center) @ rotation.T + center
    return result


@pytest.fixture
def geometry() -> dict[str, np.ndarray]:
    flat = _flat_nodes()
    first = _fold_rows(flat)
    rotated_positive = _rotate(first, 0.5 * math.pi)
    rotated_negative = _rotate(first, -0.5 * math.pi)
    return {
        "baseline": flat.copy(),
        "flat": flat,
        "first": first,
        "rotated_positive": rotated_positive,
        "rotated_negative": rotated_negative,
        "final_positive": _fold_columns(rotated_positive),
        "final_negative": _fold_columns(rotated_negative),
        "velocities": np.zeros_like(flat),
    }


def _metrics(geometry: dict[str, np.ndarray], name: str) -> FoldMetrics:
    return compute_fold_metrics(
        geometry[name],
        geometry["velocities"],
        geometry["baseline"],
        TABLE_TOP_M,
    )


def _prime_flat(
    sequence: FlexibleTowelFoldSequence,
    flat_metrics: FoldMetrics,
) -> None:
    for _ in range(FLAT_STABLE_FRAMES_REQUIRED):
        snapshot = sequence.update(
            flat_metrics,
            both_hands_engaged=False,
            hands_released=True,
        )
    assert snapshot.stage == FoldStage.FLAT.value
    assert snapshot.flat_latched


def _advance_to_second_fold(
    sequence: FlexibleTowelFoldSequence,
    geometry: dict[str, np.ndarray],
    *,
    direction: int = 1,
) -> FoldMetrics:
    _prime_flat(sequence, _metrics(geometry, "flat"))
    snapshot = sequence.update(
        _metrics(geometry, "first"),
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    suffix = "positive" if direction > 0 else "negative"
    snapshot = sequence.update(
        _metrics(geometry, f"rotated_{suffix}"),
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    final_metrics = _metrics(geometry, f"final_{suffix}")
    snapshot = sequence.update(
        final_metrics,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.SECOND_FOLD.value
    return final_metrics


def test_identity_and_material_grid_contract_are_explicit() -> None:
    assert TASK_ID == "FLEXIBLE-001"
    assert TASK_NAME == "eval_flex02_fold_towel_twice"
    assert (MATERIAL_ROWS, MATERIAL_COLUMNS, MATERIAL_NODE_COUNT) == (
        41,
        41,
        1681,
    )
    assert (TOWEL_SIZE_X_M, TOWEL_SIZE_Y_M) == pytest.approx(
        (0.27484, 0.27020)
    )


def test_metrics_distinguish_flat_first_fold_rotation_and_final_fold(
    geometry: dict[str, np.ndarray],
) -> None:
    flat = _metrics(geometry, "flat")
    first = _metrics(geometry, "first")
    rotated = _metrics(geometry, "rotated_positive")
    final = _metrics(geometry, "final_positive")

    assert flat.finite and flat.flat and flat.stable
    assert flat.supported and flat.in_workspace
    assert flat.oriented_footprint_ratio == pytest.approx(1.0)
    assert not flat.first_fold and not flat.final_fold

    assert first.first_fold and first.first_fold_maintained
    assert not first.flat and not first.final_fold
    assert first.row_mirror_xy_error == pytest.approx(0.0, abs=1e-12)
    assert first.oriented_footprint_ratio == pytest.approx(0.5)

    assert rotated.first_fold
    assert rotated.material_x_yaw == pytest.approx(0.5 * math.pi)
    assert rotated.oriented_footprint_ratio == pytest.approx(0.5)

    assert final.final_fold and final.final_fold_maintained
    assert final.first_fold_maintained
    assert final.column_mirror_xy_error == pytest.approx(0.0, abs=1e-12)
    assert final.oriented_footprint_ratio == pytest.approx(0.25)
    assert final.edge_stretch == pytest.approx(0.0, abs=1e-12)


def test_correct_order_reaches_success_only_on_fifteenth_released_stable_frame(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    final_metrics = _advance_to_second_fold(sequence, geometry)

    snapshot = sequence.update(
        final_metrics,
        both_hands_engaged=False,
        hands_released=True,
    )
    assert snapshot.stage == FoldStage.RELEASED.value
    assert snapshot.release_stable_frames == 1
    assert not snapshot.ready and not snapshot.success

    for expected in range(2, RELEASE_STABLE_FRAMES_REQUIRED):
        snapshot = sequence.update(
            final_metrics,
            both_hands_engaged=False,
            hands_released=True,
        )
        assert snapshot.release_stable_frames == expected
        assert not snapshot.success

    snapshot = sequence.update(
        final_metrics,
        both_hands_engaged=False,
        hands_released=True,
    )
    assert snapshot.release_stable_frames == RELEASE_STABLE_FRAMES_REQUIRED
    assert snapshot.ready and snapshot.success


def test_rotation_before_first_fold_is_not_credited(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    flat = _metrics(geometry, "flat")
    _prime_flat(sequence, flat)

    rotated_flat_nodes = _rotate(geometry["flat"], 0.5 * math.pi)
    rotated_flat = compute_fold_metrics(
        rotated_flat_nodes,
        geometry["velocities"],
        geometry["baseline"],
        TABLE_TOP_M,
    )
    snapshot = sequence.update(
        rotated_flat,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FLAT.value
    assert snapshot.cumulative_rotation_rad == 0.0

    pre_rotated_first = compute_fold_metrics(
        _fold_rows(rotated_flat_nodes),
        geometry["velocities"],
        geometry["baseline"],
        TABLE_TOP_M,
    )
    snapshot = sequence.update(
        pre_rotated_first,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    assert snapshot.first_fold_yaw_rad == pytest.approx(0.5 * math.pi)

    for _ in range(3):
        snapshot = sequence.update(
            pre_rotated_first,
            both_hands_engaged=True,
            hands_released=False,
        )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    assert snapshot.cumulative_rotation_rad == pytest.approx(0.0)
    assert not snapshot.rotation_latched


def test_second_fold_does_not_require_rotation_in_the_paper_protocol(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    _prime_flat(sequence, _metrics(geometry, "flat"))
    snapshot = sequence.update(
        _metrics(geometry, "first"),
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value

    unrotated_final = compute_fold_metrics(
        _fold_columns(geometry["first"]),
        geometry["velocities"],
        geometry["baseline"],
        TABLE_TOP_M,
    )
    assert unrotated_final.final_fold
    for _ in range(4):
        snapshot = sequence.update(
            unrotated_final,
            both_hands_engaged=True,
            hands_released=False,
        )
    assert snapshot.stage == FoldStage.SECOND_FOLD.value
    assert not snapshot.rotation_latched
    assert snapshot.second_fold_latched


def test_rotation_while_first_fold_is_loose_is_not_credited(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    _prime_flat(sequence, _metrics(geometry, "flat"))
    sequence.update(
        _metrics(geometry, "first"),
        both_hands_engaged=True,
        hands_released=False,
    )

    loose_rotated_nodes = _rotate(geometry["flat"], 0.5 * math.pi)
    loose_rotated = compute_fold_metrics(
        loose_rotated_nodes,
        geometry["velocities"],
        geometry["baseline"],
        TABLE_TOP_M,
    )
    assert not loose_rotated.first_fold_maintained
    snapshot = sequence.update(
        loose_rotated,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    assert snapshot.cumulative_rotation_rad == pytest.approx(0.0)

    # Re-forming the fold at the already-rotated yaw must not retroactively
    # count the loose-frame rotation.
    snapshot = sequence.update(
        _metrics(geometry, "rotated_positive"),
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    assert snapshot.cumulative_rotation_rad == pytest.approx(0.0)
    assert not snapshot.rotation_latched


def test_one_update_can_advance_at_most_one_stage(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    _prime_flat(sequence, _metrics(geometry, "flat"))
    final = _metrics(geometry, "final_positive")
    all_fold_predicates = replace(
        final,
        first_fold=True,
        first_fold_maintained=True,
        final_fold=True,
        final_fold_maintained=True,
        material_x_yaw=0.5 * math.pi,
    )

    snapshot = sequence.update(
        all_fold_predicates,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    assert snapshot.first_fold_latched
    assert not snapshot.rotation_latched

    rotated_again = replace(all_fold_predicates, material_x_yaw=math.pi)
    snapshot = sequence.update(
        rotated_again,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.SECOND_FOLD.value
    assert not snapshot.rotation_latched
    assert snapshot.second_fold_latched

    snapshot = sequence.update(
        rotated_again,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.SECOND_FOLD.value
    assert snapshot.second_fold_latched
    assert not snapshot.release_latched


@pytest.mark.parametrize("direction", (-1, 1))
def test_both_quarter_turn_directions_are_recorded_without_advancing_fold_stage(
    geometry: dict[str, np.ndarray], direction: int
) -> None:
    sequence = FlexibleTowelFoldSequence()
    _prime_flat(sequence, _metrics(geometry, "flat"))
    sequence.update(
        _metrics(geometry, "first"),
        both_hands_engaged=True,
        hands_released=False,
    )
    suffix = "positive" if direction > 0 else "negative"
    snapshot = sequence.update(
        _metrics(geometry, f"rotated_{suffix}"),
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FIRST_FOLD.value
    assert not snapshot.rotation_latched
    assert snapshot.cumulative_rotation_rad == pytest.approx(
        direction * 0.5 * math.pi
    )


def test_final_fold_does_not_progress_until_both_hands_release(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    final_metrics = _advance_to_second_fold(sequence, geometry)
    for _ in range(RELEASE_STABLE_FRAMES_REQUIRED + 2):
        snapshot = sequence.update(
            final_metrics,
            both_hands_engaged=False,
            hands_released=False,
        )
    assert snapshot.stage == FoldStage.SECOND_FOLD.value
    assert not snapshot.release_latched
    assert snapshot.release_stable_frames == 0
    assert not snapshot.success


def test_transient_looseness_restarts_hold_without_terminating_or_clearing_latches(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    final_metrics = _advance_to_second_fold(sequence, geometry)
    sequence.update(
        final_metrics,
        both_hands_engaged=False,
        hands_released=True,
    )
    for _ in range(5):
        snapshot = sequence.update(
            final_metrics,
            both_hands_engaged=False,
            hands_released=True,
        )
    assert snapshot.release_stable_frames == 6

    snapshot = sequence.update(
        _metrics(geometry, "rotated_positive"),
        both_hands_engaged=False,
        hands_released=True,
    )
    assert snapshot.stage == FoldStage.RELEASED.value
    assert snapshot.release_stable_frames == 0
    assert snapshot.flat_latched
    assert snapshot.first_fold_latched
    assert not snapshot.rotation_latched
    assert snapshot.second_fold_latched
    assert snapshot.release_latched
    assert not snapshot.success

    for index in range(RELEASE_STABLE_FRAMES_REQUIRED):
        snapshot = sequence.update(
            final_metrics,
            both_hands_engaged=False,
            hands_released=True,
        )
        assert snapshot.success is (
            index == RELEASE_STABLE_FRAMES_REQUIRED - 1
        )


@pytest.mark.parametrize(
    "corrupt",
    ("nodes", "velocities", "baseline", "table_top"),
)
def test_non_finite_input_is_finite_fail_closed_and_cannot_advance(
    geometry: dict[str, np.ndarray], corrupt: str
) -> None:
    nodes = geometry["first"].copy()
    velocities = geometry["velocities"].copy()
    baseline = geometry["baseline"].copy()
    table_top = TABLE_TOP_M
    if corrupt == "nodes":
        nodes[0, 0] = math.nan
    elif corrupt == "velocities":
        velocities[1, 1] = math.inf
    elif corrupt == "baseline":
        baseline[2, 2] = math.nan
    else:
        table_top = math.nan

    metrics = compute_fold_metrics(nodes, velocities, baseline, table_top)
    assert not metrics.finite
    assert not metrics.flat
    assert not metrics.first_fold
    assert not metrics.final_fold
    assert not metrics.stable
    serialized = json.dumps(metrics.to_dict(), allow_nan=False)
    assert isinstance(serialized, str)

    sequence = FlexibleTowelFoldSequence()
    snapshot = sequence.update(
        metrics,
        both_hands_engaged=True,
        hands_released=False,
    )
    assert snapshot.stage == FoldStage.FLAT.value
    assert not snapshot.flat_latched
    assert not snapshot.first_fold_latched
    assert not snapshot.ready


def test_snapshot_round_trip_and_reset_cover_every_hidden_latch_counter_and_rotation(
    geometry: dict[str, np.ndarray],
) -> None:
    sequence = FlexibleTowelFoldSequence()
    _advance_to_second_fold(sequence, geometry, direction=-1)
    snapshot = sequence.snapshot()
    encoded = json.dumps(snapshot.to_dict(), allow_nan=False)
    restored = FlexibleTowelFoldSequence.from_snapshot(json.loads(encoded))
    assert restored.snapshot() == snapshot

    reset = restored.reset()
    assert reset.stage == FoldStage.FLAT.value
    assert reset.update_count == 0
    assert reset.flat_stable_frames == 0
    assert reset.release_stable_frames == 0
    assert reset.cumulative_rotation_rad == 0.0
    assert not any(
        (
            reset.flat_latched,
            reset.first_fold_latched,
            reset.rotation_latched,
            reset.second_fold_latched,
            reset.release_latched,
            reset.ready,
            reset.success,
        )
    )
