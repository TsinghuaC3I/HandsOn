"""Pure geometry and ordered-success contract for FLEXIBLE-001.

The live environment supplies one row-major 41 x 41 material grid, its nodal
velocities, the settled flat grid captured at reset, and the current table-top
height.  This module deliberately has no Isaac, USD, PhysX, or Torch import so
the exact success semantics can be tested and replayed on an ordinary CPU.

A stable single fold must be followed by a distinct two-fold shape. Fold
axes and fold directions are unrestricted, including two parallel creases;
no tabletop quarter turn is required. Material-point correspondence rejects
mere compression into a smaller footprint. The released final shape must
remain supported and stable for 15 consecutive control frames.

The public task key and 16-value event layout retain their historical names
for dataset compatibility. Rotation fields are diagnostic only in contract v2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from functools import lru_cache
import math
from types import MappingProxyType
from typing import Any, ClassVar, Iterator, Mapping

import numpy as np


TASK_ID = "FLEXIBLE-001"
TASK_NAME = "eval_flex02_fold_towel_twice"
FLEXIBLE_TOWEL_TASK_ID = TASK_ID
FLEXIBLE_TOWEL_TASK_NAME = TASK_NAME
FOLD_CONTRACT_VERSION = 3

MATERIAL_ROWS = 41
MATERIAL_COLUMNS = 41
MATERIAL_NODE_COUNT = MATERIAL_ROWS * MATERIAL_COLUMNS
TOWEL_SIZE_X_M = 0.27484
TOWEL_SIZE_Y_M = 0.27020

# Geometry thresholds are expressed in metres or dimensionless ratios.  The
# 22 mm fit tolerance permits imperfect alignment of a real cloth fold.
# Labeled material points and a preference over the competing fold count
# distinguish folds from compression. Hysteresis permits a looser final hold.
# Oblique creases can overhang, so their convex envelope need not be exactly
# one half / one quarter of the original area.
FLAT_FOOTPRINT_RATIO_MIN = 0.80
FLAT_FOOTPRINT_RATIO_MAX = 1.20
FLAT_MIRROR_ERROR_MIN_M = 0.060
FIRST_FOLD_FOOTPRINT_RATIO_MIN = 0.34
FIRST_FOLD_FOOTPRINT_RATIO_MAX = 0.72
FIRST_FOLD_MAINTAIN_FOOTPRINT_RATIO_MIN = 0.10
FIRST_FOLD_MAINTAIN_FOOTPRINT_RATIO_MAX = 0.72
FINAL_FOLD_FOOTPRINT_RATIO_MIN = 0.12
FINAL_FOLD_FOOTPRINT_RATIO_MAX = 0.60
FINAL_FOLD_MAINTAIN_FOOTPRINT_RATIO_MIN = 0.09
FINAL_FOLD_MAINTAIN_FOOTPRINT_RATIO_MAX = 0.66
FOLD_SHAPE_ERROR_MAX_M = 0.022
FOLD_SHAPE_SEPARATION_M = 0.006
FOLD_SHAPE_MAINTAIN_ERROR_MAX_M = 0.035
MAX_EDGE_STRETCH_RATIO = 0.30
MAX_FOLD_HEIGHT_M = 0.12
MAX_FLAT_HEIGHT_M = 0.035

# The standard B320 worktable spans x=[-0.57, 0.27], y=[-0.64, 0.64]. Keep the success
# envelope inset from its collision edges while leaving ample room for a
# arbitrary planar motion of this 275 x 270 mm towel.
WORKSPACE_X_BOUNDS_M = (-0.52, 0.22)
WORKSPACE_Y_BOUNDS_M = (-0.60, 0.60)
SUPPORT_PENETRATION_TOLERANCE_M = 0.012
SUPPORT_CONTACT_HEIGHT_M = 0.020
MIN_SUPPORT_NODE_FRACTION = 0.10
MAX_SUPPORTED_HEIGHT_M = 0.12

STABLE_RMS_SPEED_MPS = 0.040
STABLE_MAX_SPEED_MPS = 0.22
FLAT_STABLE_FRAMES_REQUIRED = 3
RELEASE_STABLE_FRAMES_REQUIRED = 15
ROTATION_TARGET_RAD = 0.5 * math.pi
ROTATION_TOLERANCE_RAD = math.radians(20.0)
MAX_ROTATION_DELTA_PER_FRAME_RAD = math.radians(120.0)

_INVALID_SCALAR = 1.0e9
_MIN_BASELINE_EDGE_M = 1.0e-6
_MIN_MATERIAL_AXIS_M = 1.0e-6


class FoldStage(str, Enum):
    """Ordered task phases; enum values are stable recording strings."""

    FLAT = "flat"
    FIRST_FOLD = "first_fold"
    ROTATED = "rotated"  # Reserved for old recorded snapshots; never entered live.
    SECOND_FOLD = "second_fold"
    RELEASED = "released"


# More explicit public alias for callers that keep several task enums nearby.
FlexibleTowelFoldStage = FoldStage


@dataclass(frozen=True)
class FoldMetrics(Mapping[str, Any]):
    """One finite, immutable geometric readback.

    Mirror errors and bounds are in metres, yaw is in radians, edge stretch is
    the maximum positive fractional change of a material-neighbour edge, and
    speeds are in metres/second.  If an input is malformed or non-finite,
    ``finite`` is false, every predicate is false, and all numeric fields use
    finite fail-closed sentinels so observations can still be serialized.
    """

    finite: bool
    node_count: int
    row_mirror_xy_error: float
    column_mirror_xy_error: float
    material_x_yaw: float
    single_fold_error: float
    double_fold_error: float
    oriented_footprint_ratio: float
    edge_stretch: float
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    support_fraction: float
    supported: bool
    in_workspace: bool
    rms_speed: float
    max_speed: float
    stable: bool
    flat: bool
    first_fold: bool
    first_fold_maintained: bool
    final_fold: bool
    final_fold_maintained: bool

    _FIELD_NAMES: ClassVar[tuple[str, ...]] = (
        "finite",
        "node_count",
        "row_mirror_xy_error",
        "column_mirror_xy_error",
        "material_x_yaw",
        "single_fold_error",
        "double_fold_error",
        "oriented_footprint_ratio",
        "edge_stretch",
        "bounds_min",
        "bounds_max",
        "support_fraction",
        "supported",
        "in_workspace",
        "rms_speed",
        "max_speed",
        "stable",
        "flat",
        "first_fold",
        "first_fold_maintained",
        "final_fold",
        "final_fold_maintained",
    )

    def __getitem__(self, key: str) -> Any:
        if key not in self._FIELD_NAMES:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(self._FIELD_NAMES)

    def __len__(self) -> int:
        return len(self._FIELD_NAMES)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible primitive values."""

        return asdict(self)

    # Unit-explicit aliases are useful to observation builders while the short
    # names keep the metric contract easy to consume as a mapping.
    @property
    def truth_valid(self) -> bool:
        return self.finite

    @property
    def row_mirror_xy_error_m(self) -> float:
        return self.row_mirror_xy_error

    @property
    def column_mirror_xy_error_m(self) -> float:
        return self.column_mirror_xy_error

    @property
    def material_x_yaw_rad(self) -> float:
        return self.material_x_yaw

    @property
    def max_edge_stretch_ratio(self) -> float:
        return self.edge_stretch

    @property
    def on_table(self) -> bool:
        return self.supported

    @property
    def rms_nodal_speed(self) -> float:
        return self.rms_speed

    @property
    def max_nodal_speed(self) -> float:
        return self.max_speed

    @property
    def second_fold(self) -> bool:
        return self.final_fold


def _invalid_metrics(node_count: int = 0) -> FoldMetrics:
    return FoldMetrics(
        finite=False,
        node_count=int(max(0, node_count)),
        row_mirror_xy_error=_INVALID_SCALAR,
        column_mirror_xy_error=_INVALID_SCALAR,
        material_x_yaw=0.0,
        single_fold_error=_INVALID_SCALAR,
        double_fold_error=_INVALID_SCALAR,
        oriented_footprint_ratio=_INVALID_SCALAR,
        edge_stretch=_INVALID_SCALAR,
        bounds_min=(0.0, 0.0, 0.0),
        bounds_max=(0.0, 0.0, 0.0),
        support_fraction=0.0,
        supported=False,
        in_workspace=False,
        rms_speed=_INVALID_SCALAR,
        max_speed=_INVALID_SCALAR,
        stable=False,
        flat=False,
        first_fold=False,
        first_fold_maintained=False,
        final_fold=False,
        final_fold_maintained=False,
    )


def _as_material_grid(value: Any) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.shape != (MATERIAL_NODE_COUNT, 3):
        return None
    if not bool(np.all(np.isfinite(array))):
        return None
    return array.reshape(MATERIAL_ROWS, MATERIAL_COLUMNS, 3)


def _signed_material_x_axis(grid: np.ndarray) -> np.ndarray | None:
    """Estimate a signed material-x axis, including after a column fold."""

    middle = MATERIAL_COLUMNS // 2
    quarter = middle // 2
    # With two parallel folds, columns 0, middle and last coincide. Quarter
    # intervals still recover a valid axis for the four-layer strip.
    candidates = tuple(
        np.mean(grid[:, end, :2] - grid[:, start, :2], axis=0)
        for start, end in (
            (0, -1), (0, middle), (middle, -1),
            (0, quarter), (quarter, middle),
            (middle, middle + quarter), (middle + quarter, -1),
        )
    )
    norms = tuple(float(np.linalg.norm(candidate)) for candidate in candidates)
    index = int(np.argmax(np.asarray(norms, dtype=np.float64)))
    if not math.isfinite(norms[index]) or norms[index] < _MIN_MATERIAL_AXIS_M:
        return None
    return candidates[index] / norms[index]


def _material_edge_lengths(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    horizontal = np.linalg.norm(np.diff(grid, axis=1), axis=2)
    vertical = np.linalg.norm(np.diff(grid, axis=0), axis=2)
    return horizontal, vertical


def _planar_area(points: np.ndarray) -> float:
    """Area of the sheet's planar envelope, also valid for diagonal folds."""
    from scipy.spatial import ConvexHull, QhullError

    try:
        return float(ConvexHull(points).volume)
    except QhullError:
        return 0.0


def _fold_points(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    projection = points @ normal
    crease = 0.5 * (float(projection.min()) + float(projection.max()))
    return points + 2.0 * np.maximum(crease - projection, 0.0)[:, None] * normal


@lru_cache(maxsize=8)
def _fold_shape_templates(baseline_xy: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    # Sample labeled material points, rather than just bounding boxes. This
    # keeps a rolled/crumpled/compressed towel from masquerading as two folds.
    # 11.25-degree axis spacing plus the 22 mm geometric tolerance covers
    # intermediate directions without requiring the operator to align a crease.
    # Preserve the selected asset's dimensions and material indexing. A fixed
    # square template misclassifies folds of rectangular downloaded towels.
    flat = np.asarray(baseline_xy, dtype=np.float64).reshape(-1, 2)
    angles = np.arange(32, dtype=np.float64) * (2.0 * math.pi / 32)
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    first = [_fold_points(flat, normal) for normal in normals]
    second = [_fold_points(sheet, normal) for sheet in first for normal in normals]
    templates = np.asarray(first + second)
    templates -= templates.mean(axis=1, keepdims=True)
    tx = np.ascontiguousarray(templates[:, :, 0])
    ty = np.ascontiguousarray(templates[:, :, 1])
    norm = np.sum(tx * tx + ty * ty, axis=1)
    for array in (tx, ty, norm):
        array.flags.writeable = False
    return tx, ty, norm, len(first)


def _fold_shape_errors(grid: np.ndarray, footprint_ratio: float, baseline: np.ndarray) -> tuple[float, float]:
    if not 0.09 <= footprint_ratio <= 0.72:
        return _INVALID_SCALAR, _INVALID_SCALAR
    points = grid[::4, ::4, :2].reshape(-1, 2)
    points = points - points.mean(axis=0)
    flat = baseline[::4, ::4, :2].reshape(-1, 2)
    flat = flat - flat.mean(axis=0)
    tx, ty, template_norm, single_count = _fold_shape_templates(tuple(flat.ravel()))
    # Closed-form planar rigid alignment: free translation/yaw, no rescaling.
    # Restrict the larger two-fold bank to plausible final footprints.
    count = len(tx) if footprint_ratio <= FINAL_FOLD_MAINTAIN_FOOTPRINT_RATIO_MAX else single_count
    px, py = points[:, 0], points[:, 1]
    # Avoid a threaded BLAS launch for these small fixed-size reductions.
    a = np.einsum("ij,j->i", tx[:count], px) + np.einsum("ij,j->i", ty[:count], py)
    b = np.einsum("ij,j->i", tx[:count], py) - np.einsum("ij,j->i", ty[:count], px)
    residual = np.maximum(
        template_norm[:count] + float(np.sum(points * points)) - 2.0 * np.hypot(a, b),
        0.0,
    ) / len(points)
    single = math.sqrt(float(np.min(residual[:single_count])))
    double = (
        math.sqrt(float(np.min(residual[single_count:])))
        if count > single_count else _INVALID_SCALAR
    )
    return single, double


def compute_fold_metrics(
    nodes: Any,
    velocities: Any,
    baseline: Any,
    table_top: Any,
) -> FoldMetrics:
    """Compute rotationally invariant fold metrics for a row-major 41 x 41 grid.

    ``baseline`` must be the settled, flat reset grid with the same material
    indexing as ``nodes``.  The function never raises for malformed simulator
    readback: invalid shape, dtype, table height, or non-finite values return a
    finite fail-closed :class:`FoldMetrics` value.
    """

    node_count = 0
    try:
        raw_nodes = np.asarray(nodes)
        node_count = int(raw_nodes.shape[0]) if raw_nodes.ndim else 0
    except Exception:
        pass

    grid = _as_material_grid(nodes)
    velocity_grid = _as_material_grid(velocities)
    baseline_grid = _as_material_grid(baseline)
    try:
        table_top_m = float(table_top)
    except (TypeError, ValueError, OverflowError):
        return _invalid_metrics(node_count)
    if (
        grid is None
        or velocity_grid is None
        or baseline_grid is None
        or not math.isfinite(table_top_m)
    ):
        return _invalid_metrics(node_count)

    baseline_horizontal, baseline_vertical = _material_edge_lengths(
        baseline_grid
    )
    if (
        float(np.min(baseline_horizontal)) < _MIN_BASELINE_EDGE_M
        or float(np.min(baseline_vertical)) < _MIN_BASELINE_EDGE_M
    ):
        return _invalid_metrics(node_count)

    axis = _signed_material_x_axis(grid)
    baseline_axis = _signed_material_x_axis(baseline_grid)
    if axis is None or baseline_axis is None:
        return _invalid_metrics(node_count)

    yaw = float(math.atan2(float(axis[1]), float(axis[0])))
    xy = grid[:, :, :2].reshape(MATERIAL_NODE_COUNT, 2)
    baseline_xy = baseline_grid[:, :, :2].reshape(MATERIAL_NODE_COUNT, 2)
    baseline_area = _planar_area(baseline_xy)
    if not math.isfinite(baseline_area) or baseline_area <= 1.0e-8:
        return _invalid_metrics(node_count)
    footprint_ratio = _planar_area(xy) / baseline_area
    single_fold_error, double_fold_error = _fold_shape_errors(grid, footprint_ratio, baseline_grid)

    row_delta = grid[:, :, :2] - grid[::-1, :, :2]
    column_delta = grid[:, :, :2] - grid[:, ::-1, :2]
    row_mirror_error = float(
        np.sqrt(np.mean(np.sum(np.square(row_delta), axis=2)))
    )
    column_mirror_error = float(
        np.sqrt(np.mean(np.sum(np.square(column_delta), axis=2)))
    )

    current_horizontal, current_vertical = _material_edge_lengths(grid)
    horizontal_stretch = current_horizontal / baseline_horizontal - 1.0
    vertical_stretch = current_vertical / baseline_vertical - 1.0
    edge_stretch = max(
        0.0,
        float(np.max(horizontal_stretch)),
        float(np.max(vertical_stretch)),
    )

    flat_nodes = grid.reshape(MATERIAL_NODE_COUNT, 3)
    bounds_min_array = np.min(flat_nodes, axis=0)
    bounds_max_array = np.max(flat_nodes, axis=0)
    bounds_min = tuple(float(value) for value in bounds_min_array)
    bounds_max = tuple(float(value) for value in bounds_max_array)
    height = bounds_max[2] - bounds_min[2]
    support_fraction = float(
        np.mean(flat_nodes[:, 2] <= table_top_m + SUPPORT_CONTACT_HEIGHT_M)
    )
    supported = bool(
        bounds_min[2] >= table_top_m - SUPPORT_PENETRATION_TOLERANCE_M
        and bounds_max[2] <= table_top_m + MAX_SUPPORTED_HEIGHT_M
        and support_fraction >= MIN_SUPPORT_NODE_FRACTION
    )
    in_workspace = bool(
        bounds_min[0] >= WORKSPACE_X_BOUNDS_M[0]
        and bounds_max[0] <= WORKSPACE_X_BOUNDS_M[1]
        and bounds_min[1] >= WORKSPACE_Y_BOUNDS_M[0]
        and bounds_max[1] <= WORKSPACE_Y_BOUNDS_M[1]
    )

    speeds = np.linalg.norm(
        velocity_grid.reshape(MATERIAL_NODE_COUNT, 3), axis=1
    )
    rms_speed = float(np.sqrt(np.mean(np.square(speeds))))
    max_speed = float(np.max(speeds))
    stable = bool(
        rms_speed <= STABLE_RMS_SPEED_MPS
        and max_speed <= STABLE_MAX_SPEED_MPS
    )

    geometry_safe = bool(
        supported
        and in_workspace
        and edge_stretch <= MAX_EDGE_STRETCH_RATIO
        and height <= MAX_FOLD_HEIGHT_M
    )
    flat = bool(
        geometry_safe
        and height <= MAX_FLAT_HEIGHT_M
        and FLAT_FOOTPRINT_RATIO_MIN
        <= footprint_ratio
        <= FLAT_FOOTPRINT_RATIO_MAX
        and row_mirror_error >= FLAT_MIRROR_ERROR_MIN_M
        and column_mirror_error >= FLAT_MIRROR_ERROR_MIN_M
    )
    first_fold = bool(
        geometry_safe
        and single_fold_error <= FOLD_SHAPE_ERROR_MAX_M
        and single_fold_error + FOLD_SHAPE_SEPARATION_M < double_fold_error
        and FIRST_FOLD_FOOTPRINT_RATIO_MIN
        <= footprint_ratio
        <= FIRST_FOLD_FOOTPRINT_RATIO_MAX
    )
    final_fold = bool(
        geometry_safe
        and double_fold_error <= FOLD_SHAPE_ERROR_MAX_M
        and double_fold_error + FOLD_SHAPE_SEPARATION_M < single_fold_error
        and FINAL_FOLD_FOOTPRINT_RATIO_MIN
        <= footprint_ratio
        <= FINAL_FOLD_FOOTPRINT_RATIO_MAX
    )
    final_fold_maintained = bool(
        geometry_safe
        and double_fold_error <= FOLD_SHAPE_MAINTAIN_ERROR_MAX_M
        and double_fold_error + FOLD_SHAPE_SEPARATION_M < single_fold_error
        and FINAL_FOLD_MAINTAIN_FOOTPRINT_RATIO_MIN
        <= footprint_ratio
        <= FINAL_FOLD_MAINTAIN_FOOTPRINT_RATIO_MAX
    )
    first_fold_maintained = bool(
        final_fold_maintained
        or (
            geometry_safe
            and single_fold_error <= FOLD_SHAPE_MAINTAIN_ERROR_MAX_M
            and FIRST_FOLD_MAINTAIN_FOOTPRINT_RATIO_MIN
            <= footprint_ratio
            <= FIRST_FOLD_MAINTAIN_FOOTPRINT_RATIO_MAX
        )
    )

    numeric_values = (
        row_mirror_error,
        column_mirror_error,
        yaw,
        footprint_ratio,
        single_fold_error,
        double_fold_error,
        edge_stretch,
        *bounds_min,
        *bounds_max,
        support_fraction,
        rms_speed,
        max_speed,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        return _invalid_metrics(node_count)

    return FoldMetrics(
        finite=True,
        node_count=MATERIAL_NODE_COUNT,
        row_mirror_xy_error=row_mirror_error,
        column_mirror_xy_error=column_mirror_error,
        material_x_yaw=yaw,
        single_fold_error=single_fold_error,
        double_fold_error=double_fold_error,
        oriented_footprint_ratio=float(footprint_ratio),
        edge_stretch=edge_stretch,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        support_fraction=support_fraction,
        supported=supported,
        in_workspace=in_workspace,
        rms_speed=rms_speed,
        max_speed=max_speed,
        stable=stable,
        flat=flat,
        first_fold=first_fold,
        first_fold_maintained=first_fold_maintained,
        final_fold=final_fold,
        final_fold_maintained=final_fold_maintained,
    )


@dataclass(frozen=True)
class FoldSequenceSnapshot:
    """JSON-serializable hidden state required for reset/replay."""

    task_id: str
    task_name: str
    stage: str
    update_count: int
    flat_latched: bool
    flat_stable_frames: int
    first_fold_latched: bool
    first_fold_yaw_rad: float
    last_yaw_rad: float
    cumulative_rotation_rad: float
    rotation_direction: int
    rotation_latched: bool
    second_fold_latched: bool
    release_latched: bool
    release_stable_frames: int
    last_metrics_finite: bool
    ready: bool
    success: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def observation(self) -> Mapping[str, tuple[float, ...]]:
        """Return fixed-order finite fields convenient for a recorder."""

        stage_index = tuple(FoldStage).index(FoldStage(self.stage))
        return MappingProxyType(
            {
                "stage": (float(stage_index),),
                "latches": (
                    float(self.flat_latched),
                    float(self.first_fold_latched),
                    float(self.rotation_latched),
                    float(self.second_fold_latched),
                    float(self.release_latched),
                ),
                "rotation": (
                    self.first_fold_yaw_rad,
                    self.last_yaw_rad,
                    self.cumulative_rotation_rad,
                    float(self.rotation_direction),
                ),
                "counters": (
                    float(self.update_count),
                    float(self.flat_stable_frames),
                    float(self.release_stable_frames),
                ),
                "terminal": (
                    float(self.last_metrics_finite),
                    float(self.ready),
                    float(self.success),
                ),
            }
        )


FlexibleTowelFoldSequenceSnapshot = FoldSequenceSnapshot


def _metric_value(metrics: Any, name: str, default: Any) -> Any:
    if isinstance(metrics, Mapping):
        return metrics.get(name, default)
    return getattr(metrics, name, default)


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class FlexibleTowelFoldSequence:
    """Resettable, replayable ordered double-fold state machine.

    One :meth:`update` call represents exactly one control frame and can move
    through at most one :class:`FoldStage`.  Temporary invalid, unsupported,
    loosened, moving, or disengaged frames never terminate the episode and
    never clear the already-recorded order latches.
    """

    def __init__(
        self,
        snapshot: FoldSequenceSnapshot | Mapping[str, Any] | None = None,
    ) -> None:
        self.reset()
        if snapshot is not None:
            self.restore(snapshot)

    @property
    def stage(self) -> FoldStage:
        return self._stage

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def success(self) -> bool:
        return self._success

    def reset(self) -> FoldSequenceSnapshot:
        self._stage = FoldStage.FLAT
        self._update_count = 0
        self._flat_latched = False
        self._flat_stable_frames = 0
        self._first_fold_latched = False
        self._first_fold_yaw_rad = 0.0
        self._last_yaw_rad = 0.0
        self._cumulative_rotation_rad = 0.0
        self._rotation_direction = 0
        self._rotation_latched = False
        self._second_fold_latched = False
        self._release_latched = False
        self._release_stable_frames = 0
        self._last_metrics_finite = False
        self._ready = False
        self._success = False
        return self.snapshot()

    def snapshot(self) -> FoldSequenceSnapshot:
        return FoldSequenceSnapshot(
            task_id=TASK_ID,
            task_name=TASK_NAME,
            stage=self._stage.value,
            update_count=int(self._update_count),
            flat_latched=bool(self._flat_latched),
            flat_stable_frames=int(self._flat_stable_frames),
            first_fold_latched=bool(self._first_fold_latched),
            first_fold_yaw_rad=float(self._first_fold_yaw_rad),
            last_yaw_rad=float(self._last_yaw_rad),
            cumulative_rotation_rad=float(self._cumulative_rotation_rad),
            rotation_direction=int(self._rotation_direction),
            rotation_latched=bool(self._rotation_latched),
            second_fold_latched=bool(self._second_fold_latched),
            release_latched=bool(self._release_latched),
            release_stable_frames=int(self._release_stable_frames),
            last_metrics_finite=bool(self._last_metrics_finite),
            ready=bool(self._ready),
            success=bool(self._success),
        )

    def restore(
        self, snapshot: FoldSequenceSnapshot | Mapping[str, Any]
    ) -> FoldSequenceSnapshot:
        """Restore a previously serialized hidden state with strict validation."""

        data = (
            snapshot.to_dict()
            if isinstance(snapshot, FoldSequenceSnapshot)
            else dict(snapshot)
        )
        if data.get("task_id", TASK_ID) != TASK_ID or data.get(
            "task_name", TASK_NAME
        ) != TASK_NAME:
            raise ValueError("snapshot belongs to a different task")
        try:
            stage = FoldStage(str(data["stage"]))
            update_count = int(data["update_count"])
            flat_stable_frames = int(data["flat_stable_frames"])
            release_stable_frames = int(data["release_stable_frames"])
            rotation_direction = int(data["rotation_direction"])
            first_yaw = float(data["first_fold_yaw_rad"])
            last_yaw = float(data["last_yaw_rad"])
            cumulative_rotation = float(data["cumulative_rotation_rad"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError("invalid fold-sequence snapshot") from exc
        if (
            update_count < 0
            or flat_stable_frames < 0
            or release_stable_frames < 0
            or rotation_direction not in (-1, 0, 1)
            or not all(
                math.isfinite(value)
                for value in (first_yaw, last_yaw, cumulative_rotation)
            )
        ):
            raise ValueError("invalid fold-sequence counters or rotation")

        self._stage = stage
        self._update_count = update_count
        self._flat_latched = bool(data.get("flat_latched", False))
        self._flat_stable_frames = flat_stable_frames
        self._first_fold_latched = bool(data.get("first_fold_latched", False))
        self._first_fold_yaw_rad = first_yaw
        self._last_yaw_rad = last_yaw
        self._cumulative_rotation_rad = cumulative_rotation
        self._rotation_direction = rotation_direction
        self._rotation_latched = bool(data.get("rotation_latched", False))
        self._second_fold_latched = bool(data.get("second_fold_latched", False))
        self._release_latched = bool(data.get("release_latched", False))
        self._release_stable_frames = release_stable_frames
        self._last_metrics_finite = bool(data.get("last_metrics_finite", False))
        self._ready = bool(data.get("ready", False))
        self._success = bool(data.get("success", False))
        return self.snapshot()

    @classmethod
    def from_snapshot(
        cls, snapshot: FoldSequenceSnapshot | Mapping[str, Any]
    ) -> "FlexibleTowelFoldSequence":
        return cls(snapshot=snapshot)

    def _track_rotation(
        self,
        *,
        yaw: float,
        maintained: bool,
        both_hands_engaged: bool,
    ) -> None:
        if not math.isfinite(yaw):
            return
        delta = _wrap_angle(yaw - self._last_yaw_rad)
        # Always re-anchor.  A rotation that happens while the first fold is
        # loose or hands are disengaged must not be credited later as one jump.
        self._last_yaw_rad = yaw
        if (
            not maintained
            or not both_hands_engaged
            or abs(delta) > MAX_ROTATION_DELTA_PER_FRAME_RAD
        ):
            return
        self._cumulative_rotation_rad += delta

    def update(
        self,
        metrics: FoldMetrics | Mapping[str, Any],
        *,
        both_hands_engaged: bool,
        hands_released: bool,
    ) -> FoldSequenceSnapshot:
        """Consume one frame and return an immutable hidden-state snapshot."""

        self._update_count += 1
        finite = bool(_metric_value(metrics, "finite", False))
        try:
            yaw = float(_metric_value(metrics, "material_x_yaw", math.nan))
        except (TypeError, ValueError, OverflowError):
            yaw = math.nan
        self._last_metrics_finite = bool(finite and math.isfinite(yaw))
        both = bool(both_hands_engaged)
        released = bool(hands_released)
        signals_consistent = not (both and released)

        if self._success:
            return self.snapshot()

        if not self._last_metrics_finite or not signals_consistent:
            if not self._flat_latched:
                self._flat_stable_frames = 0
            if self._stage is FoldStage.RELEASED:
                self._release_stable_frames = 0
            return self.snapshot()

        supported = bool(_metric_value(metrics, "supported", False))
        workspace = bool(_metric_value(metrics, "in_workspace", False))
        stable = bool(_metric_value(metrics, "stable", False))
        flat = bool(_metric_value(metrics, "flat", False))
        first = bool(_metric_value(metrics, "first_fold", False))
        first_maintained = bool(
            _metric_value(metrics, "first_fold_maintained", False)
        )
        final = bool(_metric_value(metrics, "final_fold", False))
        final_maintained = bool(
            _metric_value(metrics, "final_fold_maintained", False)
        )
        safe = supported and workspace

        if self._stage is FoldStage.FLAT:
            if not self._flat_latched:
                if flat and stable and safe:
                    self._flat_stable_frames += 1
                    if (
                        self._flat_stable_frames
                        >= FLAT_STABLE_FRAMES_REQUIRED
                    ):
                        self._flat_latched = True
                else:
                    self._flat_stable_frames = 0
                return self.snapshot()

            if first and stable and safe:
                self._first_fold_latched = True
                self._first_fold_yaw_rad = yaw
                self._last_yaw_rad = yaw
                self._cumulative_rotation_rad = 0.0
                self._rotation_direction = 0
                self._stage = FoldStage.FIRST_FOLD
            return self.snapshot()

        if self._stage in (FoldStage.FIRST_FOLD, FoldStage.ROTATED):
            # Rotation remains useful recorded telemetry, but is never a gate.
            self._track_rotation(
                yaw=yaw,
                maintained=first_maintained and safe,
                both_hands_engaged=both and not released,
            )
            if final and stable and safe:
                self._second_fold_latched = True
                self._stage = FoldStage.SECOND_FOLD
            return self.snapshot()

        if self._stage is FoldStage.SECOND_FOLD:
            if final_maintained and safe and released and not both:
                self._release_latched = True
                self._stage = FoldStage.RELEASED
                self._release_stable_frames = 1 if stable else 0
            return self.snapshot()

        # RELEASED: require a genuinely consecutive final hold.  A transient
        # loosen/regrasp/motion restarts the hold but preserves every latch and
        # never creates a terminal failure.
        if (
            final_maintained
            and stable
            and safe
            and released
            and not both
        ):
            self._release_stable_frames += 1
        else:
            self._release_stable_frames = 0
        if self._release_stable_frames >= RELEASE_STABLE_FRAMES_REQUIRED:
            self._release_stable_frames = RELEASE_STABLE_FRAMES_REQUIRED
            self._ready = True
            self._success = True
        return self.snapshot()


__all__ = [
    "FINAL_FOLD_FOOTPRINT_RATIO_MAX",
    "FINAL_FOLD_FOOTPRINT_RATIO_MIN",
    "FIRST_FOLD_FOOTPRINT_RATIO_MAX",
    "FIRST_FOLD_FOOTPRINT_RATIO_MIN",
    "FLAT_STABLE_FRAMES_REQUIRED",
    "FOLD_CONTRACT_VERSION",
    "FLEXIBLE_TOWEL_TASK_ID",
    "FLEXIBLE_TOWEL_TASK_NAME",
    "FlexibleTowelFoldSequence",
    "FlexibleTowelFoldSequenceSnapshot",
    "FlexibleTowelFoldStage",
    "FoldMetrics",
    "FoldSequenceSnapshot",
    "FoldStage",
    "MATERIAL_COLUMNS",
    "MATERIAL_NODE_COUNT",
    "MATERIAL_ROWS",
    "MAX_EDGE_STRETCH_RATIO",
    "RELEASE_STABLE_FRAMES_REQUIRED",
    "ROTATION_TARGET_RAD",
    "ROTATION_TOLERANCE_RAD",
    "STABLE_MAX_SPEED_MPS",
    "STABLE_RMS_SPEED_MPS",
    "TASK_ID",
    "TASK_NAME",
    "TOWEL_SIZE_X_M",
    "TOWEL_SIZE_Y_M",
    "compute_fold_metrics",
]
