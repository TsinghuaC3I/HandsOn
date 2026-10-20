"""Lossless, fixed-shape force telemetry for the Plan1 hammer task.

The collector is deliberately independent from the environment implementation.  A
Plan1 environment can own one :class:`Plan1ForceTelemetry`, call ``begin_frame``
before its MuJoCo substep loop, call ``capture_pre_substep`` immediately before
and ``capture_substep`` immediately after every ``mj_step``, and merge
``finalize_frame()`` into the top-level observation.

Contact wrenches follow MuJoCo's convention: ``mj_contactForce`` returns the
wrench acting on ``geom2`` in the contact frame.  The corresponding world-frame
wrench stored here is ``contact.frame.T @ wrench`` (applied independently to the
force and torque triples).  Keeping both representations avoids an otherwise
easy-to-miss sign or basis ambiguity in downstream imitation learning.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


MAX_CONTACTS = 128
MAX_HAND_JOINTS = 64
MAX_HAND_ACTUATORS = 64
MAX_TASK_OBJECTS = 4
MAX_FIELD_ELEMENTS = 65_536


# Role values are numeric in observations and named in ``schema()``.  Right/left
# intentionally use the same codes in the hand-joint and actuator ID tables.
ROLE_UNKNOWN = 0
ROLE_RIGHT_HAND = 1
ROLE_LEFT_HAND = 2
ROLE_HAMMER = 3
ROLE_WOOD = 4
ROLE_NAIL = 5
ROLE_TABLE = 6

ROLE_CODEBOOK = {
    ROLE_UNKNOWN: "unknown",
    ROLE_RIGHT_HAND: "right_hand",
    ROLE_LEFT_HAND: "left_hand",
    ROLE_HAMMER: "hammer",
    ROLE_WOOD: "wood",
    ROLE_NAIL: "nail",
    ROLE_TABLE: "table",
}

_OBJECT_ROLES = (ROLE_HAMMER, ROLE_WOOD, ROLE_NAIL, ROLE_TABLE)

_CONTACT_ID_COLUMNS = (
    "contact_index",
    "geom1_id",
    "geom2_id",
    "geom1_body_id",
    "geom2_body_id",
    "geom1_role_code",
    "geom2_role_code",
    "constraint_dimension",
    "efc_address",
)
_HAND_JOINT_ID_COLUMNS = ("joint_id", "qpos_address", "dof_address", "side_code")
_HAND_JOINT_STATE_COLUMNS = (
    "qpos",
    "qvel",
    "qacc",
    "qfrc_actuator",
)
_HAND_JOINT_FORCE_COLUMNS = (
    "qfrc_applied",
    "qfrc_bias",
    "qfrc_passive",
    "qfrc_constraint",
    "qfrc_smooth",
    "qfrc_inverse",
)
_HAND_JOINT_DYNAMICS_COLUMNS = (
    "actfrcrange_lower",
    "actfrcrange_upper",
    "damping",
    "armature",
    "frictionloss",
)
_HAND_ACTUATOR_ID_COLUMNS = (
    "actuator_id",
    "transmission_target_id",
    "transmission_type",
    "side_code",
)
_HAND_ACTUATOR_STATE_COLUMNS = (
    "ctrl",
    "actuator_force",
    "actuator_length",
    "actuator_velocity",
    "activation_first_state_or_zero",
)
_HAND_ACTUATOR_DYNAMICS_COLUMNS = (
    "ctrlrange_lower",
    "ctrlrange_upper",
    "forcerange_lower",
    "forcerange_upper",
    "gainprm_0",
    "biasprm_0",
    "biasprm_1",
    "biasprm_2",
)
_SUBSTEP_HAND_JOINT_COLUMNS = (
    "qacc",
    "qfrc_actuator",
    "qfrc_applied",
    "qfrc_bias",
    "qfrc_passive",
    "qfrc_constraint",
    "qfrc_smooth",
    "actfrcrange_lower",
    "actfrcrange_upper",
)
_SUBSTEP_HAND_ACTUATOR_COLUMNS = (
    "ctrl_input",
    "actuator_force",
    "actuator_length",
    "actuator_velocity",
    "forcerange_lower",
    "forcerange_upper",
)
_OBJECT_ID_COLUMNS = ("role_code", "body_id", "root_joint_id_or_minus_one")
_OBJECT_CONTACT_COUNT_COLUMNS = ("last_substep", "whole_control_frame")


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    if obj_id < 0:
        return ""
    return mujoco.mj_id2name(model, obj_type, int(obj_id)) or ""


def _enum_name(enum_type: Any, value: Any) -> str:
    try:
        return enum_type(int(value)).name
    except (TypeError, ValueError):
        return str(int(value))


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy/MuJoCo values to JSON-safe Python values."""

    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _as_id_set(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, np.ndarray):
        value = value.reshape(-1).tolist()
    if isinstance(value, (set, list, tuple)):
        result: set[int] = set()
        for item in value:
            try:
                item_id = int(item)
            except (TypeError, ValueError):
                continue
            if item_id >= 0:
                result.add(item_id)
        return result
    try:
        item_id = int(value)
    except (TypeError, ValueError):
        return set()
    return {item_id} if item_id >= 0 else set()


def _side_code_from_name(name: str) -> int:
    normalized = name.lstrip("/").lower()
    if normalized.startswith("right_") or normalized.endswith("_right"):
        return ROLE_RIGHT_HAND
    if normalized.startswith("left_") or normalized.endswith("_left"):
        return ROLE_LEFT_HAND
    return ROLE_UNKNOWN


def _looks_like_hand_body(name: str, side: str) -> bool:
    normalized = name.lstrip("/").lower()
    if normalized in {f"{side}_palm_link", f"allegro_palm_{side}"}:
        return True
    if normalized.startswith(f"{side}_finger"):
        return True
    if normalized.endswith(f"_{side}") and normalized.startswith(
        ("ff_", "mf_", "rf_", "th_")
    ):
        return True
    return False


class Plan1ForceTelemetry:
    """Collect per-substep contacts and frame-final dynamics for Plan1.

    ``max_contacts`` is a per-substep cap.  Excess relevant contacts are not
    silently lost: their count is stored in ``c01_force_contact_overflow``.
    The number of substep rows is fixed from ``env._n_substeps`` at creation.
    """

    def __init__(self, env: Any, max_contacts: int = MAX_CONTACTS):
        raw_env = getattr(env, "unwrapped", env)
        self.env = raw_env
        self.model: mujoco.MjModel = getattr(raw_env, "_model", None)
        self.data: mujoco.MjData = getattr(raw_env, "_data", None)
        if self.model is None or self.data is None:
            raise TypeError("Plan1ForceTelemetry requires env._model and env._data.")

        self.max_contacts = int(max_contacts)
        if not 1 <= self.max_contacts <= MAX_CONTACTS:
            raise ValueError(
                f"max_contacts must be in [1, {MAX_CONTACTS}], got {max_contacts}."
            )
        self.n_substeps = max(1, int(getattr(raw_env, "_n_substeps", 1)))

        self._geom_roles = self._build_geom_roles()
        self._relevant_geom_ids = {
            geom_id
            for geom_id, role in self._geom_roles.items()
            if role != ROLE_UNKNOWN
        }
        self._hand_joint_rows = self._discover_hand_joints()
        self._hand_actuator_rows = self._discover_hand_actuators()
        self._object_rows = self._discover_task_objects()
        self._object_slot_by_role = {
            role: slot
            for slot, (role, body_id, _) in enumerate(self._object_rows)
            if body_id >= 0
        }

        self._arrays = self._allocate_arrays()
        self._spaces = self._build_spaces()
        self._schema = self._build_schema()
        self.begin_frame()

    # ------------------------------------------------------------------
    # Public lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Clear the fixed buffers before the environment's substep loop."""

        for value in self._arrays.values():
            value.fill(0)
        self._substep_index = 0
        self._substep_overflow = 0
        self._frame_object_contact_impulse = np.zeros(
            (MAX_TASK_OBJECTS, 6), dtype=np.float64
        )
        self._frame_object_contact_count = np.zeros(
            MAX_TASK_OBJECTS, dtype=np.int32
        )
        self._last_object_contact_wrench = np.zeros(
            (MAX_TASK_OBJECTS, 6), dtype=np.float64
        )
        self._last_object_contact_count = np.zeros(
            MAX_TASK_OBJECTS, dtype=np.int32
        )
        # ``capture_pre_substep`` snapshots the exact input before integration.
        # Frame-final data.xfrc_applied can already contain a command prepared
        # for the next control frame, so object summaries use this cached copy.
        self._last_applied_xfrc = np.asarray(
            self.data.xfrc_applied,
            dtype=np.float64,
        ).copy()
        self._pending_pre_substep: int | None = None
        self._overflow_pre_pending = False
        self._finalized = False

    def capture_pre_substep(self) -> None:
        """Snapshot the exact MuJoCo state and inputs immediately before ``mj_step``.

        MuJoCo evaluates contacts, actuation and acceleration at the pre-step
        state and only then integrates ``qpos``, ``qvel`` and ``act``.  A
        post-step-only logger therefore cannot align force with the state that
        produced it.  Every call to this method must be paired with exactly one
        :meth:`capture_substep` call after the corresponding ``mj_step``.
        """

        if self._pending_pre_substep is not None or self._overflow_pre_pending:
            raise RuntimeError(
                "capture_pre_substep() called twice without capture_substep()."
            )
        if self._substep_index >= self.n_substeps:
            self._substep_overflow += 1
            self._arrays["c01_force_substep_overflow"][0] = self._substep_overflow
            self._overflow_pre_pending = True
            return

        row = self._substep_index
        self._arrays["c01_force_substep_pre_time"][row] = float(self.data.time)
        self._arrays["c01_force_substep_pre_qpos"][row] = self.data.qpos
        self._arrays["c01_force_substep_pre_qvel"][row] = self.data.qvel
        self._arrays["c01_force_substep_pre_ctrl"][row] = self.data.ctrl
        self._arrays["c01_force_substep_pre_qfrc_applied"][row] = (
            self.data.qfrc_applied
        )
        self._arrays["c01_force_substep_pre_xfrc_applied"][row] = (
            self.data.xfrc_applied
        )
        self._arrays["c01_force_substep_pre_qacc_warmstart"][row] = (
            self.data.qacc_warmstart
        )
        self._arrays["c01_force_substep_pre_mocap_pos"][row] = self.data.mocap_pos
        self._arrays["c01_force_substep_pre_mocap_quat"][row] = self.data.mocap_quat
        if "c01_force_substep_pre_act" in self._arrays:
            self._arrays["c01_force_substep_pre_act"][row] = self.data.act
        if "c01_force_substep_pre_eq_active" in self._arrays:
            self._arrays["c01_force_substep_pre_eq_active"][row] = self.data.eq_active

        self._last_applied_xfrc = np.asarray(
            self.data.xfrc_applied,
            dtype=np.float64,
        ).copy()
        self._pending_pre_substep = row
        self._finalized = False

    def capture_substep(self) -> None:
        """Capture contacts after one MuJoCo physics step.

        Call this exactly once immediately after each ``mj_step``.  Calls beyond
        ``env._n_substeps`` are counted as substep overflow so an integration bug
        remains visible in the recorded data.
        """

        if self._overflow_pre_pending:
            self._overflow_pre_pending = False
            return
        if self._pending_pre_substep is None:
            raise RuntimeError(
                "capture_substep() requires capture_pre_substep() immediately "
                "before the corresponding mj_step()."
            )
        if self._pending_pre_substep != self._substep_index:
            raise RuntimeError("Force telemetry pre/post substep indices diverged.")

        row = self._substep_index
        self._arrays["c01_force_substep_mask"][row] = 1
        self._arrays["c01_force_substep_post_time"][row] = float(self.data.time)
        # Backwards-compatible alias: this has always been the time after
        # integration, not the time at which force was evaluated.
        self._arrays["c01_force_substep_time"][row] = float(self.data.time)

        contact_slot = 0
        overflow = 0
        object_wrench = np.zeros((MAX_TASK_OBJECTS, 6), dtype=np.float64)
        object_count = np.zeros(MAX_TASK_OBJECTS, dtype=np.int32)

        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if (
                geom1 not in self._relevant_geom_ids
                and geom2 not in self._relevant_geom_ids
            ):
                continue

            local_wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                self.model,
                self.data,
                contact_index,
                local_wrench,
            )
            frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
            world_wrench = np.concatenate(
                [
                    frame.T @ local_wrench[:3],
                    frame.T @ local_wrench[3:],
                ]
            )

            role1 = self._geom_roles.get(geom1, ROLE_UNKNOWN)
            role2 = self._geom_roles.get(geom2, ROLE_UNKNOWN)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            pos = np.asarray(contact.pos, dtype=np.float64)

            # Aggregate object wrenches about each task body's origin.  The
            # contact API is on geom2, so geom1 receives the equal/opposite one.
            self._accumulate_object_contact(
                object_wrench,
                object_count,
                role1,
                -world_wrench,
                pos,
            )
            self._accumulate_object_contact(
                object_wrench,
                object_count,
                role2,
                world_wrench,
                pos,
            )

            if contact_slot >= self.max_contacts:
                overflow += 1
                continue

            self._arrays["c01_force_contact_mask"][row, contact_slot] = 1
            self._arrays["c01_force_contact_ids"][row, contact_slot] = np.asarray(
                (
                    contact_index,
                    geom1,
                    geom2,
                    body1,
                    body2,
                    role1,
                    role2,
                    int(contact.dim),
                    int(contact.efc_address),
                ),
                dtype=np.int32,
            )
            self._arrays["c01_force_contact_pos"][row, contact_slot] = pos
            self._arrays["c01_force_contact_frame"][row, contact_slot] = frame.reshape(9)
            self._arrays["c01_force_contact_dist"][row, contact_slot] = float(
                contact.dist
            )
            self._arrays["c01_force_contact_wrench_local_on_geom2"][
                row, contact_slot
            ] = local_wrench
            self._arrays["c01_force_contact_wrench_world_on_geom2"][
                row, contact_slot
            ] = world_wrench
            self._arrays["c01_force_contact_friction"][row, contact_slot] = np.asarray(
                contact.friction, dtype=np.float64
            )
            self._arrays["c01_force_contact_includemargin_mu"][
                row, contact_slot
            ] = np.asarray(
                (float(contact.includemargin), float(contact.mu)),
                dtype=np.float64,
            )
            self._arrays["c01_force_contact_solref"][row, contact_slot] = np.concatenate(
                [
                    np.asarray(contact.solref, dtype=np.float64),
                    np.asarray(contact.solreffriction, dtype=np.float64),
                ]
            )
            self._arrays["c01_force_contact_solimp"][row, contact_slot] = np.asarray(
                contact.solimp, dtype=np.float64
            )
            contact_slot += 1

        self._arrays["c01_force_contact_overflow"][row] = overflow
        self._last_object_contact_wrench = object_wrench
        self._last_object_contact_count = object_count
        dt = float(self.model.opt.timestep)
        self._frame_object_contact_impulse += object_wrench * dt
        self._frame_object_contact_count += object_count
        self._capture_substep_dynamics(row)
        self._pending_pre_substep = None
        self._substep_index += 1
        self._arrays["c01_force_substep_count"][0] = self._substep_index
        self._finalized = False

    def finalize_frame(self) -> dict[str, np.ndarray]:
        """Capture the frame-final state and return observation-ready arrays."""

        if self._pending_pre_substep is not None or self._overflow_pre_pending:
            raise RuntimeError(
                "finalize_frame() called after capture_pre_substep() without the "
                "matching post-step capture_substep()."
            )

        self._arrays["c01_force_substep_count"][0] = self._substep_index
        self._arrays["c01_force_substep_overflow"][0] = self._substep_overflow
        self._capture_hand_state()
        self._capture_object_state()
        self._capture_raw_state()
        self._finalized = True
        return self.observation_fields()

    def observation_fields(self) -> dict[str, np.ndarray]:
        """Return independent, top-level ``c01_force_*`` observation arrays."""

        return {name: value.copy() for name, value in self._arrays.items()}

    def observation_spaces(self) -> dict[str, gym.Space]:
        """Return Gym spaces matching :meth:`observation_fields`."""

        return dict(self._spaces)

    def schema(self) -> dict[str, Any]:
        """Return a JSON-safe description of fields, units, signs and mappings."""

        return _json_safe(self._schema)

    def refresh_episode_model_snapshot(self) -> None:
        """Freeze the post-reset model used at the start of this episode.

        Plan1 changes table geometry and may randomize object dynamics during
        ``reset``.  Refreshing after those changes makes ``model.mjb`` and the
        schema hash identify the actual episode-start model rather than the
        constructor-time template.
        """
        if self._pending_pre_substep is not None or self._overflow_pre_pending:
            raise RuntimeError(
                "Cannot refresh the episode model during a pending physics step."
            )
        self._schema = self._build_schema()

    def compiled_model_bytes(self) -> bytes:
        """Return the exact baseline MJB whose hash appears in the schema."""

        return bytes(self._compiled_model_mjb)

    # ------------------------------------------------------------------
    # Discovery and role mapping
    # ------------------------------------------------------------------

    def _build_geom_roles(self) -> dict[int, int]:
        roles: dict[int, int] = {}

        hand_sets = getattr(self.env, "_c01_hand_collision_geom_ids", None)
        if isinstance(hand_sets, Mapping):
            for geom_id in _as_id_set(hand_sets.get("right")):
                roles[geom_id] = ROLE_RIGHT_HAND
            for geom_id in _as_id_set(hand_sets.get("left")):
                roles[geom_id] = ROLE_LEFT_HAND

        # Older Allegro environments expose only this left-hand set.  Generic
        # name discovery below fills the corresponding right-hand set.
        for geom_id in _as_id_set(getattr(self.env, "_left_hand_geom_ids", None)):
            roles[geom_id] = ROLE_LEFT_HAND

        for geom_id in range(int(self.model.ngeom)):
            body_id = int(self.model.geom_bodyid[geom_id])
            body_name = _name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if _looks_like_hand_body(body_name, "right"):
                roles.setdefault(geom_id, ROLE_RIGHT_HAND)
            elif _looks_like_hand_body(body_name, "left"):
                roles.setdefault(geom_id, ROLE_LEFT_HAND)

        for geom_id in _as_id_set(getattr(self.env, "_hammer_geom_ids", None)):
            roles.setdefault(geom_id, ROLE_HAMMER)
        wood_ids = _as_id_set(getattr(self.env, "_wood_collision_geom_id", None))
        for geom_id in wood_ids:
            roles.setdefault(geom_id, ROLE_WOOD)
        for geom_id in _as_id_set(getattr(self.env, "_nail_geom_ids", None)):
            roles.setdefault(geom_id, ROLE_NAIL)

        table_ids = _as_id_set(getattr(self.env, "_table_geom_id", None))
        table_body_id = self._body_id("table")
        for geom_id in range(int(self.model.ngeom)):
            geom_name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
            if geom_name in {"table_collision", "table_visual"}:
                table_ids.add(geom_id)
            if table_body_id >= 0 and int(self.model.geom_bodyid[geom_id]) == table_body_id:
                table_ids.add(geom_id)
        for geom_id in table_ids:
            roles.setdefault(geom_id, ROLE_TABLE)

        # Name-based fallbacks keep the helper useful in focused MuJoCo tests
        # and in compatible Panda/Allegro Plan1 model variants.
        for geom_id in range(int(self.model.ngeom)):
            geom_name = _name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id).lower()
            body_name = _name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[geom_id]),
            ).lower()
            if body_name == "hammer_body" or geom_name in {
                "handle",
                "face",
                "head",
                "neck",
                "claw",
            }:
                roles.setdefault(geom_id, ROLE_HAMMER)
            elif body_name == "wood" or geom_name == "wood_collision":
                roles.setdefault(geom_id, ROLE_WOOD)
            elif body_name == "nail" or geom_name in {"nail_head", "nail_shaft"}:
                roles.setdefault(geom_id, ROLE_NAIL)

        return roles

    def _discover_hand_joints(self) -> list[tuple[int, int, int, int]]:
        rows: list[tuple[int, int, int, int]] = []
        joint_names: Sequence[str] = getattr(
            getattr(self.env, "_hand", None), "joint_names", ()
        )
        candidate_ids: list[int] = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, str(joint_name)
            )
            if joint_id < 0 and not str(joint_name).startswith("/"):
                joint_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"/{joint_name}",
                )
            if joint_id >= 0:
                candidate_ids.append(int(joint_id))

        if not candidate_ids:
            for joint_id in range(int(self.model.njnt)):
                joint_name = _name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                )
                if _side_code_from_name(joint_name) and (
                    "finger" in joint_name.lower()
                    or joint_name.lstrip("/").lower().startswith(
                        ("ffj", "mfj", "rfj", "thj")
                    )
                ):
                    candidate_ids.append(joint_id)

        seen: set[int] = set()
        for joint_id in candidate_ids:
            if joint_id in seen:
                continue
            seen.add(joint_id)
            if len(rows) >= MAX_HAND_JOINTS:
                raise ValueError(
                    f"Plan1 has more than {MAX_HAND_JOINTS} hand joints."
                )
            name = _name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            rows.append(
                (
                    joint_id,
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                    _side_code_from_name(name),
                )
            )
        return rows

    def _discover_hand_actuators(self) -> list[tuple[int, int, int, int]]:
        ctrl_ids = getattr(getattr(self.env, "_hand", None), "ctrl_ids", None)
        candidate_ids = sorted(_as_id_set(ctrl_ids))
        hand_joint_ids = {row[0] for row in self._hand_joint_rows}
        if not candidate_ids:
            for actuator_id in range(int(self.model.nu)):
                target_id = int(self.model.actuator_trnid[actuator_id, 0])
                name = _name(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                )
                if target_id in hand_joint_ids or (
                    _side_code_from_name(name)
                    and ("finger" in name.lower() or name.lower().startswith(
                        ("ffa", "mfa", "rfa", "tha")
                    ))
                ):
                    candidate_ids.append(actuator_id)

        rows: list[tuple[int, int, int, int]] = []
        for actuator_id in candidate_ids:
            if not 0 <= actuator_id < int(self.model.nu):
                continue
            if len(rows) >= MAX_HAND_ACTUATORS:
                raise ValueError(
                    f"Plan1 has more than {MAX_HAND_ACTUATORS} hand actuators."
                )
            target_id = int(self.model.actuator_trnid[actuator_id, 0])
            name = _name(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
            )
            side_code = _side_code_from_name(name)
            if side_code == ROLE_UNKNOWN and 0 <= target_id < int(self.model.njnt):
                side_code = _side_code_from_name(
                    _name(self.model, mujoco.mjtObj.mjOBJ_JOINT, target_id)
                )
            rows.append(
                (
                    actuator_id,
                    target_id,
                    int(self.model.actuator_trntype[actuator_id]),
                    side_code,
                )
            )
        return rows

    def _discover_task_objects(self) -> list[tuple[int, int, int]]:
        body_candidates = {
            ROLE_HAMMER: self._env_or_named_body_id("_hammer_body_id", "hammer_body"),
            ROLE_WOOD: self._env_or_named_body_id("_wood_body_id", "wood"),
            ROLE_NAIL: self._env_or_named_body_id("_nail_body_id", "nail"),
            ROLE_TABLE: self._env_or_named_body_id("_table_body_id", "table"),
        }
        joint_candidates = {
            ROLE_HAMMER: self._env_or_named_joint_id("_hammer_joint_id", "hammer_joint"),
            ROLE_WOOD: self._env_or_named_joint_id("_wood_joint_id", "wood_joint"),
            ROLE_NAIL: self._env_or_named_joint_id(
                "_nail_slide_joint_id", "nail_slide_joint"
            ),
            ROLE_TABLE: -1,
        }
        return [
            (role, int(body_candidates[role]), int(joint_candidates[role]))
            for role in _OBJECT_ROLES
        ]

    def _body_id(self, name: str) -> int:
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name))

    def _env_or_named_body_id(self, attr: str, name: str) -> int:
        value = getattr(self.env, attr, None)
        try:
            value_id = int(value)
        except (TypeError, ValueError):
            value_id = -1
        return value_id if value_id >= 0 else self._body_id(name)

    def _env_or_named_joint_id(self, attr: str, name: str) -> int:
        value = getattr(self.env, attr, None)
        try:
            value_id = int(value)
        except (TypeError, ValueError):
            value_id = -1
        if value_id >= 0:
            return value_id
        return int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name))

    # ------------------------------------------------------------------
    # Buffer allocation and capture
    # ------------------------------------------------------------------

    def _allocate_arrays(self) -> dict[str, np.ndarray]:
        substeps = self.n_substeps
        contacts = self.max_contacts
        shapes_and_dtypes: dict[str, tuple[tuple[int, ...], np.dtype[Any] | type]] = {
            "c01_force_substep_mask": ((substeps,), np.uint8),
            "c01_force_substep_time": ((substeps,), np.float64),
            "c01_force_substep_count": ((1,), np.int32),
            "c01_force_substep_overflow": ((1,), np.int32),
            "c01_force_substep_pre_time": ((substeps,), np.float64),
            "c01_force_substep_pre_qpos": (
                (substeps, int(self.model.nq)),
                np.float64,
            ),
            "c01_force_substep_pre_qvel": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_pre_ctrl": (
                (substeps, int(self.model.nu)),
                np.float64,
            ),
            "c01_force_substep_pre_qfrc_applied": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_pre_xfrc_applied": (
                (substeps, int(self.model.nbody), 6),
                np.float64,
            ),
            "c01_force_substep_pre_qacc_warmstart": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_pre_mocap_pos": (
                (substeps, int(self.model.nmocap), 3),
                np.float64,
            ),
            "c01_force_substep_pre_mocap_quat": (
                (substeps, int(self.model.nmocap), 4),
                np.float64,
            ),
            "c01_force_substep_solve_qacc": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_solve_qfrc_actuator": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_solve_qfrc_bias": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_solve_qfrc_passive": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_solve_qfrc_constraint": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_solve_qfrc_smooth": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_solve_actuator_force": (
                (substeps, int(self.model.nu)),
                np.float64,
            ),
            "c01_force_substep_solve_actuator_length": (
                (substeps, int(self.model.nu)),
                np.float64,
            ),
            "c01_force_substep_solve_actuator_velocity": (
                (substeps, int(self.model.nu)),
                np.float64,
            ),
            "c01_force_substep_post_time": ((substeps,), np.float64),
            "c01_force_substep_post_qpos": (
                (substeps, int(self.model.nq)),
                np.float64,
            ),
            "c01_force_substep_post_qvel": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_post_qacc_warmstart": (
                (substeps, int(self.model.nv)),
                np.float64,
            ),
            "c01_force_substep_hand_joint": (
                (substeps, MAX_HAND_JOINTS, len(_SUBSTEP_HAND_JOINT_COLUMNS)),
                np.float64,
            ),
            "c01_force_substep_hand_actuator": (
                (
                    substeps,
                    MAX_HAND_ACTUATORS,
                    len(_SUBSTEP_HAND_ACTUATOR_COLUMNS),
                ),
                np.float64,
            ),
            "c01_force_contact_mask": ((substeps, contacts), np.uint8),
            "c01_force_contact_ids": (
                (substeps, contacts, len(_CONTACT_ID_COLUMNS)),
                np.int32,
            ),
            "c01_force_contact_pos": ((substeps, contacts, 3), np.float64),
            "c01_force_contact_frame": ((substeps, contacts, 9), np.float64),
            "c01_force_contact_dist": ((substeps, contacts), np.float64),
            "c01_force_contact_wrench_local_on_geom2": (
                (substeps, contacts, 6),
                np.float64,
            ),
            "c01_force_contact_wrench_world_on_geom2": (
                (substeps, contacts, 6),
                np.float64,
            ),
            "c01_force_contact_friction": ((substeps, contacts, 5), np.float64),
            "c01_force_contact_includemargin_mu": (
                (substeps, contacts, 2),
                np.float64,
            ),
            "c01_force_contact_solref": ((substeps, contacts, 4), np.float64),
            "c01_force_contact_solimp": ((substeps, contacts, 5), np.float64),
            "c01_force_contact_overflow": ((substeps,), np.int32),
            "c01_force_hand_joint_mask": ((MAX_HAND_JOINTS,), np.uint8),
            "c01_force_hand_joint_ids": (
                (MAX_HAND_JOINTS, len(_HAND_JOINT_ID_COLUMNS)),
                np.int32,
            ),
            "c01_force_hand_joint_state": (
                (MAX_HAND_JOINTS, len(_HAND_JOINT_STATE_COLUMNS)),
                np.float64,
            ),
            "c01_force_hand_joint_forces": (
                (MAX_HAND_JOINTS, len(_HAND_JOINT_FORCE_COLUMNS)),
                np.float64,
            ),
            "c01_force_hand_joint_dynamics": (
                (MAX_HAND_JOINTS, len(_HAND_JOINT_DYNAMICS_COLUMNS)),
                np.float64,
            ),
            "c01_force_hand_actuator_mask": ((MAX_HAND_ACTUATORS,), np.uint8),
            "c01_force_hand_actuator_ids": (
                (MAX_HAND_ACTUATORS, len(_HAND_ACTUATOR_ID_COLUMNS)),
                np.int32,
            ),
            "c01_force_hand_actuator_state": (
                (MAX_HAND_ACTUATORS, len(_HAND_ACTUATOR_STATE_COLUMNS)),
                np.float64,
            ),
            "c01_force_hand_actuator_dynamics": (
                (MAX_HAND_ACTUATORS, len(_HAND_ACTUATOR_DYNAMICS_COLUMNS)),
                np.float64,
            ),
            "c01_force_object_mask": ((MAX_TASK_OBJECTS,), np.uint8),
            "c01_force_object_ids": (
                (MAX_TASK_OBJECTS, len(_OBJECT_ID_COLUMNS)),
                np.int32,
            ),
            "c01_force_object_pose_world": ((MAX_TASK_OBJECTS, 7), np.float64),
            "c01_force_object_twist_world": ((MAX_TASK_OBJECTS, 6), np.float64),
            "c01_force_object_acceleration_world": (
                (MAX_TASK_OBJECTS, 6),
                np.float64,
            ),
            "c01_force_object_com_world": ((MAX_TASK_OBJECTS, 3), np.float64),
            "c01_force_object_mass_gravcomp": ((MAX_TASK_OBJECTS, 2), np.float64),
            "c01_force_object_inertia": ((MAX_TASK_OBJECTS, 3), np.float64),
            "c01_force_object_inertial_pose_local": (
                (MAX_TASK_OBJECTS, 7),
                np.float64,
            ),
            "c01_force_object_contact_wrench_world": (
                (MAX_TASK_OBJECTS, 6),
                np.float64,
            ),
            "c01_force_object_contact_impulse_world": (
                (MAX_TASK_OBJECTS, 6),
                np.float64,
            ),
            "c01_force_object_contact_counts": ((MAX_TASK_OBJECTS, 2), np.int32),
            "c01_force_object_assist_wrench_world": (
                (MAX_TASK_OBJECTS, 6),
                np.float64,
            ),
            "c01_force_object_gravity_wrench_world": (
                (MAX_TASK_OBJECTS, 6),
                np.float64,
            ),
            "c01_force_object_net_wrench_world": (
                (MAX_TASK_OBJECTS, 6),
                np.float64,
            ),
            "c01_force_qpos": ((int(self.model.nq),), np.float64),
            "c01_force_qvel": ((int(self.model.nv),), np.float64),
            "c01_force_qacc": ((int(self.model.nv),), np.float64),
            "c01_force_ctrl": ((int(self.model.nu),), np.float64),
            "c01_force_qfrc_applied": ((int(self.model.nv),), np.float64),
            "c01_force_xfrc_applied": ((int(self.model.nbody), 6), np.float64),
            "c01_force_mocap_pos": ((int(self.model.nmocap), 3), np.float64),
            "c01_force_mocap_quat": ((int(self.model.nmocap), 4), np.float64),
            "c01_force_sim_time": ((1,), np.float64),
        }
        if int(self.model.na) > 0:
            shapes_and_dtypes["c01_force_substep_pre_act"] = (
                (substeps, int(self.model.na)),
                np.float64,
            )
            shapes_and_dtypes["c01_force_substep_post_act"] = (
                (substeps, int(self.model.na)),
                np.float64,
            )
        if int(self.model.neq) > 0:
            shapes_and_dtypes["c01_force_substep_pre_eq_active"] = (
                (substeps, int(self.model.neq)),
                np.uint8,
            )
        if int(self.model.nJmom) > 0:
            shapes_and_dtypes["c01_force_substep_solve_actuator_moment"] = (
                (substeps, int(self.model.nJmom)),
                np.float64,
            )
            shapes_and_dtypes["c01_force_substep_solve_moment_rownnz"] = (
                (substeps, int(self.model.nu)),
                np.int32,
            )
            shapes_and_dtypes["c01_force_substep_solve_moment_rowadr"] = (
                (substeps, int(self.model.nu)),
                np.int32,
            )
            shapes_and_dtypes["c01_force_substep_solve_moment_colind"] = (
                (substeps, int(self.model.nJmom)),
                np.int32,
            )
        arrays: dict[str, np.ndarray] = {}
        for name, (shape, dtype) in shapes_and_dtypes.items():
            element_count = int(np.prod(shape, dtype=np.int64))
            if element_count > MAX_FIELD_ELEMENTS:
                raise ValueError(
                    f"Telemetry field {name} has {element_count} elements; "
                    f"limit is {MAX_FIELD_ELEMENTS}. Reduce physics substeps or contacts."
                )
            arrays[name] = np.zeros(shape, dtype=dtype)
        return arrays

    def _accumulate_object_contact(
        self,
        destination: np.ndarray,
        counts: np.ndarray,
        role: int,
        wrench_at_contact: np.ndarray,
        contact_pos: np.ndarray,
    ) -> None:
        slot = self._object_slot_by_role.get(role)
        if slot is None:
            return
        body_id = self._object_rows[slot][1]
        if body_id < 0:
            return
        force = wrench_at_contact[:3]
        torque_at_contact = wrench_at_contact[3:]
        # MuJoCo's xfrc_applied and gravity forces are referenced at the body
        # centre of mass, so all resultant task-object wrenches use that same
        # point.  This makes the contact/assist/gravity sum physically valid.
        lever = contact_pos - np.asarray(self.data.xipos[body_id], dtype=np.float64)
        destination[slot, :3] += force
        destination[slot, 3:] += torque_at_contact + np.cross(lever, force)
        counts[slot] += 1

    def _capture_substep_dynamics(self, substep: int) -> None:
        """Capture solve-stage outputs and the integration-result state."""

        self._arrays["c01_force_substep_solve_qacc"][substep] = self.data.qacc
        self._arrays["c01_force_substep_solve_qfrc_actuator"][substep] = (
            self.data.qfrc_actuator
        )
        self._arrays["c01_force_substep_solve_qfrc_bias"][substep] = (
            self.data.qfrc_bias
        )
        self._arrays["c01_force_substep_solve_qfrc_passive"][substep] = (
            self.data.qfrc_passive
        )
        self._arrays["c01_force_substep_solve_qfrc_constraint"][substep] = (
            self.data.qfrc_constraint
        )
        self._arrays["c01_force_substep_solve_qfrc_smooth"][substep] = (
            self.data.qfrc_smooth
        )
        self._arrays["c01_force_substep_solve_actuator_force"][substep] = (
            self.data.actuator_force
        )
        self._arrays["c01_force_substep_solve_actuator_length"][substep] = (
            self.data.actuator_length
        )
        self._arrays["c01_force_substep_solve_actuator_velocity"][substep] = (
            self.data.actuator_velocity
        )
        if "c01_force_substep_solve_actuator_moment" in self._arrays:
            self._arrays["c01_force_substep_solve_actuator_moment"][substep] = (
                self.data.actuator_moment
            )
            self._arrays["c01_force_substep_solve_moment_rownnz"][substep] = (
                self.data.moment_rownnz
            )
            self._arrays["c01_force_substep_solve_moment_rowadr"][substep] = (
                self.data.moment_rowadr
            )
            self._arrays["c01_force_substep_solve_moment_colind"][substep] = (
                self.data.moment_colind
            )

        self._arrays["c01_force_substep_post_qpos"][substep] = self.data.qpos
        self._arrays["c01_force_substep_post_qvel"][substep] = self.data.qvel
        self._arrays["c01_force_substep_post_qacc_warmstart"][substep] = (
            self.data.qacc_warmstart
        )
        if "c01_force_substep_post_act" in self._arrays:
            self._arrays["c01_force_substep_post_act"][substep] = self.data.act

        joint_rows = self._arrays["c01_force_substep_hand_joint"][substep]
        for row, (joint_id, _qpos_adr, dof_adr, _side_code) in enumerate(
            self._hand_joint_rows
        ):
            joint_rows[row] = (
                float(self.data.qacc[dof_adr]),
                float(self.data.qfrc_actuator[dof_adr]),
                float(
                    self._arrays["c01_force_substep_pre_qfrc_applied"][
                        substep, dof_adr
                    ]
                ),
                float(self.data.qfrc_bias[dof_adr]),
                float(self.data.qfrc_passive[dof_adr]),
                float(self.data.qfrc_constraint[dof_adr]),
                float(self.data.qfrc_smooth[dof_adr]),
                float(self.model.jnt_actfrcrange[joint_id, 0]),
                float(self.model.jnt_actfrcrange[joint_id, 1]),
            )

        actuator_rows = self._arrays["c01_force_substep_hand_actuator"][substep]
        for row, (actuator_id, _, _, _) in enumerate(self._hand_actuator_rows):
            actuator_rows[row] = (
                float(
                    self._arrays["c01_force_substep_pre_ctrl"][
                        substep, actuator_id
                    ]
                ),
                float(self.data.actuator_force[actuator_id]),
                float(self.data.actuator_length[actuator_id]),
                float(self.data.actuator_velocity[actuator_id]),
                float(self.model.actuator_forcerange[actuator_id, 0]),
                float(self.model.actuator_forcerange[actuator_id, 1]),
            )

    def _capture_hand_state(self) -> None:
        joint_mask = self._arrays["c01_force_hand_joint_mask"]
        joint_ids = self._arrays["c01_force_hand_joint_ids"]
        joint_state = self._arrays["c01_force_hand_joint_state"]
        joint_forces = self._arrays["c01_force_hand_joint_forces"]
        joint_dynamics = self._arrays["c01_force_hand_joint_dynamics"]
        for row, (joint_id, qpos_adr, dof_adr, side_code) in enumerate(
            self._hand_joint_rows
        ):
            joint_mask[row] = 1
            joint_ids[row] = (joint_id, qpos_adr, dof_adr, side_code)
            joint_state[row] = (
                float(self.data.qpos[qpos_adr]),
                float(self.data.qvel[dof_adr]),
                float(self.data.qacc[dof_adr]),
                float(self.data.qfrc_actuator[dof_adr]),
            )
            joint_forces[row] = (
                float(self.data.qfrc_applied[dof_adr]),
                float(self.data.qfrc_bias[dof_adr]),
                float(self.data.qfrc_passive[dof_adr]),
                float(self.data.qfrc_constraint[dof_adr]),
                float(self.data.qfrc_smooth[dof_adr]),
                float(self.data.qfrc_inverse[dof_adr]),
            )
            joint_dynamics[row] = (
                float(self.model.jnt_actfrcrange[joint_id, 0]),
                float(self.model.jnt_actfrcrange[joint_id, 1]),
                float(self.model.dof_damping[dof_adr]),
                float(self.model.dof_armature[dof_adr]),
                float(self.model.dof_frictionloss[dof_adr]),
            )

        actuator_mask = self._arrays["c01_force_hand_actuator_mask"]
        actuator_ids = self._arrays["c01_force_hand_actuator_ids"]
        actuator_state = self._arrays["c01_force_hand_actuator_state"]
        actuator_dynamics = self._arrays["c01_force_hand_actuator_dynamics"]
        for row, (actuator_id, target_id, trn_type, side_code) in enumerate(
            self._hand_actuator_rows
        ):
            actuator_mask[row] = 1
            actuator_ids[row] = (actuator_id, target_id, trn_type, side_code)
            act_value = 0.0
            act_adr = int(self.model.actuator_actadr[actuator_id])
            if act_adr >= 0 and act_adr < self.data.act.size:
                act_value = float(self.data.act[act_adr])
            actuator_state[row] = (
                float(self.data.ctrl[actuator_id]),
                float(self.data.actuator_force[actuator_id]),
                float(self.data.actuator_length[actuator_id]),
                float(self.data.actuator_velocity[actuator_id]),
                act_value,
            )
            actuator_dynamics[row] = (
                float(self.model.actuator_ctrlrange[actuator_id, 0]),
                float(self.model.actuator_ctrlrange[actuator_id, 1]),
                float(self.model.actuator_forcerange[actuator_id, 0]),
                float(self.model.actuator_forcerange[actuator_id, 1]),
                float(self.model.actuator_gainprm[actuator_id, 0]),
                float(self.model.actuator_biasprm[actuator_id, 0]),
                float(self.model.actuator_biasprm[actuator_id, 1]),
                float(self.model.actuator_biasprm[actuator_id, 2]),
            )

    def _capture_object_state(self) -> None:
        mask = self._arrays["c01_force_object_mask"]
        ids = self._arrays["c01_force_object_ids"]
        pose = self._arrays["c01_force_object_pose_world"]
        twist = self._arrays["c01_force_object_twist_world"]
        acceleration = self._arrays["c01_force_object_acceleration_world"]
        com = self._arrays["c01_force_object_com_world"]
        mass_gravcomp = self._arrays["c01_force_object_mass_gravcomp"]
        inertia = self._arrays["c01_force_object_inertia"]
        inertial_pose = self._arrays["c01_force_object_inertial_pose_local"]
        contact_wrench = self._arrays["c01_force_object_contact_wrench_world"]
        contact_impulse = self._arrays["c01_force_object_contact_impulse_world"]
        contact_counts = self._arrays["c01_force_object_contact_counts"]
        assist = self._arrays["c01_force_object_assist_wrench_world"]
        gravity = self._arrays["c01_force_object_gravity_wrench_world"]
        net = self._arrays["c01_force_object_net_wrench_world"]

        for slot, (role, body_id, joint_id) in enumerate(self._object_rows):
            ids[slot] = (role, body_id, joint_id)
            if body_id < 0:
                continue
            mask[slot] = 1
            pose[slot, :3] = self.data.xpos[body_id]
            pose[slot, 3:] = self.data.xquat[body_id]
            spatial_velocity = np.zeros(6, dtype=np.float64)
            spatial_acceleration = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                spatial_velocity,
                0,
            )
            mujoco.mj_objectAcceleration(
                self.model,
                self.data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                spatial_acceleration,
                0,
            )
            # MuJoCo spatial vectors are [angular, linear]; observations use
            # [linear, angular] to match the force/torque wrench ordering.
            twist[slot] = np.concatenate(
                (spatial_velocity[3:], spatial_velocity[:3])
            )
            acceleration[slot] = np.concatenate(
                (spatial_acceleration[3:], spatial_acceleration[:3])
            )
            com[slot] = self.data.xipos[body_id]
            body_mass = float(self.model.body_mass[body_id])
            gravcomp = float(self.model.body_gravcomp[body_id])
            mass_gravcomp[slot] = (body_mass, gravcomp)
            inertia[slot] = self.model.body_inertia[body_id]
            inertial_pose[slot, :3] = self.model.body_ipos[body_id]
            inertial_pose[slot, 3:] = self.model.body_iquat[body_id]

            contact_wrench[slot] = self._last_object_contact_wrench[slot]
            contact_impulse[slot] = self._frame_object_contact_impulse[slot]
            contact_counts[slot] = (
                self._last_object_contact_count[slot],
                self._frame_object_contact_count[slot],
            )
            assist[slot] = self._last_applied_xfrc[body_id]

            gravity_force = (
                body_mass
                * (1.0 - gravcomp)
                * np.asarray(self.model.opt.gravity, dtype=np.float64)
            )
            gravity[slot, :3] = gravity_force
            # Gravity acts through the centre of mass, hence zero torque when
            # expressed about the common COM reference point.
            gravity[slot, 3:] = 0.0
            net[slot] = contact_wrench[slot] + assist[slot] + gravity[slot]

    def _capture_raw_state(self) -> None:
        self._arrays["c01_force_qpos"][...] = self.data.qpos
        self._arrays["c01_force_qvel"][...] = self.data.qvel
        self._arrays["c01_force_qacc"][...] = self.data.qacc
        self._arrays["c01_force_ctrl"][...] = self.data.ctrl
        self._arrays["c01_force_qfrc_applied"][...] = self.data.qfrc_applied
        self._arrays["c01_force_xfrc_applied"][...] = self.data.xfrc_applied
        self._arrays["c01_force_mocap_pos"][...] = self.data.mocap_pos
        self._arrays["c01_force_mocap_quat"][...] = self.data.mocap_quat
        self._arrays["c01_force_sim_time"][0] = float(self.data.time)

    # ------------------------------------------------------------------
    # Spaces and static metadata
    # ------------------------------------------------------------------

    def _build_spaces(self) -> dict[str, gym.Space]:
        result: dict[str, gym.Space] = {}
        int_info = np.iinfo(np.int32)
        for name, array in self._arrays.items():
            if array.dtype == np.uint8:
                result[name] = gym.spaces.Box(
                    low=0,
                    high=1,
                    shape=array.shape,
                    dtype=np.uint8,
                )
            elif array.dtype == np.int32:
                result[name] = gym.spaces.Box(
                    low=int_info.min,
                    high=int_info.max,
                    shape=array.shape,
                    dtype=np.int32,
                )
            else:
                result[name] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=array.shape,
                    dtype=np.float64,
                )
        return result

    def _build_schema(self) -> dict[str, Any]:
        fields: dict[str, dict[str, Any]] = {}
        for name, array in self._arrays.items():
            fields[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "element_count": int(array.size),
            }

        field_details: dict[str, dict[str, Any]] = {
            "c01_force_substep_time": {
                "unit": "s",
                "clock": "MuJoCo data.time",
                "meaning": "deprecated alias of c01_force_substep_post_time",
            },
            "c01_force_substep_pre_time": {
                "unit": "s",
                "clock": "MuJoCo data.time",
                "stage": "immediately before mj_step",
                "meaning": "time t at which contacts, actuation and qacc are evaluated",
            },
            "c01_force_substep_pre_qpos": {
                "unit": "joint-specific m/rad/quaternion",
                "stage": "immediately before mj_step",
            },
            "c01_force_substep_pre_qvel": {
                "unit": "DOF-specific m/s or rad/s",
                "stage": "immediately before mj_step",
            },
            "c01_force_substep_pre_ctrl": {
                "unit": "actuator-specific; position servos use m or rad",
                "meaning": "exact data.ctrl input used by this mj_step",
            },
            "c01_force_substep_pre_qfrc_applied": {
                "unit": "generalized N for slide/free translation or N*m for rotation",
                "meaning": "exact data.qfrc_applied input used by this mj_step",
            },
            "c01_force_substep_pre_xfrc_applied": {
                "columns_last_axis": ("fx", "fy", "fz", "tx", "ty", "tz"),
                "unit": ("N", "N", "N", "N*m", "N*m", "N*m"),
                "coordinates": "world about each body COM",
                "meaning": "exact data.xfrc_applied input used by this mj_step",
            },
            "c01_force_substep_pre_qacc_warmstart": {
                "unit": "DOF-specific m/s^2 or rad/s^2",
                "meaning": "constraint-solver warmstart input at time t",
            },
            "c01_force_substep_pre_mocap_pos": {
                "unit": "m",
                "coordinates": "world",
                "meaning": "mocap body input at time t",
            },
            "c01_force_substep_pre_mocap_quat": {
                "columns_last_axis": ("qw", "qx", "qy", "qz"),
                "coordinates": "world orientation",
                "meaning": "mocap body input at time t",
            },
            "c01_force_substep_solve_qacc": {
                "unit": "DOF-specific m/s^2 or rad/s^2",
                "stage": "forward-dynamics solve at pre-step state t",
            },
            "c01_force_substep_solve_qfrc_actuator": {
                "unit": "generalized N or N*m",
                "stage": "forward-dynamics solve at pre-step state t",
            },
            "c01_force_substep_solve_qfrc_bias": {
                "unit": "generalized N or N*m",
                "meaning": "Coriolis, centrifugal and gravitational bias force at t",
            },
            "c01_force_substep_solve_qfrc_passive": {
                "unit": "generalized N or N*m",
                "meaning": "spring, damping, gravcomp and fluid passive force at t",
            },
            "c01_force_substep_solve_qfrc_constraint": {
                "unit": "generalized N or N*m",
                "meaning": "all resolved constraint forces, including contacts, at t",
            },
            "c01_force_substep_solve_qfrc_smooth": {
                "unit": "generalized N or N*m",
                "meaning": "MuJoCo net unconstrained generalized force at t",
            },
            "c01_force_substep_solve_actuator_force": {
                "unit": "actuation-space force; transmission-specific",
                "meaning": "data.actuator_force from the forward solve at t",
            },
            "c01_force_substep_solve_actuator_length": {
                "unit": "transmission-specific m or rad",
                "meaning": "data.actuator_length evaluated at pre-step state t",
            },
            "c01_force_substep_solve_actuator_velocity": {
                "unit": "transmission-specific m/s or rad/s",
                "meaning": "data.actuator_velocity evaluated at pre-step state t",
            },
            "c01_force_substep_post_time": {
                "unit": "s",
                "clock": "MuJoCo data.time",
                "stage": "immediately after mj_step integration",
                "meaning": "time t+h",
            },
            "c01_force_substep_post_qpos": {
                "unit": "joint-specific m/rad/quaternion",
                "stage": "integration result at t+h",
            },
            "c01_force_substep_post_qvel": {
                "unit": "DOF-specific m/s or rad/s",
                "stage": "integration result at t+h",
            },
            "c01_force_substep_post_qacc_warmstart": {
                "unit": "DOF-specific m/s^2 or rad/s^2",
                "meaning": "warmstart state after the solve/integration",
            },
            "c01_force_substep_hand_joint": {
                "columns": _SUBSTEP_HAND_JOINT_COLUMNS,
                "unit": "column-specific acceleration/generalized N or N*m",
                "meaning": "solve-stage values and active force limits at pre-step state t, aligned with contact rows",
            },
            "c01_force_substep_hand_actuator": {
                "columns": _SUBSTEP_HAND_ACTUATOR_COLUMNS,
                "meaning": "ctrl input, solve-stage outputs, and active force limits at pre-step state t",
            },
            "c01_force_contact_ids": {"columns": _CONTACT_ID_COLUMNS},
            "c01_force_contact_pos": {"unit": "m", "coordinates": "world"},
            "c01_force_contact_frame": {
                "unit": "dimensionless",
                "layout": "row-major flattened 3x3 matrix",
                "coordinates": "matrix rows are contact-frame axes expressed in world coordinates",
            },
            "c01_force_contact_dist": {"unit": "m"},
            "c01_force_contact_wrench_local_on_geom2": {
                "columns": ("fx", "fy", "fz", "tx", "ty", "tz"),
                "unit": ("N", "N", "N", "N*m", "N*m", "N*m"),
                "coordinates": "MuJoCo contact frame",
                "sign": "wrench acting on geom2 exactly as returned by mj_contactForce",
            },
            "c01_force_contact_wrench_world_on_geom2": {
                "columns": ("fx", "fy", "fz", "tx", "ty", "tz"),
                "unit": ("N", "N", "N", "N*m", "N*m", "N*m"),
                "coordinates": "world",
                "sign": "wrench acting on geom2; force and torque each equal contact.frame.T @ local triple",
            },
            "c01_force_contact_friction": {
                "columns": (
                    "sliding_tangent_1",
                    "sliding_tangent_2",
                    "torsional",
                    "rolling_tangent_1",
                    "rolling_tangent_2",
                )
            },
            "c01_force_contact_includemargin_mu": {
                "columns": ("includemargin_m", "friction_pyramid_mu")
            },
            "c01_force_contact_solref": {
                "columns": (
                    "normal_solref_0",
                    "normal_solref_1",
                    "friction_solref_0",
                    "friction_solref_1",
                )
            },
            "c01_force_contact_solimp": {
                "columns": ("dmin", "dmax", "width", "midpoint", "power")
            },
            "c01_force_contact_overflow": {
                "meaning": "relevant contacts omitted after max_contacts in each substep"
            },
            "c01_force_hand_joint_ids": {"columns": _HAND_JOINT_ID_COLUMNS},
            "c01_force_hand_joint_state": {
                "columns": _HAND_JOINT_STATE_COLUMNS,
                "unit": ("rad_or_m", "rad/s_or_m/s", "rad/s^2_or_m/s^2", "N*m_or_N"),
            },
            "c01_force_hand_joint_forces": {
                "columns": _HAND_JOINT_FORCE_COLUMNS,
                "unit": "generalized N*m for hinge or N for slide",
            },
            "c01_force_hand_joint_dynamics": {
                "columns": _HAND_JOINT_DYNAMICS_COLUMNS,
                "meaning": "frame-current joint force limits and passive dynamics; includes runtime squeeze scaling",
            },
            "c01_force_hand_actuator_ids": {"columns": _HAND_ACTUATOR_ID_COLUMNS},
            "c01_force_hand_actuator_state": {
                "columns": _HAND_ACTUATOR_STATE_COLUMNS,
                "note": "actuator_force is scalar transmission force before moment-arm projection",
            },
            "c01_force_hand_actuator_dynamics": {
                "columns": _HAND_ACTUATOR_DYNAMICS_COLUMNS,
                "note": "frame-current values; preserves runtime squeeze-force scaling",
            },
            "c01_force_object_ids": {"columns": _OBJECT_ID_COLUMNS},
            "c01_force_object_pose_world": {
                "columns": ("x", "y", "z", "qw", "qx", "qy", "qz"),
                "unit": ("m", "m", "m", "dimensionless", "dimensionless", "dimensionless", "dimensionless"),
            },
            "c01_force_object_twist_world": {
                "columns": ("vx", "vy", "vz", "wx", "wy", "wz"),
                "unit": ("m/s", "m/s", "m/s", "rad/s", "rad/s", "rad/s"),
                "coordinates": "world, evaluated at body frame",
            },
            "c01_force_object_acceleration_world": {
                "columns": ("ax", "ay", "az", "alphax", "alphay", "alphaz"),
                "unit": ("m/s^2", "m/s^2", "m/s^2", "rad/s^2", "rad/s^2", "rad/s^2"),
                "coordinates": "world, evaluated at body frame",
            },
            "c01_force_object_mass_gravcomp": {
                "columns": ("body_mass_kg", "body_gravity_compensation_fraction"),
                "dynamic": True,
            },
            "c01_force_object_inertia": {
                "columns": ("principal_Ixx", "principal_Iyy", "principal_Izz"),
                "unit": "kg*m^2",
                "coordinates": "body inertial frame",
                "dynamic": True,
            },
            "c01_force_object_inertial_pose_local": {
                "columns": ("x", "y", "z", "qw", "qx", "qy", "qz"),
                "coordinates": "inertial frame pose relative to body frame",
                "dynamic": True,
            },
            "c01_force_object_contact_wrench_world": {
                "columns": ("fx", "fy", "fz", "tx", "ty", "tz"),
                "unit": ("N", "N", "N", "N*m", "N*m", "N*m"),
                "coordinates": "world; torque shifted to task body centre of mass (data.xipos)",
                "meaning": "resultant from the last captured physics substep",
            },
            "c01_force_object_contact_impulse_world": {
                "columns": ("jx", "jy", "jz", "lx", "ly", "lz"),
                "unit": ("N*s", "N*s", "N*s", "N*m*s", "N*m*s", "N*m*s"),
                "coordinates": "world about task body COM; rectangular integration over captured substeps",
            },
            "c01_force_object_contact_counts": {
                "columns": _OBJECT_CONTACT_COUNT_COLUMNS
            },
            "c01_force_object_assist_wrench_world": {
                "columns": ("fx", "fy", "fz", "tx", "ty", "tz"),
                "unit": ("N", "N", "N", "N*m", "N*m", "N*m"),
                "coordinates": "world about body COM",
                "meaning": "xfrc_applied actually used by the last captured mj_step; frame-final raw xfrc_applied may already target the next frame",
            },
            "c01_force_object_gravity_wrench_world": {
                "columns": ("fx", "fy", "fz", "tx", "ty", "tz"),
                "unit": ("N", "N", "N", "N*m", "N*m", "N*m"),
                "coordinates": "world about body COM; effective gravity after body_gravcomp, torque is zero",
            },
            "c01_force_object_net_wrench_world": {
                "meaning": "last contact + assist + effective gravity; excludes unrecorded internal/passive/constraint terms",
                "coordinates": "world about task body COM",
            },
            "c01_force_qpos": {"meaning": "exact frame-final data.qpos"},
            "c01_force_qvel": {"meaning": "exact frame-final data.qvel"},
            "c01_force_qacc": {"meaning": "exact frame-final data.qacc"},
            "c01_force_ctrl": {"meaning": "exact frame-final data.ctrl"},
            "c01_force_qfrc_applied": {"meaning": "exact frame-final data.qfrc_applied"},
            "c01_force_xfrc_applied": {
                "meaning": "exact frame-final data.xfrc_applied for every body",
                "coordinates": "world; force then torque",
            },
            "c01_force_mocap_pos": {"unit": "m", "coordinates": "world"},
            "c01_force_mocap_quat": {
                "columns": ("qw", "qx", "qy", "qz"),
                "coordinates": "world orientation",
            },
            "c01_force_sim_time": {"unit": "s", "clock": "MuJoCo data.time"},
        }
        if "c01_force_substep_pre_act" in self._arrays:
            field_details["c01_force_substep_pre_act"] = {
                "unit": "actuator-specific activation state",
                "stage": "immediately before mj_step",
            }
            field_details["c01_force_substep_post_act"] = {
                "unit": "actuator-specific activation state",
                "stage": "integration result at t+h",
            }
        if "c01_force_substep_pre_eq_active" in self._arrays:
            field_details["c01_force_substep_pre_eq_active"] = {
                "unit": "boolean",
                "meaning": "exact equality-constraint enable inputs at time t",
            }
        if "c01_force_substep_solve_actuator_moment" in self._arrays:
            field_details["c01_force_substep_solve_actuator_moment"] = {
                "unit": "transmission derivative / moment-arm, transmission-specific",
                "layout": "MuJoCo sparse actuator_moment values; use same-substep rownnz/rowadr/colind fields",
                "stage": "forward-dynamics solve at pre-step state t",
            }
            field_details["c01_force_substep_solve_moment_rownnz"] = {
                "meaning": "nonzero count for each actuator_moment sparse row at t",
            }
            field_details["c01_force_substep_solve_moment_rowadr"] = {
                "meaning": "start address for each actuator_moment sparse row at t",
            }
            field_details["c01_force_substep_solve_moment_colind"] = {
                "meaning": "generalized-DOF column index for each sparse moment value at t",
            }
        for field_name, details in field_details.items():
            fields[field_name].update(_json_safe(details))

        geometry_mapping = []
        for geom_id in sorted(self._relevant_geom_ids):
            body_id = int(self.model.geom_bodyid[geom_id])
            geometry_mapping.append(
                {
                    "geom_id": geom_id,
                    "geom_name": _name(
                        self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
                    ),
                    "body_id": body_id,
                    "body_name": _name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    ),
                    "role_code": self._geom_roles[geom_id],
                    "role": ROLE_CODEBOOK[self._geom_roles[geom_id]],
                    "type": int(self.model.geom_type[geom_id]),
                    "size": self.model.geom_size[geom_id],
                    "pos": self.model.geom_pos[geom_id],
                    "quat": self.model.geom_quat[geom_id],
                    "contype": int(self.model.geom_contype[geom_id]),
                    "conaffinity": int(self.model.geom_conaffinity[geom_id]),
                    "condim": int(self.model.geom_condim[geom_id]),
                    "priority": int(self.model.geom_priority[geom_id]),
                    "margin": float(self.model.geom_margin[geom_id]),
                    "gap": float(self.model.geom_gap[geom_id]),
                    "solref": self.model.geom_solref[geom_id],
                    "solimp": self.model.geom_solimp[geom_id],
                    "friction_at_schema_time": self.model.geom_friction[geom_id],
                }
            )

        joint_mapping = []
        for joint_id, qpos_adr, dof_adr, side_code in self._hand_joint_rows:
            joint_mapping.append(
                {
                    "joint_id": joint_id,
                    "joint_name": _name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                    ),
                    "qpos_address": qpos_adr,
                    "dof_address": dof_adr,
                    "side_code": side_code,
                    "side": ROLE_CODEBOOK.get(side_code, "unknown"),
                    "joint_type": int(self.model.jnt_type[joint_id]),
                    "range": self.model.jnt_range[joint_id],
                }
            )

        actuator_mapping = []
        for actuator_id, target_id, trn_type, side_code in self._hand_actuator_rows:
            actuator_mapping.append(
                {
                    "actuator_id": actuator_id,
                    "actuator_name": _name(
                        self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id
                    ),
                    "target_id": target_id,
                    "transmission_type": trn_type,
                    "side_code": side_code,
                    "side": ROLE_CODEBOOK.get(side_code, "unknown"),
                    "gear": self.model.actuator_gear[actuator_id],
                    "gainprm_at_schema_time": self.model.actuator_gainprm[actuator_id],
                    "biasprm_at_schema_time": self.model.actuator_biasprm[actuator_id],
                    "ctrlrange_at_schema_time": self.model.actuator_ctrlrange[actuator_id],
                    "forcerange_at_schema_time": self.model.actuator_forcerange[actuator_id],
                }
            )

        body_mapping = []
        for role, body_id, joint_id in self._object_rows:
            body_mapping.append(
                {
                    "role_code": role,
                    "role": ROLE_CODEBOOK[role],
                    "body_id": body_id,
                    "body_name": _name(
                        self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
                    ),
                    "root_joint_id": joint_id,
                    "root_joint_name": _name(
                        self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
                    ),
                    "parent_body_id": (
                        int(self.model.body_parentid[body_id]) if body_id >= 0 else -1
                    ),
                    "mass_at_schema_time": (
                        float(self.model.body_mass[body_id]) if body_id >= 0 else 0.0
                    ),
                    "inertia_at_schema_time": (
                        self.model.body_inertia[body_id]
                        if body_id >= 0
                        else np.zeros(3, dtype=np.float64)
                    ),
                    "ipos_at_schema_time": (
                        self.model.body_ipos[body_id]
                        if body_id >= 0
                        else np.zeros(3, dtype=np.float64)
                    ),
                    "iquat_at_schema_time": (
                        self.model.body_iquat[body_id]
                        if body_id >= 0
                        else np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
                    ),
                }
            )

        opt = self.model.opt
        model_buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
        mujoco.mj_saveModel(self.model, buffer=model_buffer)
        self._compiled_model_mjb = model_buffer.tobytes()
        return {
            "schema_name": "handson.plan1_force_telemetry",
            "schema_version": 1,
            "observation_prefix": "c01_force_",
            "max_contacts_per_substep": self.max_contacts,
            "max_hand_joints": MAX_HAND_JOINTS,
            "max_hand_actuators": MAX_HAND_ACTUATORS,
            "max_task_objects": MAX_TASK_OBJECTS,
            "role_codebook": {str(key): value for key, value in ROLE_CODEBOOK.items()},
            "contact_convention": {
                "local": "mj_contactForce wrench acting on geom2",
                "world": "apply contact.frame.T separately to force and torque",
                "geom1": "receives equal-and-opposite wrench",
            },
            "fields": fields,
            "mujoco": {
                "version": mujoco.mj_versionString(),
                "compiled_model_sha256": hashlib.sha256(
                    self._compiled_model_mjb
                ).hexdigest(),
                "physics_dt_s": float(opt.timestep),
                "control_dt_s": float(
                    getattr(self.env, "control_dt", self.n_substeps * opt.timestep)
                ),
                "n_substeps": self.n_substeps,
                "gravity_m_per_s2": opt.gravity,
                "solver": {
                    "type": _enum_name(mujoco.mjtSolver, opt.solver),
                    "iterations": int(opt.iterations),
                    "ls_iterations": int(opt.ls_iterations),
                    "noslip_iterations": int(opt.noslip_iterations),
                    "tolerance": float(opt.tolerance),
                    "noslip_tolerance": float(opt.noslip_tolerance),
                },
                "integrator": _enum_name(mujoco.mjtIntegrator, opt.integrator),
                "cone": _enum_name(mujoco.mjtCone, opt.cone),
                "jacobian": _enum_name(mujoco.mjtJacobian, opt.jacobian),
                "disableflags": int(opt.disableflags),
                "enableflags": int(opt.enableflags),
                "model_sizes": {
                    "nq": int(self.model.nq),
                    "nv": int(self.model.nv),
                    "nu": int(self.model.nu),
                    "nbody": int(self.model.nbody),
                    "ngeom": int(self.model.ngeom),
                    "nmocap": int(self.model.nmocap),
                },
            },
            "mappings": {
                "hand_joints": joint_mapping,
                "hand_actuators": actuator_mapping,
                "task_bodies": body_mapping,
                "relevant_geoms": geometry_mapping,
            },
            "dynamic_parameters": {
                "mass_and_inertia": (
                    "Read c01_force_object_mass_gravcomp, c01_force_object_inertia, "
                    "and c01_force_object_inertial_pose_local every frame. They are "
                    "observations, not schema constants, because domain randomization "
                    "may change them between episodes."
                ),
                "actuator_strength": (
                    "Read c01_force_substep_hand_actuator and "
                    "c01_force_substep_hand_joint for the active actuator/joint "
                    "force limits at every solve; frame-current limits also appear "
                    "in the corresponding hand dynamics fields."
                ),
                "contact_material": (
                    "Per-contact friction/solref/solimp are recorded at every physics "
                    "substep and are authoritative over the schema-time geom snapshot."
                ),
            },
            "capture_contract": (
                "begin_frame once; for each env._n_substeps iteration call "
                "capture_pre_substep immediately before mj_step and capture_substep "
                "immediately after it; then call finalize_frame once"
            ),
        }


__all__ = [
    "MAX_CONTACTS",
    "MAX_FIELD_ELEMENTS",
    "Plan1ForceTelemetry",
    "ROLE_CODEBOOK",
]
