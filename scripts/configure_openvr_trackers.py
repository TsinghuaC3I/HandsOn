#!/usr/bin/env python3
"""Resolve wrist trackers once and write a machine-local OpenVR config."""

from __future__ import annotations

import argparse
import itertools
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import yaml


WRIST_ROLES = ("right_wrist", "left_wrist")
ROLE_ENV_NAMES = {
    "right_wrist": ("tracker_right", "TRACKER_RIGHT"),
    "left_wrist": ("tracker_left", "TRACKER_LEFT"),
}
AUTO_REFERENCES = frozenset({"", "auto", "any", "*", "lhr-xxxxxxxx"})
INDEX_REFERENCE = re.compile(r"^(?:(?:index\s*[:=]\s*)|#)?([0-9]+)$", re.IGNORECASE)
AUTO_POSE_WAIT_SECONDS = 5.0
AUTO_POSE_POLL_SECONDS = 0.1


class TrackerBindingError(RuntimeError):
    """Raised when tracker references cannot be resolved safely."""


@dataclass(frozen=True)
class TrackerDevice:
    index: int
    serial: str
    x: float | None = None
    y: float | None = None
    z: float | None = None


@dataclass(frozen=True)
class TrackingReferenceDevice:
    index: int
    serial: str
    x: float | None = None
    y: float | None = None
    z: float | None = None


def is_auto_reference(value: object) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in AUTO_REFERENCES


def environment_requests(environ: Mapping[str, str]) -> dict[str, str]:
    """Return explicit role requests, preferring lowercase compatibility names."""
    requests: dict[str, str] = {}
    for role, names in ROLE_ENV_NAMES.items():
        for name in names:
            value = environ.get(name, "").strip()
            if value:
                requests[role] = value
                break
    return requests


def parse_index_reference(value: str) -> int | None:
    match = INDEX_REFERENCE.fullmatch(value.strip())
    return int(match.group(1)) if match else None


def bindings_need_devices(
    configured: Mapping[str, object], requests: Mapping[str, str]
) -> bool:
    for role in WRIST_ROLES:
        if role in requests:
            requested = requests[role]
            if is_auto_reference(requested) or parse_index_reference(requested) is not None:
                return True
        elif is_auto_reference(configured.get(role)):
            return True
    return False


def automatic_roles(
    configured: Mapping[str, object], requests: Mapping[str, str]
) -> list[str]:
    roles: list[str] = []
    for role in WRIST_ROLES:
        reference: object = requests.get(role, configured.get(role, "auto"))
        if is_auto_reference(reference):
            roles.append(role)
    return roles


def _serial_key(serial: str) -> str:
    return serial.strip().casefold()


def _has_valid_x(device: TrackerDevice) -> bool:
    return device.x is not None and math.isfinite(float(device.x))


def _position(
    device: TrackerDevice | TrackingReferenceDevice,
) -> tuple[float, float, float] | None:
    values = (device.x, device.y, device.z)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        return None
    return tuple(float(value) for value in values)  # type: ignore[arg-type, return-value]


def _resolve_explicit_reference(
    role: str, reference: str, devices: Sequence[TrackerDevice]
) -> str:
    index = parse_index_reference(reference)
    if index is None:
        return reference.strip()

    matches = [device for device in devices if device.index == index]
    if not matches:
        available = ", ".join(
            str(device.index)
            for device in sorted(devices, key=lambda item: item.index)
        )
        raise TrackerBindingError(
            f"{role} requests OpenVR index {index}, but available tracker indices are: "
            f"{available or 'none'}"
        )
    return matches[0].serial


def auto_bind_roles(
    roles: Sequence[str],
    devices: Sequence[TrackerDevice],
    assigned_serials: set[str],
) -> dict[str, str]:
    """Bind automatic roles only when their identity is unambiguous."""
    candidates = [
        device
        for device in sorted(devices, key=lambda item: item.index)
        if _serial_key(device.serial) not in assigned_serials
    ]
    remaining_roles = list(roles)
    if {"right_wrist", "left_wrist"}.issubset(remaining_roles):
        if len(candidates) != 2:
            raise TrackerBindingError(
                "automatic right/left binding needs exactly two unassigned OpenVR "
                f"trackers, but found {len(candidates)}; set tracker_right and "
                "tracker_left to explicit device indices"
            )
        posed = [device for device in candidates if _has_valid_x(device)]
        if len(posed) < 2:
            raise TrackerBindingError(
                "automatic right/left binding needs two valid OpenVR tracker poses, "
                f"but found {len(posed)}; set tracker_right and tracker_left to the "
                "device indices printed by list_vive_trackers.py"
            )
        posed.sort(key=lambda item: float(item.x), reverse=True)
        right, left = posed
        if abs(float(right.x) - float(left.x)) < 1e-4:
            raise TrackerBindingError(
                "automatic right/left binding cannot distinguish tracker X positions; "
                "set tracker_right and tracker_left to explicit device indices"
            )
        return {"right_wrist": right.serial, "left_wrist": left.serial}

    if len(remaining_roles) != 1:
        raise TrackerBindingError(
            "automatic tracker binding requires a right_wrist or left_wrist role"
        )
    if len(candidates) != 1:
        raise TrackerBindingError(
            f"automatic binding for {remaining_roles[0]} needs exactly one unassigned "
            f"OpenVR tracker, but found {len(candidates)}; set its tracker_left or "
            "tracker_right environment variable to an explicit device index"
        )
    return {remaining_roles[0]: candidates[0].serial}


def resolve_bindings(
    configured: Mapping[str, object],
    requests: Mapping[str, str],
    devices: Sequence[TrackerDevice],
) -> dict[str, str]:
    """Resolve environment overrides, persisted bindings, then remaining auto roles."""
    bindings: dict[str, str] = {}
    assigned_serials: set[str] = set()
    auto_roles: list[str] = []

    for role in WRIST_ROLES:
        if role in requests:
            reference: object = requests[role]
        else:
            reference = configured.get(role, "auto")

        if is_auto_reference(reference):
            auto_roles.append(role)
            continue

        serial = _resolve_explicit_reference(role, str(reference), devices)
        key = _serial_key(serial)
        if key in assigned_serials:
            raise TrackerBindingError(
                f"the same tracker is assigned more than once: {serial}"
            )
        bindings[role] = serial
        assigned_serials.add(key)

    if auto_roles:
        automatic = auto_bind_roles(auto_roles, devices, assigned_serials)
        for role in auto_roles:
            serial = automatic[role]
            key = _serial_key(serial)
            if key in assigned_serials:
                raise TrackerBindingError(
                    f"the same tracker is assigned more than once: {serial}"
                )
            bindings[role] = serial
            assigned_serials.add(key)

    return {role: bindings[role] for role in WRIST_ROLES}


def enumerate_openvr_trackers(
    *,
    wait_for_poses: bool = False,
    pose_wait_seconds: float = AUTO_POSE_WAIT_SECONDS,
    pose_poll_seconds: float = AUTO_POSE_POLL_SECONDS,
) -> list[TrackerDevice]:
    try:
        import openvr
    except ImportError as exc:
        raise TrackerBindingError(
            "OpenVR Python package is unavailable; source the ROS/OpenVR workspace first"
        ) from exc

    initialized = False
    try:
        openvr.init(openvr.VRApplication_Other)
        initialized = True
        vr_system = openvr.VRSystem()
        deadline = time.monotonic() + max(0.0, float(pose_wait_seconds))
        while True:
            try:
                poses = vr_system.getDeviceToAbsoluteTrackingPose(
                    openvr.TrackingUniverseStanding,
                    0,
                    openvr.k_unMaxTrackedDeviceCount,
                )
            except Exception:
                poses = []

            devices: list[TrackerDevice] = []
            for index in range(openvr.k_unMaxTrackedDeviceCount):
                if (
                    vr_system.getTrackedDeviceClass(index)
                    != openvr.TrackedDeviceClass_GenericTracker
                ):
                    continue
                serial = vr_system.getStringTrackedDeviceProperty(
                    index, openvr.Prop_SerialNumber_String
                ).strip()
                if not serial:
                    continue
                x: float | None = None
                y: float | None = None
                z: float | None = None
                if index < len(poses) and poses[index].bPoseIsValid:
                    candidate_position = tuple(
                        float(poses[index].mDeviceToAbsoluteTracking.m[axis][3])
                        for axis in range(3)
                    )
                    if all(math.isfinite(value) for value in candidate_position):
                        x, y, z = candidate_position
                devices.append(
                    TrackerDevice(index=index, serial=serial, x=x, y=y, z=z)
                )

            valid_pose_count = sum(_has_valid_x(device) for device in devices)
            if (
                not wait_for_poses
                or valid_pose_count >= 2
                or time.monotonic() >= deadline
            ):
                return devices
            time.sleep(max(0.0, float(pose_poll_seconds)))
    except TrackerBindingError:
        raise
    except Exception as exc:
        raise TrackerBindingError(
            f"cannot enumerate Vive Trackers through OpenVR: {exc}"
        ) from exc
    finally:
        if initialized:
            try:
                openvr.shutdown()
            except Exception:
                pass


def enumerate_openvr_tracking_references(
    *,
    pose_wait_seconds: float = AUTO_POSE_WAIT_SECONDS,
    pose_poll_seconds: float = AUTO_POSE_POLL_SECONDS,
) -> list[TrackingReferenceDevice]:
    """Return pose-ready Lighthouse references visible in the standing universe."""
    try:
        import openvr
    except ImportError as exc:
        raise TrackerBindingError(
            "OpenVR Python package is unavailable; source the ROS/OpenVR workspace first"
        ) from exc

    initialized = False
    try:
        openvr.init(openvr.VRApplication_Other)
        initialized = True
        vr_system = openvr.VRSystem()
        reference_class = getattr(openvr, "TrackedDeviceClass_TrackingReference", 4)
        deadline = time.monotonic() + max(0.0, float(pose_wait_seconds))

        while True:
            poses = vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding,
                0,
                openvr.k_unMaxTrackedDeviceCount,
            )
            references: list[TrackingReferenceDevice] = []
            for index in range(openvr.k_unMaxTrackedDeviceCount):
                if vr_system.getTrackedDeviceClass(index) != reference_class:
                    continue
                serial = vr_system.getStringTrackedDeviceProperty(
                    index, openvr.Prop_SerialNumber_String
                ).strip()
                if not serial:
                    continue
                x: float | None = None
                y: float | None = None
                z: float | None = None
                if index < len(poses) and poses[index].bPoseIsValid:
                    candidate_position = tuple(
                        float(poses[index].mDeviceToAbsoluteTracking.m[axis][3])
                        for axis in range(3)
                    )
                    if all(math.isfinite(value) for value in candidate_position):
                        x, y, z = candidate_position
                references.append(
                    TrackingReferenceDevice(
                        index=index,
                        serial=serial,
                        x=x,
                        y=y,
                        z=z,
                    )
                )

            if (
                sum(_position(reference) is not None for reference in references) >= 2
                or time.monotonic() >= deadline
            ):
                return references
            time.sleep(max(0.0, float(pose_poll_seconds)))
    except TrackerBindingError:
        raise
    except Exception as exc:
        raise TrackerBindingError(
            f"cannot enumerate SteamVR base stations through OpenVR: {exc}"
        ) from exc
    finally:
        if initialized:
            try:
                openvr.shutdown()
            except Exception:
                pass


def configured_base_station_lock(
    template: Mapping[str, object], existing: Mapping[str, object] | None
) -> tuple[str, str] | tuple[()] | None:
    """Return a persisted pair, () for first-time auto selection, or None if disabled."""
    if existing is not None and "base_station_serials" in existing:
        raw = existing.get("base_station_serials")
    elif "base_station_serials" in template:
        raw = template.get("base_station_serials")
    else:
        return None

    if raw is None or is_auto_reference(raw):
        return ()
    if not isinstance(raw, (list, tuple)):
        raise TrackerBindingError(
            "base_station_serials must be an empty auto-selection list or two serials"
        )
    serials = tuple(str(value).strip() for value in raw if str(value).strip())
    if not serials:
        return ()
    if len(serials) != 2 or len({_serial_key(serial) for serial in serials}) != 2:
        raise TrackerBindingError(
            "base_station_serials must contain exactly two distinct base-station serials"
        )
    return serials


def choose_nearest_base_station_pair(
    bindings: Mapping[str, str],
    trackers: Sequence[TrackerDevice],
    references: Sequence[TrackingReferenceDevice],
) -> tuple[str, str]:
    """Choose a nearby balanced pair on opposite sides of the operator Y axis.

    OpenVR standing space uses X for left/right and Y for vertical.  Treat the
    vertical line through the wrist midpoint as the operator's Y axis.  A
    valid local pair must place one station on each side of that line.  Among
    such pairs, prefer the pair whose farther station is closest to the axis,
    then the pair whose two axis distances are most alike.
    """
    tracker_by_serial = {
        _serial_key(device.serial): device for device in trackers
    }
    wrist_positions: list[tuple[float, float, float]] = []
    for role in WRIST_ROLES:
        device = tracker_by_serial.get(_serial_key(bindings[role]))
        position = _position(device) if device is not None else None
        if position is None:
            raise TrackerBindingError(
                f"cannot auto-lock base stations because {role} has no valid OpenVR pose"
            )
        wrist_positions.append(position)

    anchor = tuple(
        sum(position[axis] for position in wrist_positions) / len(wrist_positions)
        for axis in range(3)
    )
    posed_references = tuple(
        (reference, position)
        for reference in references
        if (position := _position(reference)) is not None
    )
    if len(posed_references) < 2:
        raise TrackerBindingError(
            "automatic base-station lock needs two pose-ready SteamVR base stations; "
            f"found {len(posed_references)}"
        )

    pair_candidates: list[
        tuple[
            tuple[float, float, float, str, str],
            TrackingReferenceDevice,
            TrackingReferenceDevice,
        ]
    ] = []
    for first, second in itertools.combinations(posed_references, 2):
        first_reference, first_position = first
        second_reference, second_position = second
        first_dx = first_position[0] - anchor[0]
        second_dx = second_position[0] - anchor[0]
        if first_dx * second_dx >= 0.0:
            continue

        # Distance to the vertical OpenVR Y axis ignores the Y coordinate.
        first_axis_distance = math.hypot(first_dx, first_position[2] - anchor[2])
        second_axis_distance = math.hypot(second_dx, second_position[2] - anchor[2])
        farther_axis_distance = max(first_axis_distance, second_axis_distance)
        balance_error = abs(first_axis_distance - second_axis_distance)
        mean_3d_distance = 0.5 * sum(
            math.sqrt(
                sum((position[axis] - anchor[axis]) ** 2 for axis in range(3))
            )
            for position in (first_position, second_position)
        )
        serial_keys = sorted(
            (_serial_key(first_reference.serial), _serial_key(second_reference.serial))
        )
        score = (
            farther_axis_distance,
            balance_error,
            mean_3d_distance,
            serial_keys[0],
            serial_keys[1],
        )
        pair_candidates.append((score, first_reference, second_reference))

    if not pair_candidates:
        raise TrackerBindingError(
            "automatic base-station lock needs one pose-ready station on each side "
            "of the wrist midpoint Y axis"
        )

    _, first_reference, second_reference = min(
        pair_candidates, key=lambda item: item[0]
    )
    if float(first_reference.x) > float(second_reference.x):
        right_reference, left_reference = first_reference, second_reference
    else:
        right_reference, left_reference = second_reference, first_reference
    return right_reference.serial, left_reference.serial


def load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    except OSError as exc:
        raise TrackerBindingError(f"cannot read OpenVR config {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise TrackerBindingError(f"invalid YAML in OpenVR config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackerBindingError(f"OpenVR config must contain a YAML mapping: {path}")
    return payload


def build_machine_config(
    template: Mapping[str, object],
    existing: Mapping[str, object] | None,
    bindings: Mapping[str, str],
    base_station_serials: Sequence[str] | None = None,
) -> dict[str, object]:
    config = dict(existing if existing is not None else template)
    tracker_serials = config.get("tracker_serials")
    serial_mapping = dict(tracker_serials) if isinstance(tracker_serials, dict) else {}
    serial_mapping.update(bindings)
    config["tracker_serials"] = serial_mapping
    if base_station_serials is not None:
        config["base_station_serials"] = list(base_station_serials)
    return config


def write_machine_config(path: Path, config: Mapping[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        header = (
            "# Generated machine-local OpenVR tracker binding. "
            "Do not commit this file.\n"
        )
        payload = header + yaml.safe_dump(
            dict(config), sort_keys=False, allow_unicode=False
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        path.chmod(0o600)
    except OSError as exc:
        raise TrackerBindingError(
            f"cannot write machine-local OpenVR config {path}: {exc}"
        ) from exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass


def masked_serial(serial: str) -> str:
    if len(serial) <= 8:
        return "***"
    return f"{serial[:4]}...{serial[-4:]}"


def configure(
    template_path: Path,
    output_path: Path,
    environ: Mapping[str, str],
) -> dict[str, str]:
    template_resolved = template_path.expanduser().resolve()
    output_resolved = output_path.expanduser().resolve()
    if template_resolved == output_resolved:
        raise TrackerBindingError(
            "refusing to overwrite the repository OpenVR template; use a "
            "machine-local output path"
        )
    if not template_resolved.is_file():
        raise TrackerBindingError(
            f"OpenVR config template does not exist: {template_resolved}"
        )

    template = load_yaml_mapping(template_resolved)
    existing = load_yaml_mapping(output_resolved) if output_resolved.exists() else None
    existing_serials = existing.get("tracker_serials", {}) if existing else {}
    configured = existing_serials if isinstance(existing_serials, dict) else {}
    requests = environment_requests(environ)
    auto_roles = automatic_roles(configured, requests)
    base_station_lock = configured_base_station_lock(template, existing)
    selecting_base_stations = base_station_lock == ()
    devices = (
        enumerate_openvr_trackers(
            wait_for_poses=len(auto_roles) >= 2 or selecting_base_stations
        )
        if bindings_need_devices(configured, requests) or selecting_base_stations
        else []
    )
    bindings = resolve_bindings(configured, requests, devices)
    if selecting_base_stations:
        references = enumerate_openvr_tracking_references()
        base_station_lock = choose_nearest_base_station_pair(
            bindings, devices, references
        )
    config = build_machine_config(
        template,
        existing,
        bindings,
        base_station_serials=base_station_lock,
    )
    write_machine_config(output_resolved, config)
    return bindings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        bindings = configure(args.template, args.output, os.environ)
    except TrackerBindingError as exc:
        print(f"[ERROR] OpenVR tracker binding: {exc}", file=sys.stderr)
        return 1

    summary = ", ".join(
        f"{role}={masked_serial(bindings[role])}" for role in WRIST_ROLES
    )
    print(f"[INFO] Fixed OpenVR tracker binding: {summary}")
    config = load_yaml_mapping(args.output.expanduser().resolve())
    base_station_serials = config.get("base_station_serials")
    if isinstance(base_station_serials, list) and len(base_station_serials) == 2:
        base_summary = ", ".join(
            masked_serial(str(serial)) for serial in base_station_serials
        )
        print(f"[INFO] Fixed nearest base-station pair: {base_summary}")
    print(f"[INFO] Machine-local OpenVR config: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
