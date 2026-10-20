"""Local D20 save selection and pre-action demonstration/continuation.

Playback reconstructs recorded states for the operator. It is not a physical
replay certification and never runs after the new episode's first action.
"""
from __future__ import annotations

import hashlib
import json
import os
import secrets
import tempfile
from pathlib import Path
import time

import numpy as np

from ..sim.isaaclab import d20_stages as stages


# A sidecar keeps the immutable replay manifest/hash intact while recording
# that this successful stage has already been consumed as a splice source.
# The marker is deliberately outside replay_manifest.json: lineage references
# include that manifest hash and must remain valid after a source is claimed.
STITCH_MARKER = "d20_stitched.json"
REJECTION_MARKER = "d20_continuation_rejected.json"
DEFERRED_MARKER = "d20_continuation_deferred.json"


class ContinuationRejected(ValueError):
    """A definitive source-state failure, not a GPU/runtime/operator failure."""

    def __init__(self, message, *, code="fast_forward_unachievable", metrics=None):
        super().__init__(message)
        self.code = code
        self.metrics = dict(metrics or {})


class NoUsableSource(ValueError):
    """All complete, unused predecessors have been exhausted."""


def continuation_rejection(path, consumer_task="", *, sha256=None):
    """Read a manifest-bound label without opening Zarr (also used by the UI)."""
    try:
        marker = _json(Path(path) / REJECTION_MARKER)
        if (
            not isinstance(marker, dict)
            or marker.get("format") != "handson.d20_continuation_rejection"
            or marker.get("version") != 1
            or marker.get("rejected") is not True
            or marker.get("source_manifest_sha256") != (sha256 or manifest_hash(path))
            or (consumer_task and marker.get("consumer_task") != consumer_task)
        ):
            return {}
        return marker
    except (OSError, ValueError, TypeError):
        return {}


def mark_continuation_rejected(source, consumer_task, error):
    """Keep successful data immutable; reject only its use as this predecessor."""
    if not isinstance(source, SavedEpisode) or not isinstance(error, ContinuationRejected):
        raise TypeError("Only a validated source and a definitive continuation rejection may be tagged")
    if manifest_hash(source.path) != source.sha256:
        raise ValueError("D20 存档在续接检查期间发生变化，不能标记旧版本")
    payload = {
        "format": "handson.d20_continuation_rejection", "version": 1,
        "rejected": True, "label": "不可续接", "consumer_task": consumer_task,
        "source_manifest_sha256": source.sha256, "reason_code": error.code,
        "reason": str(error), "metrics": error.metrics,
        "marked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=source.path,
            prefix=f".{REJECTION_MARKER}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, source.path / REJECTION_MARKER)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload



def _atomic_json(path, payload):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def source_is_stitched(path):
    try:
        marker = _json(Path(path) / STITCH_MARKER)
        return bool(
            isinstance(marker, dict) and marker.get("stitched") is True
            and marker.get("consumer_success") is True
            and str(marker.get("consumer_episode") or "").strip()
        )
    except (OSError, ValueError, TypeError):
        return False


def validate_source_mode(source_mode):
    if source_mode not in {"normal", "deferred"}:
        raise ValueError("D20 队列模式必须是 normal（继续做）或 deferred（做暂时不行的数据）")
    return source_mode


def continuation_deferral(path, consumer_task="", *, sha256=None):
    """Active manual deferral; keep the historical sidecar after consumption."""
    try:
        marker = _json(Path(path) / DEFERRED_MARKER)
        if (
            not isinstance(marker, dict)
            or marker.get("format") != "handson.d20_continuation_deferral"
            or marker.get("version") != 1 or marker.get("deferred") is not True
            or marker.get("source_manifest_sha256") != (sha256 or manifest_hash(path))
            or (consumer_task and marker.get("consumer_task") != consumer_task)
            or not isinstance(marker.get("marked_at_ns"), int)
            or source_is_stitched(path)
            or continuation_rejection(path, marker.get("consumer_task", ""), sha256=sha256)
        ):
            return {}
        return marker
    except (OSError, ValueError, TypeError):
        return {}


def mark_continuation_deferred(source, consumer_task, *, reason="", collector_id="", session_id=""):
    stage = stages.stage_number(consumer_task)
    if not isinstance(source, SavedEpisode) or not 2 <= stage <= len(stages.STAGE_TASKS):
        raise ValueError("只能暂缓有效的 D20 前序阶段存档")
    if source.manifest["task"]["name"] != stages.STAGE_TASKS[stage - 2]:
        raise ValueError("D20 前序存档阶段不匹配")
    if manifest_hash(source.path) != source.sha256:
        raise ValueError("D20 存档发生变化，不能标记旧版本")
    if source.stitched or continuation_rejection(source.path, consumer_task, sha256=source.sha256):
        raise ValueError("这个 D20 存档已拼接或不可续接，不能加入暂缓队列")
    return _atomic_json(source.path / DEFERRED_MARKER, {
        "format": "handson.d20_continuation_deferral", "version": 1,
        "deferred": True, "label": "暂时不行", "consumer_task": consumer_task,
        "source_manifest_sha256": source.sha256,
        "reason": str(reason).strip()[:2000], "collector_id": str(collector_id),
        "session_id": str(session_id), "marked_at_ns": time.time_ns(),
        "marked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })


def write_source_status(path, *, session_id, task_name, source_mode, source, phase, worker_pid=0):
    """Dedicated worker cursor: never overwritten by supervisor runtime status."""
    validate_source_mode(source_mode)
    return _atomic_json(path, {
        "format": "handson.d20_source_status", "version": 1,
        "session_id": str(session_id), "task_name": task_name,
        "source_mode": source_mode, "source": source.reference() if source is not None else None,
        "phase": phase, "worker_pid": int(worker_pid), "token": secrets.token_hex(24),
    })


def read_source_status(path, *, session_id, task_name):
    try:
        status = _json(path)
        if (not isinstance(status, dict) or status.get("format") != "handson.d20_source_status"
                or status.get("version") != 1 or status.get("session_id") != str(session_id)
                or status.get("task_name") != task_name or not status.get("token")):
            return {}
        return status
    except (OSError, ValueError, TypeError):
        return {}


def _report_rejection(source, task_name, error, progress):
    mark_continuation_rejected(source, task_name, error)
    progress(f"[D20] {source.path.name} 已标记为不可续接：{error}；自动尝试下一个存档。")


def _print_progress(message):
    print(message, flush=True)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def manifest_hash(path):
    return hashlib.sha256((Path(path) / "replay_manifest.json").read_bytes()).hexdigest()


def episode_directories(root):
    """Include the ordinary local pending-save directories, but never scan chunks."""
    for directory, children, files in os.walk(Path(root).expanduser()):
        children[:] = [n for n in children if not n.endswith(".zarr") and n not in {".git", "videos", "source"}]
        if "replay_manifest.json" in files:
            yield Path(directory)
            children[:] = []


class SavedEpisode:
    def __init__(self, path, expected_task):
        import zarr
        self.path = Path(path).expanduser().resolve()
        self.manifest = _json(self.path / "replay_manifest.json")
        if self.manifest.get("task", {}).get("name") != expected_task:
            raise ValueError(f"存档任务不匹配，需要 {expected_task}: {self.path.name}")
        self.recording = self.manifest.get("recording", {})
        self.schema = _json(self.path / "recording_schema.json")["environment_schema"]
        if not self.schema.get("physx"):
            raise ValueError("存档缺少完整 PhysX 状态，请重新采集上一阶段")
        self.data = zarr.open_group(str(self.path / "replay.zarr"), mode="r")["data"]
        self.count = int(self.data["action"].shape[0])
        if self.count < 1 or not bool(np.asarray(self.data["succeed"][-1]).all()):
            raise ValueError("只能从成功完成的上一阶段存档继续")
        if self.count != int(self.recording.get("step_count", -1)):
            raise ValueError("上一阶段存档尚未完整写入")
        if "next_state" not in self.data or self.data["next_state"].shape[0] != self.count:
            raise ValueError("存档缺少最后动作之后的状态")
        self.layout = self.recording.get("d20_state_layout")
        if not self.layout or not {"d20_hidden_state", "d20_assist_state", "d20_tool_grip_state", "d20_stage_state"}.issubset({item[0] for item in self.layout}):
            raise ValueError("存档缺少阶段和夹子状态，请重新采集上一阶段")
        self.fields = [key for key in self.data.array_keys()
                       if key.startswith("physx_") and not key.startswith("physx_substep_")]
        for key in self.fields:
            if "next_" + key not in self.data or self.data["next_" + key].shape[0] != self.count:
                raise ValueError(f"存档缺少完整后继状态: {key}")
        if "physx_robot_joint_position" not in self.fields and "physx_robot_joint_q" not in self.fields:
            raise ValueError("存档缺少机器人关节状态")
        if "terminal" not in self.data or not bool(np.asarray(self.data["terminal"][-1]).all()):
            raise ValueError("上一阶段记录尚未正常结束")
        self.sha256 = manifest_hash(self.path)
        # Validate the entire terminal task state before launching Isaac.
        self.parts(-1, next_state=True)

    @property
    def stitched(self):
        """Whether this episode has already been consumed as a D20 source."""
        return source_is_stitched(self.path)

    def parts(self, index, *, next_state=False):
        values = np.asarray(self.data["next_state" if next_state else "state"][index])
        result = {}
        offset = 0
        for key, shape in self.layout:
            size = int(np.prod(shape))
            result[key] = values[offset:offset + size].reshape(shape).copy()
            offset += size
        if offset != values.size or not np.isfinite(values).all():
            raise ValueError("D20 存档状态尺寸不匹配或包含无效数值")
        return result

    def snapshot(self, index, *, next_state=False):
        prefix = "next_" if next_state else ""
        return {key: np.asarray(self.data[prefix + key][index]) for key in self.fields}

    def reference(self):
        return {"episode_id": self.path.name, "path": str(self.path), "manifest_sha256": self.sha256,
                "task_name": self.manifest["task"]["name"]}


def resolve_reference(root, reference, expected_task):
    candidates = [Path(reference.get("path", ""))]
    candidates.extend(p for p in episode_directories(root) if p.name == reference.get("episode_id"))
    for path in dict.fromkeys(candidates):
        if (path / "replay_manifest.json").is_file() and manifest_hash(path) == reference.get("manifest_sha256"):
            return SavedEpisode(path, expected_task)
    raise ValueError("找不到这个存档绑定的前序记录；请恢复对应的本地上一阶段数据")


def mark_stitched(source, consumer_task, *, consumer_episode=""):
    """Claim a source only after its continuation was saved successfully.

    The claim is a sidecar so the source manifest remains hash-stable.  A
    second claimant receives a clear error instead of silently reusing the
    same predecessor.
    """
    if not isinstance(source, SavedEpisode):
        raise TypeError("mark_stitched expects a validated SavedEpisode")
    consumer_path = Path(consumer_episode).expanduser().resolve()
    if not consumer_path.is_dir():
        raise ValueError("成功续接存档目录不存在，不能标记 D20 前序存档")
    consumer_manifest = _json(consumer_path / "replay_manifest.json")
    if consumer_manifest.get("task", {}).get("name") != str(consumer_task):
        raise ValueError("续接存档任务与 D20 当前阶段不匹配")
    recording = consumer_manifest.get("recording") or {}
    reference = recording.get("d20_stage_resume", {}).get("source")
    if not isinstance(reference, dict) or (
        reference.get("episode_id") != source.path.name
        or reference.get("manifest_sha256") != source.sha256
    ):
        raise ValueError("续接存档没有绑定本次使用的 D20 前序存档")
    try:
        existing = _json(source.path / STITCH_MARKER)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        existing = None
    if isinstance(existing, dict) and existing.get("consumer_success") is True:
        raise ValueError(f"这个 D20 前序存档已经标记为已拼接：{source.path.name}")
    marker = source.path / STITCH_MARKER
    payload = {
        "label": "已拼接",
        "stitched": True,
        "consumer_success": True,
        "consumer_task": str(consumer_task),
        "consumer_episode": str(consumer_path),
        "source_episode_id": source.path.name,
        "source_manifest_sha256": source.sha256,
        "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # The marker is written after the consumer is complete. Replace a legacy
    # pre-action marker atomically so it cannot claim a failed continuation.
    temporary = marker.with_name(f".{marker.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, marker)
    return payload


def select_source(root, task_name, requested="", *, source_mode="normal", excluded=(), progress=_print_progress):
    validate_source_mode(source_mode)
    stage = stages.stage_number(task_name)
    if stage <= 1 or stage > len(stages.STAGE_TASKS):
        return None
    expected = stages.STAGE_TASKS[stage - 2]

    def eligible(source):
        if (source.path.name, source.sha256) in excluded:
            return False
        deferred = continuation_deferral(source.path, task_name, sha256=source.sha256)
        if bool(deferred) != (source_mode == "deferred"):
            return False
        if source.stitched or continuation_rejection(source.path, task_name, sha256=source.sha256):
            return False
        try:
            validate_source_continuation(source, root, task_name)
        except ContinuationRejected as error:
            _report_rejection(source, task_name, error, progress)
            return False
        return True

    requested_path = None
    if requested:
        path = Path(requested).expanduser()
        source = SavedEpisode(path if path.is_absolute() else Path(root) / path, expected)
        requested_path = source.path
        if source.stitched:
            raise ValueError(f"这个 D20 前序存档已经标记为已拼接：{source.path.name}")
        if eligible(source):
            return source
        # An explicitly selected but rejected source also advances to the pool.
    candidates = [p for p in episode_directories(root) if p.name.startswith(expected)]
    def queue_order(path):
        if source_mode == "deferred":
            return continuation_deferral(path, task_name).get("marked_at_ns", 0), path.name
        try:
            return -(path / "replay_manifest.json").stat().st_mtime_ns, path.name
        except OSError:
            return 0, path.name

    for path in sorted(candidates, key=queue_order):
        if requested_path is not None and path.resolve() == requested_path:
            continue
        try:
            source = SavedEpisode(path, expected)
        except (ValueError, KeyError, OSError, TypeError):
            # Incomplete/in-flight files are not definitive source failures.
            continue
        if eligible(source):
            return source
    next_step = ("可切换到继续做，或稍后重试暂缓数据。" if source_mode == "deferred"
                 else f"请先完成并保存 {expected}，再启动 {task_name}。")
    raise NoUsableSource(
        f"本地 D20 {'暂时不行' if source_mode == 'deferred' else '继续做'}队列没有可用的前序阶段成功存档（已用完、已暂缓或已标记为不可续接）。"
        f"{next_step}查找目录：{root}"
    )


def with_source_retries(root, task_name, source, attempt, *, source_mode="normal", progress=_print_progress):
    """Each attempt must perform a full stable reset before any live action."""
    validate_source_mode(source_mode)
    while True:
        try:
            return attempt(source)
        except ContinuationRejected as error:
            _report_rejection(source, task_name, error, progress)
            source = select_source(root, task_name, source_mode=source_mode, progress=progress)


def validate_source_continuation(source, root, task_name):
    """Preflight the same deterministic heating gate before creating Isaac."""
    stage = stages.stage_number(task_name)
    if stage < 3:
        return
    controls = source
    # D20-3 replays its D20-2 predecessor, including the control-stage heating.
    if stage == 4:
        resume = source.recording.get("d20_stage_resume")
        reference = resume.get("source") if isinstance(resume, dict) else None
        if not isinstance(reference, dict) or not reference:
            raise ContinuationRejected("存档缺少绑定的 D20-1.5 前序记录", code="missing_predecessor")
        try:
            controls = resolve_reference(root, reference, stages.STAGE_TASKS[1])
        except (ValueError, KeyError, TypeError) as exc:
            raise ContinuationRejected(str(exc), code="invalid_predecessor") from exc
    parts = controls.parts(-1, next_state=True)
    state = np.asarray(parts.get("microwave_state", []), dtype=float).ravel()
    knobs = np.asarray(parts.get("microwave_controls", []), dtype=float).ravel()
    if state.size != 10 or knobs.size != 4:
        raise ContinuationRejected("存档缺少完整的微波炉加热状态", code="invalid_saved_heating_state")
    index = int(knobs[2])
    if not 0 <= index < len(stages.d20.POWER_RATES):
        raise ContinuationRejected("存档功率档位无效", code="invalid_saved_heating_state")
    _heating_plan({
        "d20_running": bool(state[2]), "d20_cook_progress": float(state[3]),
        "d20_hot_dog_inside": bool(state[6]), "d20_door_closed": bool(state[7]),
        "d20_power_index": index, "d20_timer_minutes": float(knobs[3]),
        "d20_power_rate": float(stages.d20.POWER_RATES[index]),
    })


def _heating_plan(initial):
    """Keep preflight and live checks identical; do not weaken the cook gate."""
    remaining = max(0.0, float(initial.get("d20_timer_minutes", 0.0)) * 60.0)
    power_rate = float(initial.get("d20_power_rate", 0.0))
    medium_target = None
    if (
        int(initial.get("d20_power_index", -1)) == stages.STAGE1_POWER_INDEX
        and 0.0 < remaining <= stages.STAGE1_TIMER_MAX_SECONDS
    ):
        medium_target = remaining
        power_rate = stages.d20.COOK_SECONDS / remaining
    cook_progress = float(initial.get("d20_cook_progress", 0.0))
    predicted = cook_progress + remaining * power_rate / 60.0
    if (
        not initial.get("d20_running")
        or not initial.get("d20_hot_dog_inside")
        or not initial.get("d20_door_closed")
        or not np.isfinite(predicted)
        or predicted < stages.d20.DONE_PROGRESS - stages.d20.COOK_PROGRESS_EPSILON
        or predicted > stages.d20.MAX_OK_PROGRESS + stages.d20.COOK_PROGRESS_EPSILON
    ):
        raise ContinuationRejected(
            "快进预测无法达到有效熟度窗口（90%-110%）",
            metrics={
                "cook_progress": cook_progress if np.isfinite(cook_progress) else None,
                "remaining_seconds": remaining if np.isfinite(remaining) else None,
                "predicted_cook_progress": predicted if np.isfinite(predicted) else None,
            },
        )
    return medium_target


def state_layout(env, config):
    raw = env.unwrapped
    return [[key, list(raw.observation_space["state"][key].shape)] for key in config.proprio_keys]


def _restore(raw, episode, index, *, next_state=False):
    parts = episode.parts(index, next_state=next_state)
    raw.restore_recorded_initial_state(parts)
    snapshot = episode.snapshot(index, next_state=next_state)
    raw._physx_ground_truth_trace.restore(snapshot, episode.schema)
    roots = snapshot.get("physx_robot_root_state")
    valid = snapshot.get("physx_robot_root_valid")
    if roots is not None and valid is not None:
        sides = episode.schema["physx"].get("robot_sides", ["right", "left"])
        for i, side in enumerate(sides):
            if bool(valid[i]) and side in raw._robots:
                robot = raw._robots[side]
                robot.write_root_state_to_sim(raw._physx_ground_truth_trace._array_like(
                    robot.data.root_state_w, np.asarray(roots[i:i+1])))
        raw._physx_ground_truth_trace._sync_after_restore()
    raw._long_metrics = raw._metrics(update=False)
    raw._d20_update_knob_drives()
    raw._d20_update_visuals()


def _render(raw, message):
    raw._d20_prelude_status = message
    raw._update_success_condition_overlay()
    if raw._sim is not None:
        # Only the existing human viewport; no recording-camera products.
        raw._sim.render()
    app = getattr(raw, "_simulation_app", None)
    if app is not None and not app.is_running():
        raise RuntimeError("D20 前序演示已取消")


def playback(raw, episode, speed, progress):
    hz = max(1.0, float(episode.recording.get("data_fps", 30)))
    stride = max(1, int(hz * speed / 30))
    indices = list(range(0, episode.count, stride))
    if indices[-1] != episode.count - 1:
        indices.append(episode.count - 1)
    for index in indices:
        started = time.monotonic()
        _restore(raw, episode, index)
        message = f"D20 saved demonstration | {episode.manifest['task']['name']} | {speed:g}x | {index+1}/{episode.count}"
        _render(raw, message)
        progress(message)
        time.sleep(max(0.0, 1 / 30 - (time.monotonic() - started)))
    # state[-1] precedes the final action. Resume only from next_state[-1].
    _restore(raw, episode, -1, next_state=True)
    _render(raw, "D20 saved demonstration complete")


def finish_heating(raw, speed, progress):
    """Accelerate only microwave heat/timer progression before live action 0."""
    def _unachievable(message):
        marker = getattr(raw, "_d20", None)
        if marker is not None and hasattr(marker, "mark_failure"):
            marker.mark_failure("fast_forward_unachievable")
        raise ContinuationRejected(message)

    raw._d20_heating_time_scale = speed
    try:
        raw.env_step = 0
        initial = raw._metrics(update=False)
        try:
            medium_target = _heating_plan(initial)
        except ContinuationRejected:
            marker = getattr(raw, "_d20", None)
            if marker is not None and hasattr(marker, "mark_failure"):
                marker.mark_failure("fast_forward_unachievable")
            raise
        marker = getattr(raw, "_d20", None)
        if medium_target is not None and marker is not None and hasattr(marker, "_medium_target_seconds"):
            marker._medium_target_seconds = medium_target
        for _ in range(int(75 / raw._control_dt) + 1):
            metrics = raw._metrics(update=False)
            if metrics["d20_cook_ok"] and metrics["d20_stop_ok"]:
                return
            if metrics["d20_failed"] or not metrics["d20_hot_dog_inside"] or not metrics["d20_door_closed"]:
                _unachievable("上一阶段存档无法继续加热：快进后无法达到有效熟度，本次续接记为失败")
            if not metrics["d20_running"] and metrics["d20_timer_minutes"] == 0:
                _unachievable("快进结束但热狗未达到原 D20 的有效熟度，本次续接记为失败")
            raw._d20_update_knob_drives()
            raw.step_with_current_joint_targets()
            message = f"D20 heating wait | {speed:g}x | {raw._metrics(update=False)['d20_timer_minutes'] * 60:.1f} s left"
            _render(raw, message)
            progress(message)
        _unachievable("D20 快进等待超时，无法确认有效熟度，本次续接记为失败")
    finally:
        raw._d20_heating_time_scale = 1.0
        raw._d20_update_knob_drives()


def resume_saved_stage(env, source, root, *, speed=8.0, progress=lambda message: None):
    raw = env.unwrapped
    speed = float(speed)
    if not np.isfinite(speed) or not 1 <= speed <= 32:
        raise ValueError("D20 演示/等待速度必须在 1 到 32 倍之间")
    trace = getattr(raw, "_physx_ground_truth_trace", None)
    if trace is None:
        raise RuntimeError("D20 存档恢复需要完整 PhysX 状态录制")
    trace.validate_restore_schema(source.schema)
    stage = stages.stage_number(raw.task_name)
    def _playback_chain(episode):
        """Show all saved ancestors, heating only after the control stage."""
        task = episode.manifest.get("task", {}).get("name", "")
        index = stages.STAGE_TASKS.index(task) if task in stages.STAGE_TASKS else -1
        if index > 0:
            resume = episode.recording.get("d20_stage_resume")
            reference = resume.get("source") if isinstance(resume, dict) else None
            if not isinstance(reference, dict) or not reference:
                raise ContinuationRejected(
                    f"{task} 存档没有绑定前序 D20 阶段", code="missing_predecessor",
                )
            parent_task = stages.STAGE_TASKS[index - 1]
            try:
                parent = resolve_reference(root, reference, parent_task)
            except (ValueError, KeyError, TypeError) as exc:
                raise ContinuationRejected(str(exc), code="invalid_predecessor") from exc
            trace.validate_restore_schema(parent.schema)
            _playback_chain(parent)
            # D20-1.5 ends with a running timer; wait only when handing off to
            # the retrieval stage so the live operator starts from cooked food.
            if task == "d20_2_retrieve_and_plate":
                finish_heating(raw, speed, progress)
        playback(raw, episode, speed, progress)
    wrappers = []
    wrapper = env
    while wrapper is not raw:
        wrappers.append(wrapper)
        wrapper = wrapper.env
    teleop = [(wrapper, bool(wrapper.intervened)) for wrapper in wrappers
              if callable(getattr(type(wrapper), "set_intervened", None))]
    for wrapper, _active in teleop:
        wrapper.set_intervened(False, start_delay=False)
    resumed = False
    try:
        _playback_chain(source)
        if stage == 3:
            # D20-2 starts from the D20-1.5 terminal state; cook only after
            # the saved rotary-control stage has been shown.
            finish_heating(raw, speed, progress)
        # Keep restored task/robot/drive state; only the new episode's counters restart.
        raw.env_step = 0
        raw._success_counter = 0
        raw._failure_counter = 0
        raw._d20_stage.release_seconds = 0.0
        raw._d20.state.release_seconds = 0.0
        raw._long_metrics = raw._metrics(update=False)
        raw._d20_stage_resume = {"source": source.reference(), "speed": speed,
                                  "playback": "pre_action_recorded_state_demonstration",
                                  "heating_wait_before_action_zero": stage == 3}
        observation = raw._attach_physx_ground_truth(raw._compute_observation())
        # Re-anchor both held poses and trackers to the restored physical scene.
        for wrapper, active in teleop:
            wrapper.set_intervened(False, start_delay=False)
            if active:
                wrapper.set_intervened(True, start_delay=False)
        for wrapper in reversed(wrappers):
            adapt = getattr(type(wrapper), "observation", None)
            if callable(adapt):
                observation = adapt(wrapper, observation)
        resumed = True
        return observation, dict(raw._d20_stage_resume)
    finally:
        raw._d20_prelude_status = ""
        if not resumed:
            for wrapper, active in teleop:
                if active:
                    wrapper.set_intervened(True, start_delay=False)
