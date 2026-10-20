"""Render a read-only GEN6 episode excerpt through canonical replay functions.

This diagnostic derivative is not a training episode or a new physical replay
certificate. Source videos, Zarr, manifests and certification remain untouched.
Run only via the task launcher on the allocated, isolated idle GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

TASK_DIR = Path(__file__).resolve().parent
ROOT = TASK_DIR.parents[2]
TASK = "eval_gen05_set_mixer_knobs"


def file_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def physics_definition_fingerprint(stage):
    """Hash authored physical properties and collider geometry, not visuals."""
    from pxr import UsdPhysics
    rows = {}
    for prim in stage.Traverse():
        physical_schemas = [str(s) for s in prim.GetAppliedSchemas()
                            if str(s).startswith(("Physics", "Physx"))]
        if not physical_schemas and not prim.IsA(UsdPhysics.Joint):
            continue
        props = {}
        for prop in prim.GetProperties():
            name = prop.GetName()
            physical = name.startswith(("physics:", "physx", "drive:", "xformOp:"))
            collider_shape = prim.HasAPI(UsdPhysics.CollisionAPI) and name in {
                "points", "faceVertexCounts", "faceVertexIndices", "size", "radius", "height", "axis"
            }
            if physical or collider_shape:
                props[name] = str(prop.GetTargets() if hasattr(prop, "GetTargets") else prop.Get())
        rows[str(prim.GetPath())] = {"type": prim.GetTypeName(), "schemas": physical_schemas, "properties": props}
    payload = json.dumps(rows, sort_keys=True).encode()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "prim_count": len(rows)}


def direct_physics_state(raw):
    import numpy as np
    trace = raw._physx_ground_truth_trace
    if trace is None:
        raise RuntimeError("Direct PhysX readback is required for this visual diagnostic")
    fields = trace._sample_robot_fields()
    fields.update(trace._sample_rigid_body_fields())
    fields.update(trace._sample_articulation_fields())
    fields["physx_sim_time"] = np.asarray([trace._sim_time()])
    return {name: np.asarray(value).copy() for name, value in fields.items()
            if any(word in name for word in ("position", "velocity", "state", "time"))}


def visual_transform_readback(raw):
    """Read direct link tensors and USD/Fabric visual transforms; never author."""
    import numpy as np
    import omni.usd
    import usdrt
    from pxr import Usd, UsdGeom
    fabric = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    rows = {}
    for side in ("left", "right"):
        robot = raw._control_robot(side)
        poses = robot.root_physx_view.get_link_transforms().detach().cpu().numpy()[0]
        for name in (side + "_palm_link", side + "_finger2_link2"):
            index = list(robot.body_names).index(name)
            prim = next(p for p in raw._stage.Traverse() if p.GetName() == name)
            row = {"link_prim": str(prim.GetPath()), "direct_link_xyz_xyzw": poses[index].tolist(), "paths": {}}
            for path in (str(prim.GetPath()), str(prim.GetPath()) + "/visuals/BlenderVisuals"):
                usd_prim = raw._stage.GetPrimAtPath(path)
                if not usd_prim:
                    continue
                value = {"usd_world": np.asarray(UsdGeom.Xformable(usd_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).tolist()}
                rt_prim = fabric.GetPrimAtPath(path)
                if rt_prim:
                    xf = usdrt.Rt.Xformable(rt_prim)
                    for key, method in (("fabric_world_position", "GetWorldPositionAttr"), ("fabric_world_matrix", "GetFabricHierarchyWorldMatrixAttr")):
                        attr = getattr(xf, method)()
                        if attr and attr.Get() is not None:
                            value[key] = np.asarray(attr.Get()).tolist()
                row["paths"][path] = value
            link_matrix = row["paths"][row["link_prim"]].get("fabric_world_matrix")
            visual_matrix = row["paths"].get(row["link_prim"] + "/visuals/BlenderVisuals", {}).get("fabric_world_matrix")
            if visual_matrix is not None and link_matrix is not None:
                row["visual_origin_direct_link_error_m"] = float(np.max(np.abs(np.asarray(visual_matrix)[3, :3] - poses[index, :3])))
                row["visual_parent_matrix_max_abs_error"] = float(np.max(np.abs(np.asarray(visual_matrix) - np.asarray(link_matrix))))
            rows[name] = row
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=360)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--sample-indices", type=int, nargs="+")
    parser.add_argument("--render-sync", choices=("paused_fk", "playing"), default="playing")
    args = parser.parse_args()
    episode, output = args.episode.resolve(), args.output.resolve()
    if output == episode or output.is_relative_to(episode) or episode.is_relative_to(output):
        raise ValueError("The diagnostic output must be separate from the source episode")
    if args.start < 0 or args.frames < 2:
        raise ValueError("Require a nonnegative start and at least two frames")
    output.mkdir(parents=True, exist_ok=False)
    source_before = file_hashes(episode)
    report = {"task": TASK, "scope": "state-guided canonical camera excerpt; not physical certification",
              "source_episode": str(episode), "source_start": args.start,
              "source_end_exclusive": args.start + args.frames,
              "source_files_sha256": source_before, "render_only_readbacks": [],
              "video_encoder_requested": os.environ.get("HANDSON_VIDEO_ENCODER", "auto"),
              "started_unix": time.time(), "status": "running"}
    code = 1
    try:
        from validation_runtime import validate_runtime
        report["runtime"] = validate_runtime()
        import numpy as np
        from PIL import Image
        from scripts import isaac_render_arm_replay as post
        from handson.tasks.eval_gen05_set_mixer_knobs.config import TaskConfig

        post.FLAGS(["gen6_visual_clip", "--exp_name=" + TASK, "--sim_backend=isaacsim"])
        sys.argv = [sys.argv[0], "--/log/file=" + str(output / "kit.log")]
        post._configure_canonical_camera_postrender(source_backend="isaacsim")
        randomize, env_seed, reset_seed = post._recorded_replay_settings(episode)
        post.FLAGS.randomize = randomize
        actions = post._load_episode_actions(episode / "replay.zarr")
        states = post._load_episode_states(episode / "replay.zarr")
        end = args.start + args.frames
        selected = args.sample_indices or list(range(args.start, end))
        if states is None or len(selected) < 2 or min(selected) < 0 or max(selected) >= min(len(actions), len(states)):
            raise ValueError("The requested excerpt exceeds the recorded action/state arrays")
        actions, states = actions[selected].copy(), states[selected].copy()
        args.frames = len(selected)
        report["source_frame_indices"] = selected
        report["source_start"], report["source_end_exclusive"] = min(selected), max(selected) + 1
        report["sparse_diagnostic"] = bool(args.sample_indices)
        report["render_sync"] = args.render_sync
        report["selected_arrays_sha256"] = {"action": hashlib.sha256(actions.tobytes()).hexdigest(),
                                            "state": hashlib.sha256(states.tobytes()).hexdigest()}
        config = TaskConfig()
        factory = config.get_environment
        inspected = {}

        def inspected_environment(*a, **kw):
            env = factory(*a, **kw)
            raw = env.unwrapped
            inspected["raw"] = raw
            report["physics_definition"] = physics_definition_fingerprint(raw._stage)
            report["task_custom_data"] = dict(raw._stage.GetPrimAtPath("/World/Task").GetCustomData())
            original_render = raw.render
            calls = 0

            def reject_fallback(*a, **kw):
                raise RuntimeError("Native visual validation encountered a fallback camera image")

            raw._fallback_camera_frame = reject_fallback

            def inspected_render(*a, **kw):
                nonlocal calls
                import omni.timeline
                timeline = omni.timeline.get_timeline_interface()
                was_playing = timeline.is_playing()
                if args.render_sync == "paused_fk":
                    timeline.pause()
                    timeline.commit()
                check = calls in {0, 30, 60, 90} or bool(args.sample_indices)
                before = direct_physics_state(raw) if check else None
                try:
                    # Lab forward() skips the articulation FK update while
                    # paused. Flush it explicitly so Hydra sees current links.
                    # This is rendering synchronization, not a physics step.
                    if args.render_sync == "paused_fk":
                        raw._sim.physics_sim_view.update_articulations_kinematic()
                        raw._sim.forward()
                    result = original_render(*a, **kw)
                    if check:
                        after = direct_physics_state(raw)
                        errors = {key: float(np.max(np.abs(after[key].astype(float) - value.astype(float)), initial=0))
                                  for key, value in before.items()}
                        transforms = visual_transform_readback(raw)
                        report["render_only_readbacks"].append({"render_call": calls,
                            "before": {key: value.tolist() for key, value in before.items()},
                            "max_abs_error": errors, "visual_transforms": transforms})
                        if any(value != 0 for value in errors.values()):
                            raise RuntimeError("PhysX state changed across a render-only call")
                        if os.environ.get("HANDSON_ISAACSIM_TASK_VISUAL_PROFILE") == "real_v12":
                            if any(row.get("visual_origin_direct_link_error_m", float("inf")) > 1e-5
                                   or row.get("visual_parent_matrix_max_abs_error", float("inf")) > 1e-6
                                   for row in transforms.values()):
                                raise RuntimeError("Fabric robot visual transform differs from its direct link pose")
                finally:
                    if was_playing and args.render_sync == "paused_fk":
                        timeline.play()
                        timeline.commit()
                calls += 1
                return result

            raw.render = inspected_render
            return env

        config.get_environment = inspected_environment
        _, trajectory, camera_metadata, camera_sources, _ = post._replay_single_demo(
            actions, states[0], states, TASK, config, env_seed=env_seed,
            reset_seed=reset_seed, desc="GEN6 visual excerpt", sim_backend="isaacsim", close_env=False)
        if len(trajectory) != args.frames:
            raise RuntimeError("Canonical excerpt frame count differs from selected actions")
        if not report["render_only_readbacks"]:
            raise RuntimeError("No guarded native render calls were observed")
        raw = inspected["raw"]
        source_profile = post._recorded_camera_profile(episode)
        profile = dict(raw.recording_camera_profile)
        report["camera_geometry_unchanged"] = all(
            profile.get(key) == source_profile.get(key) for key in ("head", "wrist", "version", "name"))
        if not report["camera_geometry_unchanged"]:
            raise RuntimeError("Runtime camera geometry differs from the source episode")
        report["appearance"] = getattr(raw, "task_visual_appearance_report", None)
        # Only camera provenance is needed by the shared writer. Do not copy
        # old success, physical certification, publication or upload metadata.
        (output / "collection_meta.json").write_text(json.dumps({"recording_camera_profile": profile}, indent=2))
        videos = post._write_arm_videos(output, trajectory, fps=30)
        source_map = post.select_canonical_camera_sources_for_trajectory(trajectory)
        for index in (0, args.frames // 2, args.frames - 1):
            for camera, source in source_map.items():
                frame = post.postprocess_canonical_image(
                    post._safe_rgb(post.observation_image(trajectory[index]["observations"], source)), camera, source)
                Image.fromarray(frame).save(output / f"{index:04d}_{camera}.png")
        report.update(status="rendered_pending_visual_review", videos=str(videos),
                      frame_count=len(trajectory), camera_sources=camera_sources,
                      camera_metadata=camera_metadata)
        import av
        media = {}
        for camera in ("main", "wrist_left", "wrist_right"):
            video = videos / f"{camera}.mp4"
            signatures, means, deviations, shapes, timestamps = [], [], [], [], []
            with av.open(str(video)) as container:
                for frame in container.decode(video=0):
                    rgb = frame.to_ndarray(format="rgb24")
                    signatures.append(hashlib.sha256(rgb.tobytes()).hexdigest())
                    means.append(float(rgb.mean()))
                    deviations.append(float(rgb.std()))
                    shapes.append(list(rgb.shape))
                    timestamps.append(float(frame.time))
            passed = (len(signatures) == args.frames and len(set(signatures)) > 1
                      and min(deviations, default=0) > 1 and min(means, default=0) > 1
                      and all(b > a for a, b in zip(timestamps, timestamps[1:])))
            media[camera] = {"frames": len(signatures), "unique_frames": len(set(signatures)),
                             "shape": shapes[0] if shapes else None, "passed_decode_checks": passed,
                             "mean_rgb_min_max": [min(means, default=0), max(means, default=0)],
                             "sha256": hashlib.sha256(video.read_bytes()).hexdigest()}
        report["media"] = media
        if not all(row["passed_decode_checks"] for row in media.values()):
            raise RuntimeError("Canonical excerpt decode/motion checks failed")
        code = 0
    except Exception as exc:
        import traceback
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        report["source_unchanged"] = file_hashes(episode) == source_before
        if not report["source_unchanged"]:
            code = 1
        report["ended_unix"] = time.time()
        (output / "validation.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
        print("GEN6_VISUAL_CLIP", report["status"], "source_unchanged", report["source_unchanged"], flush=True)
        # The normal canonical renderer also exits its dedicated Kit process
        # after saving output; no other task/process is affected.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
