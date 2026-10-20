from __future__ import annotations

import os
import textwrap
import time
from typing import Optional

import glfw
import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import (
    MujocoRenderer as GymnasiumMujocoRenderer,
    WindowViewer as GymnasiumWindowViewer,
)


_HUMAN_CAMERA_AZIMUTH_DEG = 160.0
_HUMAN_CAMERA_ELEVATION_DEG = -45.0
_HUMAN_CAMERA_DISTANCE = 1.70
_ROBOT_OCCLUSION_ALPHA = 0.18
_ROBOT_OCCLUSION_HOLD_FRAMES = 5
_RAY_VISIBLE_GEOM_GROUPS = np.asarray((1, 1, 1, 0, 0, 0), dtype=np.uint8)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class _HumanWindowViewer(GymnasiumWindowViewer):
    """Window viewer with a scene-only postprocess hook.

    The hook runs after MuJoCo rebuilds the visual scene and before OpenGL
    renders it. Changes therefore never leak into physics or recorded cameras.
    """

    scene_postprocess = None

    def render(self):
        def update():
            self._create_overlay()

            render_start = time.time()
            if self.window is None:
                return
            if glfw.window_should_close(self.window):
                glfw.destroy_window(self.window)
                glfw.terminate()
                return
            self.viewport.width, self.viewport.height = glfw.get_framebuffer_size(
                self.window
            )
            mujoco.mjv_updateScene(
                self.model,
                self.data,
                self.vopt,
                mujoco.MjvPerturb(),
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL.value,
                self.scn,
            )

            if callable(self.scene_postprocess):
                self.scene_postprocess(self)

            for marker in self._markers:
                self._add_marker_to_scene(marker)

            mujoco.mjr_render(self.viewport, self.scn, self.con)
            if not self._hide_menu:
                for gridpos, (text_left, text_right) in self._overlays.items():
                    mujoco.mjr_overlay(
                        mujoco.mjtFontScale.mjFONTSCALE_150,
                        gridpos,
                        self.viewport,
                        text_left,
                        text_right,
                        self.con,
                    )

            glfw.swap_buffers(self.window)
            glfw.poll_events()
            self._time_per_render = 0.9 * self._time_per_render + 0.1 * (
                time.time() - render_start
            )

        if self._paused:
            while self._paused:
                update()
                if self._advance_by_one_step:
                    self._advance_by_one_step = False
                    break
        else:
            self._loop_count += self.model.opt.timestep / (
                self._time_per_render * self._run_speed
            )
            if self._render_every_frame:
                self._loop_count = 1
            while self._loop_count > 0:
                update()
                self._loop_count -= 1

        self._overlays.clear()
        self._markers.clear()


class MujocoRenderer(GymnasiumMujocoRenderer):
    """Compatibility wrapper for Gymnasium 0.29 and 1.x MuJoCo renderers."""

    def __init__(self, model, data, *args, width=None, height=None, **kwargs):
        if width is None:
            width = int(model.vis.global_.offwidth)
        if height is None:
            height = int(model.vis.global_.offheight)
        super().__init__(model, data, *args, width=width, height=height, **kwargs)
        (
            self._robot_render_groups,
            self._robot_geom_to_group,
            self._robot_body_ids,
        ) = self._collect_robot_render_groups()
        self._robot_occlusion_hold = {
            group: 0 for group in self._robot_render_groups
        }
        self._robot_occlusion_fade = {
            group: 1.0 for group in self._robot_render_groups
        }
        self._robot_occlusion_last_update = time.monotonic()

    def _get_viewer(self, render_mode: Optional[str]):
        if render_mode != "human":
            return super()._get_viewer(render_mode)

        self.viewer = self._viewers.get(render_mode)
        if self.viewer is None:
            self.viewer = _HumanWindowViewer(
                self.model,
                self.data,
                self.width,
                self.height,
                self.max_geom,
                self._vopt,
            )
            self.viewer.scene_postprocess = self._postprocess_human_scene
            self._set_cam_config()
            self._viewers[render_mode] = self.viewer

        if len(self._viewers) > 1:
            self.viewer.make_context_current()
        return self.viewer

    def _set_cam_config(self):
        if self.default_cam_config is not None:
            return super()._set_cam_config()
        viewer = getattr(self, "viewer", None)
        if viewer is not None and hasattr(viewer, "window"):
            # Several environments create their viewer before their first reset.
            # Refresh derived positions so table sites/geoms are not still zero.
            mujoco.mj_forward(self.model, self.data)
            mujoco.mjv_defaultFreeCamera(self.model, viewer.cam)
            focus, _ = self._table_view_geometry()
            viewer.cam.lookat[:] = focus
            viewer.cam.azimuth = _env_float(
                "HANDSON_HUMAN_CAMERA_AZIMUTH_DEG",
                _HUMAN_CAMERA_AZIMUTH_DEG,
            )
            viewer.cam.elevation = _env_float(
                "HANDSON_HUMAN_CAMERA_ELEVATION_DEG",
                _HUMAN_CAMERA_ELEVATION_DEG,
            )
            viewer.cam.distance = max(
                0.1,
                _env_float("HANDSON_HUMAN_CAMERA_DISTANCE", _HUMAN_CAMERA_DISTANCE),
            )

    def _body_subtree_ids(self, root_body_id: int) -> set[int]:
        body_ids = {int(root_body_id)}
        for body_id in range(1, int(self.model.nbody)):
            parent = int(self.model.body_parentid[body_id])
            while parent > 0:
                if parent == root_body_id:
                    body_ids.add(body_id)
                    break
                parent = int(self.model.body_parentid[parent])
        return body_ids

    def _named_body_subtree(self, name: str) -> set[int]:
        body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            name,
        )
        return self._body_subtree_ids(int(body_id)) if body_id >= 0 else set()

    def _collect_robot_render_groups(
        self,
    ) -> tuple[dict[str, tuple[int, ...]], dict[int, str], set[int]]:
        specs = (
            ("right", "base_right", ("right_palm_link", "allegro_palm_right")),
            ("left", "base_left", ("left_palm_link", "allegro_palm_left")),
            (
                "center",
                "rokae_center_components",
                ("rokae_center_head_fixed_frame",),
            ),
            ("robot", "base", ("allegro_palm",)),
        )
        render_groups: dict[str, tuple[int, ...]] = {}
        geom_to_group: dict[int, str] = {}
        all_robot_bodies: set[int] = set()

        for group, root_name, hand_roots in specs:
            root_bodies = self._named_body_subtree(root_name)
            if not root_bodies:
                continue
            all_robot_bodies.update(root_bodies)
            arm_bodies = set(root_bodies)
            for hand_root in hand_roots:
                arm_bodies.difference_update(self._named_body_subtree(hand_root))
            geom_ids = tuple(
                geom_id
                for geom_id in range(int(self.model.ngeom))
                if int(self.model.geom_bodyid[geom_id]) in arm_bodies
                and int(self.model.geom_group[geom_id]) <= 3
            )
            if not geom_ids:
                continue
            render_groups[group] = geom_ids
            for geom_id in geom_ids:
                geom_to_group[geom_id] = group
        return render_groups, geom_to_group, all_robot_bodies

    def _table_view_geometry(self) -> tuple[np.ndarray, np.ndarray]:
        table_geom_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_GEOM,
            "table_visual",
        )
        if table_geom_id >= 0:
            focus = np.asarray(self.data.geom_xpos[table_geom_id], dtype=np.float64).copy()
            size = np.asarray(self.model.geom_size[table_geom_id], dtype=np.float64)
            focus[2] += float(size[2]) + 0.06
            return focus, np.maximum(size[:2], 0.1)

        table_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "table_top",
        )
        if table_site_id >= 0:
            focus = np.asarray(self.data.site_xpos[table_site_id], dtype=np.float64).copy()
            focus[2] += 0.06
            return focus, np.asarray((0.45, 0.70), dtype=np.float64)

        focus = np.asarray(self.model.stat.center, dtype=np.float64).copy()
        extent = float(np.clip(self.model.stat.extent, 0.5, 2.0))
        return focus, np.asarray((0.4 * extent, 0.4 * extent), dtype=np.float64)

    @staticmethod
    def _scene_camera_position(scene) -> np.ndarray:
        position = np.zeros(3, dtype=np.float64)
        forward = np.zeros(3, dtype=np.float64)
        up = np.zeros(3, dtype=np.float64)
        mujoco.mjv_cameraInModel(position, forward, up, scene)
        return position

    def _body_has_visible_geom(self, body_id: int) -> bool:
        geom_adr = int(self.model.body_geomadr[body_id])
        geom_num = int(self.model.body_geomnum[body_id])
        for geom_id in range(geom_adr, geom_adr + geom_num):
            if (
                int(self.model.geom_group[geom_id]) <= 2
                and float(self.model.geom_rgba[geom_id, 3]) > 0.01
            ):
                return True
        return False

    def _human_view_focus_points(self) -> list[np.ndarray]:
        table_focus, table_half_size = self._table_view_geometry()
        table_top_z = float(table_focus[2] - 0.06)
        points = [table_focus]
        candidates: list[tuple[float, np.ndarray]] = []
        for body_id in range(1, int(self.model.nbody)):
            if body_id in self._robot_body_ids or not self._body_has_visible_geom(body_id):
                continue
            body_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            ) or ""
            if body_name in {"table", "target"}:
                continue
            pos = np.asarray(self.data.xpos[body_id], dtype=np.float64)
            delta_xy = np.abs(pos[:2] - table_focus[:2])
            if np.any(delta_xy > table_half_size + 0.08):
                continue
            if not (table_top_z - 0.02 <= float(pos[2]) <= table_top_z + 0.60):
                continue
            distance = float(np.linalg.norm(pos[:2] - table_focus[:2]))
            candidates.append((distance, pos.copy()))
        candidates.sort(key=lambda item: item[0])
        points.extend(pos for _, pos in candidates[:10])
        return points

    def _ray_blocking_robot_group(
        self,
        camera_pos: np.ndarray,
        target_pos: np.ndarray,
    ) -> str | None:
        ray = np.asarray(target_pos, dtype=np.float64) - np.asarray(
            camera_pos,
            dtype=np.float64,
        )
        target_distance = float(np.linalg.norm(ray))
        if target_distance <= 1e-6:
            return None
        ray /= target_distance
        geom_id = np.asarray((-1,), dtype=np.int32)
        hit_distance = float(
            mujoco.mj_ray(
                self.model,
                self.data,
                np.asarray(camera_pos, dtype=np.float64),
                ray,
                _RAY_VISIBLE_GEOM_GROUPS,
                1,
                -1,
                geom_id,
            )
        )
        if hit_distance < 0.0 or hit_distance >= target_distance - 0.015:
            return None
        hit_geom = int(geom_id[0])
        if hit_geom < 0 or float(self.model.geom_rgba[hit_geom, 3]) <= 0.01:
            return None
        return self._robot_geom_to_group.get(hit_geom)

    def _occluding_robot_groups(self, scene) -> set[str]:
        if not self._robot_render_groups:
            return set()
        camera_pos = self._scene_camera_position(scene)
        return {
            group
            for target in self._human_view_focus_points()
            if (group := self._ray_blocking_robot_group(camera_pos, target))
            is not None
        }

    def _update_robot_occlusion_fade(self, blocked: set[str]) -> None:
        active = set()
        for group in self._robot_render_groups:
            if group in blocked:
                self._robot_occlusion_hold[group] = _ROBOT_OCCLUSION_HOLD_FRAMES
            elif self._robot_occlusion_hold[group] > 0:
                self._robot_occlusion_hold[group] -= 1
            if group in blocked or self._robot_occlusion_hold[group] > 0:
                active.add(group)

        now = time.monotonic()
        dt = float(np.clip(now - self._robot_occlusion_last_update, 1.0 / 240.0, 0.1))
        self._robot_occlusion_last_update = now
        transparent_alpha = float(
            np.clip(
                _env_float(
                    "HANDSON_ROBOT_OCCLUSION_ALPHA",
                    _ROBOT_OCCLUSION_ALPHA,
                ),
                0.02,
                1.0,
            )
        )
        for group, current in self._robot_occlusion_fade.items():
            target = transparent_alpha if group in active else 1.0
            tau = 0.12 if target < current else 0.20
            blend = 1.0 - np.exp(-dt / tau)
            value = current + (target - current) * blend
            if abs(value - target) < 1e-3:
                value = target
            self._robot_occlusion_fade[group] = float(value)

    def _postprocess_human_scene(self, viewer) -> None:
        scene = viewer.scn
        self._update_robot_occlusion_fade(self._occluding_robot_groups(scene))
        for index in range(int(scene.ngeom)):
            scene_geom = scene.geoms[index]
            if int(scene_geom.objtype) != int(mujoco.mjtObj.mjOBJ_GEOM):
                continue
            group = self._robot_geom_to_group.get(int(scene_geom.objid))
            if group is None:
                continue
            fade = self._robot_occlusion_fade[group]
            if fade >= 0.999:
                continue
            scene_geom.rgba[3] *= fade
            scene_geom.transparent = 1

    def _human_viewer(self):
        viewers = getattr(self, "_viewers", None)
        if isinstance(viewers, dict):
            viewer = viewers.get("human")
            if viewer is not None:
                return viewer
        if hasattr(self, "_get_viewer"):
            try:
                return self._get_viewer("human")
            except Exception:
                pass
        viewer = getattr(self, "viewer", None)
        if viewer is not None and hasattr(viewer, "window"):
            return viewer
        return None

    def _task_overlay_prompt(self) -> str:
        getter = getattr(self, "task_overlay_prompt_getter", None)
        if callable(getter):
            try:
                prompt = getter()
            except Exception:
                prompt = ""
        else:
            prompt = getattr(self, "task_overlay_prompt", "")
        return str(prompt or "").strip()

    def _add_task_prompt_overlay(self) -> None:
        prompt = self._task_overlay_prompt()
        if not prompt:
            return
        viewer = self._human_viewer()
        if viewer is None or not hasattr(viewer, "add_overlay"):
            return
        title = str(getattr(self, "task_overlay_title", "Task") or "Task")
        wrapped_prompt = "\n".join(textwrap.wrap(prompt, width=52)) or prompt
        viewer.add_overlay(
            mujoco.mjtGridPos.mjGRID_TOPRIGHT,
            "Task\nPrompt",
            f"{title}\n{wrapped_prompt}",
        )

    def _camera_name_from_args(
        self,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ) -> str:
        if camera_name is not None:
            return str(camera_name)
        if camera_id is None:
            return ""
        try:
            name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                int(camera_id),
            )
        except Exception:
            return ""
        return str(name or "")

    def _rokae_wuji_self_occluding_wrist_geoms(
        self,
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ) -> list[int]:
        name = self._camera_name_from_args(camera_id=camera_id, camera_name=camera_name)
        if name not in {
            "wrist_fisheye_left",
            "handcam_rgb_left",
            "wrist_fisheye_right",
            "handcam_rgb_right",
            "handcam_rgb",
        }:
            return []

        side = "left" if name.endswith("_left") else "right"
        geoms = []
        for suffix in ("visual", "collision"):
            geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"{side}_wrist_fisheye_flange_assembly_{suffix}",
            )
            if geom_id >= 0:
                geoms.append(int(geom_id))
        return geoms

    def _render_without_rokae_wuji_wrist_self_occlusion(
        self,
        geoms: list[int],
        render_mode: Optional[str],
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        if not geoms:
            return super().render(
                render_mode=render_mode,
                camera_id=camera_id,
                camera_name=camera_name,
            )

        original = {geom_id: self.model.geom_rgba[geom_id].copy() for geom_id in geoms}
        try:
            for geom_id in geoms:
                self.model.geom_rgba[geom_id, 3] = 0.0
            return super().render(
                render_mode=render_mode,
                camera_id=camera_id,
                camera_name=camera_name,
            )
        finally:
            for geom_id, rgba in original.items():
                self.model.geom_rgba[geom_id] = rgba

    def render(
        self,
        render_mode: Optional[str],
        camera_id: Optional[int] = None,
        camera_name: Optional[str] = None,
    ):
        if render_mode == "human":
            self._add_task_prompt_overlay()
        try:
            hidden_geoms = self._rokae_wuji_self_occluding_wrist_geoms(
                camera_id=camera_id,
                camera_name=camera_name,
            )
            return self._render_without_rokae_wuji_wrist_self_occlusion(
                hidden_geoms,
                render_mode=render_mode,
                camera_id=camera_id,
                camera_name=camera_name,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise

        if camera_id is not None and camera_name is not None:
            raise ValueError("camera_id and camera_name cannot both be specified.")

        if camera_name is not None:
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                camera_name,
            )

        if camera_id is not None:
            self.camera_id = camera_id

        hidden_geoms = self._rokae_wuji_self_occluding_wrist_geoms(
            camera_id=camera_id
        )
        if not hidden_geoms:
            return super().render(render_mode=render_mode)

        original = {
            geom_id: self.model.geom_rgba[geom_id].copy()
            for geom_id in hidden_geoms
        }
        try:
            for geom_id in hidden_geoms:
                self.model.geom_rgba[geom_id, 3] = 0.0
            return super().render(render_mode=render_mode)
        finally:
            for geom_id, rgba in original.items():
                self.model.geom_rgba[geom_id] = rgba
