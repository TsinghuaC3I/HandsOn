"""Preserve early paper tasks' physical RNG use before the V16 appearance pass.

The archived bimanual_assembly_scene.randomize_task_scene_visuals used the
environment RNG. Later tasks use a separate visual stream. V16 supersedes the
old material/light values, but the earlier draws still affect the RNG state.
The ten source cohorts use the bundled 14-texture pool (no external textures).
Source helper AST SHA256: 166cf5da8b02a85aa6f016b25b30341714a7abb4b89c56630c63718b6eed4809
"""
import numpy as np

TASKS = frozenset({
    't03_cover_box', 't04_cover_pitcher', 't09_insert_peg',
    't14_move_and_click_mouse', 't19_orient_cup_upright',
    't24_place_ball_in_bowl', 't25_place_cup_on_plate',
    't26_place_fruit_in_bucket', 't44_stack_blocks', 't50_water_plant',
})


def sample(random_state, *, key_light_radius):
    """The original draw order, including randint's variable rejection count."""
    texture = int(random_state.randint(0, 14))
    scale = (float(random_state.uniform(0.8, 1.8)),
             float(random_state.uniform(1.2, 3.0)))
    roughness = float(random_state.uniform(0.45, 0.8))
    tint = np.asarray(random_state.uniform(0.82, 1.16, size=3), dtype=np.float32)
    dome_color = tuple(np.clip(tint * random_state.uniform(0.86, 1.08), 0.65, 1.25).tolist())
    key_color = tuple(np.clip(tint * random_state.uniform(0.90, 1.18), 0.65, 1.25).tolist())
    dome_intensity = float(random_state.uniform(280.0, 900.0))
    key_intensity = float(random_state.uniform(1800.0, 5200.0))
    key_position = tuple(np.asarray([
        random_state.uniform(-0.75, 0.25),
        random_state.uniform(-0.75, 0.75),
        random_state.uniform(1.65, 2.65),
    ], dtype=np.float64).tolist())
    radius = float(random_state.uniform(0.08, 0.30)) if key_light_radius else None
    return dict(texture_index=texture, scale=scale, roughness=roughness,
                dome_color=dome_color, key_color=key_color,
                dome_intensity=dome_intensity, key_intensity=key_intensity,
                key_position=key_position, key_radius=radius)


def preserve(env):
    if (env.task_name not in TASKS or env._physics_backend != 'physx'
            or not env.randomize or env._randomize_task_scene_visuals is None):
        return
    import omni.usd
    from pxr import Usd, UsdGeom
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath('/World/Task')
    # The original helper rejects these cases before its first random draw.
    if not root.IsValid() or not any(
            p.GetName().lower() == 'table' and p.IsA(UsdGeom.Xformable)
            for p in Usd.PrimRange(root)):
        return
    key = stage.GetPrimAtPath('/World/HandsOnKeyLight')
    radius = key.GetAttribute('inputs:radius') if key.IsValid() else None
    env._paper_reset_visual_draw = sample(
        env._random, key_light_radius=bool(radius and radius.IsValid()))
