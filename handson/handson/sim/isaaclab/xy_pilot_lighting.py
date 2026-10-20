"""Opt-in neutral soft lighting for native XY appearance review."""
import os


def apply_xy_pilot_lighting(environment):
    profile = os.environ.get("HANDSON_XY_VISUAL_LIGHTING", "source").strip()
    if profile != "soft_fill_v1" or environment.task_name not in {
        "eval_comb02_open_laptop_and_click_mouse", "eval_comb01_cover_pitcher_and_box",
        "eval_comb06_season_and_stir", "eval_comb04_place_fruit_and_photograph_logo",
    }:
        return None
    from pxr import Gf, UsdGeom, UsdLux
    from .blender_visual_layer import physics_signature

    stage = environment._stage
    before = physics_signature(stage)
    root = "/World/XYVisualPilotLighting"
    UsdGeom.Xform.Define(stage, root)
    dome = UsdLux.DomeLight.Define(stage, root + "/NeutralFill")
    dome.CreateIntensityAttr(320.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key = UsdLux.RectLight.Define(stage, root + "/Softbox")
    key.CreateIntensityAttr(1000.0)
    key.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    key.CreateWidthAttr(1.0)
    key.CreateHeightAttr(1.0)
    key.CreateNormalizeAttr(True)
    key.AddTranslateOp().Set(Gf.Vec3d(0.1, -0.25, 2.2))
    after = physics_signature(stage)
    if before != after:
        raise RuntimeError("XY lighting changed the physical contract")
    report = {"profile": profile, "fill_intensity": 320.0,
              "softbox_intensity": 1000.0, "softbox_position_m": [0.1, -0.25, 2.2],
              "physics_before": before, "physics_after": after}
    environment.xy_lighting_appearance_report = report
    print("[HandsOn/XY] Native soft lighting: " + str(report), flush=True)
    return report
