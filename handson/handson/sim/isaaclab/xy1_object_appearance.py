"""XY1 visual finishes, preserving the existing box/lid physics and geometry."""
import os


def apply_xy1_object_appearance(environment):
    if environment.task_name != "eval_comb03_pass_block_and_close_box":
        return None
    if os.environ.get("HANDSON_XY1_OBJECT_APPEARANCE", "source").strip() != "polymer_v1":
        return None
    from .xy_container_appearance import apply_clear_box_materials
    paths = {}
    for name in ("Clear_Box", "Box_lid"):
        prim = environment._find_rigid_body_prim(name)
        if prim is None or not prim.IsValid():
            raise RuntimeError("XY1 visual container body missing: " + name)
        paths[name] = str(prim.GetPath())
    report = apply_clear_box_materials(environment._stage, paths)
    environment.xy1_object_appearance_report = report
    print("[HandsOn/XY1] Container appearance: " + str(report), flush=True)
    return report
