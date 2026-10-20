"""Keep the selected Newton/Warp installation intact across Kit startup."""
import importlib
import importlib.machinery
from pathlib import Path
import sys


class _WarpPackageFinder:
    def __init__(self, root):
        self.root = root

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "warp" and not fullname.startswith("warp."):
            return None
        search = self.root.parent.joinpath(*fullname.split(".")[:-1])
        spec = importlib.machinery.PathFinder.find_spec(fullname, [str(search)], target)
        if spec is None:
            raise ModuleNotFoundError(f"{fullname} is absent from the selected Warp package", name=fullname)
        return spec


def pin_warp():
    # Kit's FastFinder can otherwise import old omni.warp.core submodules into
    # the newer package selected before SimulationApp starts.
    warp = importlib.import_module("warp")
    root = Path(warp.__file__).resolve().parent
    sys.meta_path[:] = [x for x in sys.meta_path if not isinstance(x, _WarpPackageFinder)]
    sys.meta_path.insert(0, _WarpPackageFinder(root))


def prepare_render():
    """Use Newton's CPU Fabric hierarchy propagation for the pinned RTX runtime.

    Its cubric ctypes adapter targets interface 0.1 while Sim 6 supplies 0.2.
    That route left visual descendants stale, including the first policy image.
    Keep GPU physics and Fabric body writes; only hierarchy propagation uses
    the upstream CPU fallback. Refresh after reset and visual visibility edits.
    """
    from isaaclab_newton.physics.newton_manager import NewtonManager
    NewtonManager._cubric = None
    NewtonManager._cubric_adapter = None
    NewtonManager._transforms_dirty = True


def publish_camera_pose(camera, target):
    """Newton FrameView owns sites; RTX reads the camera's USD/Fabric pose."""
    from pxr import Gf, UsdGeom
    import omni.usd
    import usdrt

    prim = camera._sensor_prims[0].GetPrim()
    world = Gf.Matrix4d(target.T.tolist())
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    xform = UsdGeom.Xformable(prim)
    xform.MakeMatrixXform().Set(world * parent.GetInverse())
    stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    fabric_prim = stage.GetPrimAtPath(str(prim.GetPath()))
    if fabric_prim:
        fabric_prim.CreateAttribute("omni:fabric:worldMatrix", usdrt.Sdf.ValueTypeNames.Matrix4d).Set(
            usdrt.Gf.Matrix4d(*target.T.reshape(-1).tolist()))


def rendered_camera_pose(camera):
    """Read the transform consumed by RTX rather than Newton's site cache."""
    import numpy as np
    from pxr import UsdGeom
    import omni.usd
    import usdrt

    prim = camera._sensor_prims[0].GetPrim()
    usd = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim)).T
    stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    fabric_prim = stage.GetPrimAtPath(str(prim.GetPath()))
    attr = fabric_prim.GetAttribute("omni:fabric:worldMatrix") if fabric_prim else None
    value = attr.Get() if attr else None
    fabric = np.asarray(value).T if value is not None else usd
    return usd, fabric
