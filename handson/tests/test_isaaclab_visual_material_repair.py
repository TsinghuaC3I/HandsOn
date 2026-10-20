from __future__ import annotations

import ast
import os
from pathlib import Path
import sys
import types


_SCENE_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "handson"
    / "sim"
    / "isaaclab"
    / "bimanual_assembly_scene.py"
)


def _load_material_repair_function():
    source = _SCENE_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_SCENE_SOURCE))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_repair_imported_visual_materials"
    )
    isolated_module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(isolated_module)
    namespace = {
        "os": os,
        "_is_robot_collision_render_path": lambda _path: False,
    }
    exec(compile(isolated_module, str(_SCENE_SOURCE), "exec"), namespace)
    return namespace["_repair_imported_visual_materials"]


class _InvalidPrim:
    def IsValid(self) -> bool:
        return False

    def GetPath(self) -> str:
        return ""

    def GetParent(self):
        return self


_INVALID_PRIM = _InvalidPrim()


class _MaterialPrim:
    def __init__(self, path: str) -> None:
        self.path = path

    def IsValid(self) -> bool:
        return True

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]


class _Material:
    def __init__(self, path: str) -> None:
        self.path = path
        self.prim = _MaterialPrim(path)

    def GetPath(self) -> str:
        return self.path

    def GetPrim(self) -> _MaterialPrim:
        return self.prim


class _Prim:
    def __init__(
        self,
        path: str,
        *,
        parent: _Prim | None = None,
        gprim: bool = False,
        instance_proxy: bool = False,
        instance: bool = False,
        material: _Material | None = None,
    ) -> None:
        self.path = path
        self.parent = parent
        self.gprim = gprim
        self.instance_proxy = instance_proxy
        self.instance = instance
        self.material = material
        self.children: list[_Prim] = []
        self.bound_material: _Material | None = None
        self.set_instanceable_calls: list[bool] = []
        if parent is not None:
            parent.children.append(self)

    def IsValid(self) -> bool:
        return True

    def GetPath(self) -> str:
        return self.path

    def GetName(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    def GetParent(self):
        return self.parent if self.parent is not None else _INVALID_PRIM

    def IsA(self, _schema) -> bool:
        return self.gprim

    def IsInstanceProxy(self) -> bool:
        return self.instance_proxy

    def IsInstance(self) -> bool:
        return self.instance

    def SetInstanceable(self, value: bool) -> None:
        self.set_instanceable_calls.append(value)
        self.instance = value
        if not value:
            for descendant in self.walk()[1:]:
                descendant.instance_proxy = False

    def walk(self) -> list[_Prim]:
        result = [self]
        for child in self.children:
            result.extend(child.walk())
        return result


class _Stage:
    def __init__(self, root: _Prim) -> None:
        self.root = root

    def GetPrimAtPath(self, path: str):
        return next(
            (prim for prim in self.root.walk() if prim.path == path),
            _INVALID_PRIM,
        )


class _DirectBinding:
    def __init__(self, material: _Material | None) -> None:
        self.material = material

    def GetMaterial(self) -> _Material | None:
        return self.material


class _MaterialBindingAPI:
    def __init__(self, prim: _Prim) -> None:
        self.prim = prim

    @classmethod
    def Apply(cls, prim: _Prim):
        return cls(prim)

    def GetDirectBinding(self) -> _DirectBinding:
        return _DirectBinding(self.prim.material)

    def Bind(self, material: _Material, *, bindingStrength: str) -> None:
        assert bindingStrength == "strongerThanDescendants"
        self.prim.bound_material = material


def _install_fake_pxr(monkeypatch) -> None:
    pxr = types.ModuleType("pxr")
    pxr.Usd = types.SimpleNamespace(
        PrimRange=lambda root, _predicate: root.walk(),
        TraverseInstanceProxies=lambda: object(),
    )
    pxr.UsdGeom = types.SimpleNamespace(Gprim=object())
    pxr.UsdShade = types.SimpleNamespace(
        MaterialBindingAPI=_MaterialBindingAPI,
        Tokens=types.SimpleNamespace(
            strongerThanDescendants="strongerThanDescendants"
        ),
    )
    monkeypatch.setitem(sys.modules, "pxr", pxr)


def _visual_instance(material_paths: tuple[str, str]):
    root = _Prim("/World/Task")
    microwave = _Prim("/World/Task/microwave_object", parent=root)
    visuals = _Prim(
        "/World/Task/microwave_object/visuals",
        parent=microwave,
        instance=True,
    )
    meshes = []
    for name, material_path in zip(("body", "cavity"), material_paths, strict=True):
        material = _Material(material_path)
        xform = _Prim(
            f"/World/Task/microwave_object/visuals/{name}",
            parent=visuals,
            instance_proxy=True,
            material=material,
        )
        meshes.append(
            _Prim(
                f"/World/Task/microwave_object/visuals/{name}/mesh",
                parent=xform,
                gprim=True,
                instance_proxy=True,
            )
        )
    return _Stage(root), visuals, tuple(meshes)


def test_heterogeneous_visual_instance_is_expanded_and_bound_per_geom(
    monkeypatch,
) -> None:
    monkeypatch.delenv("HANDSON_ISAACLAB_DEBUG_MATERIALS", raising=False)
    _install_fake_pxr(monkeypatch)
    repair = _load_material_repair_function()
    stage, visuals, (body_mesh, cavity_mesh) = _visual_instance(
        ("/Materials/body", "/Materials/cavity")
    )

    rebound, hidden = repair(stage, "/World/Task")

    assert visuals.set_instanceable_calls == [False]
    assert visuals.bound_material is None
    assert body_mesh.bound_material.path == "/Materials/body"
    assert cavity_mesh.bound_material.path == "/Materials/cavity"
    assert (rebound, hidden) == (2, 0)


def test_homogeneous_visual_instance_stays_instanced(monkeypatch) -> None:
    monkeypatch.delenv("HANDSON_ISAACLAB_DEBUG_MATERIALS", raising=False)
    _install_fake_pxr(monkeypatch)
    repair = _load_material_repair_function()
    stage, visuals, meshes = _visual_instance(
        ("/Materials/shared", "/Materials/shared")
    )

    rebound, hidden = repair(stage, "/World/Task")

    assert visuals.set_instanceable_calls == []
    assert visuals.bound_material.path == "/Materials/shared"
    assert all(mesh.bound_material is None for mesh in meshes)
    assert (rebound, hidden) == (1, 0)


def test_heterogeneous_asset_root_is_not_broadly_expanded(monkeypatch) -> None:
    monkeypatch.delenv("HANDSON_ISAACLAB_DEBUG_MATERIALS", raising=False)
    _install_fake_pxr(monkeypatch)
    repair = _load_material_repair_function()
    root = _Prim("/World/Task", instance=True)
    visuals = _Prim(
        "/World/Task/visuals",
        parent=root,
        instance_proxy=True,
    )
    for name in ("body", "cavity"):
        xform = _Prim(
            f"/World/Task/visuals/{name}",
            parent=visuals,
            instance_proxy=True,
            material=_Material(f"/Materials/{name}"),
        )
        _Prim(
            f"/World/Task/visuals/{name}/mesh",
            parent=xform,
            gprim=True,
            instance_proxy=True,
        )

    rebound, hidden = repair(_Stage(root), "/World/Task")

    assert root.set_instanceable_calls == []
    assert root.bound_material is None
    assert (rebound, hidden) == (0, 0)
