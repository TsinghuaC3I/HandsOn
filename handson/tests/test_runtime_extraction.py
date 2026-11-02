"""The shared implementation must preserve the verified method bodies.

The fixture was captured before extraction. This checks effective lookup,
including methods intentionally retained as historical overrides, without
requiring an Isaac Sim process.
"""

import ast
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "handson/sim/isaaclab"
FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/runtime_extraction.json").read_text()
)


def methods(module, class_name):
    tree = ast.parse((ROOT / (module + ".py")).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return cls, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


@pytest.mark.parametrize("module", FIXTURE["modules"])
def test_resolved_runtime_methods_preserve_baseline(module):
    reference = FIXTURE["modules"][module]
    cls, own = methods(module, reference["class"])
    support_name = reference["support_module"]
    support_class = FIXTURE["support_modules"][support_name]["class_name"]
    _, shared = methods(support_name, support_class)
    assert isinstance(cls.bases[0], ast.Name) and cls.bases[0].id == support_class
    assert set(own) | set(shared) == set(reference["methods"])
    for name, expected in reference["methods"].items():
        node = own.get(name, shared.get(name))
        actual = hashlib.sha256(
            ast.dump(node, include_attributes=False).encode()
        ).hexdigest()
        assert actual == expected, (module, name)
    assert not set(reference["shared_methods"]) & set(own)


@pytest.mark.parametrize("module", FIXTURE["support_modules"])
def test_shared_methods_do_not_depend_on_old_module_or_super(module):
    _, shared = methods(module, FIXTURE["support_modules"][module]["class_name"])
    forbidden = {
        "super",
        "globals",
        "eval",
        "exec",
        "__class__",
        "__file__",
        "__name__",
    }
    for name, method in shared.items():
        assert not any(
            isinstance(n, ast.Name) and n.id in forbidden for n in ast.walk(method)
        ), name


@pytest.mark.parametrize("module", FIXTURE["support_modules"])
def test_shared_method_annotations_resolve(module):
    import importlib
    import inspect
    import typing

    runtime = importlib.import_module("handson.sim.isaaclab." + module)
    cls = getattr(runtime, FIXTURE["support_modules"][module]["class_name"])
    for _, method in inspect.getmembers(cls, inspect.isfunction):
        typing.get_type_hints(method)
