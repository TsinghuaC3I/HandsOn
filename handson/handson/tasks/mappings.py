from __future__ import annotations

from collections.abc import Iterator, Mapping
from importlib import import_module


from .registry import TASKS

_TASK_CONFIG_MODULES = {name: spec.config_module for name, spec in TASKS.items()}


class _LazyConfigMapping(Mapping[str, type]):
    def __init__(self, modules: Mapping[str, str]) -> None:
        self._modules = dict(modules)
        self._cache: dict[str, type] = {}

    def __getitem__(self, key: str) -> type:
        module_name = self._modules[key]
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        module = import_module(module_name, __package__)
        config_cls = getattr(module, "TaskConfig")
        self._cache[key] = config_cls
        return config_cls

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    def __len__(self) -> int:
        return len(self._modules)


CONFIG_MAPPING = _LazyConfigMapping(_TASK_CONFIG_MODULES)
