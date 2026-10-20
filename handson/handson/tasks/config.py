from abc import ABC, abstractmethod
import ast
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _read_multitask_prompt(task_name: str) -> str:
    if not task_name:
        return ""
    config_path = _REPO_ROOT / "configs" / "multi_task" / f"{task_name}.yaml"
    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("prompt:"):
            continue
        raw = stripped.split(":", 1)[1].strip()
        if not raw:
            return ""
        try:
            return str(ast.literal_eval(raw))
        except (SyntaxError, ValueError):
            return raw.strip("\"'")
    return ""


class TaskConfigBase(ABC):
    """Base config for task environments."""

    proprio_keys = None
    task_id = ""
    task_name = ""

    def _task_name(self) -> str:
        task_name = str(getattr(self, "task_name", "") or "")
        if task_name:
            return task_name
        module_parts = self.__class__.__module__.split(".")
        if len(module_parts) >= 2:
            return module_parts[-2]
        return ""

    def _task_overlay_title(self) -> str:
        task_id = str(getattr(self, "task_id", "") or "").strip()
        task_name = self._task_name()
        if task_id and task_name:
            return f"{task_id} {task_name}"
        return task_id or task_name or "Task"

    def _task_overlay_prompt(self) -> str:
        prompt = str(getattr(self, "task_prompt", "") or "").strip()
        if prompt:
            return prompt
        prompt = _read_multitask_prompt(self._task_name())
        return prompt or self._task_name()

    def attach_prompt_overlay(self, env):
        prompt_getter = getattr(env, "_target_prompt", None)
        title = self._task_overlay_title()
        prompt = "" if callable(prompt_getter) else self._task_overlay_prompt()

        for attr in ("_viewer", "_mj_viewer"):
            renderer = getattr(env, attr, None)
            if renderer is None:
                continue
            setattr(renderer, "task_overlay_title", title)
            if callable(prompt_getter):
                setattr(renderer, "task_overlay_prompt_getter", prompt_getter)
            else:
                setattr(renderer, "task_overlay_prompt", prompt)

        setattr(env, "task_overlay_title", title)
        if callable(prompt_getter):
            setattr(env, "task_overlay_prompt_getter", prompt_getter)
        else:
            setattr(env, "task_overlay_prompt", prompt)
        return env

    def get_environment(self, policy_mode=False, render_mode="human", randomize=True, **kwargs):
        """Build a collection environment or the canonical EEF52 policy view.

        Policy mode returns measured state and accepts world EEF52 actions for
        every task. For complete paper reset/render/budget orchestration use
        handson.evaluation; this factory retains the task's native lifecycle.
        """
        if not policy_mode:
            return self.get_native_environment(policy_mode=False, render_mode=render_mode,
                                               randomize=randomize, **kwargs)
        from ..interfaces import CONTROL_HZ
        from .registry import TASKS
        contract = TASKS[self._task_name()].native_contract()
        for key, value in (contract or {}).get("constructor", {}).items():
            if key not in {"randomize", "render_mode", "policy_mode", "image_obs", "headless"}:
                kwargs.setdefault(key, value)
        from .eef52_wrapper import EEF52PolicyWrapper
        if kwargs.get("hand_type", "wuji1") != "wuji1":
            raise ValueError("The EEF52 policy interface requires wuji1")
        if abs(kwargs.get("control_dt", 1 / CONTROL_HZ) - 1 / CONTROL_HZ) > 1e-12:
            raise ValueError("The EEF52 policy interface requires 30 Hz control")
        kwargs.setdefault("hand_type", "wuji1")
        kwargs.setdefault("control_dt", 1 / CONTROL_HZ)
        wrapped = self.get_native_environment(policy_mode=True, render_mode=render_mode,
                                              randomize=randomize, **kwargs)
        try:
            return EEF52PolicyWrapper(wrapped, self._task_name())
        except BaseException:
            wrapped.close()
            raise

    @abstractmethod
    def get_native_environment(self, policy_mode=False, render_mode="human", randomize=True, **kwargs):
        """Task-owned collection factory with its native action/observation layout."""
        pass

    @abstractmethod
    def process_demos(self, demo):
        pass
