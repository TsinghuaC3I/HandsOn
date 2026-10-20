from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from typing import Any


class IsaacSimUnavailable(RuntimeError):
    """Raised when the Isaac Sim runtime cannot be loaded."""


_SIMULATION_APP: Any | None = None


def isaacsim_is_importable() -> bool:
    return importlib.util.find_spec("isaacsim") is not None


@dataclass(frozen=True)
class IsaacSimLaunchConfig:
    headless: bool = True
    width: int = 640
    height: int = 640
    renderer: str = "RaytracedLighting"


def launch_simulation_app(config: IsaacSimLaunchConfig):
    """Start or reuse the process-wide Isaac Sim SimulationApp.

    Isaac Sim requires the application helper to be constructed before importing
    Omniverse modules. Keeping this in a small helper makes that ordering hard
    to accidentally violate from task code.
    """

    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return _SIMULATION_APP

    try:
        from isaacsim import SimulationApp
    except Exception as exc:  # pragma: no cover - exercised in Isaac runtime
        raise IsaacSimUnavailable(
            "Isaac Sim is not importable. Run this backend with Isaac Sim's "
            "python.sh, or set launch_isaac=False for interface-only tests."
        ) from exc

    _SIMULATION_APP = SimulationApp(
        {
            "headless": bool(config.headless),
            "width": int(config.width),
            "height": int(config.height),
            "renderer": str(config.renderer),
        }
    )
    return _SIMULATION_APP


def close_simulation_app() -> None:
    global _SIMULATION_APP
    if _SIMULATION_APP is None:
        return
    _SIMULATION_APP.close()
    _SIMULATION_APP = None
