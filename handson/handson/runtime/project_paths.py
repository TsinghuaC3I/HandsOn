"""Resolve portable project configuration without depending on process cwd."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def resolve_project_path(value: str | Path) -> Path:
    """Relative settings belong to the checkout; explicit external paths work."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()
