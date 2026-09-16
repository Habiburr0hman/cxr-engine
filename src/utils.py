from pathlib import Path
from typing import Any

import yaml

from src.constants import PROJECT_ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found at: {file_path.resolve()}"
        )

    with file_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_project_relative_path(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()
