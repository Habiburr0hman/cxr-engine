import logging
from pathlib import Path
from typing import Any

import yaml

from src.constants import PROJECT_ROOT

logger = logging.getLogger(__name__)


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


class DvcPointerError(Exception):
    """Raised when a DVC pointer cannot be read or parsed."""


def get_dvc_hash(pointer_file: Path | str, strict: bool = False) -> str | None:
    """Extracts the MD5/directory hash (.dir) from a DVC tracking file."""
    path = Path(pointer_file)

    try:
        if not path.is_file():
            raise DvcPointerError(f"DVC pointer file not found: {path}")

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        outs = data.get("outs", []) if isinstance(data, dict) else []
        md5 = outs[0].get("md5") if outs and isinstance(outs[0], dict) else None

        if not md5:
            raise DvcPointerError(f"No MD5 hash found in DVC pointer: {path}")

        return str(md5)

    except (OSError, yaml.YAMLError) as err:
        if strict:
            raise DvcPointerError(
                f"Failed reading DVC pointer '{path}': {err}"
            ) from err
        logger.warning("Failed reading DVC pointer '%s': %s", path, err)
        return None
    except DvcPointerError as err:
        if strict:
            raise
        logger.warning(str(err))
        return None
