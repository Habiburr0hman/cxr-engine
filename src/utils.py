import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.constants import PROJECT_ROOT


def setup_reproducibility(seed) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


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
