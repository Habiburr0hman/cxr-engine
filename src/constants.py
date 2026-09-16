from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

BINARY_TARGET_MAP = {"NORMAL": 0, "PNEUMONIA": 1}
MULTI_TARGET_MAP = {"NORMAL": 0, "BACTERIA": 1, "VIRUS": 2}

SEED = 20925010
