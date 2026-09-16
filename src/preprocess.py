# scripts/preprocess_offline.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import DATA_DIR
from src.utils import get_project_relative_path

IN_CATALOG = DATA_DIR / "metadata" / "catalog.csv"
OUT_RAW_DIR = DATA_DIR / "processed" / "raw_resized_1"
OUT_CLAHE_DIR = DATA_DIR / "processed" / "clahe_resized_1"
OUT_CATALOG = DATA_DIR / "metadata" / "catalog_processed.csv"

TARGET_SIZE = (256, 256)
CLAHE_CLIP_LIMIT = 2
CLAHE_GRID_SIZE = (8, 8)


def apply_square_pad(image: np.ndarray) -> np.ndarray:
    """
    Symmetrically pad the shorter dimension with zeros to a 1:1 aspect ratio.
    The image must be a 2-D (H, W) or 3-D (H, W, C) uint8 or float array.
    Padding is applied on **both sides** of the shorter axis so the original
    content remains centred.
    Args:
        image: Input numpy array of shape (H, W) or (H, W, C).
    Returns:
        Square numpy array of shape (M, M) or (M, M, C) where
        M = max(H, W). Dtype is preserved.
    """
    if image.ndim not in (2, 3):
        raise ValueError(
            f"apply_square_pad expects a 2-D or 3-D array, got shape {image.shape}"
        )
    h, w = image.shape[:2]
    if h == w:
        return image

    max_dim: int = max(h, w)
    # Compute symmetric padding amounts (split remainder evenly; extra goes to end)
    if h < w:
        # Pad top and bottom
        pad_total: int = max_dim - h
        pad_before: int = pad_total // 2
        pad_after: int = pad_total - pad_before
        pad_width = (
            ((pad_before, pad_after), (0, 0))
            if image.ndim == 2
            else ((pad_before, pad_after), (0, 0), (0, 0))
        )
    else:
        # Pad left and right
        pad_total = max_dim - w
        pad_before = pad_total // 2
        pad_after = pad_total - pad_before
        pad_width = (
            ((0, 0), (pad_before, pad_after))
            if image.ndim == 2
            else ((0, 0), (pad_before, pad_after), (0, 0))
        )

    padded = np.pad(image, pad_width, mode="constant", constant_values=0)
    assert padded.shape[0] == max_dim and padded.shape[1] == max_dim, (
        f"Square-pad assertion failed: output shape {padded.shape}"
    )
    return padded


def main():
    OUT_RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_CLAHE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(IN_CATALOG)
    clahe_engine = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_GRID_SIZE
    )

    path_raw_list = []
    path_clahe_list = []

    print(f"Preprocessing {len(df)} images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        src_path = Path(row["filepath"])
        stem_name = f"{src_path.stem}.png"
        img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {src_path}")

        # Variant A: Square Pad -> Resize
        img_padded = apply_square_pad(img)
        img_raw_resized = cv2.resize(
            img_padded, TARGET_SIZE, interpolation=cv2.INTER_AREA
        )
        img_raw_path = OUT_RAW_DIR / stem_name
        cv2.imwrite(str(img_raw_path), img_raw_resized)

        # Variant B: CLAHE -> Square Pad -> Resize
        img_clahe = clahe_engine.apply(img)
        img_clahe_padded = apply_square_pad(img_clahe)
        img_clahe_resized = cv2.resize(
            img_clahe_padded, TARGET_SIZE, interpolation=cv2.INTER_AREA
        )
        img_clahe_path = OUT_CLAHE_DIR / stem_name
        cv2.imwrite(str(img_clahe_path), img_clahe_resized)

        path_raw_list.append(get_project_relative_path(img_raw_path))
        path_clahe_list.append(get_project_relative_path(img_clahe_path))

    df["path_raw_resized"] = path_raw_list
    df["path_clahe_resized"] = path_clahe_list
    df.to_csv(OUT_CATALOG, index=False)

    print("\nPreprocessing complete.")
    print(f"""
        Raw square-padded images saved to: {OUT_RAW_DIR.resolve()}
        CLAHE square-padded images saved to: {OUT_CLAHE_DIR.resolve()}
        Updated catalog saved to: {OUT_CATALOG.resolve()}
    """)


if __name__ == "__main__":
    main()
