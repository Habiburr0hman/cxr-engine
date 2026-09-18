# src/dataset.py
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.constants import SEED
from src.utils import get_project_relative_path


def parse_preprocessed_image(
    filepath: tf.Tensor, label: tf.Tensor
) -> tuple[tf.Tensor, tf.Tensor]:
    img_bytes = tf.io.read_file(filepath)
    image = tf.io.decode_png(img_bytes, channels=1)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)
    return image, label


def create_tf_dataset(
    df: pd.DataFrame,
    image_dir: str | Path,
    label_col: str,
    batch_size: int,
    filename_col: str = "filename",
    is_training: bool = False,
    seed: int = SEED,
) -> tf.data.Dataset:
    base_dir = Path(image_dir)
    sorted_df = df.sort_values(filename_col).reset_index(drop=True)
    paths = [
        (base_dir / f"{Path(name).stem}.png").as_posix()
        for name in sorted_df[filename_col]
    ]
    labels = sorted_df[label_col].values[:, None].astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)
    if is_training:
        ds = ds.shuffle(
            buffer_size=len(sorted_df), seed=seed, reshuffle_each_iteration=True
        )
    ds = ds.map(
        map_func=parse_preprocessed_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return ds


def sample_patients(
    sub_df: pd.DataFrame,
    target_n: int,
    rng: np.random.Generator,
    patient_col: str = "patient_id",
) -> pd.Index:
    # 1. Fallback if patient_col is not present
    if patient_col not in sub_df.columns:
        # Sort indices first so rng.choice is order-invariant
        sorted_indices = np.sort(sub_df.index.to_numpy())
        chosen = rng.choice(
            sorted_indices, size=min(len(sorted_indices), target_n), replace=False
        )
        return pd.Index(chosen)

    if len(sub_df) <= target_n:
        return sub_df.index

    # 2. Extract and strictly sort patient IDs to ensure order-invariant permutation
    patient_to_indices = sub_df.groupby(patient_col, sort=True).indices
    sorted_unique_patients = np.array(sorted(patient_to_indices.keys()))
    permuted_patients = rng.permutation(sorted_unique_patients)

    selected_indices = []
    accumulated_count = 0

    for pid in permuted_patients:
        row_pos = patient_to_indices[pid]
        actual_indices = sub_df.index[row_pos].tolist()
        selected_indices.extend(actual_indices)
        accumulated_count += len(actual_indices)

        if accumulated_count >= target_n:
            break

    return pd.Index(selected_indices)


def balance_split_by_patients(
    split_df: pd.DataFrame,
    label_col: str,
    patient_col: str,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.Index]:
    # 1. Enforce a clean, unique index to prevent .loc duplicates
    working_df = split_df.reset_index(drop=True)

    # 2. Deterministic class ordering (sort by class label to break ties consistently)
    class_counts = working_df[label_col].value_counts()
    min_class_count = int(class_counts.min())
    deterministic_classes = sorted(class_counts.index.tolist())

    balanced_indices = []
    for cls in deterministic_classes:
        cls_sub_df = working_df[working_df[label_col] == cls]
        sampled_idx = sample_patients(
            sub_df=cls_sub_df,
            target_n=min_class_count,
            rng=rng,
            patient_col=patient_col,
        )
        balanced_indices.extend(sampled_idx)

    selected_index = pd.Index(balanced_indices)

    # 3. Direct slice without redundant PRNG draw (create_tf_dataset sorts by filename)
    balanced_df = working_df.loc[selected_index].reset_index(drop=True)

    return balanced_df, selected_index


def get_train_val_test_datasets_with_audit(
    image_dir: str | Path,
    catalog_path: str | Path,
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_size: int = 32,
    label_col: str = "label",
    patient_col: str = "patient_id",
    filename_col: str = "filename",
    random_state: int = SEED,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    tf.data.Dataset,
    pd.DataFrame,
    dict[str, Any],
]:
    # 1. Deterministic file loading and ordering
    raw_catalog_df = pd.read_csv(catalog_path)

    # Sorted working copy for deterministic internal partitioning
    catalog_df = raw_catalog_df.sort_values(by=[patient_col, filename_col]).reset_index(
        drop=True
    )
    image_dir = Path(image_dir)

    # 2. Independent RNG streams via SeedSequence (prevents cross-split PRNG coupling)
    seed_seq = np.random.SeedSequence(random_state)
    train_seed, val_seed, test_seed = seed_seq.spawn(3)
    train_rng = np.random.default_rng(train_seed)
    val_rng = np.random.default_rng(val_seed)
    test_rng = np.random.default_rng(test_seed)

    # 3. Patient-level stratification with stable tie-breaking
    def _deterministic_mode(s: pd.Series):
        modes = s.mode()
        return min(modes)

    patient_table = (
        catalog_df.groupby(patient_col, sort=True)[label_col]
        .agg(_deterministic_mode)
        .reset_index()
        .sort_values(by=patient_col)
        .reset_index(drop=True)
    )

    # 4. Stage 1: Hold out Test patients
    train_val_pts, test_pts = train_test_split(
        patient_table,
        test_size=test_size,
        stratify=patient_table[label_col],
        random_state=random_state,
    )

    # Stage 2: Split Train and Val patients (synchronize sorted rows with stratification array)
    sorted_train_val_pts = train_val_pts.sort_values(by=patient_col).reset_index(
        drop=True
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_pts, val_pts = train_test_split(
        sorted_train_val_pts,
        test_size=relative_val_size,
        stratify=sorted_train_val_pts[label_col],
        random_state=random_state,
    )

    test_pt_set = set(test_pts[patient_col])
    val_pt_set = set(val_pts[patient_col])
    train_pt_set = set(train_pts[patient_col])

    # 5. Leakage Assertions
    assert train_pt_set.isdisjoint(val_pt_set), "Leakage: Train and Val share patients!"
    assert train_pt_set.isdisjoint(test_pt_set), (
        "Leakage: Train and Test share patients!"
    )
    assert val_pt_set.isdisjoint(test_pt_set), "Leakage: Val and Test share patients!"

    # 6. Partition raw catalog
    df_train_raw = catalog_df[catalog_df[patient_col].isin(train_pt_set)].copy()
    df_val_raw = catalog_df[catalog_df[patient_col].isin(val_pt_set)].copy()
    df_test_raw = catalog_df[catalog_df[patient_col].isin(test_pt_set)].copy()

    # 7. Balance subsets using independent RNG generators
    df_train_balanced, _ = balance_split_by_patients(
        df_train_raw, label_col=label_col, patient_col=patient_col, rng=train_rng
    )
    df_val_balanced, _ = balance_split_by_patients(
        df_val_raw, label_col=label_col, patient_col=patient_col, rng=val_rng
    )
    df_test_balanced, _ = balance_split_by_patients(
        df_test_raw, label_col=label_col, patient_col=patient_col, rng=test_rng
    )

    # 8. Robust audit tracking preserving exact original input catalog row order
    audit_df = raw_catalog_df.copy()
    audit_df["cohort"] = "unassigned"
    audit_df["status"] = "unassigned"
    audit_df["split"] = "unassigned"

    split_tracking = [
        ("train", set(df_train_balanced[filename_col]), train_pt_set),
        ("val", set(df_val_balanced[filename_col]), val_pt_set),
        ("test", set(df_test_balanced[filename_col]), test_pt_set),
    ]
    for split_name, retained_files, pt_set in split_tracking:
        is_in_split_pts = audit_df[patient_col].isin(pt_set)
        retained_mask = audit_df[filename_col].isin(retained_files)
        dropped_mask = is_in_split_pts & ~retained_mask

        # 1. Decoupled dimensions (Best Practice)
        audit_df.loc[is_in_split_pts, "cohort"] = split_name
        audit_df.loc[retained_mask, "status"] = "retained"
        audit_df.loc[dropped_mask, "status"] = "dropped"

        # 2. Simplified composite tag
        audit_df.loc[retained_mask, "split"] = f"{split_name}_retained"
        audit_df.loc[dropped_mask, "split"] = f"{split_name}_dropped"

    # 9. Audit summary
    def _get_class_dist(df: pd.DataFrame) -> dict[str, int]:
        return {
            str(k): int(v) for k, v in df[label_col].value_counts().sort_index().items()
        }

    rel_catalog_path = get_project_relative_path(Path(catalog_path))
    rel_image_dir = get_project_relative_path(Path(image_dir))
    audit_report = {
        "metadata": {
            "catalog_path": str(rel_catalog_path),
            "image_dir": str(rel_image_dir),
            "label_column": label_col,
            "patient_column": patient_col,
            "filename_column": filename_col,
            "random_state": random_state,
            "batch_size": batch_size,
            "split_ratios": {
                "train_target": round(1.0 - val_size - test_size, 4),
                "val_target": val_size,
                "test_target": test_size,
            },
        },
        "leakage_verification": {
            "train_val_overlap": len(train_pt_set & val_pt_set),
            "train_test_overlap": len(train_pt_set & test_pt_set),
            "val_test_overlap": len(val_pt_set & test_pt_set),
        },
        "patient_counts": {
            "total_catalog_patients": len(patient_table),
            "allocated_at_split": {
                "train": len(train_pt_set),
                "val": len(val_pt_set),
                "test": len(test_pt_set),
            },
            "retained_after_balancing": {
                "train": int(df_train_balanced[patient_col].nunique()),
                "val": int(df_val_balanced[patient_col].nunique()),
                "test": int(df_test_balanced[patient_col].nunique()),
            },
            "dropped": {
                "train": len(train_pt_set)
                - int(df_train_balanced[patient_col].nunique()),
                "val": len(val_pt_set) - int(df_val_balanced[patient_col].nunique()),
                "test": len(test_pt_set) - int(df_test_balanced[patient_col].nunique()),
            },
        },
        "image_counts": {
            "total_catalog_images": len(catalog_df),
            "train": {
                "raw": len(df_train_raw),
                "balanced": len(df_train_balanced),
                "dropped": len(df_train_raw) - len(df_train_balanced),
                "retention_rate": round(len(df_train_balanced) / len(df_train_raw), 4)
                if len(df_train_raw) > 0
                else 0.0,
            },
            "val": {
                "raw": len(df_val_raw),
                "balanced": len(df_val_balanced),
                "dropped": len(df_val_raw) - len(df_val_balanced),
                "retention_rate": round(len(df_val_balanced) / len(df_val_raw), 4)
                if len(df_val_raw) > 0
                else 0.0,
            },
            "test": {
                "raw": len(df_test_raw),
                "balanced": len(df_test_balanced),
                "dropped": len(df_test_raw) - len(df_test_balanced),
                "retention_rate": round(len(df_test_balanced) / len(df_test_raw), 4)
                if len(df_test_raw) > 0
                else 0.0,
            },
            "total_retained": (
                len(df_train_balanced) + len(df_val_balanced) + len(df_test_balanced)
            ),
            "total_dropped": len(catalog_df)
            - (len(df_train_balanced) + len(df_val_balanced) + len(df_test_balanced)),
        },
        "class_distributions": {
            "catalog_raw": _get_class_dist(catalog_df),
            "train_raw": _get_class_dist(df_train_raw),
            "train_balanced": _get_class_dist(df_train_balanced),
            "val_raw": _get_class_dist(df_val_raw),
            "val_balanced": _get_class_dist(df_val_balanced),
            "test_raw": _get_class_dist(df_test_raw),
            "test_balanced": _get_class_dist(df_test_balanced),
        },
        "audit_status_breakdown": {
            str(k): int(v)
            for k, v in audit_df["split"].value_counts().sort_index().items()
        },
    }

    # 10. Construct Datasets
    train_ds = create_tf_dataset(
        df=df_train_balanced,
        image_dir=image_dir,
        label_col=label_col,
        batch_size=batch_size,
        filename_col=filename_col,
        is_training=True,
        seed=random_state,
    )
    val_ds = create_tf_dataset(
        df=df_val_balanced,
        image_dir=image_dir,
        label_col=label_col,
        batch_size=batch_size,
        filename_col=filename_col,
        is_training=False,
        seed=random_state,
    )
    test_ds = create_tf_dataset(
        df=df_test_balanced,
        image_dir=image_dir,
        label_col=label_col,
        batch_size=batch_size,
        filename_col=filename_col,
        is_training=False,
        seed=random_state,
    )

    return train_ds, val_ds, test_ds, audit_df, audit_report
