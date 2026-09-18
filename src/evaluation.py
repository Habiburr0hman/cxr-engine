# src/evaluation.py
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from src.dataset import parse_preprocessed_image
from src.plots import (
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    render_report_image,
)


def evaluate_cohort(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    split_name: str,
    class_names: list[str],
    output_dir: Path,
    default_threshold: float = 0.50,
) -> tuple[dict[str, Any], list[Path]]:
    """Evaluates a dataset split, generates plots, text reports, and scalar metrics."""
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    generated_artifacts: list[Path] = []

    # 1. Extract ground truth and probability predictions
    y_true_list = []
    y_prob_list = []
    for images, labels in dataset:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(labels.numpy().flatten())
        y_prob_list.extend(preds.flatten())

    y_true = np.array(y_true_list, dtype=np.int32)
    y_prob = np.array(y_prob_list, dtype=np.float32)

    # 2. Compute ROC Curve & Youden's J
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    roc_auc_val = float(auc(fpr, tpr))

    # Youden's Index: J = Sensitivity + Specificity - 1 = TPR - FPR
    youden_j = tpr - fpr
    best_j_idx = int(np.argmax(youden_j))
    clean_roc_thresholds = np.where(np.isinf(roc_thresholds), 1.0, roc_thresholds)
    optimal_roc_threshold = float(clean_roc_thresholds[best_j_idx])

    # 3. Compute PR Curve & Optimal F1 Threshold
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc_val = float(auc(recalls, precisions))
    baseline_prev = float(np.mean(y_true))

    # Optimal F1 calculation across PR thresholds
    denom = precisions[:-1] + recalls[:-1]
    f1_scores = np.divide(
        2 * (precisions[:-1] * recalls[:-1]),
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    best_f1_idx = int(np.argmax(f1_scores)) if len(f1_scores) > 0 else 0
    optimal_pr_threshold = (
        float(pr_thresholds[best_f1_idx]) if len(pr_thresholds) > 0 else 0.50
    )

    # 4. Compute Confusion Matrix & Classification Report at Default (0.50)
    y_pred_default = (y_prob >= default_threshold).astype(int)
    cm_default = confusion_matrix(y_true, y_pred_default)

    # Save Classification Report as text and JSON
    report_text = classification_report(
        y_true, y_pred_default, target_names=class_names, digits=2
    )
    txt_path = split_dir / f"classification_report_{split_name}.txt"
    txt_path.write_text(
        report_text + f"\nConfusion Matrix:\n{cm_default}\n", encoding="utf-8"
    )
    generated_artifacts.append(txt_path)

    # Preserve full class-specific breakdown into JSON artifact
    report_dict = classification_report(
        y_true, y_pred_default, target_names=class_names, output_dict=True
    )
    json_path = split_dir / f"classification_report_{split_name}.json"
    json_path.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    generated_artifacts.append(json_path)

    # 5. Render Plots
    cm_plot_path = plot_confusion_matrix(
        cm=cm_default,
        classes=class_names,
        output_path=split_dir / f"confusion_matrix_{split_name}.png",
        title=f"Confusion Matrix ({split_name.capitalize()})",
        subtitle=f"Threshold = {default_threshold:.2f}",
    )
    generated_artifacts.append(cm_plot_path)

    roc_plot_path = plot_roc_curve(
        fpr=fpr,
        tpr=tpr,
        roc_auc=roc_auc_val,
        thresholds=roc_thresholds,
        output_path=split_dir / f"roc_curve_{split_name}.png",
        title=f"ROC Curve ({split_name.capitalize()})",
        default_threshold=default_threshold,
        optimal_threshold=optimal_roc_threshold,
    )
    generated_artifacts.append(roc_plot_path)

    pr_plot_path = plot_pr_curve(
        recalls=recalls,
        precisions=precisions,
        pr_auc=pr_auc_val,
        thresholds=pr_thresholds,
        baseline_prevalence=baseline_prev,
        output_path=split_dir / f"pr_curve_{split_name}.png",
        title=f"Precision-Recall ({split_name.capitalize()})",
        default_threshold=default_threshold,
        optimal_threshold=optimal_pr_threshold,
    )
    generated_artifacts.append(pr_plot_path)

    report_img_path = render_report_image(
        report_text=report_text,
        output_path=split_dir / f"classification_report_{split_name}.png",
        title=f"Classification Report ({split_name.capitalize()})",
    )
    generated_artifacts.append(report_img_path)

    # 6. Structured scalar metrics for MLflow logging
    scalar_metrics = {
        f"{split_name}_accuracy": float(report_dict["accuracy"]),
        f"{split_name}_precision": float(report_dict["macro avg"]["precision"]),
        f"{split_name}_recall": float(report_dict["macro avg"]["recall"]),
        f"{split_name}_f1": float(report_dict["macro avg"]["f1-score"]),
        # f"{split_name}_weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        f"{split_name}_roc_auc": roc_auc_val,
        f"{split_name}_pr_auc": pr_auc_val,
        # f"{split_name}_optimal_roc_threshold_youden": optimal_roc_threshold,
        # f"{split_name}_optimal_pr_threshold_f1": optimal_pr_threshold,
    }

    # Add class-specific precision and recall
    # for cls in class_names:
    #     if cls in report_dict:
    #         scalar_metrics[f"{split_name}_{cls}_precision"] = float(
    #             report_dict[cls]["precision"]
    #         )
    #         scalar_metrics[f"{split_name}_{cls}_recall"] = float(
    #             report_dict[cls]["recall"]
    #         )
    #         scalar_metrics[f"{split_name}_{cls}_f1"] = float(
    #             report_dict[cls]["f1-score"]
    #         )

    return scalar_metrics, generated_artifacts


def enrich_audit_manifest(
    model: tf.keras.Model,
    audit_df: pd.DataFrame,
    image_dir: Path | str,
    batch_size: int,
    filename_col: str = "filename",
    label_col: str = "label",
) -> pd.DataFrame:
    """Enriches audit_df with model inference probabilities, thresholded predictions,

    and clinical error types using the canonical preprocessing pipeline.
    """
    img_dir_path = Path(image_dir)
    inference_df = audit_df.copy()

    # 1. Resolve filepaths on disk (matching .png stem variants)
    resolved_paths = []
    for fn in inference_df[filename_col]:
        stem = Path(fn).stem
        png_path = img_dir_path / f"{stem}.png"
        resolved_paths.append(
            str(png_path if png_path.is_file() else img_dir_path / fn)
        )

    labels = inference_df[label_col].to_numpy(dtype=np.float32)

    # 2. Deterministic pipeline reusing canonical parse_preprocessed_image
    predict_ds = (
        tf.data.Dataset.from_tensor_slices((resolved_paths, labels))
        .map(parse_preprocessed_image, num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda img, _: img)  # Strip labels for model.predict
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 3. Model Inference
    raw_preds = model.predict(predict_ds, verbose=1).flatten()
    inference_df["pred_prob"] = raw_preds

    # 4. Predictions
    inference_df["pred_label"] = (raw_preds >= 0.50).astype(int)

    # 5. Clinical Error Categorization at Optimal Threshold
    y_true = inference_df[label_col].to_numpy()
    y_pred = inference_df["pred_label"].to_numpy()
    inference_df["is_correct"] = inference_df[label_col] == inference_df["pred_label"]

    error_types = []
    for true, pred in zip(y_true, y_pred, strict=False):
        if true == 1 and pred == 1:
            error_types.append("TP")
        elif true == 0 and pred == 0:
            error_types.append("TN")
        elif true == 0 and pred == 1:
            error_types.append("FP")
        else:
            error_types.append("FN")

    inference_df["error_type"] = error_types
    return inference_df


def benchmark_inference_latency(
    model: tf.keras.Model,
    sample_batch: tf.Tensor,
    warmup_runs: int = 2,
    benchmark_runs: int = 5,
) -> float:
    """Calculates mean per-image inference latency in milliseconds."""
    # Warm up TensorFlow execution graph
    for _ in range(warmup_runs):
        _ = model.predict(sample_batch[:2], verbose=0)

    durations: list[float] = []
    batch_size = int(tf.shape(sample_batch)[0])

    for _ in range(benchmark_runs):
        t0 = time.perf_counter()
        _ = model.predict(sample_batch, verbose=0)
        durations.append((time.perf_counter() - t0) / batch_size)

    mean_latency_ms = float(np.mean(durations) * 1000.0)
    return mean_latency_ms
