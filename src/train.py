# src/train.py
import src.reproducibility  # noqa: F401

# isort: split

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import keras
import mlflow
import mlflow.tensorflow
import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models.signature import infer_signature

from src.config import AppConfig
from src.dataset import get_train_val_test_datasets_with_audit
from src.evaluation import (
    benchmark_inference_latency,
    enrich_audit_manifest,
    evaluate_cohort,
)
from src.model import DepthwiseSeparableResBlock, create_model, get_model_complexity
from src.plots import (
    plot_class_dist_bar,
    plot_class_dist_pie,
    plot_dual_training_log,
    plot_learning_rate,
    plot_split_dist_pie,
    plot_training_log,
)
from src.registry import promote_champion_model
from src.tracking import setup_mlflow
from src.utils import get_dvc_hash, get_project_relative_path


def run_training():
    config = AppConfig.load()
    setup_mlflow(config)

    catalog_path = config.data.catalog_path
    img_size = (config.data.image_height, config.data.image_width)
    channels = config.data.channels
    batch_size = config.data.batch_size
    val_split = config.data.val_split
    test_split = config.data.test_split
    data_variant = config.data.variant
    img_dir = config.data.processed_dir
    model_variant = config.model.variant

    variant_dvc_hash = get_dvc_hash(config.data.dvc_pointer_path) or "untracked"
    raw_dvc_hash = get_dvc_hash(config.data.raw_dvc_pointer_path) or "untracked"

    rel_img_dir = get_project_relative_path(img_dir)

    local_timestamp = datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M")
    run_name = f"{model_variant}_{data_variant}_{local_timestamp}"
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Folder Scaffolding
        run_dir = Path("artifacts/runs") / run_id
        checkpoints_dir = run_dir / "checkpoints"
        manifests_dir = run_dir / "manifests"
        vis_train_dir = run_dir / "visualization" / "training"
        vis_eval_dir = run_dir / "visualization" / "evaluation"
        tb_log_dir = run_dir / "graph_inspection"

        for directory in [
            checkpoints_dir,
            manifests_dir,
            vis_train_dir,
            vis_eval_dir,
            tb_log_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # 1. Track dataset and dvc binaries
        mlflow.set_tags(
            {
                "data.source_name": config.data.source_name,
                "data.variant": data_variant,
                "data.processed_dvc_hash": variant_dvc_hash,
                "data.raw_dvc_hash": raw_dvc_hash,
                "data.processed_folder": rel_img_dir,
            }
        )

        # 2. Register dataset catalog
        rel_catalog_path = get_project_relative_path(catalog_path)
        catalog_df = pd.read_csv(catalog_path)
        source_uri = (
            f"{config.mlflow.dagshub.web_repo_url}/src/main/{rel_catalog_path}"
            if config.project.tracking_mode == "dagshub"
            else rel_catalog_path
        )
        print(f"\n[MLFLOW SOURCE CHECK] -> {source_uri}\n")
        mlflow_dataset = from_pandas(
            df=catalog_df,
            source=source_uri,
            name=f"{config.data.source_name}_{data_variant}_v1",
            targets=config.data.label_col,
        )
        mlflow.log_input(mlflow_dataset, context="training_and_evaluation")

        # 3. Split dataset
        train_ds, val_ds, test_ds, audit_df, audit_report = (
            get_train_val_test_datasets_with_audit(
                image_dir=img_dir,
                catalog_path=catalog_path,
                val_size=val_split,
                test_size=test_split,
                batch_size=batch_size,
                label_col="label",
                patient_col="patient_id",
                filename_col="filename",
            )
        )

        # 4a. Summary JSON report
        summary_json_path = manifests_dir / "audit_summary.json"
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(audit_report, f, indent=2)

        mlflow.log_artifact(str(summary_json_path), artifact_path="reproducibility")
        mlflow.log_dict(audit_report, "reproducibility/audit_summary.json")

        # 4d. Log metadata parameters directly from audit_report
        mlflow.log_params(
            {
                **{
                    f"data_{k}": v
                    for k, v in config.data.model_dump(mode="json").items()
                },
                **{
                    f"model_{k}": v
                    for k, v in config.model.model_dump(mode="json").items()
                },
                **{
                    f"train_{k}": v
                    for k, v in config.training.model_dump(mode="json").items()
                },
                "sampling_strategy": "patient_aware_stratified_undersampling",
                "patient_leakage_status": audit_report["leakage_verification"],
                # Image counts (raw vs. balanced)
                "images_train_raw": audit_report["image_counts"]["train"]["raw"],
                "images_train_balanced": audit_report["image_counts"]["train"][
                    "balanced"
                ],
                "images_train_retention_rate": audit_report["image_counts"]["train"][
                    "retention_rate"
                ],
                "images_val_raw": audit_report["image_counts"]["val"]["raw"],
                "images_val_balanced": audit_report["image_counts"]["val"]["balanced"],
                "images_val_retention_rate": audit_report["image_counts"]["val"][
                    "retention_rate"
                ],
                "images_test_raw": audit_report["image_counts"]["test"]["raw"],
                "images_test_balanced": audit_report["image_counts"]["test"][
                    "balanced"
                ],
                "images_test_retention_rate": audit_report["image_counts"]["test"][
                    "retention_rate"
                ],
                "images_total_retained": audit_report["image_counts"]["total_retained"],
                "images_total_dropped": audit_report["image_counts"]["total_dropped"],
                # Patient counts (allocated vs. retained after balancing)
                "patients_train_allocated": audit_report["patient_counts"][
                    "allocated_at_split"
                ]["train"],
                "patients_train_retained": audit_report["patient_counts"][
                    "retained_after_balancing"
                ]["train"],
                "patients_val_allocated": audit_report["patient_counts"][
                    "allocated_at_split"
                ]["val"],
                "patients_val_retained": audit_report["patient_counts"][
                    "retained_after_balancing"
                ]["val"],
                "patients_test_allocated": audit_report["patient_counts"][
                    "allocated_at_split"
                ]["test"],
                "patients_test_retained": audit_report["patient_counts"][
                    "retained_after_balancing"
                ]["test"],
                "patients_total_catalog": audit_report["patient_counts"][
                    "total_catalog_patients"
                ],
                # Audit integrity paths
                "catalog_path": rel_catalog_path,
            }
        )

        # 4e. Render Data Distribution Figures (Retained vs All)
        # Prepare label mapping
        audit_display = audit_df.copy()
        audit_display["class_label"] = (
            audit_display["label"]
            .map({0: "Normal", 1: "Pneumonia"})
            .fillna(audit_display["label"])
        )

        cohort_order = ["train", "val", "test"]
        distribution_targets = [
            ("retained", audit_display[audit_display["status"] == "retained"]),
            ("all", audit_display[audit_display["cohort"].isin(cohort_order)]),
        ]

        for subset_key, subset_df in distribution_targets:
            subset_title = (
                "Balanced Retained"
                if subset_key == "retained"
                else "All Allocated (Raw)"
            )
            prefix = f"{subset_key}_"

            # 1. Split Allocation Ratio Pie
            split_counts = subset_df["cohort"].value_counts().reindex(cohort_order)
            pie_path = plot_split_dist_pie(
                split_counts=split_counts,
                output_path=vis_train_dir / f"{prefix}split_dist_pie.png",
                title=f"Split Allocation ({subset_title})",
            )
            mlflow.log_artifact(str(pie_path), artifact_path="visualization/training")

            # 2. Split x Class Grouped Bar Chart
            split_class_cross = pd.crosstab(
                subset_df["cohort"], subset_df["class_label"]
            ).reindex(cohort_order)

            bar_path = plot_class_dist_bar(
                split_class_df=split_class_cross,
                output_path=vis_train_dir / f"{prefix}class_dist_bar.png",
                title=f"Class Balance Across Splits ({subset_title})",
            )
            mlflow.log_artifact(str(bar_path), artifact_path="visualization/training")

            # 3. Triple Class Pie Chart
            class_series_dict = {
                cohort: subset_df[subset_df["cohort"] == cohort][
                    "class_label"
                ].value_counts()
                for cohort in cohort_order
            }
            triple_pie_path = plot_class_dist_pie(
                df_dict=class_series_dict,
                output_path=vis_train_dir / f"{prefix}class_dist_pie.png",
                suptitle=f"Class Balance per Split ({subset_title})",
            )
            mlflow.log_artifact(
                str(triple_pie_path), artifact_path="visualization/training"
            )

        input_shape = (img_size[0], img_size[1], channels)
        model = create_model(config.model, input_shape=input_shape)

        model_complexity = get_model_complexity(model)
        mlflow.log_params({f"arch.{k}": v for k, v in model_complexity.items()})

        loss_fn = tf.keras.losses.BinaryCrossentropy()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=config.training.learning_rate
            ),
            loss=loss_fn,
            metrics=[
                "accuracy",
                keras.metrics.AUC(name="auc"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.F1Score(name="f1", threshold=0.5, average="macro"),
            ],
        )

        # 7. Model Training
        # 7a. Training & Callback setup
        best_model_path = checkpoints_dir / "model_best.keras"
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(best_model_path),
                monitor="val_loss",
                mode="min",
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.1,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir=str(tb_log_dir),
                write_graph=True,
                histogram_freq=1,
            ),
        ]

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.training.epochs,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # have been shuffled on tf dataset creation
        )

        # 7b. Checkpoint Extraction & Epoch Provenance
        val_losses = history.history["val_loss"]
        best_epoch_idx = int(np.argmin(val_losses))
        best_epoch = best_epoch_idx + 1
        last_epoch = len(val_losses)

        mlflow.log_params(
            {
                "epoch.best": best_epoch,
                "epoch.last": last_epoch,
                "epoch.best_val_loss": float(val_losses[best_epoch_idx]),
                "epoch.last_val_loss": float(val_losses[-1]),
            }
        )

        # 7c. Save Final Epoch State (Diagnostic Artifact)
        last_model_path = checkpoints_dir / f"model_last_e{last_epoch}.keras"
        model.save(str(last_model_path))
        mlflow.log_artifact(str(last_model_path), artifact_path="checkpoints")

        # 7d. Restore Best Weights for Evaluation & Registration
        print(
            f"\n[CHECKPOINT] Reloading complete Best Model (Epoch {best_epoch}, "
            f"val_loss: {val_losses[best_epoch_idx]:.4f})..."
        )

        # Replaces in-memory model object with the restored best state
        custom_objects = {"DepthwiseSeparableResBlock": DepthwiseSeparableResBlock}
        model = tf.keras.models.load_model(
            str(best_model_path),
            custom_objects=custom_objects,
        )
        mlflow.log_artifact(str(best_model_path), artifact_path="checkpoints")

        # 8. Training History: JSON Export & Selective Curve Plotting
        serializable_history = {
            k: [float(epoch_val) for epoch_val in v] for k, v in history.history.items()
        }
        history_json_path = vis_train_dir / "training_history.json"
        with open(history_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=2)

        mlflow.log_artifact(
            str(history_json_path), artifact_path="visualization/training"
        )

        # 8b. Primary Dual Plot (Binary Cross-Entropy Loss vs. Macro F1)
        history_plot_path = plot_dual_training_log(
            history_dict=history.history,
            metric1_name="loss",
            metric1_label="Binary Cross Entropy Loss",
            metric2_name="f1",
            metric2_label="F1-Score",
            output_path=vis_train_dir / "training_history_log.png",
        )
        mlflow.log_artifact(
            str(history_plot_path), artifact_path="visualization/training"
        )

        # 8c. Selective Single Metric Plots
        # Explicitly specify only the metrics you want rendered into plots
        selected_metrics = ["loss", "accuracy", "precision", "recall", "f1", "auc"]

        metric_display_names = {
            "loss": "Binary Cross-Entropy Loss",
            "accuracy": "Accuracy",
            "auc": "AUC",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1-Score",
        }

        for metric in selected_metrics:
            if metric not in history.history:
                continue

            train_arr = history.history[metric]
            val_arr = history.history.get(f"val_{metric}")
            label = metric_display_names.get(metric, metric.replace("_", " ").title())

            metric_plot_path = plot_training_log(
                train_arr=train_arr,
                val_arr=val_arr,
                title=f"Training Log: {label}",
                metric_label=label,
                output_path=vis_train_dir / f"{metric}_history.png",
            )
            mlflow.log_artifact(
                str(metric_plot_path), artifact_path="visualization/training"
            )

        # 8d. Learning Rate Schedule Plot
        for lr_key in ("lr", "learning_rate"):
            if lr_key in history.history:
                lr_plot_path = plot_learning_rate(
                    lr_arr=history.history[lr_key],
                    output_path=vis_train_dir / "learning_rate_history.png",
                    title="Learning Rate History",
                )
                mlflow.log_artifact(
                    str(lr_plot_path), artifact_path="visualization/training"
                )
                break

        # 8e. Log Step-Level Epoch Metrics to MLflow (Preserves all metrics in UI)
        for epoch in range(len(history.history["loss"])):
            metrics_step = {
                metric: float(np.mean(values[epoch]))
                for metric, values in history.history.items()
            }
            mlflow.log_metrics(metrics_step, step=epoch)

        # 9. Systematic Multi-Cohort Evaluation (Train, Val, Test)
        # 9a. Evaluation
        class_labels = ["Normal", "Pneumonia"]
        splits_to_evaluate = [
            ("train", train_ds),
            ("val", val_ds),
            ("test", test_ds),
        ]

        all_eval_metrics: dict[str, float] = {}
        for split_name, ds_cohort in splits_to_evaluate:
            print(f"\nEvaluating cohort: {split_name.upper()}...")
            metrics, artifacts = evaluate_cohort(
                model=model,
                dataset=ds_cohort,
                split_name=split_name,
                class_names=class_labels,
                output_dir=vis_eval_dir,
                default_threshold=0.50,
            )
            all_eval_metrics.update(metrics)
            mlflow.log_metrics(metrics)

            for art in artifacts:
                mlflow.log_artifact(
                    str(art),
                    artifact_path=f"visualization/evaluation/{split_name}",
                )

        # 9b. Benchmark Latency
        test_batch_images = next(iter(test_ds))[0]
        latency_ms = benchmark_inference_latency(
            model=model, sample_batch=test_batch_images
        )
        mlflow.log_metric("test_latency_ms_per_image", latency_ms)
        print(f"[BENCHMARK] Inference Latency: {latency_ms:.2f} ms/image")

        # 9c. Whole-Catalog Inference & Manifest Enrichment
        enriched_manifest_df = enrich_audit_manifest(
            model=model,
            audit_df=audit_df,
            image_dir=img_dir,
            batch_size=batch_size,
            filename_col="filename",
            label_col="label",
        )

        # Serialize finalized manifest
        manifest_csv_path = manifests_dir / "audit_manifest.csv"
        enriched_manifest_df.to_csv(manifest_csv_path, index=False)
        sha256_hash = hashlib.sha256(manifest_csv_path.read_bytes()).hexdigest()

        # Log as single artifact, queryable table, and mutable tag
        mlflow.log_artifact(str(manifest_csv_path), artifact_path="reproducibility")
        mlflow.log_table(
            data=enriched_manifest_df,
            artifact_file="reproducibility/audit_manifest_table.json",
        )
        mlflow.set_tag("reproducibility.manifest_sha256", sha256_hash)

        # Audit dropped cohort performance
        dropped_mask = enriched_manifest_df["status"] == "dropped"
        if dropped_mask.any():
            dropped_df = enriched_manifest_df[dropped_mask]
            dropped_acc = float(
                (dropped_df["label"] == dropped_df["pred_label"]).mean()
            )
            mlflow.log_metric("dropped_cohort_accuracy", dropped_acc)
            print(
                f"[AUDIT] Dropped cohort accuracy ({len(dropped_df)} samples): {dropped_acc:.4%}"
            )

        # 10. Model Registration & Promotion
        sample_batch = next(iter(test_ds))
        sample_images = sample_batch[0][:2].numpy()
        sample_predictions = model.predict(sample_images, verbose=0)
        signature = infer_signature(sample_images, sample_predictions)

        registered_name = f"cxr_{model_variant}_{data_variant}"
        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=registered_name,
        )

        promote_champion_model(
            model_name=registered_name,
            run_id=run.info.run_id,
            metric_name="test_roc_auc",
            higher_is_better=True,
        )

        print(f"\n[Training Complete] MLflow Run ID: {run.info.run_id}")
        print(f"Registered Model: {registered_name}")
        print(f"Enriched Manifest SHA-256: {sha256_hash}")


if __name__ == "__main__":
    run_training()
