# src/train.py
import src.reproducibility  # noqa: F401

# isort: split

import hashlib
import json
from pathlib import Path

import keras
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
from mlflow.models.signature import infer_signature

from src.config import AppConfig
from src.dataset import get_train_val_test_datasets_with_audit
from src.model import build_separable_resnet
from src.tracking import setup_mlflow


def run_training():
    config = AppConfig.load()
    setup_mlflow(config)

    catalog_path = "data/metadata/catalog.csv"
    img_size = (config.data.image_height, config.data.image_width)
    channels = config.data.channels
    batch_size = config.data.batch_size
    val_split = config.data.val_split
    test_split = config.data.test_split
    data_variant = config.data.variant
    img_dir = config.data.processed_dir

    with mlflow.start_run(run_name="balanced_separable_resnet_run") as run:
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

        manifest_dir = Path("artifacts/manifests")
        manifest_dir.mkdir(parents=True, exist_ok=True)

        manifest_csv_path = manifest_dir / f"run_{run.info.run_id}_audit_manifest.csv"
        audit_df.to_csv(manifest_csv_path, index=False)
        sha256_hash = hashlib.sha256(manifest_csv_path.read_bytes()).hexdigest()

        # 4b. Aggregated summary JSON report
        summary_json_path = manifest_dir / f"run_{run.info.run_id}_audit_summary.json"
        with open(summary_json_path, "w") as f:
            json.dump(audit_report, f, indent=2)

        # 4c. Log to MLflow artifacts
        mlflow.log_artifact(str(manifest_csv_path), artifact_path="reproducibility")
        mlflow.log_artifact(str(summary_json_path), artifact_path="reproducibility")
        mlflow.log_dict(audit_report, "reproducibility/audit_summary.json")
        mlflow.log_table(
            data=audit_df, artifact_file="reproducibility/sampling_table.json"
        )

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
                # Audit integrity hashes and paths
                "manifest_sha256": sha256_hash,
                "catalog_path": str(catalog_path),
            }
        )

        input_shape = (img_size[0], img_size[1], channels)
        model = build_separable_resnet(input_shape=input_shape)

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

        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                mode="min",
                factor=0.1,
                patience=3,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir="artifacts/graph_inspection",
                write_graph=True,
                histogram_freq=1,
            ),
        ]

        # 7. Model Training
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.training.epochs,
            callbacks=callbacks,
            verbose=1,
            shuffle=False,  # have been shuffled on tf dataset creation
        )

        # 8. Log Epoch Metrics
        for epoch in range(len(history.history["loss"])):
            metrics_step = {
                metric: float(np.mean(values[epoch]))
                for metric, values in history.history.items()
            }
            mlflow.log_metrics(metrics_step, step=epoch)

        # 9. Test Evaluation
        print("\n" + "=" * 50)
        print("Evaluating Model on Raw Test Set...")
        print("=" * 50)
        test_eval = model.evaluate(test_ds, return_dict=True, verbose=1)
        test_metrics = {f"test_{k}": float(np.mean(v)) for k, v in test_eval.items()}
        mlflow.log_metrics(test_metrics)

        # 10. Model Artifact & Signature Registration
        sample_batch = next(iter(test_ds))
        sample_images = sample_batch[0][:2].numpy()
        sample_predictions = model.predict(sample_images, verbose=0)
        signature = infer_signature(sample_images, sample_predictions)

        mlflow.tensorflow.log_model(
            model=model,
            artifact_path="model",
            signature=signature,
            registered_model_name=f"separable_resnet_{data_variant}",
        )

        print(f"\n[Training Complete] MLflow Run ID: {run.info.run_id}")
        print(f"Manifest SHA-256: {sha256_hash}")


if __name__ == "__main__":
    run_training()
