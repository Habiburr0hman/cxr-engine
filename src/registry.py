# src/registry.py
import logging
from pathlib import Path

import keras
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.model import DepthwiseSeparableResBlock

logger = logging.getLogger(__name__)


def promote_champion_model(
    model_name: str,
    run_id: str,
    metric_name: str = "test_roc_auc",
    higher_is_better: bool = True,
) -> None:
    """Evaluates the newly logged model version against the current '@champion'

    alias and promotes it if performance improved.
    """
    client = MlflowClient()

    # Find the version that was just registered for this run
    versions = client.search_model_versions(f"name='{model_name}'")
    run_versions = [v for v in versions if v.run_id == run_id]
    if not run_versions:
        logger.warning(
            "No registered model version found for run %s under '%s'.",
            run_id,
            model_name,
        )
        return

    # Sort to pick the latest version registered by this run
    new_version = max(run_versions, key=lambda x: int(x.version))
    v_num = new_version.version

    # Check for existing champion
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
    except MlflowException:
        champion = None

    if not champion:
        client.set_registered_model_alias(model_name, "champion", v_num)
        print(f"[REGISTRY] Model '{model_name}' v{v_num} set as initial @champion.")
        return

    # Compare scores
    candidate_run = client.get_run(run_id)
    champion_run = client.get_run(champion.run_id)

    cand_score = candidate_run.data.metrics.get(metric_name)
    champ_score = champion_run.data.metrics.get(metric_name)

    if cand_score is None:
        client.set_registered_model_alias(model_name, "challenger", v_num)
        return

    is_better = (
        (cand_score > champ_score) if higher_is_better else (cand_score < champ_score)
    )

    if champ_score is None or is_better:
        client.set_registered_model_alias(model_name, "champion", v_num)
        client.set_registered_model_alias(
            model_name, "previous_champion", champion.version
        )
        print(
            f"[REGISTRY] PROMOTED v{v_num} to @champion! ({metric_name}: {cand_score:.4f} vs previous {champ_score:.4f})"
        )
    else:
        client.set_registered_model_alias(model_name, "challenger", v_num)
        print(
            f"[REGISTRY] Retained v{v_num} as @challenger ({metric_name}: {cand_score:.4f} <= champion {champ_score:.4f})"
        )


def load_saved_model_pair(
    run_id: str,
    compile_for_training: bool = False,
    base_runs_dir: Path | str = "artifacts/runs",
) -> tuple[keras.Model, keras.Model, dict[str, int]]:
    """Loads twin best and last models directly from disk or DagsHub/MLflow

    conforming to Option A folder-scoped layout.

    Args:
        run_id: 32-character MLflow run identifier.
        compile_for_training: Whether to compile upon loading. Defaults to False
            for downstream inference, evaluation, and Grad-CAM inspection.
        base_runs_dir: Root directory where run-scoped artifacts are stored locally.

    Returns:
        tuple: (model_best, model_last, epoch_info_dict)
    """
    client = mlflow.MlflowClient()
    run_data = client.get_run(run_id).data

    best_epoch = int(run_data.params.get("epoch.best", 1))
    last_epoch = int(run_data.params.get("epoch.last", 1))
    epoch_info = {"best_epoch": best_epoch, "last_epoch": last_epoch}

    # 1. Resolve Option A run-scoped local paths
    run_dir = Path(base_runs_dir) / run_id
    checkpoints_dir = run_dir / "checkpoints"
    best_file = checkpoints_dir / "model_best.keras"
    last_file = checkpoints_dir / f"model_last_e{last_epoch}.keras"

    # 2. Download from MLflow/DagsHub if missing on the local machine
    if not best_file.is_file() or not last_file.is_file():
        print(
            f"[SYNC] Downloading checkpoints from MLflow run {run_id} into {checkpoints_dir}..."
        )
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        # Downloads 'checkpoints/' into run_dir, reconstructing run_dir/checkpoints/*
        mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="checkpoints",
            dst_path=str(run_dir),
        )

    # 3. Load self-contained .keras models
    custom_objects = {"DepthwiseSeparableResBlock": DepthwiseSeparableResBlock}
    model_best = keras.models.load_model(
        str(best_file),
        custom_objects=custom_objects,
        compile=compile_for_training,
    )
    model_last = keras.models.load_model(
        str(last_file),
        custom_objects=custom_objects,
        compile=compile_for_training,
    )
    model_best = keras.models.load_model(str(best_file), compile=compile_for_training)
    model_last = keras.models.load_model(str(last_file), compile=compile_for_training)

    return model_best, model_last, epoch_info
