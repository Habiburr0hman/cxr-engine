from pathlib import Path
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    model_validator,
)

from src.constants import DATA_DIR
from src.utils import load_yaml


class FrozenBaseModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError as err:
            raise KeyError(
                f"Key '{item}' not found in {self.__class__.__name__}"
            ) from err

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)

    def items(self):
        return self.model_dump().items()


class MLflowLocalConfig(FrozenBaseModel):
    tracking_uri: str = "sqlite:///mlflow.db"


class MLflowDagshubConfig(FrozenBaseModel):
    tracking_uri: str


class MLflowConfig(FrozenBaseModel):
    local: MLflowLocalConfig = Field(default_factory=MLflowLocalConfig)
    dagshub: MLflowDagshubConfig | None = None


class ProjectConfig(FrozenBaseModel):
    experiment_name: str
    tracking_mode: Literal["local", "dagshub"] = "local"
    artifact_dir: Path
    seed: int = Field(ge=0, default=20925010)


class DataConfig(FrozenBaseModel):
    catalog_path: Path = Path("data/metadata/catalog.csv")
    variant: str
    image_height: int = Field(gt=0)
    image_width: int = Field(gt=0)
    channels: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    val_split: float = Field(ge=0.0, lt=1.0)
    test_split: float = Field(ge=0.0, lt=1.0)
    label_col: str = "label"
    patient_col: str = "patient_id"

    @property
    def processed_dir(self) -> Path:
        return DATA_DIR / "processed" / f"{self.variant}"


class ModelConfig(FrozenBaseModel):
    variant: str = "separable_resnet"


class TrainingConfig(FrozenBaseModel):
    epochs: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)


class AppConfig(FrozenBaseModel):
    project: ProjectConfig
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

    @model_validator(mode="after")
    def validate_tracking_mode(self) -> "AppConfig":
        if self.project.tracking_mode == "dagshub" and self.mlflow.dagshub is None:
            raise ValueError(
                "tracking_mode is set to 'dagshub', but 'mlflow.dagshub' section is missing in configuration."
            )
        return self

    @property
    def active_tracking_uri(self) -> str:
        if self.project.tracking_mode == "dagshub":
            assert self.mlflow.dagshub is not None
            return self.mlflow.dagshub.tracking_uri
        return self.mlflow.local.tracking_uri

    @classmethod
    def load(cls, path: str | Path = Path("configs/config.yaml")) -> "AppConfig":
        config_path = Path(path)
        raw_data = load_yaml(config_path)

        try:
            return cls.model_validate(raw_data)
        except ValidationError as err:
            raise ValueError(
                f"Invalid configuration in '{config_path}':\n{err}"
            ) from err
