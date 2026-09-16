from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.constants import DATA_DIR
from src.utils import load_yaml


class FrozenBaseModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )


class ProjectConfig(FrozenBaseModel):
    experiment_name: str
    tracking_uri: str
    artifact_dir: Path
    seed: int = Field(ge=0)


class DataConfig(FrozenBaseModel):
    variant: str
    image_height: int = Field(gt=0)
    image_width: int = Field(gt=0)
    channels: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    val_split: float = Field(ge=0.0, lt=1.0)
    test_split: float = Field(ge=0.0, lt=1.0)

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
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig

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
