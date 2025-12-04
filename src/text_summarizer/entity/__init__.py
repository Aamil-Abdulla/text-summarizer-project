from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path 
    STATUS_FILE : str 
    ALL_REQUIRED_FILES  : list 




@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    train_path: Path
    validation_path: Path
    test_path: Path
    tokenizer_name: str


from dataclasses import dataclass
from pathlib import Path

@dataclass
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str


@dataclass
class ModelEvaluationConfig:
    root_dir: Path
    model_path: Path
    tokenizer_path: Path
    data_path: Path
    metric_file_name: Path
    column_text: str = "article"
    column_summary: str = "highlights"
