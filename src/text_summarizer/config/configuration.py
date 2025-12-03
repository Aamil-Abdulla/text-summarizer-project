from text_summarizer.constants import *
from text_summarizer.utils.common import read_yaml, create_directories
from text_summarizer.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)
from pathlib import Path


class ConfigurationManager:
    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    # -------------------- DATA INGESTION --------------------
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([Path(config.root_dir)])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir)
        )

    # -------------------- DATA VALIDATION --------------------
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=Path(config.status_file),
            ALL_REQUIRED_FILES=config.ALL_REQUIRED_FILES
        )

    # -------------------- DATA TRANSFORMATION --------------------
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            train_path=Path(config.train_path),
            validation_path=Path(config.validation_path),
            test_path=Path(config.test_path),
            tokenizer_name=config.tokenizer_name
        )

    # -------------------- MODEL TRAINER --------------------
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer

        create_directories([Path(config.root_dir)])

        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt
        )
