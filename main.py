from text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from text_summarizer.logging import logger
from text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.initiate_data_ingestion()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f"Error occurred in stage {STAGE_NAME}: {e}")
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    from text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline

    data_validation_pipeline = DataValidationTrainingPipeline()
    data_validation_pipeline.initiate_data_validation()
    logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.error(f"Error occurred in stage {STAGE_NAME}: {e}")
    raise e