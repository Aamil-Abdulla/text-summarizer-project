from text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from text_summarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from text_summarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from text_summarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from text_summarizer.logging import logger


if __name__ == "__main__":
    try:
        # STAGE_NAME = "Data Ingestion Stage"
        # logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        # data_ingestion = DataIngestionTrainingPipeline()
        # data_ingestion.initiate_data_ingestion()
        # logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # STAGE_NAME = "Data Validation Stage"
        # logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        # data_validation = DataValidationTrainingPipeline()
        # data_validation.initiate_data_validation()
        # logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # STAGE_NAME = "Data Transformation Stage"
        # logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        # data_transformation = DataTransformationTrainingPipeline()
        # data_transformation.initiate_data_transformation()
        # logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        # STAGE_NAME = "Model Trainer Stage"
        # logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        # model_trainer = ModelTrainerTrainingPipeline()
        # model_trainer.main()
        # logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")

        STAGE_NAME= "Model Evaluation Stage"
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        model_evaluation = ModelEvaluationTrainingPipeline()
        model_evaluation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
        logger.info(">>> All stages completed successfully <<<")

    except Exception as e:
        logger.exception(e)
        raise e

 
# STAGE_NAME = "Model Evaluation Stage"
# try:
#     logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
#     model_evaluation = ModelEvaluationTrainingPipeline()
#     model_evaluation.main()
#     logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e
