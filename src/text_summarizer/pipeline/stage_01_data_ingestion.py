from text_summarizer.components.data_ingestion import DataIngestion
from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.logging import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        self.config = ConfigurationManager()
        self.data_ingestion_config = self.config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=self.data_ingestion_config)

    def initiate_data_ingestion(self):
        try:
            zip_file_path = self.data_ingestion.download_data()
            self.data_ingestion.extract_zip_file(zip_file_path=zip_file_path)
        except Exception as e:
            logger.error(f"Error in data ingestion pipeline: {e}")
            raise e