import os
import urllib.request as request
from zipfile import ZipFile
from text_summarizer.utils.common import get_size
from text_summarizer.logging import logger
from text_summarizer.entity import DataIngestionConfig


class DataIngestion:
        def __init__(self, config: DataIngestionConfig):
            self.config = config
        
        def download_data(self) -> str:
            logger.info("Starting data download...")
            filename = os.path.basename(self.config.source_URL)
            download_path = self.config.root_dir / filename

            logger.info(f"Downloading data from {self.config.source_URL} to {download_path}")
            request.urlretrieve(self.config.source_URL, download_path)
            logger.info(f"Data downloaded successfully! File size: {get_size(download_path)}")

            return str(download_path)
        
        def extract_zip_file(self, zip_file_path: str):
            logger.info(f"Extracting zip file: {zip_file_path} to directory: {self.config.unzip_dir}")
            with ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(self.config.unzip_dir)
            logger.info("Extraction completed successfully.")