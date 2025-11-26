from text_summarizer.entity import DataValidationConfig
from text_summarizer.logging import logger

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            logger.info("Starting data validation for all required files.")

            all_files_present = True
            for file_name in self.config.ALL_REQUIRED_FILES:
                file_path = self.config.root_dir / file_name
                if not file_path.exists():
                    logger.error(f"Required file missing: {file_path}")
                    all_files_present = False
                else:
                    logger.info(f"Required file found: {file_path}")

            if all_files_present:
                logger.info("All required files are present.")
            else:
                logger.warning("Some required files are missing.")

            return all_files_present

        except Exception as e:
            logger.exception(f"An error occurred during data validation: {e}")
            raise e
        
        