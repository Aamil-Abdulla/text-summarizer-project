from text_summarizer.entity import DataValidationConfig
from text_summarizer.logging import logger
import os


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            logger.info("Starting data validation for all required files.")

            # Use the correct ingestion directory from CONFIG
            ingestion_dir = "data_ingestion"

            all_files_present = True
            missing_files = []

            # Check all required files
            for file_name in self.config.ALL_REQUIRED_FILES:
                file_path = os.path.join(ingestion_dir, file_name)

                if not os.path.exists(file_path):
                    logger.error(f"Required file missing: {file_path}")
                    all_files_present = False
                    missing_files.append(file_name)
                else:
                    logger.info(f"Required file found: {file_path}")

            # Ensure status file directory exists
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)

            # Write validation status
            with open(self.config.STATUS_FILE, "w") as f:
                if all_files_present:
                    f.write("Validation Status: Success — All required files exist.\n")
                else:
                    f.write("Validation Status: Failed — Missing files:\n")
                    for file in missing_files:
                        f.write(f"- {file}\n")

            logger.info(f"Validation results written to {self.config.STATUS_FILE}")

            return all_files_present

        except Exception as e:
            logger.exception(f"An error occurred during data validation: {e}")
            raise e
