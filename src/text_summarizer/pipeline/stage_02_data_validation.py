from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.components.data_validation import DataValidation

try:
    config_manager = ConfigurationManager()
    data_validation_config = config_manager.get_data_validation_config()

    data_validation = DataValidation(config=data_validation_config)
    data_validation.validate_all_files_exist()
except Exception as e:
    raise e


from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.components.data_validation import DataValidation

class DataValidationTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.data_validation_config = self.config_manager.get_data_validation_config()

    def initiate_data_validation(self):
        """
        This method runs the data validation process.
        """
        data_validation = DataValidation(config=self.data_validation_config)
        data_validation.validate_all_files_exist()
