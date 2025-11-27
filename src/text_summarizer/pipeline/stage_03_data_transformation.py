from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.components.data_transformation import DataTransformation


class DataTransformationTrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()
        self.data_transformation_config = self.config_manager.get_data_transformation_config()

    def initiate_data_transformation(self):
        """
        This method runs the data validation process.
        """
        data_transformation = DataTransformation(config=self.data_transformation_config)
        data_transformation.convert()
        
        
