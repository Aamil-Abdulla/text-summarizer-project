from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.components.data_transformation import DataTransformation

try:
    config_manager = ConfigurationManager()
    data_transformation_config = config_manager.get_data_transformation_config()
    data_transformation = DataTransformation(config=data_transformation_config)
    data_transformation.convert()
except Exception as e:
    raise e