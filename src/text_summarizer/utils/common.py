import os
import yaml
from box.exceptions import BoxValueError
from box import ConfigBox
from text_summarizer.logging import logger
from pathlib import Path
from typing import Any
from ensure import ensure_annotations
from typing import List
from pathlib import Path
@ensure_annotations
def read_yaml(path_to_yaml:(str,Path)) -> ConfigBox:
    """Reads a YAML file and returns its contents as a ConfigBox object.

    Args:
        path_to_yaml (Path): The path to the YAML file. 
    Returns:
        ConfigBox: The contents of the YAML file as a ConfigBox object. 
    """
    try:
        with open(path_to_yaml, 'r') as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError as box_error:
        logger.error(f"Error converting YAML content to ConfigBox: {box_error}")
        raise box_error
    except Exception as e:
        logger.error(f"Error reading YAML file at {path_to_yaml}: {e}")
        raise e 
    

def create_directories(path_to_directories: list) -> None:
    """
    Create directories safely.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory created at: {path}")
    except Exception as e:
        logger.error(f"Error creating directories {path_to_directories}: {e}")
        raise

@ensure_annotations
def get_size(path: Path) -> str:
    """Returns the size of the file at the given path in KB.

    Args:
        path (Path): The path to the file.
    Returns:
        str: The size of the file in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024, 2)
    logger.info(f"Size of file at {path}: {size_in_kb} KB")
    return f"{size_in_kb} KB"