import random
import pickle
import random
import logging
from typing import Dict, Union
from pathlib import Path
import yaml
from box.exceptions import BoxValueError
from box import ConfigBox
from ensure import ensure_annotations

import os 
from src.logDet import logger


@ensure_annotations
def read_yaml(path_to_yaml : Path) -> ConfigBox:
    """
    This function reads a yaml file and returns a ConfigBox object. 

    Parameters
    ----------
    path_to_yaml : Path
        path to yaml file.

    Raises:
        ValueError: if yaml file is empty.
        e: if any other error occurs.
    
    Returns:
    -------
        ConfigBox : ConfigBox object.
    """
    try: 
        with open(path_to_yaml, "r") as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Yaml file : {os.path.normpath(path_to_yaml)} loaded successfully.")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty.")
    except Exception as e:
        raise e  


@ensure_annotations
def decision(probability : float) -> bool:
    """
    This function returns True with a probability of probability.
    
    Parameters
    ----------
    probability : float
        probability of True.    
    
    Returns
    -------
        bool : True or False.
    """
    return random.random() < probability


@ensure_annotations
def dump_pickle(obj : Dict, file_path : Path):
    """
    This function dumps an object to a pickle file.
    
    Parameters
    ----------
    obj : Union[Dict, any]
        Object to be dumped.
        
    file_path : Path
        Path to the pickle file.
        
    Returns
    -------
    None
    """
    logger.info(f"Dumping to {file_path}")
    with open(file_path, "wb") as fw:
        pickle.dump(obj, fw)
        

@ensure_annotations
def read_pickle(file_path: Path):
    """
    This function reads a pickle file and returns the object.
    
    Parameters
    ----------
    file_path : Path
        Path to the pickle file.    
        
    Returns
    -------
    Any : Object from the pickle file.
    """
    logger.info(f"Loading from {file_path}")
    with open(file_path, "rb") as fr:
        return pickle.load(fr)


