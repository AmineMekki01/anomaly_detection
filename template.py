"""
Script name : template.py
Author : Amine MEKKI
Date : 05/20/2024 mm/dd/yyyy
Description : This script is a template for python scripts architecture.
"""


import os
from pathlib import Path
import logging


project_name = "logDet"

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common_functions.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "setup.py",
    "research/testing.ipynb",
    "Makefile",
    "pyproject.toml"
    
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)
    
    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True),
        logging.info(f"Creating directory : {file_dir} for the file {file_name}.")
    
    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Creating empty file : {file_name}.")
    else:
        logging.info(f"File {file_name} already exists.")
        
        