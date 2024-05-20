import os
import sys
import logging
from datetime import datetime

logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y')}.log"
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging_str = (
    "[ %(asctime)s ] PID: %(process)d TID: %(thread)d Module: %(module)s "
    "Function: %(funcName)s Line: %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

logging.basicConfig(
    format=logging_str,
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_FILE_PATH, mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("logDet")
