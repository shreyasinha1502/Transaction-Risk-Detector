import logging
import os
from datetime import datetime

# logs folder create
LOG_DIR = "logs"
LOG_PATH = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(LOG_PATH, exist_ok=True)

# log file name
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_PATH, LOG_FILE)

# logger object
logger = logging.getLogger("transactionLogger")
logger.setLevel(logging.INFO)

# avoid duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

# file handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)

# console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# formatter
formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s %(filename)s:%(lineno)d - %(message)s"
)

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)
