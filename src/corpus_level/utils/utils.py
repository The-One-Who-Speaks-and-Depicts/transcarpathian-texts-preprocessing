# NOTE: currently, a full duplicate of text_level.utils.utils

from logging import getLogger, FileHandler, Formatter
import os
from datetime import datetime

def set_logger(folder: str, logger_level: int = 10):
    logger = getLogger('preprocessor')
    logger.setLevel(logger_level)
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    log_handler = FileHandler(filename=os.path.join(folder, f'preprocessor{date_time}.log'), encoding='utf-8')
    log_formatter = Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    return logger

def perform_initial_check_config(config: any) -> None:
    if not bool(config):
        raise ValueError("Config is not set")
    if not bool(config["exp_name"]):
        raise ValueError("No folder for the experiment results")

