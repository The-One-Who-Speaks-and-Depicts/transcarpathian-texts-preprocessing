"""
Utilities required for running the parts of the program not directly connected to the analysis.

* set_logger (PathLike, int) -> Logger: creates a logger that writes data to a provided folder with
a level expressed through a provided integer (0/10/20/30/40/50, see the Python logging library
documentation for the details)
* perform_initial_check_config (Any): checks, whether a basic configuration exists
"""

from datetime import datetime

from logging import getLogger, FileHandler, Formatter, Logger
from os import PathLike
from os.path import exists

from typing import Any

def set_logger(folder: PathLike, logger_level: int = 10) -> Logger:
    """
    Sets a logger that writes logs to a specific folder with a specific level.

    Params:
        folder (PathLike): a folder where the logger will write logs
        logger_level (int): a numerical representation of a level of logs,
        from which an instantiated logger should start writing (0/10/20/30/40/50,
        see the Python logging library documentation for the details)
    
    Returns:
        Logger: an instantiated logger
    """
    if not isinstance(folder, PathLike):
        raise TypeError(f"folder should be a PathLike object, received {folder}")
    if not exists(folder):
        raise ValueError(f"folder is not an existing directory, received {folder}")
    if not isinstance(logger_level, int):
        raise TypeError(f"logger_level is not an integer, receive {logger_level}")
    if logger_level not in [0, 10, 20, 30, 40, 50]:
        raise ValueError("logger_level has a value {logger_level}, which is non-convertable " \
        "to a NOTSET, DEBUG, INFO, WARNING, ERROR or CRITICAL level (see the Python " \
        "logging library documentation for the details)")
    logger = getLogger('preprocessor')
    logger.setLevel(logger_level)
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    log_handler = FileHandler(filename= folder / f'preprocessor{date_time}.log', encoding='utf-8')
    log_formatter = Formatter(fmt="%(asctime)s %(levelname)s %(funcName)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    logger.log(logger_level, "Logger is set, params: %s", logger)
    return logger

def perform_initial_check_config(config: Any) -> None:
    """
    Checks the existense of configuration
    and a key-value pair for the experiment folder name in it.

    Params:
        config (Any): an experiment configuration
    """
    if not bool(config):
        raise ValueError("Config is not set")
    if not bool(config["exp_name"]):
        raise ValueError("No folder for the experiment results")