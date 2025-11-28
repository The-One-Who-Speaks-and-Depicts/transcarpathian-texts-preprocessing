import logging
import os
from os import PathLike
from os.path import isfile
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger('preprocessor')

@dataclass
class ClientParams():
    api: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    seed: int = 1590
    retries: int = 3
    temperature: float = 0
    prompt: str = ""

def set_logger(folder: str, logger_level: int = 10):
    logger = logging.getLogger('preprocessor')
    logger.setLevel(logger_level)
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
    log_handler = logging.FileHandler(filename=os.path.join(folder, f'preprocessor{date_time}.log'), encoding='utf-8')
    log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    return logger

def perform_initial_check_config(config: any) -> None:
    if not bool(config):
        raise ValueError("Config is not set")
    if not bool(config["exp_name"]):
        raise ValueError("No folder for the experiment results")


def load_prompt(user_params: dict) -> str:
    if bool(user_params["prompt"]):
        if isfile(user_params["prompt"]):
            logger.info("Loading user-defined prompt from %s", user_params["prompt"])
            return open(user_params["prompt"], 'r', encoding='utf-8').read()
        logger.info("Loading user-defined prompt directly from config, value is %s", user_params["prompt"])
        return user_params["prompt"]
    default_prompt = "Generate Universal Dependencies CoNLL-U from {sent}"
    logger.warning("No user-defined prompt, using default {default_prompt}")
    return default_prompt
            

def load_llm_params(user_params: dict) -> ClientParams:
    logger.info("Input: %s", locals())
    params_set = ClientParams(
        api = user_params["api"] if bool(user_params["api"]) else "OpenAI",
        api_key = os.environ.get("API_KEY") if "API_KEY" in os.environ else "",
        base_url=user_params["base_url"] if bool(user_params["base_url"]) else "http://localhost:1234/v1/",
        model=user_params["model"] if bool(user_params["model"]) else "google/gemma-3en-4b",
        retries=user_params["retries"] if bool(user_params["retries"]) else 3,
        seed=user_params["seed"] if bool(user_params["seed"]) else 1590,
        temperature=user_params["temperature"] if bool(user_params["temperature"]) else 0,
        prompt=load_prompt(user_params)
    )
    logger.info("The LLM tagging parameters set to %s", params_set)
    return params_set
    
