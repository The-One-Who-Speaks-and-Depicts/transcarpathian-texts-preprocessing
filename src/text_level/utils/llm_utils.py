
from dataclasses import dataclass
from logging import getLogger
from os import environ
from os.path import isfile

from openai import OpenAI

logger = getLogger('preprocessor')
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
        api_key = environ.get("API_KEY") if "API_KEY" in environ else "",
        base_url=user_params["base_url"] if bool(user_params["base_url"]) else "http://localhost:1234/v1/",
        model=user_params["model"] if bool(user_params["model"]) else "google/gemma-3en-4b",
        retries=user_params["retries"] if bool(user_params["retries"]) else 3,
        seed=user_params["seed"] if bool(user_params["seed"]) else 1590,
        temperature=user_params["temperature"] if bool(user_params["temperature"]) else 0,
        prompt=load_prompt(user_params)
    )
    logger.info("The LLM tagging parameters set to %s", params_set)
    return params_set




def load_llm_client(api: str, key: str, url: str) -> OpenAI:
    match api:
        case "OpenAI":            
            client = OpenAI(
                api_key=key,
                base_url=url
            )
            logger.info("Loading an OpenAI-like API client, base_url: %s")
            return client
        case _:
            raise ValueError(f"Incorrect API type, received {api}")