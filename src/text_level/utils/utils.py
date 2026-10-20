import logging
import os
import re
from os.path import isfile
from datetime import datetime
from dataclasses import dataclass
import ntpath
from copy import deepcopy

from stanza.utils.conll import CoNLL
from stanza.models.common.doc import Document

from openai import OpenAI

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

def check_eval_parameters(settings: dict) -> None:
    if not bool(settings["pred_file"]) or not isfile(settings["pred_file"]):
        raise ValueError(f"File with predictions does not exist, received {settings['pred_file']}")
    if not bool(settings["gold_file"]) or not isfile(settings["gold_file"]):
        raise ValueError(f"File with gold annotation does not exist" \
                         f", received {settings['gold_file']}")
    logger.info("The evaluation parameters successfully checked; " \
    "checking predictions in %s against gold annotation in %s",
    settings["pred_file"], settings["gold_file"])


def store_predictions(
        stage_name: str,
        model_name: str,
        exp_folder: os.PathLike,
        dataset_name: str,
        doc: Document
        ) -> None:
    model_name = re.sub(r'\/', '_', model_name)
    file_to_store = os.path.join(exp_folder, f"{stage_name}_{model_name}_{dataset_name}")
    CoNLL.write_doc2conll(doc, file_to_store)
    logger.info("See results of %s tagging in %s", stage_name, file_to_store)
    file_to_edit = os.path.join(exp_folder, f"{stage_name}_gold_{dataset_name}")
    CoNLL.write_doc2conll(doc, file_to_edit)
    logger.info("Edit results of %s in %s", stage_name, file_to_edit)


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

def load_input_data(input_file_path: str) -> tuple[Document, Document]:
    logger.info("Input: %s", locals())
    doc = CoNLL.conll2doc(input_file_path)
    doc_for_processing = deepcopy(doc)
    for sent in doc_for_processing.sentences:
        for token in sent.tokens:
            for word in token.words:
                misc_keys = word.misc.split('|')
                for key in misc_keys:
                    if 'wf' in key:
                        new_form = re.sub('\"', '', key)
                        new_form = re.sub('wf=', '', new_form)
                        word.text = new_form
    logger.debug("%s loaded", input_file_path)
    return doc, doc_for_processing