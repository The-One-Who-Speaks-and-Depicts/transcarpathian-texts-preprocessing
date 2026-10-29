from argparse import ArgumentParser
from dotenv import load_dotenv

from logging import getLogger, StreamHandler, Formatter
from os import getcwd
from os.path import isfile, join, exists
from sys import stdout

from yaml import safe_load

from corpus_distance.pipeline import create_and_set_storage_directory

from utils.utils import (
    perform_initial_check_config, set_logger
)

from utils.llm_utils import load_llm_params
from utils.tagging_utils import check_eval_parameters

from pos import pos_tag_with_stanza, pos_tag_with_llm, eval_morphology
from lemma import lemmatise_with_stanza, lemmatise_with_llm, evaluate_lemma
from depparse import depparse_with_stanza, depparse_with_llm, evaluate_depparse
from topic_modelling import perform_topic_modelling  


def main(args):
    # Starting with declaring a simple stdout logger to inform the user
    # of the program execution before the initialisation of the experiment
    # directory
    logger = getLogger('preprocessor_initialize')
    logger.setLevel(10)
    handler = StreamHandler(stdout)
    handler.setFormatter(Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    load_dotenv()
    logger.debug("Environment is loaded!")
    config_path = args.config if exists(args.config) else join(getcwd(), 'config.yaml')
    if not isfile(config_path):
        raise ValueError("Configuration path %s is not valid", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        logger.info("Config is being loaded from %s", config_path)
        config = safe_load(f)
    logger.debug("Config is loaded successfully")
    experiments_storage = config["exp_dir"] if "exp_dir" in config.keys() and exists(config["exp_dir"]) else getcwd()
    logger.info("Experiment storage set to %s", experiments_storage)
    exp_dir = join(experiments_storage, config["exp_name"])
    perform_initial_check_config(config)
    create_and_set_storage_directory(exp_dir)
    if not bool(config["tagging_settings"]):
        raise ValueError("No settings for tagging")
    logger.debug("Initial check finished, see further report in %s", exp_dir)
    logger = set_logger(exp_dir)
    logger.debug('Configuration loaded: %s', config)
    if not bool(config["tagging_settings"]):
        raise ValueError("No settings for tagging")
    if not bool(config["phase"]):
        raise ValueError("Phase parameter is not set")
    phase = config["phase"]
    if phase not in ["eval", "pred"]:
        logger.error(
            "Use either pred to generate file with automatic tagging" +
            ", or eval to contrast automatic tagging with gold standard"
            )
        raise ValueError("Phase is set incorrectly, see log for details")    
    settings = config["tagging_settings"] if phase == "pred" else config["evaluation_settings"]    
    if not bool(config["stage"]):
        raise ValueError("Stage parameter is not set")
    if not bool(config["model"]):
        raise ValueError("Model parameter is not set")
    match config["stage"]:
        case "pos":
            match phase:
                case "pred":
                    if not bool(settings["input_file"]) or not exists(settings["input_file"]):
                        raise ValueError("Input file does not exist")
                    match config["model"]:
                        case "stanza":
                            pos_tag_with_stanza(settings["input_file"], exp_dir)
                        case "llm":
                            if not bool(settings["llm_params"]):
                                raise ValueError("LLM parameters not given")
                            llm_params = load_llm_params(settings["llm_params"])
                            pos_tag_with_llm(settings["input_file"], exp_dir, llm_params)
                        case _:
                            raise ValueError(f"Incorrect model parameter, received {config["model"]}")
                case "eval":                    
                    check_eval_parameters(settings)
                    if not "train_result" in settings.keys() or not bool(settings["train_result"]) or settings["train_result"] == "not_set" or type(settings["train_result"]) != float or settings["train_result"] < 0:
                        logger.warning("Train result not supplied, setting to default value")
                        settings["train_result"] = -1
                    if not "train_ufeats" in settings.keys() or not bool(settings["train_ufeats"]) or settings["train_ufeats"] == "not_set" or type(settings["train_ufeats"]) != float or settings["train_ufeats"] < 0:
                        logger.warning("Train ufeats accuracy not supplied, setting to default value")
                        settings["train_ufeats"] = -1
                    eval_morphology(settings["gold_file"], settings["pred_file"], settings["train_result"], settings["train_ufeats"], exp_dir)
        case "lemma":
            match phase:
                case "pred":
                    if not bool(settings["input_file"]) or not exists(settings["input_file"]):
                        raise ValueError("Input file does not exist")
                    if not "pretagged" in settings.keys():
                        logger.warning("Information on tagging not supplied")
                        settings["pretagged"] = False
                    match config["model"]:
                        case "stanza":
                            lemmatise_with_stanza(settings["input_file"], exp_dir, pretagged=settings["pretagged"])
                        case "llm":
                            if not bool(settings["llm_params"]):
                                raise ValueError("LLM parameters not given")
                            llm_params = load_llm_params(settings["llm_params"])
                            lemmatise_with_llm(settings["input_file"], exp_dir, llm_params, pretagged=settings["pretagged"])
                        case _:
                            raise ValueError(f"Incorrect model parameter, received {config["model"]}")
                case "eval":                    
                    check_eval_parameters(settings)
                    if not "train_result" in settings.keys() or not bool(settings["train_result"]) or settings["train_result"] == "not_set" or type(settings["train_result"]) != float or settings["train_result"] < 0:
                        logger.warning("Train result not supplied, setting to default value")
                        settings["train_result"] = -1
                    evaluate_lemma(settings["gold_file"], settings["pred_file"], settings["train_result"], exp_dir)               
        case "depparse":
            match phase:
                case "pred":
                    if not bool(settings["input_file"]) or not exists(settings["input_file"]):
                        raise ValueError("Input file does not exist")
                    if not "pretagged" in settings.keys():
                        logger.warning("Information on tagging not supplied")
                        settings["pretagged"] = False
                    match config["model"]:
                        case "stanza":
                            depparse_with_stanza(settings["input_file"], exp_dir, pretagged=settings["pretagged"])
                        case "llm":
                            llm_params = load_llm_params(settings["llm_params"])
                            depparse_with_llm(settings["input_file"], exp_dir, llm_params, pretagged=settings["pretagged"])
                        case _:
                            raise ValueError(f"Incorrect model parameter, received {config["model"]}")
                case "eval":
                    check_eval_parameters(settings)
                    if not "train_result" in settings.keys() or not bool(settings["train_result"]) or settings["train_result"] == "not_set" or type(settings["train_result"]) != float or settings["train_result"] < 0:
                        logger.warning("Train result not supplied, setting to default value")
                        settings["train_result"] = -1
                    if not "train_ufeats" in settings.keys() or not bool(settings["train_ufeats"]) or settings["train_ufeats"] == "not_set" or type(settings["train_ufeats"]) != float or settings["train_ufeats"] < 0:
                        logger.warning("Train ufeats accuracy not supplied, setting to default value")
                        settings["train_ufeats"] = -1
                    evaluate_depparse(settings["gold_file"], settings["pred_file"], gold_uas= settings["train_result"], gold_las=settings["train_ufeats"])
        case "topic":
            match phase:
                case "pred":
                    if not bool(settings["input_file"]) or not exists(settings["input_file"]):
                        raise ValueError("Input file does not exist")
                    perform_topic_modelling(settings["input_file"], exp_dir)
                case "eval":
                    logger.critical("No %s phase for stage %s", phase, config["stage"])
        case _:
            logger.error("Use one of the following:" +
                         "pos for joined part-of-speech and morphological tagging," +
                         "lemma for lemmatisation," +
                         "deps for syntactic parsing," +
                         # "ner for named entity recognition," +
                         # "swadesh for basic vocabulary detection," +
                         "topic for thematic modelling,"# + 
                         # "var for linguistic variation markers"
                         )
            raise ValueError("Stage is set incorrectly, see log for details")
            



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to config.yaml file')
    args = parser.parse_args()
    main(args)
