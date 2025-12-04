import os
import logging

import yaml

from datetime import datetime

from corpus_distance.pipeline import create_and_set_storage_directory

from pos import pos_tag_with_stanza, eval_pos
from lemma import lemmatise_with_stanza, evaluate_lemma

def set_logger(folder: str):
    logger = logging.getLogger('preprocessor')
    # TODO: logger level in config
    logger.setLevel(logging.INFO)
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

def main():
    with open(os.path.join(os.getcwd(), 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # TODO: add alternative checks if .env does not exist
    # TODO: set separate path to dir with exps and hide it in .env, only each exp dir should be in .yaml
    experiments_storage = config["exp_dir"] if "exp_dir" in config.keys() and os.path.exists(config["exp_dir"]) else os.getcwd()
    exp_dir = os.path.join(experiments_storage, config["exp_name"])
    perform_initial_check_config(config)
    create_and_set_storage_directory(exp_dir)
    if not bool(config["tagging_settings"]):
        raise ValueError("No settings for tagging")
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
    match config["stage"]:
        case "pos":
            match phase:
                case "pred":
                    if not bool(settings["input_file"]) or not os.path.exists(settings["input_file"]):
                        raise ValueError("Input file does not exist")
                    pos_tag_with_stanza(settings["input_file"], exp_dir)
                case "eval":
                    if not bool(settings["pred_file"]) or not os.path.exists(settings["pred_file"]):
                        raise ValueError("Input file does not exist")
                    if not bool(settings["gold_file"]) or not os.path.exists(settings["gold_file"]):
                        raise ValueError("Input file does not exist")
                    if not "train_result" in settings.keys() or not bool(settings["train_result"]) or settings["train_result"] == "not_set" or type(settings["train_result"]) != float or settings["train_result"] < 0:
                        logger.warning("Train result not supplied, setting to default value")
                        settings["train_result"] = -1
                    if not "train_ufeats" in settings.keys() or not bool(settings["train_ufeats"]) or settings["train_ufeats"] == "not_set" or type(settings["train_ufeats"]) != float or settings["train_ufeats"] < 0:
                        logger.warning("Train ufeats accuracy not supplied, setting to default value")
                        settings["train_ufeats"] = -1
                    eval_pos(settings["gold_file"], settings["pred_file"], settings["train_result"], settings["train_ufeats"], exp_dir)
        case "lemma":
            match phase:
                case "pred":
                    if not bool(settings["input_file"]) or not os.path.exists(settings["input_file"]):
                        raise ValueError("Input file does not exist")
                    if not "pretagged" in settings.keys():
                        logger.warning("Information on tagging not supplied, performing the whole pipeline")
                        settings["pretagged"] = False
                    lemmatise_with_stanza(settings["input_file"], exp_dir, pretagged=settings["pretagged"])
                case "eval":
                    if not bool(settings["pred_file"]) or not os.path.exists(settings["pred_file"]):
                        raise ValueError("Input file does not exist")
                    if not bool(settings["gold_file"]) or not os.path.exists(settings["gold_file"]):
                        raise ValueError("Input file does not exist")
                    if not "train_result" in settings.keys() or not bool(settings["train_result"]) or settings["train_result"] == "not_set" or type(settings["train_result"]) != float or settings["train_result"] < 0:
                        logger.warning("Train result not supplied, setting to default value")
                        settings["train_result"] = -1
                    evaluate_lemma(settings["gold_file"], settings["pred_file"], settings["train_result"], exp_dir)               
        case _:
            logger.error("Use one of the following:" +
                         "pos for joined part-of-speech and morphological tagging," +
                         "lemma for lemmatisation," +
                         "deps for syntactic parsing," +
                         "ner for named entity recognition," +
                         "swadesh for basic vocabulary detection," +
                         "topic for thematic modelling," + 
                         "var for linguistic variation markers"
                         )
            raise ValueError("Stage is set incorrectly, see log for details")
            



if __name__ == '__main__':
    main()
