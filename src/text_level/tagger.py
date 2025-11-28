import os
import logging

import yaml

from corpus_distance.pipeline import create_and_set_storage_directory

from pos import pos_tag_with_stanza

def set_logger(folder: str):
    logger = logging.getLogger('preprocessor')
    logger.setLevel(logging.DEBUG)
    log_handler = logging.FileHandler(filename=os.path.join(folder, 'preprocessor.log'), encoding='utf-8')
    log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S')
    log_handler.setFormatter(log_formatter)
    logger.addHandler(log_handler)
    return logger

def perform_initial_check_config(config: any, exp_dir: str) -> None:
    if not bool(config):
        raise ValueError("Config is not set")
    if not bool(config["folder_name"]):
        raise ValueError("No folder for the experiment results")
    create_and_set_storage_directory(config["folder_name"])


def main():
    with open(os.path.join(os.getcwd(), 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    exp_dir = os.path.join(os.getcwd(), config["folder_name"])
    perform_initial_check_config(config, exp_dir)
    logger = set_logger(exp_dir)
    logger.debug('Configuration loaded: %s', config)
    if not bool(config["tagging_settings"]):
        raise ValueError("No settings for tagging")
    settings = config["tagging_settings"]
    if not bool(settings["input_file"]) or not os.path.exists(settings["input_file"]):
        raise ValueError("Input file does not exist")
    if not bool(settings["stage"]):
        raise ValueError("Stage parameter is not set")
    if not bool(settings["phase"]):
        raise ValueError("Phase parameter is not set")    
    match settings["stage"]:
        case "pos":
            match settings["phase"]:
                case "pred":
                    pos_tag_with_stanza(settings["input_file"], exp_dir)
                case _:
                    logger.error(
                        "Use either pred to generate file with automatic tagging" +
                        ", or eval to contrast automatic tagging with gold standard"
                        )
                    raise ValueError("Phase is set incorrectly, see log for details")
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
