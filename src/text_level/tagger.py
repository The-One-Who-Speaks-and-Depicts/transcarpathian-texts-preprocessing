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
from utils.eval_utils import check_eval_parameters_by_stage

from annotation import annotate_with_stanza, annotate_with_llm, LLMAnnotationSettings
from utils.data_schema import get_depparse_schema
from pos import get_stanza_morphology_pipeline, pos_tag_with_llm, eval_morphology
from lemma import get_stanza_lemmatisation_pipeline, lemmatise_with_llm, evaluate_lemma
from depparse import  get_stanza_depparse_pipeline, get_depparse_prompt, evaluate_depparse
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
    # ==================== Setting configuration and performing initial checks ====================
    config_path = args.config if exists(args.config) else join(getcwd(), 'config.yaml')
    if not isfile(config_path):
        raise ValueError("Configuration path %s is not valid", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = safe_load(f)
    config = perform_initial_check_config(config)
    logger.debug('Configuration loaded: %s', config)
    exp_dir = join(config["exp_dir"], config["exp_name"])
    create_and_set_storage_directory(exp_dir)
    logger.debug("Initial check finished, configuration is:\n%s\n\nSee further report in %s",
                 config,
                 exp_dir)
    logger = set_logger(exp_dir)
    # ==================================== Tagging/evaluation ====================================    
    match config["phase"]:
        case "pred":
            if not bool(config["tagging_settings"]):
                raise ValueError(f"Tagging settings do not exist, configuration:\n{config}")
            settings = config["tagging_settings"]
            if not bool(settings["input_file"]) or not exists(settings["input_file"]):
                raise ValueError("Input file does not exist")
            if not "pretagged" in settings.keys():
                logger.warning("Information on tagging not supplied")
                settings["pretagged"] = False
            llm_params = None
            if config["model"] == "llm":
                llm_params = load_llm_params(settings["llm_params"])
            match config["stage"]:
                case "pos":
                    match config["model"]:
                        case "stanza":
                            annotate_with_stanza(
                                settings["input_file"], exp_dir, config["stage"],
                                get_stanza_morphology_pipeline('uk', settings["pretagged"]),
                                settings["pretagged"])
                        case "llm":
                            # TODO: LLM annotation unification
                            pos_tag_with_llm(settings["input_file"], exp_dir, llm_params)
                case "lemma":
                    match config["model"]:
                        case "stanza":
                            annotate_with_stanza(
                                settings["input_file"], exp_dir, config["stage"],
                                get_stanza_lemmatisation_pipeline('uk', settings["pretagged"]),
                                settings["pretagged"])
                        case "llm":
                            lemmatise_with_llm(
                                settings["input_file"], exp_dir,
                                llm_params, pretagged=settings["pretagged"]
                                )
                case "depparse":
                    match config["model"]:
                        case "stanza":
                            annotate_with_stanza(
                                # TODO: these three are the same everywhere, do the same class and a united check
                                settings["input_file"], exp_dir, config["stage"],
                                get_stanza_depparse_pipeline('uk', settings["pretagged"]),
                                settings["pretagged"])
                        case "llm":
                            annotate_with_llm(
                                settings, exp_dir, config["stage"],
                                LLMAnnotationSettings(
                                    get_depparse_prompt,
                                    get_depparse_schema,
                                    ['head', 'deprel']
                                )                                
                            )
                case "topic":
                    match config["model"]:
                        case "LDA":
                            perform_topic_modelling(settings["input_file"], exp_dir)
        case "eval":
            settings = check_eval_parameters_by_stage(config)
            match config["stage"]:
                case "pos":
                    eval_morphology(
                        settings["gold_file"],
                        settings["pred_file"],
                        settings["train_class"],
                        settings["train_feats"],
                        exp_dir
                    )
                case "lemma":
                    evaluate_lemma(
                        settings["gold_file"],
                        settings["pred_file"],
                        settings["train_class"],
                        exp_dir
                    )
                case "depparse":
                    evaluate_depparse(
                        settings["gold_file"],
                        settings["pred_file"],
                        gold_uas=settings["train_class"],
                        gold_las=settings["train_feats"]
                    )
            



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to config.yaml file')
    args = parser.parse_args()
    main(args)
