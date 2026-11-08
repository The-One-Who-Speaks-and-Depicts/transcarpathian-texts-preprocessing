from argparse import ArgumentParser
from dotenv import load_dotenv

from logging import getLogger, StreamHandler, Formatter
from os import getcwd
from os.path import isfile, join, exists
from sys import stdout

from yaml import safe_load

from corpus_distance.pipeline import create_and_set_storage_directory

from utils import (
    perform_initial_check_config, set_logger
)

from text_level.utils.eval_utils import check_eval_parameters_by_stage
from text_level.utils.tagging_utils import check_config_for_annotation

from text_level.annotation import annotate_with_stanza, annotate_with_llm, LLMAnnotationSettings
from text_level.utils.data_schema import get_pos_schema, get_depparse_schema, get_lemma_schema
from text_level.pos import get_stanza_morphology_pipeline, evaluate_morphology, get_pos_prompt
from text_level.lemma import get_stanza_lemmatisation_pipeline, evaluate_lemma, get_lemma_prompt
from text_level.depparse import  get_stanza_depparse_pipeline, get_depparse_prompt, evaluate_depparse
from text_level.topic_modelling import perform_topic_modelling

from corpus_level.feature_collection import collect_conllu_features_to_dataframe


def main(args):
    # Starting with declaring a simple stdout logger to inform the user
    # of the program execution before the initialisation of the experiment
    # directory
    logger = getLogger('initializer')
    logger.setLevel(10)
    handler = StreamHandler(stdout)
    handler.setFormatter(Formatter(
        fmt="%(asctime)s %(levelname)s: %(message)s",
                                datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(handler)
    load_dotenv()
    logger.debug("Environment is loaded!")
    # =================== Setting configuration and performing initial checks ===================
    config_path = args.config if exists(args.config) else join(getcwd(), 'config.yaml')
    if not isfile(config_path):
        raise ValueError("Configuration path %s is not valid", config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = safe_load(f)
    config = perform_initial_check_config(config)
    exp_dir = create_and_set_storage_directory(join(config["exp_dir"], config["exp_name"]))    
    logger.debug("Initial check finished, configuration is:\n%s\n\nSee further report in %s",
                 config,
                 exp_dir)
    logger = set_logger(exp_dir)
    # =================================== Tagging/evaluation ===================================
    match config["phase"]:
        case "pred":
            settings = check_config_for_annotation(config)
            match config["stage"]:
                case "pos":
                    match config["model"]:
                        case "stanza":
                            annotate_with_stanza(
                                settings["input_file"], exp_dir, config["stage"],
                                get_stanza_morphology_pipeline('uk', settings["pretagged"]),
                                settings["pretagged"])
                        case "llm":
                            annotate_with_llm(
                                settings, exp_dir, config["stage"],
                                LLMAnnotationSettings(
                                    get_pos_prompt,
                                    get_pos_schema,
                                    ['upos', 'feats']
                                )
                                )
                case "lemma":
                    match config["model"]:
                        case "stanza":
                            annotate_with_stanza(
                                settings["input_file"], exp_dir, config["stage"],
                                get_stanza_lemmatisation_pipeline('uk', settings["pretagged"]),
                                settings["pretagged"])
                        case "llm":
                            annotate_with_llm(
                                settings, exp_dir, config["stage"],
                                LLMAnnotationSettings(
                                    get_lemma_prompt,
                                    get_lemma_schema,
                                    ['lemma']
                                )
                                )
                case "depparse":
                    match config["model"]:
                        case "stanza":
                            annotate_with_stanza(
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
                    evaluate_morphology(
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
        case "gather":
            if not bool(config["tagging_settings"]):
                raise KeyError(f"Experiment settings is absent, config is:\n{config}")
            settings = config["tagging_settings"]
            if not bool(settings["inputs"]):
                raise KeyError("Information on input files is absent, " \
                f"experiment settings are:\n{settings}")
            inputs = settings["inputs"]
            if not isinstance(inputs, list) and not all(
                isinstance(
                    i, dict
                    ) and not set(["file", "lemma", "alias"]).issubset(i.keys()) for i in inputs
            ):
                raise ValueError("inputs should be a list of dictionaries, each containing" \
                f"keys for a file to load and a lemma to gather, received:\n{inputs}")
            collect_conllu_features_to_dataframe(exp_dir, inputs)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', '-c', help='Path to config.yaml file')
    args = parser.parse_args()
    main(args)
