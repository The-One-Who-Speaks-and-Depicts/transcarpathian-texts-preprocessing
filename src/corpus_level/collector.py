import argparse
import os

import yaml

import pandas as pd

from stanza.utils.conll import CoNLL

from corpus_distance.pipeline import create_and_set_storage_directory

from utils.utils import perform_initial_check_config, set_logger

def main(args):
    config_path = args.config if os.path.exists(args.config) else os.path.join(os.getcwd(), 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
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
    settings = config["tagging_settings"]
    overall_glosses = []
    for input in settings["inputs"]:
        doc = CoNLL.conll2doc(input["file"])
        for sent in doc.sentences:
            be_ids = []
            be_heads = []
            be_surr = []
            for token in sent.tokens:
                for word in token.words:
                    if word.lemma == input["lemma"]:
                        be_ids.append(word.id)
                        be_heads.append(word.head)
                        be_surr.append(word.pos)
                        if word.feats:
                            for i in word.feats.split('|'):
                                be_surr.append(i)
            for token in sent.tokens:
                for word in token.words:
                    if word.id in be_heads:
                        be_surr.append(word.pos)
                        if word.feats:
                            for i in word.feats.split('|'):
                                be_surr.append(i)                        
                    if word.head in be_ids:
                        be_surr.append(word.pos)
                        if word.feats:
                            for i in word.feats.split('|'):
                                be_surr.append(i)
            overall_glosses.extend(be_surr)
    overall_glosses = list(set(overall_glosses))
    bees = {}
    for input in settings["inputs"]:
        name = input["alias"]
        doc = CoNLL.conll2doc(input["file"])
        be_ids_with_heads = []
        for sent in doc.sentences:
            for token in sent.tokens:
                for word in token.words:
                    if word.lemma == input["lemma"]:
                        be_ids_with_heads.append((word.id, word.head))
        for id, head in be_ids_with_heads:
            uid = name + '_' + str(id)
            glosses = []
            for sent in doc.sentences:
                for token in sent.tokens:
                    for word in token.words:
                        if word.lemma == input["lemma"] or word.id == head or word.head == id:
                            if word.pos in overall_glosses:
                                glosses.append(word.pos)
                            if word.feats:
                                for i in word.feats.split('|'):
                                    glosses.append(i)
            partial_glosses = [1 if i in glosses else 0 for i in overall_glosses]
            params = [name]
            params.extend(partial_glosses)
            bees[uid] = params
    logger.info(bees)
    params = ['lect']
    params.extend(overall_glosses)
    bees = pd.DataFrame.from_dict(bees, columns=params, orient='index')
    logger.info(bees.head())
    bees.to_csv(os.path.join(exp_dir, 'bees.csv'))
    logger.info('Saved to %s', os.path.join(exp_dir, 'bees.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', help='.yaml file with config')
    args = parser.parse_args()
    main(args)
