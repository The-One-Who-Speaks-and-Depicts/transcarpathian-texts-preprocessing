import logging
import os
import re


import ntpath

import pandas as pd

from corpus_distance.data_preprocessing.topic_modelling import build_topic_words_for_lect
from stanza.utils.conll import CoNLL

from stanza.models.common.doc import Word


import random
from numpy.random import RandomState

logger = logging.getLogger('preprocessor')

def set_random_seeds() -> None:
    SEED = 1590
    RandomState(SEED)
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = "0"


def select_form_for_topic_modelling(word: Word) -> str:
    if word.lemma is not None:
        return word.lemma
    misc_keys = word.misc.split('|')
    for key in misc_keys:
        if 'wf' in key:
            new_form = re.sub('\"', '', key)
            new_form = re.sub('wf=', '', new_form)
            return new_form               


def perform_topic_modelling(input_file_path: str, exp_dir: str) -> None:
    set_random_seeds()
    doc = CoNLL.conll2doc(input_file_path)
    full_text = []
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                full_text.append(select_form_for_topic_modelling(word))
    full_text = ' '.join(full_text)
    df = pd.DataFrame.from_dict({'0': [full_text, '_']}, orient='index', columns=['text', 'lect'])
    topics = build_topic_words_for_lect(df, '_', 2, 7)
    logger.debug("Collected topics are: %s", ", ".join(topics))
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                form_to_check = select_form_for_topic_modelling(word)
                if form_to_check in topics:
                    if word.misc == None:
                        word.misc = 'Topic=True'
                    else:
                        word.misc = word.misc + '|Topic=True'
    name = ntpath.basename(input_file_path)
    file_to_store = os.path.join(exp_dir, f"topics_{name}")
    CoNLL.write_doc2conll(doc, file_to_store)

