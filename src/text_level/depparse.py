import logging

import re
import stanza
from stanza.utils.conll import CoNLL
import ntpath
from copy import deepcopy
import os

logger = logging.getLogger('preprocessor')

# TODO: technically, all tagging prediction schema are very similar and can be joined into a single one

def depparse_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk', pretagged: bool = True):
    logger.debug("%s is called with arguments %s", input_file_path, locals())
    name = ntpath.basename(input_file_path)
    stanza.download(lang)
    doc = CoNLL.conll2doc(input_file_path)
    doc_for_procesing = deepcopy(doc)
    for sent in doc_for_procesing.sentences:
        for token in sent.tokens:
            for word in token.words:
                misc_keys = word.misc.split('|')
                for key in misc_keys:
                    if 'wf' in key:
                        new_form = re.sub('\"', '', key)
                        new_form = re.sub('wf=', '', new_form)
                        word.text = new_form
                if not pretagged:
                    word.upos = '_'
                    word.feats = '_'
                    word.lemma = '_'
    logger.debug("%s loaded", input_file_path)
    nlp = stanza.Pipeline(
        lang=lang,
        processors=['tokenize', 'depparse'],
        use_gpu=True,
        tokenize_pretokenized=True,
        depparse_pretagged=True
        ) if pretagged else stanza.Pipeline(
            lang=lang,
            processors=['tokenize', 'pos', 'lemma', 'depparse'],
            use_gpu=True,
            tokenize_pretokenized=True
            )
    logger.debug("Dependency parsing pipeline prepared, settings: %s", nlp)
    doc_processed = nlp(doc_for_procesing)
    for sent_idx, sent in enumerate(doc_processed.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, word in enumerate(token.words):
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
                if not pretagged:
                    doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    pretagging_info = 'gold' if pretagged else 'silver'
    file_to_store = os.path.join(exp_folder, f"depparse_{pretagging_info}_stanza_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_store)
    logger.debug("See results of dependency parsing in %s", file_to_store)
    file_to_edit = os.path.join(exp_folder, f"depparse_{pretagging_info}_gold_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_edit)
    logger.debug("Edit results of dependency parsing in %s", file_to_edit)


# def calculate_las_uas(gold_heads, gold_labels, pred_heads, pred_labels):
#     """Calculate LAS and UAS scores"""
#     total = len(gold_heads)
#     uas_correct = sum(1 for g, p in zip(gold_heads, pred_heads) if g == p)
#     las_correct = sum(1 for g_h, g_l, p_h, p_l in 
#                      zip(gold_heads, gold_labels, pred_heads, pred_labels) 
#                      if g_h == p_h and g_l == p_l)
    
#     uas = uas_correct / total if total > 0 else 0
#     las = las_correct / total if total > 0 else 0
    
#     return uas, las

# Example usage


# def evaluate_depparse():
#     uas, las = calculate_las_uas(gold_heads, gold_labels, pred_heads, pred_labels)