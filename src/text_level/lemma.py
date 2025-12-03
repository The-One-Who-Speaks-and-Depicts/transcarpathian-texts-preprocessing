import logging

import re
import stanza
from stanza.utils.conll import CoNLL
import ntpath
from copy import deepcopy
import os

logger = logging.getLogger('preprocessor')

def lemmatise_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk', pretagged: bool = True):
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
    logger.debug("%s loaded", input_file_path)
    nlp = stanza.Pipeline(
        lang=lang,
        processors=['tokenize', 'lemma'],
        use_gpu=True,
        tokenize_pretokenized=True,
        lemma_pretagged=True
        ) if pretagged else stanza.Pipeline(
            lang=lang,
            processors=['tokenize', 'pos', 'lemma'],
            use_gpu=True,
            tokenize_pretokenized=True
            )
    logger.debug("Lemmatisation tagging pipeline prepared, settings: %s", nlp)
    doc_processed = nlp(doc_for_procesing)
    for sent_idx, sent in enumerate(doc_processed.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, word in enumerate(token.words):
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
                if not pretagged:
                    doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    pretagging_info = 'gold-pos' if pretagged else 'silver-pos'
    file_to_store = os.path.join(exp_folder, f"lemma_{pretagging_info}_stanza_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_store)
    logger.debug("See results of lemmatisation in %s", file_to_store)
    file_to_edit = os.path.join(exp_folder, f"lemma_{pretagging_info}_gold_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_edit)
    logger.debug("Edit results of lemmatisation in %s", file_to_edit)


def longest_common_substring_indices(str1, str2):
    """Вяртае індэксы найдоўжага агульнага падрадка ў форме:
    ((start1, end1), (start2, end2))
    дзе end1 і end2 - індэксы ПАСЛЯ апошняга сімвала
    """
    m, n = len(str1), len(str2)
    
    # Матрыца для дынамічнага праграмавання
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    max_length = 0
    end_pos1 = 0  # end пазіцыя ў str1 (індыкс ПАСЛЯ апошняга сімвала)
    end_pos2 = 0  # end пазіцыя ў str2
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_pos1 = i  # i - гэта пазіцыя ў str1 (з 1)
                    end_pos2 = j  # j - гэта пазіцыя ў str2 (з 1)
    
    if max_length == 0:
        return ((-1, -1), (-1, -1))
    
    # Вылічым пачатковыя пазіцыі
    start1 = end_pos1 - max_length  # індыкс першага сімвала ў str1
    start2 = end_pos2 - max_length  # індыкс першага сімвала ў str2
    
    return ((start1, end_pos1), (start2, end_pos2))