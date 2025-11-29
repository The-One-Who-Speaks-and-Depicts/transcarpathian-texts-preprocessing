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
    logger.debug("Part-of-speech tagging pipeline prepared")
    doc_processed = nlp(doc_for_procesing)
    for sent_idx, sent in enumerate(doc_processed.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, word in enumerate(token.words):
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
    file_to_store = os.path.join(exp_folder, f"pos_stanza_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_store)
    logger.debug("See results of morphological tagging in %s", file_to_store)
    file_to_edit = os.path.join(exp_folder, f"pos_gold_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_edit)
    logger.debug("Edit results of morphological tagging in %s", file_to_edit)