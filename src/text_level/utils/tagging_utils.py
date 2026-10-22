"""
Contains utilitary function for annotating texts.

* restore_original_documents (
    stanza.models.common.doc.Document,
    stanza.models.common.doc.Document,
    bool
) -> stanza.models.common.doc.Document: takes the annotated document,
the original document, and restores original token forms and/or
language-specific parts-of-speech (in case there was no pretagging).
Returns the annotated document with restored annotation. 
"""


from copy import deepcopy
from logging import getLogger
from os import PathLike
from os.path import isfile, join
from re import sub

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

logger = getLogger('preprocessor')

def check_eval_parameters(settings: dict) -> None:
    if not bool(settings["pred_file"]) or not isfile(settings["pred_file"]):
        raise ValueError(f"File with predictions does not exist, received {settings['pred_file']}")
    if not bool(settings["gold_file"]) or not isfile(settings["gold_file"]):
        raise ValueError(f"File with gold annotation does not exist" \
                         f", received {settings['gold_file']}")
    logger.info("The evaluation parameters successfully checked; " \
    "checking predictions in %s against gold annotation in %s",
    settings["pred_file"], settings["gold_file"])



def store_predictions(
        stage_name: str,
        model_name: str,
        exp_folder: PathLike,
        dataset_name: str,
        doc: Document
        ) -> None:
    logger.info("Input: %s", locals())
    # models usually contain slash in their name ("google/gemma-3en-4b"),
    # I replace it with underscore
    model_name = sub(r'\/', '_', model_name)
    file_to_store = join(exp_folder, f"{stage_name}_{model_name}_{dataset_name}")
    CoNLL.write_doc2conll(doc, file_to_store)
    logger.info("See results of %s tagging in %s", stage_name, file_to_store)
    file_to_edit = join(exp_folder, f"{stage_name}_gold_{dataset_name}")
    CoNLL.write_doc2conll(doc, file_to_edit)
    logger.info("Edit results of %s in %s", stage_name, file_to_edit)

def load_input_data(input_file_path: str) -> tuple[Document, Document]:
    logger.info("Input: %s", locals())
    doc = CoNLL.conll2doc(input_file_path)
    doc_for_processing = deepcopy(doc)
    for sent in doc_for_processing.sentences:
        for token in sent.tokens:
            for word in token.words:
                misc_keys = word.misc.split('|')
                for key in misc_keys:
                    if 'wf' in key:
                        new_form = sub('\"', '', key)
                        new_form = sub('wf=', '', new_form)
                        word.text = new_form
    logger.debug("%s loaded", input_file_path)
    return doc, doc_for_processing

def restore_original_tags(
        doc_for_processing: Document, original_document: Document, pretagged: bool = False
        ) -> Document:
    """
    Restores original annotation in a CoNLL-U document: puts non-normalised tokens back to their
    positions as the original tokens of the word, and assings empty value to a None
    xpos (a language-specific part-of-speech), set by a Stanza tagger, in case it is required. 

    Params:
        doc_for_processing (stanza.models.common.doc.Document): the annotated document.
        original_document (stanza.models.common.doc.Document): the document in its original form
        pretagged (bool): the information on earlier annotation, used for assigning or
        not assigning (because it may exist) an xpos to the tokens of the preprocessed document.
    
    Returns:
        stanza.models.common.doc.Document: the annotated document with the original tokenisation
        and, optionally, xpos)
    """
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, _ in enumerate(token.words):
                doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = original_document.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
                if not pretagged:
                    doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    return doc_for_processing