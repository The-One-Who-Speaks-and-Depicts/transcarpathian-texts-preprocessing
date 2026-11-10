"""
Contains utilitary functions for annotating texts.

* store_predictions (str, str, PathLike, str, Document): takes information about
experiment parameters and results to store them.
* load_input_data (str, bool | str) -> tuple[Document, Document]: takes a path
to input data and splits it into two documents, one using normalised tokens, and 
one still containing non-normalised ones.
* restore_original_documents (
    stanza.models.common.doc.Document,
    stanza.models.common.doc.Document,
    bool
) -> stanza.models.common.doc.Document: takes the annotated document,
the original document, and restores original token forms and/or
language-specific parts-of-speech (in case there was no pretagging).
Returns the annotated document with restored annotation. 
* check_config_for_annotation (Any) -> dict: checks, whether the provided
annotation parameters are correct, amending, if possible. 
"""


from copy import deepcopy
from logging import getLogger
from os import PathLike
from os.path import exists, isfile
from re import sub
from typing import Any

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

logger = getLogger('preprocessor')


def store_predictions(
        stage_name: str,
        model_name: str,
        exp_folder: PathLike,
        dataset_name: str,
        doc: Document
        ) -> None:
    """
    Takes experiment parameters and results to create two files, the first
    required to store results of annotation by a particular model for further evaluation,
    and the second required for a human-in-the-loop check.

    Params:
        stage_name (str): an annotation stage (for instance, "pos" for morphological annotation)
        model_name (str): a model that performed the annotation (could be "google/gemma-3-en-4b")
        exp_folder (PathLike): a directory to store the results and logs in
        dataset_name (str): a user-defined dataset designation
        doc (Document): a Stanza Document object that contains the annotation results
    """
    logger.debug("Input: %s", locals())
    # models usually contain slash in their name ("google/gemma-3en-4b"),
    # I replace it with underscore
    model_name = sub(r'\/', '_', model_name)
    file_to_store = exp_folder / f"{stage_name}_{model_name}_{dataset_name}"
    CoNLL.write_doc2conll(doc, file_to_store)
    logger.info("See results of %s tagging in %s", stage_name, file_to_store)
    file_to_edit = exp_folder / f"{stage_name}_gold_{dataset_name}"
    CoNLL.write_doc2conll(doc, file_to_edit)
    logger.info("Edit results of %s in %s", stage_name, file_to_edit)

def load_input_data(
        input_file_path: PathLike, pretagged: bool | str = False
        ) -> tuple[Document, Document]:
    """
    Takes a path to a CoNLL-U file and a flag that shows whether the file
    already contains the results of morphosyntactic annotation previous stage
    (morphological tagging for lemmatisation, for instance), and returns 
    two copies of the file parsed into a Stanza Document, one containing
    only the normalised version of tokens and the other preserving the original form.

    Params:
        input_file_path (PathLike): a path to a CoNLL-U file
        retagged (bool | str): information on the previously performed annotation available:
            - True: tokens and morphological tags
            - False: only tokens

    Returns:
        tuple[Document, Document]: a pair of Stanza Document objects, representing data
        through non-normalised and normalised versions of tokens 
    """
    logger.debug("Input: %s", locals())
    if not isinstance(
        input_file_path, PathLike
        ) or not exists(input_file_path) or not isfile(input_file_path):
        raise ValueError("Incorrect input_file_path parameter, " \
        f"received {input_file_path}")
    if not isinstance(pretagged, bool) and not isinstance(pretagged, str):
        raise ValueError(f"Incorrect pretagged parameter, received {pretagged}")
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
                if not pretagged:
                    if not hasattr(word, 'upos') or word.upos is None:
                        word.upos = '_'
                    if not hasattr(word, 'feats') or word.feats is None:
                        word.feats = '_'
                    if not hasattr(word, 'lemma') or word.lemma is None:
                        word.lemma = '_'
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
        doc_for_processing (stanza.models.common.doc.Document): an annotated document
        original_document (stanza.models.common.doc.Document): the document in its original form
        pretagged (bool): information on earlier annotation, used for assigning or
        not assigning (because it may exist) an xpos to the tokens of the preprocessed document
    
    Returns:
        stanza.models.common.doc.Document: the annotated document with the original tokenisation
        and, optionally, xpos
    """
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, _ in enumerate(token.words):
                original_token = original_document.sentences[
                    sent_idx
                    ].tokens[tkn_idx].words[wrd_idx].text
                doc_for_processing.sentences[
                    sent_idx
                    ].tokens[tkn_idx].words[wrd_idx].text = original_token
                if not pretagged:
                    doc_for_processing.sentences[
                        sent_idx
                        ].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    return doc_for_processing

def check_config_for_annotation(config: Any) -> dict:
    """
    Performs the initial check on the annotation parameters in the configuration file,
    provided by a user. In case there is no information provided, attempts to amend,
    assigning default values (only in case of the previous evaluation).
    
    Params:
        config (Any): an experiment configuration
    
    Returns:
        dict: an experiment configuration object, either amended by a function, or preserved as is
    """
    if not bool(config["tagging_settings"]):
        raise ValueError(f"Tagging settings do not exist, configuration:\n{config}")
    if not bool(config["stage"]) or config["stage"] not in ["pos", "lemma", "depparse", "topic"]:
        raise ValueError("Stage parameter is set incorrectly")
    if not bool(config["model"]) or config["model"] not in ["stanza", "llm"]:
        raise ValueError("Model parameter is set incorrectly")
    settings = config["tagging_settings"]
    if not bool(settings["input_file"]) or not exists(settings["input_file"]):
        raise ValueError("Input file does not exist")
    if not "pretagged" in settings.keys():
        logger.warning("Information on tagging not supplied")
        settings["pretagged"] = False
    return settings