
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