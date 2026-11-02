"""
Contains the functions that facilitate dataset annotation, regardless
of the particular annotation type.

* annotate_with_stanza (str, str, str, stanza.Pipeline, bool) -> None:
a master function for Stanza-based annotation
"""

from logging import getLogger
from ntpath import basename

from stanza import Pipeline

from utils.tagging_utils import store_predictions, load_input_data, restore_original_tags

logger = getLogger('preprocessor')

def annotate_with_stanza(
        input_file_path: str, exp_folder: str, stage: str, nlp: Pipeline, pretagged: bool = True
        ) -> None:
    """
    Takes experiment settings and performs annotation with help of a Stanza package.

    Params:
        input_file_path (str): a path to a file to annotate
        exp_folder (str): a folder that should contain experiment results
        stage (str): an exact stage of annotation (usually, pos/lemma/depparse)
        nlp (stanza.Pipeline) a Stanza Pipeline for this stage
        pretagged (bool): information on whether there was an annotation for an earlier stage
        (pos for lemma, for instance)
    """
    logger.debug("Called with arguments %s", locals())
    if not isinstance(nlp, Pipeline):
        raise ValueError(f"nlp is not a stanza.Pipeline object, received {nlp}")
    doc, doc_for_processing = load_input_data(input_file_path, pretagged)
    logger.debug("%s loaded", input_file_path)    
    logger.debug("%s pipeline prepared, settings:\n %s", stage, nlp)
    doc_processed = nlp(doc_for_processing)
    doc_processed = restore_original_tags(doc_processed, doc, pretagged)
    pretagging_info = 'gold' if pretagged else 'silver'
    store_predictions(f"{stage}_{pretagging_info}", "stanza", exp_folder, basename(input_file_path), doc_processed)
