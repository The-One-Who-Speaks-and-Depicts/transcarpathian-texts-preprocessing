"""
Contains the functions that facilitate dataset annotation, regardless
of the particular annotation type.

* annotate_with_stanza (str, str, str, stanza.Pipeline, bool) -> None:
a master function for Stanza-based annotation
"""

from logging import getLogger
from ntpath import basename
from typing import Optional

from collections.abc import Callable

from stanza import Pipeline
from stanza.models.common.doc import Sentence

from utils.llm_utils import load_llm_client, ClientParams, process_llm_errors, process_llm_response, transfer_llm_annotation
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

# TODO: draft function
def annotate_with_llm(input_file_path: str, exp_folder: str, stage: str, prompt_assembler: Callable[[str, Sentence, str, bool | str], str], schema_assembler: Callable[[Sentence, Optional[list[str]]], dict], fields_to_gather: list[str], params: ClientParams = ClientParams(), pretagged: bool | str = True) -> None:
    logger.debug("Called with arguments %s", locals())
    doc, doc_for_processing = load_input_data(input_file_path)
    client = load_llm_client(params.api, params.api_key, params.base_url)
    logger.debug("LLM pipeline prepared, stage %s", stage)
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        sent_tokens = [w.text for w in sent.words]
        sent_tokenised = ' '.join(sent_tokens)
        prompt = prompt_assembler(params.prompt, sent, sent_tokenised, pretagged)        
        annotated_sent = {}
        retries = 0
        while True:
            try:
                response = None
                response_content = None
                match params.api:
                    case "OpenAI":           
                        response = client.chat.completions.create(
                            model=params.model,
                            messages=[
                                {"role": "system", "content": ""},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=params.temperature,
                            seed=params.seed,
                            response_format= schema_assembler(sent, None),
                            stream=False
                        )
                        response_content = response.choices[0].messsage.content
                    case _:
                        raise AttributeError(f"Invalid LLM call API type, received {params.api}")
                annotated_sent = process_llm_response(
                    sent,
                    sent_tokenised,
                    response,
                    response_content
                    )
                break
            except ValueError as e:
                retries, prompt = process_llm_errors(
                    e,
                    retries, params.retries,
                    sent_tokenised,
                    prompt.format(sent=sent_tokenised)
                    )
                continue

        doc_for_processing = transfer_llm_annotation(
            doc_for_processing,
            sent_idx,
            sent,
            fields_to_gather,
            annotated_sent
        )
    doc_for_processing = restore_original_tags(doc_for_processing, doc, pretagged)
    pretagging_info = None
    match pretagged:
        case 'detokenised':
            pretagging_info = 'gold_detokenised'
        case True:
            pretagging_info = 'gold'
        case _:
            pretagging_info = 'silver'
    store_predictions(f"{stage}_{pretagging_info}", params.model, exp_folder, basename(input_file_path), doc_for_processing)