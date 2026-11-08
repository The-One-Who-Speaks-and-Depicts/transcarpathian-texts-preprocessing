"""
Contains the functions that facilitate dataset annotation, regardless
of the particular annotation type.

* annotate_with_stanza (str, str, str, stanza.Pipeline, bool) -> None:
a master function for Stanza-based annotation
* annotate_with_llm (dict, str, Callable, Callable, list) -> None:
a master function for LLM-based annotation
"""

from logging import getLogger
from ntpath import basename
from os.path import exists, isdir, isfile
from typing import Optional, NamedTuple

from collections.abc import Callable

from stanza import Pipeline
from stanza.models.common.doc import Sentence

from utils.llm_utils import load_llm_client, process_llm_errors, process_llm_response, transfer_llm_annotation, load_llm_params
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

class LLMAnnotationSettings(NamedTuple):
    prompt_assembler: Callable[[str, Sentence, str, bool | str], str]
    schema_assembler: Callable[[Sentence, Optional[list[str]]], dict]
    fields_to_gather: list[str]


def annotate_with_llm(
        settings: dict,
        exp_folder: str,
        stage: str,
        llm_annotation_settings: LLMAnnotationSettings
        ) -> None:
    """
    Takes an LLM annotation settings for a specific stage, and performs an annotation.

    Params:
        settings (dict): an experiment configuration object, provided by a user
        exp_folder (str): a folder that should contain experiment results
        prompt_assembler (Callable): a function that assembles a prompt
        schema_assembler (Callable): a function that assembles a data schema
        fields_to_gather (list[str]): a list of fields to annotate (e. g., ["lemma"])
    """
    logger.debug("Called with arguments %s", locals())
    if not bool(settings["input_file"]) or not bool(settings["pretagged"]) or not bool(
        settings["stage"]
        ):
        raise ValueError(f"Incorrect experiment settings object, received:\n{settings}")
    if not isinstance(
        settings["input_file"], str
        ) or not settings["input_file"] or not settings["input_file"].strip() or not exists(
            settings["input_file"]
        ) or not isfile(settings["input_file"]):
        raise ValueError("input_file parameter in settings should be a valid path to file, " \
                         f"received: {settings["input_file"]}")
    input_file_path = settings["input_file"]
    if not isinstance(pretagged, bool) and not isinstance(pretagged, str):
        raise TypeError(f"pretagged should be either a boolean or a string, received {pretagged}")
    pretagged = settings["pretagged"]
    if not isinstance(
        stage, str
    ) or not stage or not stage.strip():
        raise ValueError(f"Stage parameter is set incorrectly, received {stage}")
    doc, doc_for_processing = load_input_data(input_file_path, pretagged)
    params = load_llm_params(settings["llm_params"])
    client = load_llm_client(params.api, params.api_key, params.base_url)
    if not isinstance(exp_folder, str) or not exists(exp_folder) or not isdir(exp_folder):
        raise TypeError(f"Incorrectly set experiment folder, received:{exp_folder}")
    if not isinstance(llm_annotation_settings, LLMAnnotationSettings):
        raise TypeError("LLM annotation settings are not of respective type, " \
        f"received {llm_annotation_settings}")
    if not hasattr(
        llm_annotation_settings, "prompt_assembler"
        ) or not isinstance(llm_annotation_settings.prompt_assembler, Callable):
        raise TypeError("prompt_assembler is not a valid function, " \
        f"experiment settings: {llm_annotation_settings}")
    if not hasattr(
        llm_annotation_settings, "schema_assembler"
        ) or not isinstance(llm_annotation_settings.schema_assembler, Callable):
        raise TypeError("schema_assembler is not a valid function, " \
        f"experiment settings: {llm_annotation_settings}")
    if not hasattr(
        llm_annotation_settings, "fields_to_gather"
        ) or not isinstance(llm_annotation_settings.fields_to_gather, list) or not all(
        isinstance(i, str) for i in list
    ):
        raise ValueError("fields_to_gather is not a list of strings, " \
        f"received:\n{llm_annotation_settings.fields_to_gather}")
    logger.debug("LLM pipeline prepared, stage %s", stage)
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        sent_tokens = [w.text for w in sent.words]
        sent_tokenised = ' '.join(sent_tokens)
        prompt = llm_annotation_settings.prompt_assembler(
            params.prompt, sent, sent_tokenised, pretagged
            )        
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
                            response_format= llm_annotation_settings.schema_assembler(sent, None),
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
            llm_annotation_settings.fields_to_gather,
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