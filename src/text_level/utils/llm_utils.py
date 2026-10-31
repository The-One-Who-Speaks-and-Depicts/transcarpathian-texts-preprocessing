
"""
Contains functions that aid in LLM-based annotation.

* load_llm_params (dict) -> ClientParams: checks the initial configuration
for the specific parameters to run an LLM with, inserting the default ones where possible
* transfer_llm_annotation (Document, int, Sentence, list, dict) -> Document: 
takes a Sentence with an integer index in a stanza Document object and annotates the fields,
determined by a corresponding list, with LLM annotation results,
provided through a dictionary that has integers (token indices) 
as keys and dictionaries for each token as values. The values contain field names as keys and
suggested annotations as values.
* process_llm_errors (Exception, int, int, str, str) -> tuple[int, str]:
takes data for error processing, returns updated retries counter and prompt.
* process_llm_response(Sentence, list[str], dict, str) -> dict: takes a sentence and its tokens,
alongside with the whole response of an LLM and its text content, to partailly clean it,
perform a preliminary check on correctness and parse it into a JSON object
that the code down the line may use to annotate the data. 
"""
from copy import deepcopy
from dataclasses import dataclass
from json import loads
from logging import getLogger
from os import environ
from os.path import isfile
from re import sub

from openai import OpenAI

from stanza.models.common.doc import Document, Sentence

logger = getLogger('preprocessor')
@dataclass
class ClientParams():
    api: str = ""
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    seed: int = 1590
    retries: int = 3
    temperature: float = 0
    prompt: str = ""


def load_prompt(user_params: dict) -> str:
    logger.info("Input: %s", locals())
    if bool(user_params["prompt"]):
        if isfile(user_params["prompt"]):
            logger.info("Loading user-defined prompt from %s", user_params["prompt"])
            return open(user_params["prompt"], 'r', encoding='utf-8').read()
        logger.info("Loading user-defined prompt directly from config, value is %s", user_params["prompt"])
        return user_params["prompt"]
    default_prompt = "Generate Universal Dependencies CoNLL-U from {sent}"
    logger.warning("No user-defined prompt, using default prompt:\n%s", default_prompt)
    return default_prompt
            

def load_llm_params(settings: dict) -> ClientParams:
    """
    Takes an experiment configuration object and returns an LLM client parameters object.

    Params:
        settings (dict): an experiment configuration object, provided by a user

    Returns:
        ClientParams: a ClientParams object that stores a set of parameters that 
        regulate an LLM run in the experiment for the reproducibility purposes
    """
    logger.info("Input: %s", locals())
    if not bool(settings["llm_params"]):
        raise ValueError("LLM parameters not given")
    user_params = settings["llm_params"]
    params_set = ClientParams(
        api = user_params["api"] if bool(user_params["api"]) else "OpenAI",
        api_key = environ.get("API_KEY") if "API_KEY" in environ else "not-provided",
        base_url=user_params["base_url"] if bool(user_params["base_url"]) else "http://localhost:1234/v1/",
        model=user_params["model"] if bool(user_params["model"]) else "google/gemma-3en-4b",
        retries=user_params["retries"] if bool(user_params["retries"]) else 3,
        seed=user_params["seed"] if bool(user_params["seed"]) else 1590,
        temperature=user_params["temperature"] if bool(user_params["temperature"]) else 0,
        prompt=load_prompt(user_params)
    )
    logged_params = deepcopy(params_set)
    logged_params.api_key = '******' + logged_params.api_key[-6:] 
    logger.info("The LLM tagging parameters set to %s", logged_params)
    return params_set




def load_llm_client(api: str, key: str, url: str) -> OpenAI:
    match api:
        case "OpenAI":            
            client = OpenAI(
                api_key=key,
                base_url=url
            )
            logger.info("Loading an OpenAI-like API client, base_url: %s")
            return client
        case _:
            raise ValueError(f"Incorrect API type, received {api}")

def process_llm_response(
        sent: Sentence, sent_tokenised: list[str], response: dict, response_content: str
        ) -> dict:
    """
    Transforms data that an LLM annotated, alongside with its full response
    and the text content of this response that contains the annotation the LLM performed
    into a dictionary that contains annotation in the structured format.

    Params:
        sent (Sentence): a CoNLL-U sentence
        sent_tokenised (list[str]): a list of tokens from the sentence
        response (dict): a full LLM response
        response_content (str): a text part of an LLM response
    
    Returns:
        dict: a dictionary with each item representing a token from the original sentence,
        its id being a key, and its LLM-performed annotation being a value dictionary,
        with annotated CoNLL-U fields ("pos", "feats", "lemma", "head", "deprel") as keys
        and LLM predictions as values
    """
    if not isinstance(sent, Sentence):
        raise TypeError(f"sent is not a CoNLL-U sentence in Stanza format, received {sent}")
    if not isinstance(sent_tokenised, list) or not all(
        isinstance(i, str) for i in sent_tokenised
    ):
        raise TypeError(f"sent_tokenised should be a list of strings, received {sent_tokenised}")
    if not isinstance(response, dict):
        raise TypeError(f"LLM response should be a dictionary, received {response}")
    if not isinstance(response_content, str):
        raise TypeError(f"LLM text response should be a string, received {response_content}")
    logger.debug('LLM response for %s:\n%s', sent, response) 
    logger.debug('LLM annotation for tokens %s:\n%s', sent_tokenised, response_content)
    # small models tend to incontrollably generate code blocks 
    # designations around the actual output
    cleared_content = sub(r'```(json)?\n?', '', response_content)
    annotated = loads(cleared_content)
    if len(annotated) != len(sent.words):
        raise ValueError(
            f"Token count mismatch: expected {len(sent.words)} from  {sent_tokenised}, "
            f"received {len(annotated)}"
        )
    logger.debug("LLM response processed correctly")
    return annotated


def process_llm_errors(
        e: Exception,
        retries: int,
        max_retries: int,
        sent_tokenised: str,
        original_prompt: str
        ) -> tuple[int, str]:
    """
    Takes the information on the LLM error and its processing to update prompt
    and retries counter.

    Params:
        e (Exception): the received exception, usually a ValueError
        retries (int): the current value of the counter that scores how many times
        an LLM attempted to perform the task
        max_retries (int): the maximum possible value of the counter that scores how many times
        an LLM attempted to perform the task
        sent_tokenised (str): a tokenised sentence
        original_prompt (str): a prompt used in the previous run

    Returns:
        int: an updated counter that scores how many times
        an LLM attempted to perform the task
        str: a prompt updated given the received exception
    """
    logger.error(f"Invalid LLM output, the error is {e}")
    retries = retries + 1
    if retries > max_retries:
        raise ValueError("Too many retries")
    prompt = f"The result of {sent_tokenised} processing was incorrect, " \
    f"there was an exception {e}." \
    f"Please, re-perform the task, keeping the exception in mind. The task is {original_prompt}"
    logger.debug("Retry: %s out of %s, \nPrompt:\n%s", retries, max_retries, prompt)
    return retries, prompt

def transfer_llm_annotation(
        doc_for_processing: Document,
        sent_idx: int,
        sent: Sentence,
        fields_to_annotate: list[str],
        annotation: dict[int, dict[str, str]]
) -> Document:
    """
    Transfers morphosyntactic annotation from a dictionary, produced by an LLM,
    to a CoNLL-U document.

    Params:
        doc_for_processing (Document): the CoNLL-U document, presented through
        a stanza.models.common.doc.Document class
        sent_idx (int): the index of an analysed sentence in the document
        sent (Sentence): the analysed sentence
        fields_to_annotate (list[str]): a list of CoNLL-U fields
        words (dict): a dictionary with LLM-based annotation results, with
        token indices as keys and annotations as values. The annotations are
        dictionaries with fields names as keys, and LLM suggestions for annotation
        as values.

    Returns:
        Document: the annotated CoNLL-U document        
    """
    logger.info("Input: %s", locals())
    if not isinstance(doc_for_processing, Document):
        raise TypeError("Incorrect object for a document annotation," \
        f"received {doc_for_processing}")
    if not isinstance(sent_idx, int):
        raise TypeError(f"{sent_idx} is not an integer, when expected")
    if not isinstance(sent, Sentence):
        raise TypeError(f"Incorrect sentence object, received {sent}")
    if not isinstance(fields_to_annotate, list) and not all(
        isinstance(i, str) for i in fields_to_annotate
    ):
        raise TypeError(f"List of fields has incorrect values, received {fields_to_annotate}")
    if not isinstance(annotation, dict) or not all(
        isinstance(i, int) and (
            isinstance(k, str) for k in annotation[i].keys()
            ) and (
                isinstance(v, str) for v in list(annotation[i].values())
                ) for i in annotation.keys() 
    ):
        raise TypeError(f"Incorrect LLM response format, received:\n{annotation}")
    for tkn_idx, token in enumerate(sent.tokens):
        for wrd_idx, _ in enumerate(token.words):
            current_word = annotation[str(tkn_idx+1)]
            for field in fields_to_annotate:
                setattr(
                    doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx],
                    field,
                    getattr(current_word, field) if hasattr(current_word, field) else '_'
                )
    logger.info("Annotation of sentence %s is finished, the product is %s",
                sent,
                doc_for_processing.sentences[sent_idx])
    return doc_for_processing