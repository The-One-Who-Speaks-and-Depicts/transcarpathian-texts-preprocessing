"""
Stores the JSON data schemas for LLM-based tagging.

* get_pos_schema (stanza.models.common.doc.Sentence, list[str] -> dict):
takes a CoNLL-U sentence and a list of its tokens, and returns a schema
for morphological tagging (including both UPOS and feats detection)
* get_lemma_schema (stanza.models.common.doc.Sentence, list[str] -> dict):
takes a CoNLL-U sentence and a list of its tokens, and returns a schema
for lemmatisation
"""

from logging import getLogger
from stanza.models.common.doc import Sentence

logger = getLogger('preprocessor')

def get_pos_schema(sent: Sentence, sent_tokens: list[str]) -> dict:
    """
    Returns a schema for morphological tagging, based on the provided 
    sentence and its tokens.

    Params:
        sent (stanza.models.common.doc.Sentence): a CoNLL-U sentence to annotate
        sent_tokens (list[str]): the sentence in a pre-tokenized form, but without
        any auxiliary information (e. g., from metadata)
    
    Returns:
        dict: a JSON schema that one can afterwards feed to the annotating LLM
        and which constraints its output to the exact same tokens it received, 
        annotated with the exact set of universal part-of-speech tag and the set of 
        morphological features, determined by the model, but hinted at the prompt.
    """
    logger.info('Input: %s', locals())
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ud_pos",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    str(w.id): {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "enum": [w.text]
                            },
                        "upos": {
                            "type": "string",
                            "enum": ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
                                     'NUM', 'PART', 'PRON', 'PROPN','PUNCT', 'SCONJ', 'SYM', 
                                     'VERB', 'X']
                            },
                        "feats": {"type": "string"}
                    },
                    "required": ["text", "upos", "feats"],
                    "additionalProperties":False,
                    } for w in sent.words
                },
                "required": [w.id for w in sent.words],
                "additionalProperties": False,
                "minItems": len(sent_tokens),
                "maxItems": len(sent_tokens)
            },
        }
    }
    logger.info("Generated schema:\n%s", schema)
    return schema

def get_lemma_schema(sent: Sentence, sent_tokens: list[str]) -> dict:
    """
    Returns a schema for lemmatisation, based on the provided 
    sentence and its tokens.
    
    Params:
        sent (stanza.models.common.doc.Sentence): a CoNLL-U sentence to annotate
        sent_tokens (list[str]): the sentence in a pre-tokenized form, but without
        any auxiliary information (e. g., from metadata)
    
    Returns:
        dict: a JSON schema that one can afterwards feed to the annotating LLM
        and which constraints its output to the exact same tokens it received, 
        annotated with the lemmatisation, determined by the model, hinted at the prompt.
    """
    logger.info("Input: %s", locals())
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "ud_lemma",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    str(w.id): {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "enum": [w.text]
                        },
                        "lemma": {"type": "string"},
                    },
                    "required": ["text", "lemma"],
                    "additionalProperties":False,
                    } for w in sent.words
                },
                "required": [w.id for w in sent.words],
                "additionalProperties": False,
                "minItems": len(sent_tokens),
                "maxItems": len(sent_tokens)
            },
        }
    }
    logger.info("Generated schema:\n%s", schema)
    return schema