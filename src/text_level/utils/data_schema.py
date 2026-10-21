from logging import getLogger
from stanza.models.common.doc import Sentence

logger = getLogger('preprocessor')

def get_pos_schema(sent: Sentence, sent_tokens: list[str]) -> dict:
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
                        "text": {"type": "string"},
                        "upos": {"type": "string"},
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