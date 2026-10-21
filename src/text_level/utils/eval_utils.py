from logging import getLogger
from typing import NamedTuple

import numpy as np
from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

logger = getLogger('preprocessor')

def collect_labels(doc: Document) -> dict:
    result = {'pos': [], 'feats': ['_'], 'deprel': ['_']}    
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                result['pos'].append(word.upos)
                if word.feats:
                    result['feats'].extend([i.split('=')[0] for i in word.feats.split('|') if i and i.strip()])
                result['deprel'].append(word.deprel)
    result['pos'] = list(set(result['pos']))
    result['feats'] = list(set(result['feats']))
    result['deprel'] = list(set(result['deprel']))
    return result

def robustness_score(original_score: float, current_score: float, end: int):
    ideal_vector = np.array([end, 0])
    current_vector = np.array([end, current_score - original_score])
    cos_sim = np.dot(ideal_vector, current_vector) / (np.linalg.norm(ideal_vector) * np.linalg.norm(current_vector))
    score_with_diff = original_score * (1 - cos_sim)
    robustness_coefficient = 1 if current_score >= original_score else -1
    return score_with_diff * robustness_coefficient

class AnnotationLabels(NamedTuple):
    pos: str
    lemma: str
    feats: str
    head: str
    deprel: str


def collect_annotation(doc: Document) -> list[AnnotationLabels]:
    logger.info("Input: %s", locals())
    annotations = []
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                pos = word.upos if word.upos is not None else '_'
                lemma = word.lemma if word.lemma is not None else '_'
                feats = word.feats if word.feats is not None else '_'
                head = word.head if word.head is not None else '_'
                deprel = word.deprel  if word.deprel is not None else '_'
                annotations.append(AnnotationLabels(pos, lemma, feats, head, deprel))
    return annotations

class EvaluationDataset(NamedTuple):
    gold_doc: Document
    pred_doc: Document
    gold: list[AnnotationLabels]
    pred: list[AnnotationLabels]
    labels: list[str]

def prepare_annotation(gold_file_path: str, pred_file_path: str):
    gold_doc = CoNLL.conll2doc(gold_file_path)
    gold = collect_annotation(gold_doc)
    labels = collect_labels(gold)
    pred_doc = CoNLL.conll2doc(pred_file_path)
    pred = collect_annotation(pred_doc)
    return EvaluationDataset(gold_doc, pred_doc, gold, pred, labels)