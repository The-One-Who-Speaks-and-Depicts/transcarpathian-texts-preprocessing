"""
Contains utility functions that may occur to be useful different
evaluation procedures.

* collect_labels (stanza.models.common.doc) -> dict: gathers a 
dictionary with unique morphosyntactic tags from the Stanza Document object. 
* robustness_score (Number, Number, Number) -> Number: evalutes the ability of a model
to perform transfer learning.
* collect_annotation (stanza.models.common.doc) -> list[AnnotationLabels]: collects
annotation for each token in the Stanza Document object.
* get_lcss_indices (str, str, bool, int) -> tuple[tuple[int, int], tuple[int, int]]: takes
two strings, finds their longest common substring, and returns its indices. In case the task
is to analyse lemmatisation, or some other task that requires finding common substrings that
are not random, the function uses a combination of a boolean flag to initiate the procedure,
and an integer (by default, 1) to set the minimal length of a common substring that is
not going to be considered random.
* single_point_accuracy (str, str) -> int: a simple wrapper for comparing
the equality of two strings.
"""

from logging import getLogger
from ntpath import basename
from numbers import Number
from os.path import join
from typing import NamedTuple

from collections.abc import Callable
import numpy as np

from stanza.models.common.doc import Document
from stanza.utils.conll import CoNLL

logger = getLogger('preprocessor')

def collect_labels(doc: Document) -> dict:
    """
    Takes a Stanza Document and transforms it into a dictionary
    that contains information on the parts-of-speech, morphological features
    and dependency relatonships occurring in the document.

    Params:
        doc (Document): a Stanza Document, gathered from a CoNLL-U file

    Returns:
        dict: a dictionary with three key-value pairs, each for the 
        respective part of morphosyntactic annotation: parts-of-speech,
        morphological features and dependency relationships
    """
    logger.debug("Input: %s", locals())
    if not isinstance(doc, Document):
        raise TypeError(f"doc should be a Document, received {doc}")
    result = {'pos': [], 'feats': ['_'], 'deprel': ['_']}    
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                if hasattr(word, 'pos'):
                    result['pos'].append(word.upos)
                if hasattr(word, 'feats'):
                    result['feats'].extend(
                        [i.split('=')[0] for i in word.feats.split('|') if i and i.strip()]
                        )
                if hasattr(word, 'deprel'):
                    result['deprel'].append(word.deprel)
    result['pos'] = list(set(result['pos']))
    result['feats'] = list(set(result['feats']))
    result['deprel'] = list(set(result['deprel']))
    logger.debug("Result: %s", result)
    return result

def robustness_score(
        original_score: Number, current_score: Number,
        robustness_coefficient: Number
        ) -> Number:
    """
    An evaluation metric that takes the score a model achieved on the evaluation subset of
    original data, the score it achieved on the dataset its acquired knowledge was used to tag,
    and a numeric coefficient. Using simple geometry-based representation of an ideal transfer
    and a real transfer, the metric returns a value that takes into the account both difference
    between an original score and a current score, and how well the model performed originally.

    Params:
        original_score (Number): the score a model achieved on the original data, preferrably,
        its evaluation subset
        current_score (Number): the score a model achieved on the data it got transferred to
        robustness_coefficient (Number): a non-negative non-zero number used to geometrically
        represent the difference between the performance of the model on the original dataset
        and the dataset its knowledge got transferred to

    Returns:
        Number: a metric value    
    """
    if not isinstance(original_score, Number) or original_score < 0:
        raise ValueError("original_score should be a non-negative number, " \
        f"received {original_score}")
    if not isinstance(current_score, Number) or current_score < 0:
        raise ValueError("current_score should be a non-negative number, " \
        f"received {current_score}")
    if not isinstance(robustness_coefficient, Number) or robustness_coefficient <= 0:
        raise ValueError("robustness_coefficient should be number and higher than zero, " \
        f"received {robustness_coefficient}")
    logger.debug("Input: %s", locals())
    ideal_vector = np.array([robustness_coefficient, 0])
    current_vector = np.array([robustness_coefficient, current_score - original_score])
    cos_sim = np.dot(
        ideal_vector, current_vector
        ) / (
            np.linalg.norm(ideal_vector) * np.linalg.norm(current_vector)
            )
    score_with_diff = original_score * (1 - cos_sim)
    is_current_score_higher_than_original = 1 if current_score >= original_score else -1
    resulting_score = score_with_diff * is_current_score_higher_than_original
    logger.debug("Robustness score between %s and %s with coefficient %s is %s",
                 original_score, current_score, robustness_coefficient,
                 resulting_score)
    return resulting_score

class AnnotationLabels(NamedTuple):
    """
    Contains the morphosyntactic information, extracted from
    a CoNLL-U document, or an annotation: parts-of-speech (pos),
    lemmatisation (lemma), morphological features (feats), syntactic head (head)
    and syntactic relationship between the token and the head (deprel).
    """
    pos: str
    lemma: str
    feats: str
    head: str
    deprel: str


def collect_annotation(doc: Document) -> list[AnnotationLabels]:
    """
    Collects annotation for each word in a Stanza Document object,
    basically flattening its structure for easier evaluation.

    Params:
        doc (Document): a Stanza Document, gathered from a CoNLL-U file
    
    Returns:
        list[AnnotationLabels]: a list of morphosyntactic annotation
        for each token in the original document
    """
    logger.info("Input: %s", locals())
    if not isinstance(doc, Document):
        raise TypeError(f"doc should be a Document, received {doc}")
    annotations = []
    for sent in doc.sentences:
        for token in sent.tokens:
            for word in token.words:
                pos = word.upos if word.upos is not None else '_'
                lemma = word.lemma if word.lemma is not None else '_'
                feats = word.feats if word.feats is not None else '_'
                head = word.head if word.head is not None else '_'
                deprel = word.deprel  if word.deprel is not None else '_'
                annotations.append(
                    AnnotationLabels(pos, lemma, feats, head, deprel)
                    )
    return annotations

class EvaluationDataset(NamedTuple):
    gold_doc: Document
    pred_doc: Document
    gold: list[AnnotationLabels]
    pred: list[AnnotationLabels]
    labels: dict

def prepare_annotation(gold_file_path: str, pred_file_path: str):
    gold_doc = CoNLL.conll2doc(gold_file_path)
    gold = collect_annotation(gold_doc)
    labels = collect_labels(gold)
    pred_doc = CoNLL.conll2doc(pred_file_path)
    pred = collect_annotation(pred_doc)
    return EvaluationDataset(gold_doc, pred_doc, gold, pred, labels)

def store_rapidity_rate(
        gold_with_rapidity: Document, gold_file_path: str, exp_folder: str
        ) -> None:
    name = basename(gold_file_path)
    file_to_store = join(exp_folder, f"rapidity_{name}")
    CoNLL.write_doc2conll(gold_with_rapidity, file_to_store)
    logger.info("See results of assigned rapidity in %s", file_to_store)

def get_lcss_indices(
        comparable: str,
        compared: str,
        ignore_if_not_at_beginning: bool = False,
        min_length: int = 1
        ) -> tuple[tuple[int, int], tuple[int, int]]:
    """
    Takes two strings and returns a tuple with indices
    of their first longest common substring
    from them, with possible regulations for the sequences of length 1.

    Arguments:
        comparable, compared (str): string to compare. The difference between
        comparable and compared is crucial, as if comparable contains two LCCs between it
        and compared, the algorithm returns only the indices of first LCC.
        ignore_if_not_at_beginning (bool): if the parameter is true, uses the special rule:
        if both the sequence do not start with it, ignores LCS.
        Mostly needed for lemmatisation comparison.
        min_length (int): defines the length of a sequence to ignore,
        if ignore_if_not_at_beginning is set to True. Mostly needed for lemmatisation comparison.

    Returns:
        tuple[tuple[int, int], tuple[int, int]]: indices of the longest common substring
        in both of the compared strings
    """
    logger.debug("Input: %s", locals())
    if not isinstance(compared, str):
        raise TypeError(f"The string compared should be of str type, received {compared}")
    if not isinstance(comparable, str):
        raise TypeError("The string to which comparison should be done," \
        f"should be of str type, received {compared}")
    if not isinstance(ignore_if_not_at_beginning, bool):
        raise TypeError("ignore_at_beginning should be a boolean value, " \
        f"received {ignore_if_not_at_beginning}")
    if not isinstance(min_length, int):
        raise TypeError(f"minimal length should be an integer, received {min_length}")
    if min_length < 0:
        raise ValueError(f"min_length should be a non-negative integer, received {min_length}")
    
    comparable_len, compared_len = len(comparable), len(compared)
    
    # two-dimensional array, X being length of first string, Y being length of second string
    iterative_matrix = [[0] * (compared_len + 1) for _ in range(comparable_len + 1)]
    
    lcs_length = 0
    lcs_end_compb = 0
    lcs_end_compd = 0
    
    for i in range(1, comparable_len + 1):
        for j in range(1, compared_len + 1):
            # search for the first coinciding symbol
            if comparable[i-1] == compared[j-1]:
                # while the symbols coincide,
                # progressively increment the value of current cs length
                iterative_matrix[i][j] = iterative_matrix[i-1][j-1] + 1
                # as soon as it gets bigger than lcs, assign it as lcs 
                if iterative_matrix[i][j] > lcs_length:
                    lcs_length = iterative_matrix[i][j]
                    lcs_end_compb = i
                    lcs_end_compd = j
    
    if lcs_length == 0:
        logger.debug("No common substring between %s and %s", comparable, compared)
        return ((-1, -1), (-1, -1))
    
    lcs_start_compb = lcs_end_compb - lcs_length
    lcs_start_compd = lcs_end_compd - lcs_length

    # Lemmatisation-specific part; basically, protects the algorithm from finding
    # a single (or more than single) random coincidence
    if ignore_if_not_at_beginning and lcs_length < min_length:
        compb_starts_wit_lcs = (lcs_start_compb == 0)
        compd_starts_with_lcs = (lcs_start_compd == 0)

        if not (compb_starts_wit_lcs and compd_starts_with_lcs):
            logger.debug("Common substring of length %s" \
            "considered to be random under the required minimal length of %s",
            lcs_length, min_length)
            return ((-1, -1), (-1, -1))
        
    # while returning, subtract 1 from end index, as it is by one bigger than the
    # actual index
    logger.debug("Common substring spans from %s to %s in %s and from %s to %s in %s",
                 lcs_start_compb, lcs_end_compb - 1, comparable,
                 lcs_start_compd, lcs_end_compd - 1, compared)
    return ((lcs_start_compb, lcs_end_compb - 1), (lcs_start_compd, lcs_end_compd - 1))

class ErrorsByPos(NamedTuple):
    incorrect_pos: list[AnnotationLabels]
    correct_pos: list[AnnotationLabels]
    all_errors: list[AnnotationLabels]

def single_point_accuracy(gold_val: str, pred_val: str) -> Number:
    """
    A wrapper of a simple string-to-string comparison.

    Params:
        gold_val (str): the first value (usually, the gold annotation label)
        pred_val (str): the second value (usually, the predicted label)
    
    Returns:
        Number: either 1 (strings are equal) or 0 (strings are not equal)
    """
    if not isinstance(gold_val, str):
        raise TypeError(f"The first value is not a string, received {gold_val}")
    if not isinstance(pred_val, str):
        raise TypeError(f"The first value is not a string, received {pred_val}")
    result = int(gold_val == pred_val)
    logger.debug("The comparison of %s and %s yields %s", gold_val, pred_val, result)
    return result

def collect_errors_by_pos(
        gold: list[AnnotationLabels],
        pred: list[AnnotationLabels],
        metric: Callable[[str, str], Number]
        ) -> ErrorsByPos:
    results_with_correct_pos = [
        metric(
            i[0][1], i[1][1]
            ) for i in zip(gold, pred) if i[0][1] != i[1][1] and i[0][0] == i[1][0]
        ]
    results_with_incorrect_pos = [
        metric(
            i[0][1], i[1][1]
            ) for i in zip(gold, pred) if i[0][1] != i[1][1] and i[0][0] != i[1][0]
        ]
    all_results = results_with_correct_pos + results_with_incorrect_pos
    return ErrorsByPos(
        results_with_correct_pos,
        results_with_incorrect_pos,
        all_results
    )