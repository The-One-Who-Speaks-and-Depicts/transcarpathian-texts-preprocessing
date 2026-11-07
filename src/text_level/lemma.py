"""
Contains the functions, specific for lemmatisation annotation and evaluation.

* get_lemmatisation_pipeline (str, bool) -> Pipeline: prepares a Stanza processing pipeline
for lemmatisation.
* get_lemma_prompt (str, Sentence, str, bool | str) -> str: prepares a prompt for
LLM-assisted lemmatisation.
"""

from logging import getLogger
from numbers import Number

from collections.abc import Callable

from stanza import download, Pipeline
from stanza.models.common.doc import Document, Sentence
from Levenshtein import distance as lev
from pyjarowinkler.distance import get_jaro_distance as jw


from utils.eval_utils import prepare_annotation, robustness_score, store_rapidity_rate, AnnotationLabels, get_lcss_indices, single_point_accuracy, collect_errors_by_pos
from utils.llm_utils import check_prompt_parameters

logger = getLogger('preprocessor')

def get_stanza_lemmatisation_pipeline(lang: str = 'uk', pretagged: bool = True) -> Pipeline:
    """
    Returns a Stanza lemmatisation Pipeline for provided language and previously
    available information.

    Params:
        lang (str): a language code (see Stanza documentation for the list of available languages)
        pretagged (bool): information on whether morphological tagging already happened

    Returns:
        stanza.Pipeline: a Stanza Pipeline for lemmatisation
    """
    logger.debug("Input: %s", locals())
    download(lang)
    nlp = Pipeline(
        lang=lang,
        processors=['tokenize', 'lemma'],
        use_gpu=True,
        tokenize_pretokenized=True,
        lemma_pretagged=True
        ) if pretagged else Pipeline(
            lang=lang,
            processors=['tokenize', 'pos', 'lemma'],
            use_gpu=True,
            tokenize_pretokenized=True
            )
    return nlp 

def get_lemma_prompt(
        prompt_template: str,
        sent: Sentence, sent_tokenised: str,
        pretagged: bool | str = True
        ) -> str:
    """
    Takes a prompt template, information about a sentence to tag (including its tokenised form),
    and information on whether morphological annotation is available to prepare
    a prompt for LLM-based lemmatisation.

    Params:
        prompt_template (str): a prompt of a template
        sent (stanza.models.commom.doc.Sentence): a CoNLL-U sentence
        sent_tokenised (str): tokens of a sentence, stored in sent
        pretagged (bool | str): information on the previously performed annotation available:
            - True: tokens and morphological tags
            - False: only tokens
    
    Returns:
        str: a prompt prepared for a model
    """
    check_prompt_parameters(prompt_template, sent, sent_tokenised, pretagged)
    prompt = prompt_template.format(
        sent=f"{sent_tokenised}\nTokens:{[str(w.id) + ': ' + w.text + ' ' + (
            w.pos if w.pos is not None and w.pos.strip() else '_'
            ) + ' ' + (
                w.feats if w.feats is not None and w.feats.strip() else '_'
                ) +  '\n' for w in sent.words]}\n"
        )
    logger.debug("Prompt is assembled:\n %s", prompt)
    return prompt

def show_lemmatiser_error_spots(pred: str, gold: str) -> str:
    if pred == gold:
        return '_' * len(gold)
    lcs_pred_indices = get_lcss_indices(pred, gold, True)[0]
    pred_to_list = list(pred)
    if lcs_pred_indices[0] != 0:
        for i in range(0, lcs_pred_indices[0]):
            pred_to_list[i] = 'X'
    for i in range(lcs_pred_indices[0], lcs_pred_indices[1] + 1):
        pred_to_list[i] = '_'
    if lcs_pred_indices[1] + 1 < len(pred):
        for i in range(lcs_pred_indices[1] + 1, len(pred)):
            pred_to_list[i] = 'X'
    return ''.join(pred_to_list)


def assign_lemma_rapidity(pred: Document, gold: Document) -> Document:
    for sent_idx, sent in enumerate(pred.sentences):
        for token_idx, token in enumerate(sent.tokens):
            for word_idx, word in enumerate(token.words):
                gold_word = gold.sentences[sent_idx].tokens[token_idx].words[word_idx]
                rapidity_rate = show_lemmatiser_error_spots(word.lemma, gold_word.lemma)
                rate_param = f'LemmaErrorSpots={rapidity_rate}|TaggedLemma={word.lemma}'
                if gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc == None:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = rate_param
                else:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc + '|' + rate_param
    return gold

def score_string_similarity_measure(
        gold: list[AnnotationLabels],
        pred: list[AnnotationLabels],
        str_s_measure: Callable[[str, str], Number],
        metric_name: str,
        pos: str = 'overall part-of-speech tagging'
) -> None:
    results = collect_errors_by_pos(gold, pred, str_s_measure)
    av_str_s_with_correct_pos = "-1 (no examples found)" if len(results.correct_pos) == 0 else sum(results.correct_pos)/len(results.correct_pos)
    logger.info("Average %s, when %s is correct, is %s", metric_name, pos, av_str_s_with_correct_pos)
    av_str_s_with_incorrect_pos = "-1 (no examples found)" if len(results.incorrect_pos) == 0 else sum(results.incorrect_pos)/len(results.incorrect_pos)
    logger.info("Average %s, when %s is incorrect, is %s", metric_name, pos, av_str_s_with_incorrect_pos)
    av_levs = "-1 (no examples found)" if len(results.all_errors) == 0 else sum(results.all_errors)/len(results.all_errors)
    logger.info("Average %s is %s", metric_name, av_levs)    
    
    

def score_lemmatisation_metrics(gold: list[AnnotationLabels], pred: list[AnnotationLabels], train_result: float, pos: str = 'overall part-of-speech tagging') -> None:
    results = collect_errors_by_pos(gold, pred, single_point_accuracy)
    accuracy_score_val = (len(gold) - len(results.all_errors))/len(gold) * 100
    logger.info("Accuracy score is %s%%", accuracy_score_val)
    if (train_result and train_result >= 0):
        logger.info('Lemma robustness:  %.2f', robustness_score(train_result, accuracy_score_val))
    if (len(results.all_errors) > 0):
        correct_pos_errata_share = len(results.correct_pos)/len(results.all_errors) * 100
        logger.info("Share of errata, when %s is correct, is %s%%", pos, correct_pos_errata_share)
        incorrect_pos_errata_share = len(results.incorrect_pos)/len(results.all_errors) * 100
        logger.info("Share of errata, when %s is not correct, is %s%%", pos, incorrect_pos_errata_share)
    score_string_similarity_measure(gold, pred, lev, "Levenshtein distance", pos)
    score_string_similarity_measure(gold, pred, jw, "Jaro-Winkler distance", pos)


def evaluate_lemma(gold_file_path: str, pred_file_path: str, train_result: float, exp_folder: str) -> None:
    logger.debug("Called with arguments %s", locals())
    logger.info("Lemma: starting comparison between %s and %s", gold_file_path, pred_file_path)
    dataset_for_annotation = prepare_annotation(gold_file_path, pred_file_path)
    logger.info("PoS labels are %s", dataset_for_annotation.labels['pos'])
    score_lemmatisation_metrics(dataset_for_annotation.gold, dataset_for_annotation.pred, train_result)
    for pos in dataset_for_annotation.labels['pos']:
        gold = [i for i in dataset_for_annotation.gold if i.pos == pos]
        pred = [i for i in dataset_for_annotation.pred if i.pos == pos]
        score_lemmatisation_metrics(gold, pred, pos=pos)
    gold_with_rapidity = assign_lemma_rapidity(dataset_for_annotation.pred_doc, dataset_for_annotation.gold_doc)
    store_rapidity_rate(gold_with_rapidity, gold_file_path, exp_folder)
    logger.info("Lemmatisation evaluation is finished")

    
