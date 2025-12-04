import logging

import re
import stanza
from stanza.utils.conll import CoNLL
import ntpath
from copy import deepcopy
import os

from Levenshtein import distance as lev
from pyjarowinkler.distance import get_jaro_distance as jw


from pos import collect_labels, robustness_score

logger = logging.getLogger('preprocessor')

def lemmatise_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk', pretagged: bool = True):
    logger.debug("%s is called with arguments %s", input_file_path, locals())
    name = ntpath.basename(input_file_path)
    stanza.download(lang)
    doc = CoNLL.conll2doc(input_file_path)
    doc_for_procesing = deepcopy(doc)
    for sent in doc_for_procesing.sentences:
        for token in sent.tokens:
            for word in token.words:
                misc_keys = word.misc.split('|')
                for key in misc_keys:
                    if 'wf' in key:
                        new_form = re.sub('\"', '', key)
                        new_form = re.sub('wf=', '', new_form)
                        word.text = new_form
                if not pretagged:
                    word.upos = '_'
                    word.feats = '_'
    logger.debug("%s loaded", input_file_path)
    nlp = stanza.Pipeline(
        lang=lang,
        processors=['tokenize', 'lemma'],
        use_gpu=True,
        tokenize_pretokenized=True,
        lemma_pretagged=True
        ) if pretagged else stanza.Pipeline(
            lang=lang,
            processors=['tokenize', 'pos', 'lemma'],
            use_gpu=True,
            tokenize_pretokenized=True
            )
    logger.debug("Lemmatisation tagging pipeline prepared, settings: %s", nlp)
    doc_processed = nlp(doc_for_procesing)
    for sent_idx, sent in enumerate(doc_processed.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, word in enumerate(token.words):
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
                if not pretagged:
                    doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    pretagging_info = 'gold-pos' if pretagged else 'silver-pos'
    file_to_store = os.path.join(exp_folder, f"lemma_{pretagging_info}_stanza_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_store)
    logger.debug("See results of lemmatisation in %s", file_to_store)
    file_to_edit = os.path.join(exp_folder, f"lemma_{pretagging_info}_gold_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_edit)
    logger.debug("Edit results of lemmatisation in %s", file_to_edit)


def get_lcss_indices(
        comparable: str,
        compared: str,
        regulate_one: bool = False
        ) -> tuple[tuple[int, int]]:
    """
    Takes two strings and returns a tuple with indices
    of their first longest common substring
    from them, with possible regulations for the sequences of length 1.

    Arguments:
        comparable, compared (str): string to compare. The difference between
        comparable and compared is crucial, as if comparable contains two LCCs between it
        and compared, the algorithm returns only the indices of first LCC.
        regulated_one (bool): if the parameter is true, uses the special set of rules: if both the 
        sequence do not start with it, ignores LCS. Mostly needed for lemmatisation comparison.

    Returns:
        tuple[tuple[int, int]]: indices of the longest common substring
        in both of the compared strings
    """
    # here, it is crucial to check
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
        return ((-1, -1), (-1, -1))
    
    lcs_start_compb = lcs_end_compb - lcs_length
    lcs_start_compd = lcs_end_compd - lcs_length
    
    if regulate_one and lcs_length == 1:
        compb_starts_wit_lcs = (lcs_start_compb == 0)
        compd_starts_with_lcs = (lcs_start_compd == 0)

        if not (compb_starts_wit_lcs and compd_starts_with_lcs):
            return ((-1, -1), (-1, -1))
        
    # while returning, subtract 1 from end index, as it is by one bigger than the
    # actual index
    return ((lcs_start_compb, lcs_end_compb - 1), (lcs_start_compd, lcs_end_compd - 1))

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


def assign_lemma_rapidity(pred: stanza.Document, gold: stanza.Document) -> stanza.Document:
    for sent_idx, sent in enumerate(pred.sentences):
        for token_idx, token in enumerate(sent.tokens):
            for word_idx, word in enumerate(token.words):
                gold_word = gold.sentences[sent_idx].tokens[token_idx].words[word_idx]
                rapidity_rate = show_lemmatiser_error_spots(word, gold_word)
                rate_param = f'LemmaErrorSpots={rapidity_rate}|TaggedLemma={word}'
                if gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc == None:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = rate_param
                else:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc + '|' + rate_param
    return gold


def evaluate_lemma(gold_file_path: str, pred_file_path: str, train_result: float, exp_folder: str) -> None:
    # TODO: the evaluation phase is mostly rehashing of the same code, I really dislike that
    logger.info("Lemma: starting comparison between %s and %s", gold_file_path, pred_file_path)
    gold = CoNLL.conll2doc(gold_file_path)
    labels = collect_labels(gold)["pos"]
    logger.info("PoS labels are %s", labels)
    pred = CoNLL.conll2doc(pred_file_path)
    gold_lemma = []
    for sent in gold.sentences:
        for token in sent.tokens:
            for word in token.words:
                gold_lemma.append((word.upos, word.lemma))
    pred_lemma = []
    for sent in pred.sentences:
        for token in sent.tokens:
            for word in token.words:
                pred_lemma.append((word.upos, word.lemma))
    # =================================OVERALL ACCURACY============================================================
    errata_with_correct_pos = [i for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] == i[1][0]]
    errata_with_incorrect_pos = [i for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] != i[1][0]]
    all_errata = errata_with_correct_pos + errata_with_incorrect_pos
    accuracy_score = len(all_errata)/len(gold_lemma) * 100
    logger.info("Accuracy score is %s%%", accuracy_score)
    if (train_result and train_result >= 0):
        logger.info('PoS robustness:  %.2f', robustness_score(train_result, accuracy_score))
    if (len(all_errata) > 0):
        correct_pos_errata_share = errata_with_correct_pos/all_errata * 100
        logger.info("Share of errata, when pos is not correct, is %s%%", correct_pos_errata_share)
        incorrect_pos_errata_share = errata_with_incorrect_pos/all_errata * 100
        logger.info("Share of errata, when pos is correct, is %s%%", incorrect_pos_errata_share)
    #===============================OVERALL LEVENSHTEIN===============================================================
    levenshtein_with_correct_pos = [lev(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] == i[1][0]]
    av_leveshtein_with_correct_pos = 0 if len(levenshtein_with_correct_pos) == 0 else sum(levenshtein_with_correct_pos)/len(levenshtein_with_correct_pos)
    logger.info("Average Levenshtein distance, when pos is correct, is %s", av_leveshtein_with_correct_pos)
    levenshtein_with_incorrect_pos = [lev(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] != i[1][0]]
    av_leveshtein_with_incorrect_pos = 0 if len(levenshtein_with_incorrect_pos) == 0 else sum(levenshtein_with_incorrect_pos)/len(levenshtein_with_incorrect_pos)
    logger.info("Average Levenshtein distance, when pos is incorrect, is %s", av_leveshtein_with_incorrect_pos)
    all_levs = levenshtein_with_correct_pos + levenshtein_with_incorrect_pos
    av_levs = 0 if len(all_levs) == 0 else sum(all_levs)/len(all_levs)
    logger.info("Average Levenshtein distance is %s", av_levs)
    #===============================OVERALL JARO=======================================================================
    jw_with_correct_pos = [jw(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] == i[1][0]]
    av_jw_with_correct_pos = 0 if len(jw_with_correct_pos) == 0 else sum(jw_with_correct_pos)/len(jw_with_correct_pos)
    logger.info("Average Jaro-Winkler distance, when pos is correct, is %s", av_jw_with_correct_pos)
    jw_with_incorrect_pos = [jw(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] != i[1][0]]
    av_jw_with_incorrect_pos = 0 if len(jw_with_incorrect_pos) == 0 else sum(jw_with_incorrect_pos)/len(jw_with_incorrect_pos)
    logger.info("Average Jaro-Winkler distance, when pos is incorrect, is %s", av_jw_with_incorrect_pos)
    all_jws = jw_with_correct_pos + jw_with_incorrect_pos
    all_jws = 0 if len(all_jws) == 0 else sum(all_jws)/len(all_jws)
    logger.info("Average Jato-Winkler distance is %s", all_jws)
    #======================================SCORING BY POS==============================================================
    for pos in labels:
        #============================================FILTER============================================================
        lemma_by_pos = [i for i in zip(gold_lemma, pred_lemma) if i[0][0] == pos]
        #===========================================ACCURACY===========================================================
        errata_with_correct_pos = [i for i in lemma_by_pos if i[0][1] != i[1][1] and i[0][0] == i[1][0]]
        errata_with_incorrect_pos = [i for i in lemma_by_pos if i[0][1] != i[1][1] and i[0][0] != i[1][0]]
        all_errata = errata_with_correct_pos + errata_with_incorrect_pos
        accuracy_score = len(all_errata)/len(gold_lemma) * 100
        logger.info("Accuracy score for %s is %s%%", pos, accuracy_score)
        if (len(all_errata) > 0):
            correct_pos_errata_share = errata_with_correct_pos/all_errata * 100
            logger.info("Share of errata, when %s is not correct, is %s%%", pos, correct_pos_errata_share)
            incorrect_pos_errata_share = errata_with_incorrect_pos/all_errata * 100
            logger.info("Share of errata, when %s is correct, is %s%%", pos, incorrect_pos_errata_share)
        #==========================================LEVENSHTEIN=========================================================
        levenshtein_with_correct_pos = [lev(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] == i[1][0]]
        av_leveshtein_with_correct_pos = 0 if len(levenshtein_with_correct_pos) == 0 else sum(levenshtein_with_correct_pos)/len(levenshtein_with_correct_pos)
        logger.info("Average Levenshtein distance, when %s is correct, is %s", pos, av_leveshtein_with_correct_pos)
        levenshtein_with_incorrect_pos = [lev(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] != i[1][0]]
        av_leveshtein_with_incorrect_pos = 0 if len(levenshtein_with_incorrect_pos) == 0 else sum(levenshtein_with_incorrect_pos)/len(levenshtein_with_incorrect_pos)
        logger.info("Average Levenshtein distance, when %s is incorrect, is %s", pos, av_leveshtein_with_incorrect_pos)
        all_levs = levenshtein_with_correct_pos + levenshtein_with_incorrect_pos
        av_levs = 0 if len(all_levs) == 0 else sum(all_levs)/len(all_levs)
        logger.info("Average Levenshtein distance for %s is %s", pos, av_levs)
        #==========================================JARO=========================================================
        jw_with_correct_pos = [jw(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] == i[1][0]]
        av_jw_with_correct_pos = 0 if len(jw_with_correct_pos) == 0 else sum(jw_with_correct_pos)/len(jw_with_correct_pos)
        logger.info("Average Jaro-Winkler distance, when %s is correct, is %s", pos, av_jw_with_correct_pos)
        jw_with_incorrect_pos = [jw(i[0][1], i[1][1]) for i in zip(gold_lemma, pred_lemma) if i[0][1] != i[1][1] and i[0][0] != i[1][0]]
        av_jw_with_incorrect_pos = 0 if len(jw_with_incorrect_pos) == 0 else sum(jw_with_incorrect_pos)/len(jw_with_incorrect_pos)
        logger.info("Average Jaro-Winkler distance, when %s is incorrect, is %s", pos, av_jw_with_incorrect_pos)
        all_jws = jw_with_correct_pos + jw_with_incorrect_pos
        all_jws = 0 if len(all_jws) == 0 else sum(all_jws)/len(all_jws)
        logger.info("Average Jato-Winkler distance for %s is %s", pos, all_jws)
    #==============================================STORE HEAT===================================================
    gold_with_rapidity = assign_lemma_rapidity(pred, gold)
    name = ntpath.basename(gold_file_path)
    file_to_store = os.path.join(exp_folder, f"rapidity_{name}")
    CoNLL.write_doc2conll(gold_with_rapidity, file_to_store)
    logger.info("See results of assigned rapidity in %s", file_to_store)
    logger.info("Lemmatisation evaluation is finished")

    
