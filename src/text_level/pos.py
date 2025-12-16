import logging

import re

from sklearn.metrics import precision_recall_fscore_support
import stanza
from stanza.utils.conll import CoNLL
import ntpath
from copy import deepcopy
import os

import numpy as np
from numpy.linalg import norm

logger = logging.getLogger('preprocessor')

def pos_tag_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk') -> None:
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
    logger.debug("%s loaded", input_file_path)
    nlp = stanza.Pipeline(lang=lang, processors=['tokenize', 'pos'], use_gpu=True, tokenize_pretokenized=True)
    logger.debug("Part-of-speech tagging pipeline prepared")
    logger.debug("Called with arguments %s", locals())
    doc_processed = nlp(doc_for_procesing)
    for sent_idx, sent in enumerate(doc_processed.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, word in enumerate(token.words):
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
    file_to_store = os.path.join(exp_folder, f"pos_stanza_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_store)
    logger.info("See results of morphological tagging in %s", file_to_store)
    file_to_edit = os.path.join(exp_folder, f"pos_gold_{name}")
    CoNLL.write_doc2conll(doc_processed, file_to_edit)
    logger.info("Edit results of morphological tagging in %s", file_to_edit)


def collect_labels(doc: stanza.Document) -> dict:
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

def score_for_single_pos(pred: list, true: list, label: str) -> None:
    overall_size = len(true)
    size_of_label = len([i for i in true if i == label])
    share = size_of_label/overall_size
    correct = len([i for i in zip(pred, true) if i[1] == label and i[0] == i[1]])
    # I do not score F1 for single label, because for labels with low share it is going to be very low-informative
    acc = correct/size_of_label
    logger.info("Accuracy for %s is %.2f%s, the share of %s in dataset is %.2f%s", label, acc*100, "%", label, share*100, "%")

def score_single_feat_acc(key: str, value: str, true: dict) -> int:
    if key not in true.keys():
        return 0
    if true[key] != value:
        return 0.5
    return 1

def score_feats_acc(pred: str | None, true: str | None) -> float:
    logger.debug("Starting scoring accuracy for %s and %s", pred, true)
    if (pred == true):
        return 1
    if pred == None:
        if true == None:
            return 1
        return 0
    if true == None:
        return 0
    pred_items = [i for i in pred.split('|') if i and i.strip()]
    true_items = [i for i in true.split('|') if i and i.strip()]
    pred_items_dict = dict([i.split('=') for i in pred_items])
    true_items_dict = dict([i.split('=') for i in true_items])
    all_items = list(pred_items_dict.keys())
    all_items.extend(list(true_items_dict.keys()))
    total_len = len(set(all_items))
    logger.debug('All keys for %s and %s are %s',
                 pred,
                 true,
                 all_items)
    acc_by_feat = []
    for k, v in pred_items_dict.items():
        acc_by_feat.append(score_single_feat_acc(k, v, true_items_dict))
    for k in true_items_dict.keys():
        if k not in pred_items_dict.keys():
            acc_by_feat.append(0)
    logger.debug('Total keys: %s, analysed keys: %s', total_len, len(acc_by_feat))
    assert len(acc_by_feat) == total_len
    return sum(acc_by_feat)/total_len

        


def check_for_label_in_feats(feats: str | None, label: str) -> bool:
    logger.debug('Checking for existence of %s in %s', label, feats)
    if (feats == None):
        return label == '_'
    feats = feats.split('|')
    for feat in feats:
        if feat.split('=')[0] == label:
            return True
    return False

def check_for_correct_feat(pred_feats: str | None, correct_feats: str | None, label: str) -> bool:
    logger.debug('Evaluating %s and %s', pred_feats, correct_feats)
    if label == '_':
        return pred_feats == None and correct_feats == None
    logger.debug('Label is not underscore')
    if pred_feats == None:
        if correct_feats == None:
            return True
        return False
    if correct_feats == None:
        return False
    pred_feats_dict = dict([i.split('=') for i in pred_feats.split('|') if i and i.strip()])
    correct_feats_dict = dict([i.split('=') for i in correct_feats.split('|') if i and i.strip()])
    if label not in pred_feats_dict.keys() and label in correct_feats_dict.keys():
        logger.debug("Incoincidence! %s and %s for label %s", pred_feats, correct_feats, label)
        return False
    if pred_feats_dict[label] == correct_feats_dict[label]:
        logger.debug("Found coincidence! %s and %s for label %s", pred_feats, correct_feats, label)
        return True


def score_for_single_morph_feature(pred: list, true: list, label: str) -> None:
    relevant_tokens = [i for i in zip(pred, true) if check_for_label_in_feats(i[1], label)]
    correct = len([i for i in relevant_tokens if check_for_correct_feat(i[0], i[1], label)])
    # I do not score F1 for single label, because for labels with low share it is going to be very low-informative
    acc = correct/len(relevant_tokens)
    logger.info("Accuracy for %s is %.2f%s from %s tokens", label, acc*100, "%", len(relevant_tokens))

def robustness_score(original_score: float, current_score: float):
    ideal_vector = np.array([25, 0])
    current_vector = np.array([25, current_score - original_score])
    cos_sim = np.dot(ideal_vector, current_vector) / (norm(ideal_vector) * norm(current_vector))
    logger.debug('Sign-neutral cosine similarity is %.2f', cos_sim)
    robustness_coefficient = 1 if current_score >= original_score else -1
    return cos_sim * robustness_coefficient

def score_rapidity_rate(pred: any, true: any) -> int:
    if pred.upos != true.upos:
        return 7
    if pred.feats == None:
        if true.feats == None:
            return 0
        return len([i for i in true.feats.split('|') if i and i.strip()])
    if true.feats == None:
        return len([i for i in pred.feats.split('|') if i and i.strip()])
    gold_feats = [i for i in true.feats.split('|') if i and i.strip()]
    pred_feats = [i for i in pred.feats.split('|') if i and i.strip()]
    rapidity_rate = abs(len(gold_feats) - len(pred_feats))
    gold_feats = dict([i.split('=') for i in gold_feats if i and i.strip()])
    pred_feats = dict([i.split('=') for i in pred_feats if i and i.strip()])
    for i in gold_feats.keys():
        if i in pred_feats.keys():
            if gold_feats[i] != pred_feats[i]:
                rapidity_rate = rapidity_rate + 1
    return rapidity_rate

def assign_rapidity_rate(pred: stanza.Document, gold: stanza.Document) -> stanza.Document:
    for sent_idx, sent in enumerate(pred.sentences):
        for token_idx, token in enumerate(sent.tokens):
            for word_idx, word in enumerate(token.words):
                gold_word = gold.sentences[sent_idx].tokens[token_idx].words[word_idx]
                rapidity_rate = score_rapidity_rate(word, gold_word)
                rate_param = f'PosRapidity={rapidity_rate}'
                if gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc == None:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = rate_param
                else:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc + '|' + rate_param
    return gold



def eval_pos(gold_file_path: str, pred_file_path: str, train_result: float, train_ufeats: float, exp_folder: str) -> None:
    logger.info("POS: starting comparison between %s and %s", gold_file_path, pred_file_path)
    logger.debug("Called with arguments %s", locals())
    gold = CoNLL.conll2doc(gold_file_path)
    labels = collect_labels(gold)
    logger.info("PoS and morphological labels are %s", labels)
    pred = CoNLL.conll2doc(pred_file_path)
    gold_pos = []
    for sent in gold.sentences:
        for token in sent.tokens:
            for word in token.words:
                gold_pos.append(word.upos)
    pred_pos = []
    for sent in pred.sentences:
        for token in sent.tokens:
            for word in token.words:
                pred_pos.append(word.upos)
    common_results = precision_recall_fscore_support(gold_pos, pred_pos, average='macro')
    logger.info('PoS precision: %.2f%s', common_results[0] * 100, "%")
    logger.info('PoS recall: %.2f%s', common_results[1] * 100, "%")
    logger.info('PoS macro F1-score: %.2f%s', common_results[2] * 100, "%")
    if (train_result and train_result >= 0):
        logger.info('PoS robustness:  %.2f', robustness_score(train_result, common_results[2] * 100))
    for pos in labels['pos']:
        logger.debug('Starting scoring accuracy score for %s', pos)
        score_for_single_pos(pred_pos, gold_pos, pos)
    gold_morph = []
    for sent in gold.sentences:
        for token in sent.tokens:
            for word in token.words:
                gold_morph.append(word.feats)
    pred_morph = []
    for sent in pred.sentences:
        for token in sent.tokens:
            for word in token.words:
                pred_morph.append(word.feats)
    morph_comparison = [score_feats_acc(i[0], i[1]) for i in zip(pred_morph, gold_morph)]
    morph_acc = sum(morph_comparison)/len(morph_comparison)
    logger.info("UFeats match by keys:  %.2f%s", morph_acc*100, "%")
    ufeats_exact = len([i for i in zip(gold_morph, pred_morph) if i[0] == i[1]])/len(gold_morph)
    logger.info("UFeats exact match: %.2f%s", ufeats_exact*100, "%")
    if (train_ufeats and train_ufeats >= 0):
        logger.info('UFeats exact robustness:  %.2f', robustness_score(train_ufeats, ufeats_exact * 100))
    for morph in labels['feats']:
        logger.debug('Starting scoring accuracy score for %s', morph)
        score_for_single_morph_feature(pred_morph, gold_morph, morph)
    gold_with_rapidity = assign_rapidity_rate(pred, gold)
    name = ntpath.basename(gold_file_path)
    file_to_store = os.path.join(exp_folder, f"rapidity_{name}")
    CoNLL.write_doc2conll(gold_with_rapidity, file_to_store)
    logger.info("See results of assigned rapidity in %s", file_to_store)
    logger.info("PoS and morphological tagging evaluation is finished")
    
    