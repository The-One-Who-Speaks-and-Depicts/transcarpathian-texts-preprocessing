from logging import getLogger
from ntpath import basename
from os.path import join
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support

import stanza
from stanza.models.common.doc import Document

from utils.data_schema import get_pos_schema
from utils.eval_utils import robustness_score, AnnotationLabels, prepare_annotation, store_rapidity_rate
from utils.llm_utils import ClientParams, load_llm_client, process_llm_response, process_llm_errors, transfer_llm_annotation
from utils.tagging_utils import store_predictions, load_input_data, restore_original_tags

logger = getLogger('preprocessor')

def pos_tag_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk') -> None:
    logger.debug("Called with arguments %s", locals())
    doc, doc_for_processing = load_input_data(input_file_path, True)
    stanza.download(lang)
    nlp = stanza.Pipeline(lang=lang, processors=['tokenize', 'pos'], use_gpu=True, tokenize_pretokenized=True)
    logger.debug("Part-of-speech tagging pipeline prepared")
    doc_processed = nlp(doc_for_processing)
    doc_processed = restore_original_tags(doc_processed, doc, pretagged = False)
    store_predictions("pos", "stanza", Path(exp_folder), basename(input_file_path), doc_processed)

def pos_tag_with_llm(input_file_path: str, exp_folder: str, params: ClientParams = ClientParams()) -> None:
    logger.debug("Called with arguments %s", locals())
    doc, doc_for_processing = load_input_data(input_file_path)    
    client = load_llm_client(params.api, params.api_key, params.base_url)
    logger.debug("Part-of-speech tagging pipeline prepared")
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        sent_tokens = [w.text for w in sent.words]
        sent_tokenised = ' '.join(sent_tokens)
        prompt = params.prompt.format(sent=f"{sent_tokenised}\nTokens:{[str(w.id) + ': ' + w.text + '\n' for w in sent.words]}\n")
        logger.info("Prompt is assembled:\n %s", prompt)
        annotated_sent = {}
        retries = 0
        while True:            
            try:
                response = {}
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
                            response_format= get_pos_schema(sent, sent_tokens),
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
                    retries,
                    params.retries,
                    sent_tokenised,
                    params.prompt.format(sent=sent_tokenised)
                    )
                continue
                    
        doc_for_processing = transfer_llm_annotation(
            doc_for_processing,
            sent_idx,
            sent,
            ['upos', 'feats'],
            annotated_sent
        )
    doc_for_processing = restore_original_tags(doc_for_processing, doc, pretagged = False)
    store_predictions("pos", params.model, Path(exp_folder), basename(input_file_path), doc_for_processing)

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
    if len(acc_by_feat) != total_len:
        raise ValueError("Not all keys are analysed," \
        f"analysed {acc_by_feat} / {total_len} keys")
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


def score_pos_rapidity_rate(pred: any, true: any) -> int:
    if pred.upos != true.upos:
        # Usually there is no more than 5 to 6 morphological features in UD by
        # a single lexical item, and an error in a part-of-speech is 
        # of higher level, being more crucial than getting any morphology
        # wrong (due to morphology features being generally dependent on part-of-speech)
        # Thus, heat rapidity rate of part-of-speech error should be higher than rapidity rate
        # for getting all the morphological features incorrect; at the same time, it would be
        # unjust to punish the model for getting them incorrect as well. This leads to automatically
        # returning 7 in case of a part-of-speech error.
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

def assign_pos_rapidity_rate(pred: Document, gold: Document) -> Document:
    for sent_idx, sent in enumerate(pred.sentences):
        for token_idx, token in enumerate(sent.tokens):
            for word_idx, word in enumerate(token.words):
                gold_word = gold.sentences[sent_idx].tokens[token_idx].words[word_idx]
                rapidity_rate = score_pos_rapidity_rate(word, gold_word)
                rate_param = f'PosRapidity={rapidity_rate}'
                if gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc == None:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = rate_param
                else:
                    gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc = gold.sentences[sent_idx].tokens[token_idx].words[word_idx].misc + '|' + rate_param
    return gold


def eval_pos(
        gold_pos: list[AnnotationLabels], pred_pos: list[AnnotationLabels],
        train_result: float, labels: list[str]
        ) -> None:
    gold_pos = [a.pos for a in gold_pos]
    pred_pos = [a.pos for a in pred_pos]
    common_results = precision_recall_fscore_support(
        gold_pos, 
        pred_pos,
        average='macro'
        )
    logger.info('PoS precision: %.2f%s', common_results[0] * 100, "%")
    logger.info('PoS recall: %.2f%s', common_results[1] * 100, "%")
    logger.info('PoS macro F1-score: %.2f%s', common_results[2] * 100, "%")
    if (train_result and train_result >= 0):
        logger.info('PoS robustness:  %.2f', robustness_score(train_result, common_results[2] * 100))
    for pos in labels:
        logger.debug('Starting scoring accuracy score for %s', pos)
        score_for_single_pos(pred_pos, gold_pos, pos)


def eval_feats(gold_morph: list[AnnotationLabels], pred_morph: list[AnnotationLabels], train_result: float, labels: list[str]) -> None:
    gold_morph = [a.feats for a in gold_morph]
    pred_morph = [a.feats for a in pred_morph]
    morph_comparison = [score_feats_acc(i[0], i[1]) for i in zip(pred_morph, gold_morph)]
    morph_acc = sum(morph_comparison)/len(morph_comparison)
    logger.info("UFeats match by keys:  %.2f%s", morph_acc*100, "%")
    ufeats_exact = len([i for i in zip(gold_morph, pred_morph) if i[0] == i[1]])/len(gold_morph)
    logger.info("UFeats exact match: %.2f%s", ufeats_exact*100, "%")
    if (train_result and train_result >= 0):
        logger.info('UFeats exact robustness:  %.2f', robustness_score(train_result, ufeats_exact * 100))
    for morph in labels:
        logger.debug('Starting scoring accuracy score for %s', morph)
        score_for_single_morph_feature(pred_morph, gold_morph, morph)

def eval_morphology(
        gold_file_path: str,
        pred_file_path: str,
        train_result: float,
        train_ufeats: float,
        exp_folder: str
        ) -> None:
    logger.debug("Called with arguments %s", locals())
    logger.info("POS: starting comparison between %s and %s", gold_file_path, pred_file_path)
    dataset_for_annotation = prepare_annotation(gold_file_path, pred_file_path)
    logger.info("PoS and morphological labels are %s", dataset_for_annotation['labels'])
    eval_pos(
        dataset_for_annotation['gold'],
        dataset_for_annotation['pred'],
        train_result,
        dataset_for_annotation['labels']['pos']
        )
    eval_feats(
        dataset_for_annotation['gold'],
        dataset_for_annotation['pred'],
        train_ufeats,
        dataset_for_annotation['labels']['feats']
        )    
    gold_with_rapidity = assign_pos_rapidity_rate(dataset_for_annotation['pred_doc'], dataset_for_annotation['gold_doc'])
    store_rapidity_rate(gold_with_rapidity, gold_file_path, exp_folder)
    logger.info("PoS and morphological tagging evaluation is finished")

