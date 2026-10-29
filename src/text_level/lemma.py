from logging import getLogger
from numbers import Number
from ntpath import basename

from collections.abc import Callable

from stanza import download, Pipeline
from stanza.models.common.doc import Document
from Levenshtein import distance as lev
from pyjarowinkler.distance import get_jaro_distance as jw


from utils.data_schema import get_lemma_schema
from utils.eval_utils import prepare_annotation, robustness_score, store_rapidity_rate, AnnotationLabels, get_lcss_indices, single_point_accuracy, collect_errors_by_pos
from utils.llm_utils import ClientParams, load_llm_client, transfer_llm_annotation, process_llm_errors, process_llm_response
from utils.tagging_utils import store_predictions, load_input_data, restore_original_tags

logger = getLogger('preprocessor')

def lemmatise_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk', pretagged: bool = True):
    doc, doc_for_processing = load_input_data(input_file_path)    
    download(lang)
    doc, doc_for_processing = load_input_data(input_file_path, pretagged)
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
    logger.debug("Lemmatisation tagging pipeline prepared, settings: %s", nlp)
    logger.debug("Called with arguments %s", locals())
    doc_processed = nlp(doc_for_processing)
    doc_processed = restore_original_tags(doc_processed, doc, pretagged)
    pretagging_info = 'gold-pos' if pretagged else 'silver-pos'
    store_predictions(f"lemmatisation_{pretagging_info}", "stanza", exp_folder, basename(input_file_path), doc_processed)

def lemmatise_with_llm(input_file_path: str, exp_folder: str, params: ClientParams = ClientParams(), pretagged: bool = True) -> None:
    logger.debug("Called with arguments %s", locals())
    doc, doc_for_processing = load_input_data(input_file_path)
    client = load_llm_client(params.api, params.api_key, params.base_url)
    logger.debug("Lemmatisation pipeline prepared")
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        sent_tokens = [w.text for w in sent.words]
        sent_tokenised = ' '.join(sent_tokens)
        prompt = params.prompt.format(sent=f"{sent_tokenised}\nTokens:{[str(w.id) + ': ' + w.text + ' ' + (w.pos if w.pos is not None and w.pos.strip() else '_') + ' ' + (w.feats if w.feats is not None and w.feats.strip() else '_') +  '\n' for w in sent.words]}\n")
        logger.info("Prompt is assembled:\n %s", prompt)
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
                            response_format= get_lemma_schema(sent, sent_tokens),
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
            ['lemma'],
            annotated_sent
        )
    doc_for_processing = restore_original_tags(doc_for_processing, doc, pretagged)
    pretagging_info = 'gold-pos' if pretagged else 'silver-pos'
    store_predictions(f"lemmatisation_{pretagging_info}", params.model, exp_folder, basename(input_file_path), doc_for_processing)


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

    
