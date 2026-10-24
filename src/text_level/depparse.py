import logging
import json
import re

import stanza
from stanza.utils.conll import CoNLL
import ntpath
from copy import deepcopy

from stanza.models.common.doc import Document, Sentence, Word



from zss import Node, simple_distance

from pos import collect_labels, robustness_score
from utils.llm_utils import ClientParams, load_llm_client
from utils.tagging_utils import store_predictions, load_input_data



logger = logging.getLogger('preprocessor')

# TODO: technically, all tagging prediction schema are very similar and can be joined into a single one

def preprocess_conllu_word_for_depparse(word: Word, pretagged: bool | str) -> str:
    text = word.text
    if not pretagged:
        return text
    upos = word.pos if word.pos is not None and word.pos.strip() else '_'
    feats = word.feats if word.feats is not None and word.feats.strip() else '_'
    lemma = word.lemma if word.lemma is not None and word.lemma.strip() else '_'
    if pretagged == 'detokenised':
        return ' '.join([upos, feats])
    return ' '.join([text, lemma, upos, feats])


def depparse_with_stanza(input_file_path: str, exp_folder: str, lang: str = 'uk', pretagged: bool = True) -> None:
    logger.debug("Called with arguments %s", locals())
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
                    word.lemma = '_'
    logger.debug("%s loaded", input_file_path)
    nlp = stanza.Pipeline(
        lang=lang,
        processors=['tokenize', 'depparse'],
        use_gpu=True,
        tokenize_pretokenized=True,
        depparse_pretagged=True
        ) if pretagged else stanza.Pipeline(
            lang=lang,
            processors=['tokenize', 'pos', 'lemma', 'depparse'],
            use_gpu=True,
            tokenize_pretokenized=True
            )
    logger.debug("Dependency parsing pipeline prepared, settings: %s", nlp)
    doc_processed = nlp(doc_for_procesing)
    for sent_idx, sent in enumerate(doc_processed.sentences):
        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, word in enumerate(token.words):
                doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
                if not pretagged:
                    doc_processed.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    pretagging_info = 'gold' if pretagged else 'silver'
    store_predictions(f"depparse_{pretagging_info}", "stanza", exp_folder, ntpath.basename(input_file_path), doc_processed)

def depparse_with_llm(input_file_path: str, exp_folder: str, params: ClientParams = ClientParams(), pretagged: bool | str = True) -> None:
    logger.debug("Called with arguments %s", locals())
    doc, doc_for_processing = load_input_data(input_file_path)
    client = load_llm_client(params.api, params.api_key, params.base_url)
    logger.debug("Dependency parsing pipeline prepared")
    for sent_idx, sent in enumerate(doc_for_processing.sentences):
        sent_tokens = [w.text for w in sent.words]
        sent_tokenised = ' '.join(sent_tokens)
        words = []
        prompt = params.prompt.format(
            sent=f"{sent_tokenised}\nTokens:\n{
                '\n'.join(
                    [' '.join(
                        [
                            str(w.id),
                            ':',
                            preprocess_conllu_word_for_depparse(w, pretagged)
                        ]
                         ) for w in sent.words]
                    ) + '\n'
                }"
            )
        logger.info("Prompt is assembled:\n %s", prompt)
        retries = 0
        while True:
            match params.api:
                case "OpenAI":
                    try:
                        ids = [w.id for w in sent.words]
                        ids_with_root = [w.id for w in sent.words]
                        ids_with_root.append(0)
                        ud_schema = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "ud_deprel",
                                "strict": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        str(w.id): {
                                        "type": "object",
                                        "properties": {
                                            "text": {
                                                "type": "string",
                                                "enum": [w.text],
                                            },
                                            "head": {
                                                "type": "integer", 
                                                "enum": [
                                                    id for id in ids_with_root if id != w.id 
                                                    ],
                                            },
                                            "deprel": {
                                                "type": "string",
                                                "pattern": (
                                                    r"^(?:acl|advcl|advmod|amod|appos|aux|case|cc|"
                                                    r"ccomp|clf|compound|conj|cop|csubj|dep|det|"
                                                    r"discourse|dislocated|expl|fixed|flat|goeswith|"
                                                    r"iobj|list|mark|nmod|nsubj|nummod|obj|obl|"
                                                    r"orphan|parataxis|punct|reparandum|root|"
                                                    r"vocative|xcomp)(?::[A-Za-z0-9_-]+)?$"
                                                )
                                            }
                                        },
                                        "required": ["text", "head", "deprel"],
                                        "additionalProperties":False,
                                        } for w in sent.words
                                    },
                                    "required": [str(id) for id in ids],
                                    "additionalProperties": False,
                                },
                            }
                        }
                        response = client.chat.completions.create(
                            model=params.model,
                            messages=[
                                {"role": "system", "content": ""},
                                {"role": "user", "content": prompt},
                            ],
                            temperature=params.temperature,
                            seed=params.seed,
                            response_format= ud_schema,
                            stream=False
                        )
                        logger.info(f'Response: {response}')
                        logger.info(f'Message: {response.choices[0].message.content}')
                        # small models tend to incontrollably generate code blocks 
                        # designations around the actual output
                        cleared_content = re.sub(r'```(json)?\n?', '', response.choices[0].message.content)
                        annotated = json.loads(cleared_content)
                        if len(annotated) != len(sent.words):
                            raise ValueError(
                                f"Token count mismatch: expected {len(sent.words)} from  {sent_tokenised}, "
                                f"received {len(annotated)}"
                            )
                        words = annotated
                        break
                    except ValueError as e:
                        logger.error(f"Invalid LLM output, the error is {e}")
                        retries = retries + 1
                        if retries > params.retries:
                            raise ValueError("Too many retries")
                        original_prompt = prompt
                        prompt = f"Original prompt: {original_prompt}. Prompt processing was incorrect, there was an exception {e}. Please, re-perform the task, keeping the exception in mind."
                        continue

        for tkn_idx, token in enumerate(sent.tokens):
            for wrd_idx, _ in enumerate(token.words):
                doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].head = words[str(tkn_idx+1)]['head']
                doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].deprel = words[str(tkn_idx+1)]['deprel']
                doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text = doc.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].text
                if not pretagged:
                    doc_for_processing.sentences[sent_idx].tokens[tkn_idx].words[wrd_idx].xpos = '_'
    pretagging_info = None
    match pretagged:
        case 'detokenised':
            pretagging_info = 'gold_detokenised'
        case True:
            pretagging_info = 'gold'
        case _:
            pretagging_info = 'silver'
    # models usually contain slash in their name ("google/gemma-3en-4b"),
    # I replace it with underscore
    model_name = re.sub(r'\/', '_', params.model)
    pretagging_info = 'gold-pos' if pretagged else 'silver-pos'
    store_predictions(f"lemmatisation_{pretagging_info}", model_name, exp_folder, ntpath.basename(input_file_path), doc_for_processing)


def conllu_to_zss_node(sent: Sentence) -> Node | None:
    node_map = {}
    for token in sent.tokens:
        for word in token.words:
            node_name = "%s_%s" % (word.text, word.id)
            node = Node(node_name)
            node_map[word.id] = node    
    for token in sent.tokens:
        for word in token.words:
            parent_id = word.head
            if parent_id != 0:
                parent_node = node_map[parent_id]
                child_node = node_map[word.id]
                parent_node.addkid(child_node)
    
    root_id = None
    for token in sent.tokens:
        for word in token.words:
            if word.head == 0:
                root_id = word.id
    return node_map[root_id] if root_id else None

def check_correct_morphology_and_lemma(g: tuple[str, str, str, str, str], p: tuple[str, str, str, str, str]) -> bool:
    return g[0] == p[0] and g[1] == p[1] and g[2] == p[2]


def score_uas_and_las(
        gold_deps: tuple[str, str, str, str, str],
        pred_deps: tuple[str, str, str, str, str],
        data_part: str = "in dataset",
        gold_uas: float = -1,
        gold_las: float = -1) -> None:
    total = len(gold_deps)
    # ======================================================== TOTAL ACCURACY ======================================================================
    uas = sum(1 for g, p in zip(gold_deps, pred_deps) if g[3] == p[3])
    las = sum(1 for g, p in zip(gold_deps, pred_deps) if g[3] == p[3] and g[4] == p[4])
    uas_score = uas/total * 100 if total > 0 else 0
    logger.info("UAS %s: %.2f%%", data_part, uas_score)
    if (gold_uas >= 0):
        robustness_uas = robustness_score(uas_score, gold_uas)
        logger.info("UAS robustness %s: %.2f", data_part, robustness_uas)
    las_score = las/total * 100 if total > 0 else 0
    logger.info("LAS %s: %.2f%%", data_part, las_score)
    if (gold_las >= 0):
        robustness_las = robustness_score(las_score, gold_las)
        logger.info("LAS robustness %s: %.2f", data_part, robustness_las)
    # ======================================================= ASSESSING PREVIOUS TAGGING ===========================================================
    total_correct_tagging = sum(1 for g, p in zip(gold_deps, pred_deps) if check_correct_morphology_and_lemma(g,p))
    total_incorrect_tagging = total - total_correct_tagging
    correct_share = total_correct_tagging/total * 100 if total > 0 else 0
    logger.info("Share of correctly morphologically tagged items %s is %.2f%%", data_part, correct_share)
    # ====================================================== ASSESSING ITEMS WITH CORRECT MORPHOLOGY ================================================
    uas_with_correct_tagging = sum(1 for g, p in zip(gold_deps, pred_deps) if g[3] == p[3] and check_correct_morphology_and_lemma(g,p))
    uas_with_correct_tagging_share = uas_with_correct_tagging/total_correct_tagging * 100 if total_correct_tagging > 0 else 0
    logger.info("Share of correctly syntactically tagged items %s among correctly morphologically tagged items is %.2f%%", data_part, uas_with_correct_tagging_share)
    las_with_correct_tagging = sum(1 for g, p in zip(gold_deps, pred_deps) if g[3] == p[3] and g[4] == p[4] and check_correct_morphology_and_lemma(g,p))
    las_with_correct_tagging_share = las_with_correct_tagging/total_correct_tagging * 100 if total_correct_tagging > 0 else 0
    logger.info("Share of correctly syntactically tagged items %s among correctly morphologically tagged items is %.2f%%", data_part, las_with_correct_tagging_share)
    # ====================================================== ASSESSING ITEMS WITH INCORRECT MORPHOLOGY ================================================
    uas_with_incorrect_tagging = sum(1 for g, p in zip(gold_deps, pred_deps) if g[3] == p[3] and not check_correct_morphology_and_lemma(g,p))
    uas_with_incorrect_tagging_share = uas_with_incorrect_tagging/total_incorrect_tagging * 100 if total_incorrect_tagging > 0 else 0
    logger.info("Share of correctly syntactically tagged items %s among incorrectly morphologically tagged items is %.2f%%", data_part, uas_with_incorrect_tagging_share)
    las_with_incorrect_tagging = sum(1 for g, p in zip(gold_deps, pred_deps) if g[3] == p[3] and g[4] == p[4] and not check_correct_morphology_and_lemma(g,p))
    las_with_incorrect_tagging_share = las_with_incorrect_tagging/total_incorrect_tagging * 100 if total_incorrect_tagging > 0 else 0
    logger.info("Share of correctly syntactically tagged items %s among incorrectly morphologically tagged items is %.2f%%", data_part, las_with_incorrect_tagging_share)

def get_root(sent: Sentence) -> str:
    for token in sent.tokens:
        for word in token.words:
            if word.pos == 'VERB' and word.deprel == 'root':
                return word.id
    return ''

def get_deps(sent: Sentence, root: str) -> list[tuple[str, str]]:
    deps = []
    for token in sent.tokens:
        for word in token.words:
            if word.head == root:
                deps.append((word.id, word.deprel))
    return deps

def get_cp(doc: Document) -> list[dict]:
    cps = []
    for sent in doc.sentences:
        cp = {'id': sent.sent_id, 'root': get_root(sent)}
        if cp['root']:
            cp['deps'] = get_deps(sent, cp['root'])
            cps.append(cp)
    return cps

def get_coinciding_cps(gold: list[dict], pred: list[dict]) -> list[dict]:
    pred_ids = [cp['id'] for cp in pred]
    return [cp for cp in gold if cp['id'] in pred_ids]

def score_ucp(gold: list[tuple[str, str]], pred: list[tuple[str, str]]) -> float:
    pred_heads = [i[0] for i in pred]
    gold_heads = [i[0] for i in gold]
    all_heads = len(list(set(pred_heads + gold_heads)))
    pred_in_gold = len([i for i in pred_heads if i in gold_heads])    
    return pred_in_gold/all_heads


def score_lcp(gold: list[tuple[str, str]], pred: list[tuple[str, str]]) -> float:
    pred_heads_with_labels = [(i[0], i[1]) for i in pred]
    gold_heads_with_labels = [(i[0], i[1]) for i in gold]
    all_heads_with_labels = len(list(set(pred_heads_with_labels + gold_heads_with_labels)))
    pred_in_gold = len([i for i in pred_heads_with_labels if i in gold_heads_with_labels])    
    return pred_in_gold/all_heads_with_labels

def score_individual_cp(gold: dict, pred: list[dict]) -> tuple[float, float]:    
    for pred_cp in pred:
        if pred_cp['id'] == gold['id']:
            if len(gold['deps']) == 0:
                if len(pred_cp['deps']) == 0:
                    return (1, 1)
                return (0, 0)
            if len(pred_cp['deps']) == 0:
                return (0, 0)
            return (score_ucp(gold['deps'], pred_cp['deps']), score_lcp(gold['deps'], pred_cp['deps']))



def score_cp_metrics(gold: Document, pred: Document) -> None:
    gold_cps = get_cp(gold)
    pred_cps = get_cp(pred)
    coinciding_gold_cps = get_coinciding_cps(gold_cps, pred_cps)
    ucp = [0 for _ in range(len(gold_cps) - len(coinciding_gold_cps))]
    lcp = [0 for _ in range(len(gold_cps) - len(coinciding_gold_cps))]
    for cp in coinciding_gold_cps:
        individual_ucp, individual_lcp = score_individual_cp(cp, pred_cps)
        ucp.append(individual_ucp)
        lcp.append(individual_lcp)
    ucp = sum(ucp)/len(gold_cps) * 100
    lcp = sum(lcp)/len(gold_cps) * 100
    logger.info("UCP is %.2f%%", ucp)
    logger.info("LCP is %.2f%%", lcp)
        
                
        

def evaluate_depparse(gold_file_path: str, pred_file_path: str, gold_uas: float, gold_las: float) -> None:
    logger.info("Dependency parsing: starting comparison between %s and %s", gold_file_path, pred_file_path)
    # ====================================================== LOADING DATA =================================================================
    gold = CoNLL.conll2doc(gold_file_path)
    labels = collect_labels(gold)
    pos_labels = labels["pos"]
    logger.info("PoS labels are %s", pos_labels)
    deprel_labels = labels["deprel"]
    logger.info("Syntactic labels are %s", deprel_labels)
    pred = CoNLL.conll2doc(pred_file_path)
    gold_deps = []
    for sent in gold.sentences:
        for token in sent.tokens:
            for word in token.words:
                feats = word.feats if word.feats is not None else '_'
                deprel = word.deprel  if word.deprel is not None else '_'
                gold_deps.append((word.upos, word.lemma, feats, word.head, deprel))
    pred_deps = []
    for sent in pred.sentences:
        for token in sent.tokens:
            for word in token.words:
                feats = word.feats if word.feats is not None else '_'
                deprel = word.deprel  if word.deprel is not None else '_'
                pred_deps.append((word.upos, word.lemma, feats, word.head, deprel))
    # ================================================ DOCUMENT-SPAN METRICS =====================================================================
    ted_distances = [simple_distance(conllu_to_zss_node(sent1), conllu_to_zss_node(sent2)) for sent1, sent2 in zip(gold.sentences, pred.sentences)]
    ted = sum(ted_distances)/len(ted_distances)
    logger.info("Average tree edit distance is %.2f", ted)
    score_uas_and_las(gold_deps, pred_deps, gold_uas=gold_uas, gold_las=gold_las)
    score_cp_metrics(gold, pred)
    # ======================================= FINE-GRAINED: PoS + deprel==========================================================================
    for deprel_label in deprel_labels:
        gold_and_pred_with_deprel = [i for i in zip(gold_deps, pred_deps) if i[0][4] == deprel_label]
        gold_filtered = [i[0] for i in gold_and_pred_with_deprel]
        pred_filtered = [i[1] for i in gold_and_pred_with_deprel]
        data_part = "for " + deprel_label
        score_uas_and_las(gold_filtered, pred_filtered, data_part)

            

