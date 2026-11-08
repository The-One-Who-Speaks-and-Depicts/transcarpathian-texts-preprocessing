"""
Facilitates collecting parts-of-speech and morphological features for the tokens that have
a particular lemma and the tokens that are either their heads or dependencies
from a provided corpus.

* gather_morphological_features_from_word (word) -> list[str]: takes a CoNLL-U word and
extracts its morphological annotation to a list of strings
* collect_conllu_features_to_dataframe (PathLike, list[dict]): a pipeline function,
receives a path to store the results in, and information on a corpus and data to extract
from it
* gather_unique_features_in_immediate_syntactic_surroundings (list[dict]) -> list[str]:
takes a corpus to analyse, and processes each file in it, gathering parts-of-speech
and morphological features of tokens
* gather_features_by_instance (list[dict], list[str]) -> dict: takes a corpus to analyse,
and processes each file in it, gathering parts-of-speech and morphological features for
each token, using a pre-gathered set of features
* store_glossed_instances_in_dataframe (PathLike, list[str], dict): takes gathered data
to save it in a pandas.DataFrame-compatible comma-separated values file
"""

from logging import getLogger
from os import PathLike

from pandas import DataFrame

from stanza.models.common.doc import Word
from stanza.utils.conll import CoNLL

logger = getLogger('preprocessor')

def gather_morphological_features_from_word(word: Word) -> list[str]:
    """
    Takes a word tagged in CoNLL-U format and extracts part-of-speech and
    morphological features from it.

    Params:
        word (Word): a CoNLL-U-parsed word
    
    Returns:
        list[str]: a list that includes all the morphological features and
        part-of-speech of the word
    """
    logger.debug("Input:\n%s", word)
    if not isinstance(word, Word):
        raise TypeError(f"Received a variable {word} of type {type(word)}, " \
        "should have received a Stanza Word object")
    if not hasattr(word, 'pos') or getattr(word, 'pos') is None:
        raise ValueError("Provided word does not contain any morphological tagging")
    word_glosses = []
    word_glosses.append(word.pos)
    if word.feats:
        word_glosses.extend(
            [i for i in word.feats.split('|') if i and i.strip()]
            )
    logger.debug("Gathered %s from %s", word_glosses, word.text)
    return word_glosses


def gather_unique_features_in_immediate_syntactic_surroundings(
        inputs_list: list[dict]
        ) -> list[str]:
    """
    Takes a list of dictionaries that contain an information on CoNLL-U files
    and lemmas from them, the tokens of which are going to
    become the main object of analysis, and gathers morphological features
    that occur in immediate syntactic surrounding of these tokens (i. e., their own features,
    features of their syntactic heads, and their immediate syntactic dependencies features:
    everything that is connected to these tokens on the dependency graph).

    Params:
        inputs_list (list[dict]): a list of dictionaries that contain information about files
        in a dataset and which linguistic units, represented through their lemmata,
        the function should process by collecting their morphological features and
        morphological features of their immediate surrounding
    
    Returns: 
        list[str]: unique glosses, collected from tokens of a dataset that belong to 
        a certain lemma and immediate surrounding of its tokens 
    """
    logger.debug("Input: %s", inputs_list)
    if not isinstance(inputs_list, list) and not all(
        isinstance(
            i, dict
            ) and not set(["file", "lemma", "alias"]).issubset(set(i.keys())) for i in inputs_list
    ):
        raise ValueError("inputs_list should be a list of dictionaries, each containing " \
        f"keys for a file to load and a lemma to gather, received:\n{inputs_list}")
    overall_glosses = []
    for input in inputs_list:
        doc = CoNLL.conll2doc(input["file"])
        for sent in doc.sentences:
            instances = []
            heads = []
            for token in sent.tokens:
                for word in token.words:
                    if word.lemma == input["lemma"]:
                        instances.append(word.id)
                        heads.append(word.head)
            surroundings = []
            for token in sent.tokens:
                for word in token.words:
                    if word.id in heads or word.head in instances or word.id in instances:
                        surroundings.extend(
                            gather_morphological_features_from_word(word)
                        )
            overall_glosses.extend(surroundings)
    overall_glosses = list(set(overall_glosses))
    logger.debug("Gathered glosses:\n%s", overall_glosses)
    return overall_glosses

def gather_features_by_instance(
        inputs_list: list[dict], overall_glosses: list[str]
        ) -> dict:
    """
    Takes a list of dictionaries that contain an information on CoNLL-U files
    and lemmas from them, the tokens of which are going to
    become the main object of analysis, and previously gathered morphological features
    that occur in the immediate syntactic surrounding of these tokens (i. e., their own features,
    the features of their syntactic heads, and their immediate syntactic dependencies feature:
    everything that is connected to these tokens on the dependency graph) to gather information
    on which morphological features occur in immediate surrounding of which exact token.

    Params:
        inputs_list (list[dict]): a list of dictionaries that contain information about files
        in a dataset and which linguistic units, represented through their lemmata,
        a function should process by collecting their morphological features and
        the morphological features of their immediate surrounding
        overall_glosses (list[str]): a list of glosses to be used as dataframe column names
    
    Returns:
        dict: an information about each analysed token either having
        or not each of the glosses
    """
    logger.debug("Input: %s", locals())
    if not isinstance(inputs_list, list) and not all(
        isinstance(
            i, dict
            ) and not set(["file", "lemma", "alias"]).issubset(i.keys()) for i in inputs_list
    ):
        raise ValueError("inputs_list should be a list of dictionaries, each containing" \
        f"keys for a file to load and a lemma to gather, received:\n{inputs_list}")
    if not isinstance(overall_glosses, list) or not all(
        isinstance(i, str) for i in overall_glosses
    ):
        raise ValueError("overall_glosses should be a list of strings, received:\n" \
        f"{overall_glosses}")
    instances = {}
    for input in inputs_list:
        name = input["alias"]
        doc = CoNLL.conll2doc(input["file"])
        be_ids_with_heads = []
        for sent in doc.sentences:
            for token in sent.tokens:
                for word in token.words:
                    if word.lemma == input["lemma"]:
                        be_ids_with_heads.append(
                            (sent.sent_id + '_' + str(word.id), word.id, word.head)
                            )
        for uid, id, head in be_ids_with_heads:
            uid = name + '_' + uid
            glosses = []
            for sent in doc.sentences:
                for token in sent.tokens:
                    for word in token.words:
                        if word.lemma == input["lemma"] or word.id == head or word.head == id:
                            glosses.extend(
                                gather_morphological_features_from_word(word)
                            )
            present_glosses = [1 if i in glosses else 0 for i in overall_glosses]
            params = [name]
            params.extend(present_glosses)
            instances[uid] = params
    logger.debug("Features collected:\n%s", instances)
    return instances


def store_glossed_instances_in_dataframe(
        experiment_folder: PathLike,
        possible_glosses: list[str],
        glossed_instances: dict
        ) -> None:
    """
    Takes an experiment directory, a list of glosses, collected from all the texts in a corpus,
    and information on glosses for each analysed instance, to store the information in a 
    comma-separated file in the experiment directory.

    Params:
        experiment_folder (PathLike): a directory to store the results and logs in
        possible_glosses (list[str]): a list of glosses to be used as dataframe column names
        glossed_instances (dict): information about the each analysed token either having
        or not each of the glosses
    """
    logger.debug("Input: %s", locals())
    if not isinstance(experiment_folder, PathLike):
        raise TypeError("experiment_folder should be a PathLike object," \
        f"received {experiment_folder} of type {type(experiment_folder)}")
    if not isinstance(possible_glosses, list) or not all(
        isinstance(i, str) for i in possible_glosses
    ):
        raise ValueError("possible_glosses should be a list of strings, received:\n" \
        f"{possible_glosses}")
    if not isinstance(glossed_instances, dict):
        raise TypeError("glossed_instances should be a dictionary, received:\n" \
        f"{type(glossed_instances)}")
    params = ['lect']
    params.extend(possible_glosses)
    instances_by_parameters = DataFrame.from_dict(
        glossed_instances, columns=params, orient='index'
        )
    logger.debug(
        "Created a dataframe from the provided instances:\n%s",
        instances_by_parameters.head()
        )
    result_path = experiment_folder / 'collected_glosses.csv'
    instances_by_parameters.to_csv(result_path)
    logger.info('Glossed instances stored in %s', result_path)


def collect_conllu_features_to_dataframe(
        experiments_folder: PathLike, inputs_list: list[dict]
        ) -> None:
    """
    Takes a path to the directory and a list of dictionaries that contain 
    information on CoNLL-U files and lemmas from them, the tokens of which are going to
    become the main object of analysis, and gathers parts-of-speech and morphological features
    that occur in the immediate syntactic surrounding of these tokens (i. e., their own features,
    features of their syntactic heads, and their immediate syntactic dependencies features:
    everything that is connected to these tokens on the dependency graph). Extracts
    the information to a list of features that indeed occur with at least one of the tokens,
    and a dictionary with the tokens themselves as keys and the information
    on whether each of the features is in their syntactic surrounding, to store in a
    comma-separated values file.

    Params:
        experiments_folder (PathLike): a directory to store the results and logs in
        inputs_list (list[dict]): a list of dictionaries that contain information about files
        in a dataset and which linguistic units, represented through their lemmata,
        a function should process by collecting their morphological features and
        the morphological features of their immediate surrounding
    """
    if not isinstance(experiments_folder, PathLike):
        raise TypeError("exp_dir should be a PathLinke object," \
        f"received {experiments_folder} of type {type(experiments_folder)}")
    if not isinstance(inputs_list, list) and not all(
        isinstance(
            i, dict
            ) and not set(["file", "lemma", "alias"]).issubset(i.keys()) for i in inputs_list
    ):
        raise ValueError("inputs_list should be a list of dictionaries, each containing" \
        f"keys for a file to load and a lemma to gather, received:\n{inputs_list}")
    glosses = gather_unique_features_in_immediate_syntactic_surroundings(inputs_list)    
    instances = gather_features_by_instance(inputs_list, glosses)
    store_glossed_instances_in_dataframe(experiments_folder, glosses, instances)
    logger.info("Features collection finished")