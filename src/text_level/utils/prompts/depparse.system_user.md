# General task outline

Your task is to perform a dependency parsing in the Universal Dependecies framework for the following pre-tokenized sentence:

{sent}

Assign each token its immediate syntactic head and UD dependency relation.

The syntactic root of the sentence has head 0 and deprel "root". The number 0 represents the artificial ROOT node.


# Input

1. A sentence in a pre-tokenised form.
2. The indexed tokens that can contain following information:
   1. Raw tokens
   2. Raw tokens, part-of-speech and morphological tagging and lemmatisation information
   3. Only part-of-speech and morphological information.


# Example of input

Продавши свій шедевр Меценатові , еллінський скульптор споневажив саме мистецтво : Ти не продався , – гірше !

0: ROOT
1: Продавши продати VERB Aspect=Perf|Tense=Past|VerbForm=Conv
2: свій свій DET Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs|Reflex=Yes
3: шедевр шедевр NOUN Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing
4: Меценатові Меценат PROPN Animacy=Anim|Case=Dat|Gender=Masc|NameType=Giv|Number=Sing
5: , , PUNCT _
6: еллінський еллінський ADJ Case=Nom|Gender=Masc|Number=Sing
7: скульптор скульптор NOUN Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
8: споневажив споневажити VERB Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin
9: саме самий DET Case=Acc|Gender=Neut|Number=Sing|PronType=Prs|Reflex=Yes
10: мистецтво мистецтво NOUN Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing
11: : : PUNCT _
12: Ти ти PRON Animacy=Anim|Case=Nom|Number=Sing|Person=2|PronType=Prs
13: не не PART Polarity=Neg
14: продався продатися VERB Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin
15: , , PUNCT _
16: – – PUNCT PunctType=Dash
17: гірше гірше ADV Degree=Cmp
18: ! ! PUNCT _

=>

{{
  "1": {{
    "text": "Продавши",
    "head": 8,
    "deprel": "advcl"
  }},
  "2": {{
    "text": "свій",
    "head": 3,
    "deprel": "det"
  }},
  "3": {{
    "text": "шедевр",
    "head": 1,
    "deprel": "obj"
  }},
  "4": {{
    "text": "Меценатові",
    "head": 1,
    "deprel": "iobj"
  }},
  "5": {{
    "text": ",",
    "head": 1,
    "deprel": "punct"
  }},
  "6": {{
    "text": "еллінський",
    "head": 7,
    "deprel": "amod"
  }},
  "7": {{
    "text": "скульптор",
    "head": 8,
    "deprel": "nsubj"
  }},
  "8": {{
    "text": "споневажив",
    "head": 0,
    "deprel": "root"
  }},
  "9": {{
    "text": "саме",
    "head": 10,
    "deprel": "det"
  }},
  "10": {{
    "text": "мистецтво",
    "head": 8,
    "deprel": "obj"
  }},
  "11": {{
    "text": ":",
    "head": 14,
    "deprel": "punct"
  }},
  "12": {{
    "text": "Ти",
    "head": 14,
    "deprel": "nsubj"
  }},
  "13": {{
    "text": "не",
    "head": 14,
    "deprel": "advmod"
  }},
  "14": {{
    "text": "продався",
    "head": 8,
    "deprel": "parataxis"
  }},
  "15": {{
    "text": ",",
    "head": 17,
    "deprel": "punct"
  }},
  "16": {{
    "text": "–",
    "head": 17,
    "deprel": "punct"
  }},
  "17": {{
    "text": "гірше",
    "head": 14,
    "deprel": "parataxis"
  }},
  "18": {{
    "text": "!",
    "head": 17,
    "deprel": "punct"
  }}
}}


# Output constraints

1. Each annotated token in your answer should be exactly the annotated version of a token in the original dataset. Follow strictly the number of tokens in the original sentence as well as token themeselves, it is strictly prohibited to invent new tokens.
2. The providing of explanation, or just instruction instead of the result, is strictly prohibited. The next step in the pipeline is the immediate parsing of the result, so in case there is explanation, or code block denotation, or something like this, there will be an error.