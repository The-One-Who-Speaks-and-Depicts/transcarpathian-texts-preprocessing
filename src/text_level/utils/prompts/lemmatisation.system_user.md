# General task outline

Your task is to perform a lemmatisation in the Universal Dependecies framework for the following pre-tokenized sentence: {sent}

# Input

1. A sentence in a pre-tokenised form.
2. The indexed tokens that can either contain information on part-of-speech and morphological tagging, or not.

# Example of input

Продавши свій шедевр Меценатові , еллінський скульптор споневажив саме мистецтво : Ти не продався , – гірше !

1: Продавши VERB Aspect=Perf|Tense=Past|VerbForm=Conv
2: свій DET Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs|Reflex=Yes
3: шедевр NOUN Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing
4: Меценатові PROPN Animacy=Anim|Case=Dat|Gender=Masc|NameType=Giv|Number=Sing
5: , PUNCT _
6: еллінський ADJ Case=Nom|Gender=Masc|Number=Sing
7: скульптор NOUN Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing
8: споневажив VERB Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin
9: саме DET Case=Acc|Gender=Neut|Number=Sing|PronType=Prs|Reflex=Yes
10: мистецтво NOUN Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing
11: : PUNCT _
12: Ти PRON Animacy=Anim|Case=Nom|Number=Sing|Person=2|PronType=Prs
13: не PART Polarity=Neg
14: продався VERB Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin
15: , PUNCT _
16: – PUNCT PunctType=Dash
17: гірше ADV Degree=Cmp
18: ! PUNCT _

=>

{{
  "1": {{
    "text": "Продавши",
    "lemma": "продати"
  }},
  "2": {{
    "text": "свій",
    "lemma": "свій"
  }},
  "3": {{
    "text": "шедевр",
    "lemma": "шедевр"
  }},
  "4": {{
    "text": "Меценатові",
    "lemma": "Меценат"
  }},
  "5": {{
    "text": ",",
    "lemma": ","
  }},
  "6": {{
    "text": "еллінський",
    "lemma": "еллінський"
  }},
  "7": {{
    "text": "скульптор",
    "lemma": "скульптор"
  }},
  "8": {{
    "text": "споневажив",
    "lemma": "споневажити"
  }},
  "9": {{
    "text": "саме",
    "lemma": "самий"
  }},
  "10": {{
    "text": "мистецтво",
    "lemma": "мистецтво"
  }},
  "11": {{
    "text": ":",
    "lemma": ":"
  }},
  "12": {{
    "text": "Ти",
    "lemma": "ти"
  }},
  "13": {{
    "text": "не",
    "lemma": "не"
  }},
  "14": {{
    "text": "продався",
    "lemma": "продатися"
  }},
  "15": {{
    "text": ",",
    "lemma": ","
  }},
  "16": {{
    "text": "–",
    "lemma": "–"
  }},
  "17": {{
    "text": "гірше",
    "lemma": "гірше"
  }},
  "18": {{
    "text": "!",
    "lemma": "!"
  }}
}}

# Output constraints

1. Each annotated token in your answer should be exactly the annotated version of a token in the original dataset. Follow strictly the number of tokens in the original sentence as well as token themeselves, it is strictly prohibited to invent new tokens.
2. The providing of explanation, or just instruction instead of the result, is strictly prohibited. The next step in the pipeline is the immediate parsing of the result, so in case there is explanation, or code block denotation, or something like this, there will be an error.
