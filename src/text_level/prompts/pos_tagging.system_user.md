# General task outline

Your task is to perform a morphological annotation in the Universal Dependecies framework for the following pre-tokenized sentence: {sent}

# The idea

Your task is to transform a sentence, like "Великий Липник ." into a list of objects of the following structure:

[
    "1": {{
        "text": TOKEN 1,
        "upos": PART-OF-SPEECH IN THE UNIVERSAL DEPENDECIES FRAMEWORK,
        "feats": MORPHOLOGICAL CATEGORIES IN THE UNIVERSAL DEPENDENCIES FRAMEWORK
    }},
    "2": {{
        "text": TOKEN 2,
        "upos": PART-OF-SPEECH IN THE UNIVERSAL DEPENDECIES FRAMEWORK,
        "feats": MORPHOLOGICAL CATEGORIES IN THE UNIVERSAL DEPENDENCIES FRAMEWORK
    }},
    (...)
    "N": {{
        "text": TOKEN N
        "upos": PART-OF-SPEECH IN THE UNIVERSAL DEPENDECIES FRAMEWORK,
        "feats": MORPHOLOGICAL CATEGORIES IN THE UNIVERSAL DEPENDENCIES FRAMEWORK
    }}
]

# Input

1. A sentence in a pre-tokenised form.
2. The indexed tokens.

# Example of input

Продавши свій шедевр Меценатові , еллінський скульптор споневажив саме мистецтво : Ти не продався , – гірше !

1: Продавши
2: свій
3: шедевр
4: Меценатові
5: ,
6: еллінський
7: скульптор
8: споневажив
9: саме
10: мистецтво
11: :
12: Ти
13: не
14: продався
15: ,
16: –
17: гірше
18: !

=>

{{
  "1": {{
    "text": "Продавши",
    "upos": "VERB",
    "feats": "Aspect=Perf|Tense=Past|VerbForm=Conv"
  }},
  "2": {{
    "text": "свій",
    "upos": "DET",
    "feats": "Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs|Reflex=Yes"
  }},
  "3": {{
    "text": "шедевр",
    "upos": "NOUN",
    "feats": "Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing"
  }},
  "4": {{
    "text": "Меценатові",
    "upos": "PROPN",
    "feats": "Animacy=Anim|Case=Dat|Gender=Masc|NameType=Giv|Number=Sing"
  }},
  "5": {{
    "text": ",",
    "upos": "PUNCT",
    "feats": ""
  }},
  "6": {{
    "text": "еллінський",
    "upos": "ADJ",
    "feats": "Case=Nom|Gender=Masc|Number=Sing"
  }},
  "7": {{
    "text": "скульптор",
    "upos": "NOUN",
    "feats": "Animacy=Anim|Case=Nom|Gender=Masc|Number=Sing"
  }},
  "8": {{
    "text": "споневажив",
    "upos": "VERB",
    "feats": "Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin"
  }},
  "9": {{
    "text": "саме",
    "upos": "DET",
    "feats": "Case=Acc|Gender=Neut|Number=Sing|PronType=Prs|Reflex=Yes"
  }},
  "10": {{
    "text": "мистецтво",
    "upos": "NOUN",
    "feats": "Animacy=Inan|Case=Acc|Gender=Neut|Number=Sing"
  }},
  "11": {{
    "text": ":",
    "upos": "PUNCT",
    "feats": ""
  }},
  "12": {{
    "text": "Ти",
    "upos": "PRON",
    "feats": "Animacy=Anim|Case=Nom|Number=Sing|Person=2|PronType=Prs"
  }},
  "13": {{
    "text": "не",
    "upos": "PART",
    "feats": "Polarity=Neg"
  }},
  "14": {{
    "text": "продався",
    "upos": "VERB",
    "feats": "Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin"
  }},
  "15": {{
    "text": ",",
    "upos": "PUNCT",
    "feats": ""
  }},
  "16": {{
    "text": "–",
    "upos": "PUNCT",
    "feats": "PunctType=Dash"
  }},
  "17": {{
    "text": "гірше",
    "upos": "ADV",
    "feats": "Degree=Cmp"
  }},
  "18": {{
    "text": "!",
    "upos": "PUNCT",
    "feats": ""
  }}
}}


# Output constraints

1. Each annotated token in your answer should be exactly the annotated version of a token in the original dataset. Follow strictly the number of tokens in the original sentence as well as token themeselves, it is strictly prohibited to invent new tokens.
2. The providing of explanation, or just instruction instead of the result, is strictly prohibited. The output should be only like this:

[
    "1": {{
        "text": "Продавши",
        "upos": "VERB",
        "feats": "Aspect=Perf|Tense=Past|VerbForm=Conv"
    }},
    "2": {{
        "text": "свій",
        "upos": "DET",
        "feats": "Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing|Poss=Yes|PronType=Prs|Reflex=Yes"
    }},
]



The next step in the pipeline is the immediate parsing of the result, so in case there is explanation, or code block denotation, or something like this, there will be an error.

# Checks to perform after the input is generated

1. The number of objects in the generated array vs. the number of tokens. These should be equal. Otherwise, amend and regenerate.

