import re
import nltk
import unicodedata
import string

# This is a very basic tokenizer.
# Lowering (optional) + Accent removal + Punctuation splitting and removal + whitespace tokenizer

def strip_accents(text):  
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
        cat = unicodedata.category(char)
        if cat == "Mn":
            continue
        output.append(char)
    return "".join(output)


def is_punctuation(char):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

def split_on_punc(text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
        char = chars[i]
        if _is_punctuation(char):
            output.append([char])
            start_new_word = True
        else:
            if start_new_word:
                output.append([])
            start_new_word = False
            output[-1].append(char)
        i += 1

    return ["".join(x) for x in output]

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

# Apply WordPiece Tokenization https://github.com/google-research/bert#tokenization
def _tokenize(sentence, uncased=True, unaccented=True):
    # Uncased
    if uncased:
        sentence = sentence.lower()
    
    # Unaccented
    if unaccented:
        sentence = strip_accents(sentence)
    
    sentence = nltk.WordPunctTokenizer().tokenize(sentence)
    #sentence = "_".join(sentence)
    sentence = [word for word in sentence if not re.fullmatch('[' + string.punctuation + ']+', word)]
    return sentence

