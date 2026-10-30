"""Text processing shared by training, duplicate checks and prediction."""
from functools import lru_cache
import html
import re

from nltk.stem import PorterStemmer

_STEMMER = PorterStemmer()
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION = re.compile(r"@\w+")
_WORDS = re.compile(r"[a-z]+")
_EMOTICONS = {":)": " smile ", ":-)": " smile ", ";)": " wink ",
              ";-)": " wink ", ":d": " smile ", ":-d": " smile ",
              ":(": " sad ", ":-(": " sad ", ":p": " playful "}
_EMOTICON_PATTERN = re.compile("|".join(re.escape(k) for k in sorted(_EMOTICONS, key=len, reverse=True)))


@lru_cache(maxsize=100000)
def _stem(word):
    return _STEMMER.stem(word)


def tokenize_text(text):
    """Keep negation and sentiment words without downloading NLTK corpora."""
    if not isinstance(text, str):
        raise TypeError("Expected a string")
    text = html.unescape(text).lower().replace("\u2019", "'")
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _EMOTICON_PATTERN.sub(lambda match: _EMOTICONS[match.group()], text)
    text = text.replace("won't", "will not").replace("can't", "can not")
    text = re.sub(r"\b(\w+)n't\b", r"\1 not", text)
    text = re.sub(r"([a-z])\1{2,}", r"\1\1", text)
    return [_stem(word) for word in _WORDS.findall(text) if len(word) > 1]


def text_key(text):
    """Match texts that become identical after model preprocessing."""
    return " ".join(tokenize_text(text))
