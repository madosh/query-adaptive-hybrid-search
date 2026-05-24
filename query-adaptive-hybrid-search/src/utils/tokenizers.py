"""Language-specific tokenizers including trigram tokenization for CJK+ languages."""

import re
from typing import List, Optional

TRIGRAM_LANGUAGES = {"hi", "zh", "ja", "ko", "bn", "te"}

STOP_WORDS = {
    "en": {"a", "an", "the", "is", "it", "of", "in", "to", "and", "or", "for", "on", "at", "by", "with", "from", "as", "that", "this", "was", "are", "be", "has", "had", "have", "not", "but", "its", "do", "does"},
    "de": {"der", "die", "das", "ein", "eine", "und", "ist", "in", "von", "zu", "den", "mit", "auf", "für", "im", "dem", "nicht", "sich", "es", "auch", "an", "als"},
    "fr": {"le", "la", "les", "de", "du", "des", "un", "une", "et", "est", "en", "que", "qui", "dans", "pour", "pas", "sur", "ce", "il", "à"},
    "es": {"el", "la", "los", "las", "de", "del", "en", "un", "una", "que", "es", "y", "por", "con", "para", "se", "no", "al", "lo", "su"},
    "it": {"il", "lo", "la", "i", "gli", "le", "di", "del", "della", "un", "una", "e", "è", "in", "che", "non", "per", "con", "si", "da"},
    "pt": {"o", "a", "os", "as", "de", "do", "da", "em", "um", "uma", "que", "e", "é", "para", "com", "não", "por", "se", "no", "na"},
    "ru": {"и", "в", "не", "на", "с", "что", "это", "как", "он", "она", "по", "но", "из", "от", "за", "для", "к", "до", "о", "же"},
    "ar": {"في", "من", "على", "إلى", "أن", "هذا", "هذه", "التي", "الذي", "ما", "لا", "هو", "هي", "كان", "عن", "أو", "ذلك"},
}


def trigram_tokenize(text: str) -> List[str]:
    """Trigram-based tokenization for non-space-delimited languages.

    Generates character trigrams from the text after removing whitespace.
    Used for Hindi, Chinese, Japanese, Korean, Bengali, and Telugu
    as described in Section 3.2 (Weaviate-style trigram tokenization).
    """
    text = re.sub(r"\s+", "", text)
    if len(text) < 3:
        return [text] if text else []
    return [text[i : i + 3] for i in range(len(text) - 2)]


def whitespace_tokenize(text: str, language: str = "en") -> List[str]:
    """Whitespace tokenization with stop-word removal for space-delimited languages."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    stop = STOP_WORDS.get(language, set())
    return [t for t in tokens if t not in stop and len(t) > 1]


def get_tokenizer(language: str):
    """Return the appropriate tokenizer function for a language.

    Uses trigram tokenization for CJK+ languages (hi, zh, ja, ko, bn, te)
    and whitespace tokenization with stop-word filtering for others.
    """
    if language in TRIGRAM_LANGUAGES:
        return trigram_tokenize
    return lambda text: whitespace_tokenize(text, language)


def tokenize(text: str, language: str = "en") -> List[str]:
    """Tokenize text using the language-appropriate tokenizer."""
    tokenizer = get_tokenizer(language)
    return tokenizer(text)
