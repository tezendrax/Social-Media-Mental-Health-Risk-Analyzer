"""
preprocessing.py
----------------
Advanced text preprocessing for the Mental Health Risk Analyzer.
Handles: lowercasing, URL removal, contraction expansion, stopword removal,
special character stripping, and feature signal extraction.
"""

import re
from typing import Tuple

# ── Contraction map ──────────────────────────────────────────────────────────
CONTRACTIONS = {
    "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
    "can't": "cannot", "won't": "will not", "don't": "do not",
    "doesn't": "does not", "didn't": "did not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "could've": "could have", "should've": "should have", "would've": "would have",
    "couldn't": "could not", "shouldn't": "should not", "wouldn't": "would not",
    "let's": "let us", "who's": "who is", "what's": "what is",
}

# ── Common English stopwords (minimal, keeping sentiment words) ───────────────
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
    "for", "of", "with", "by", "from", "is", "was", "are", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "you", "your", "he", "she", "him", "her",
    "as", "if", "then", "than", "so", "up", "out", "about", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "same", "too", "very",
    "just", "also", "how", "all", "while", "which", "who", "whom",
}

# ── Negation words (kept intentionally, signal for sentiment) ─────────────────
NEGATION_WORDS = {"not", "no", "never", "none", "nothing", "nobody",
                  "nowhere", "neither", "nor", "cannot", "without"}


def expand_contractions(text: str) -> str:
    """Expand English contractions: i'm → i am, won't → will not, etc."""
    for contraction, expansion in CONTRACTIONS.items():
        text = re.sub(re.escape(contraction), expansion, text, flags=re.IGNORECASE)
    return text


def remove_urls(text: str) -> str:
    """Remove http/https URLs and bare www. links."""
    return re.sub(r'https?://\S+|www\.\S+', '', text)


def remove_mentions_hashtags(text: str) -> str:
    """Remove @mentions and #hashtags (keep the word after # for context)."""
    text = re.sub(r'@\w+', '', text)                  # remove @mention
    text = re.sub(r'#(\w+)', r'\1', text)              # #depression → depression
    return text


def remove_special_characters(text: str) -> str:
    """Remove non-alphabetic characters (keep spaces)."""
    return re.sub(r'[^a-z\s]', '', text)


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords but preserve negation words for sentiment accuracy."""
    return [t for t in tokens if t not in STOPWORDS or t in NEGATION_WORDS]


def clean_text(text: str, remove_stops: bool = True) -> str:
    """
    Full preprocessing pipeline for a raw social media post.

    Steps:
        1. Lowercase
        2. Expand contractions
        3. Remove URLs
        4. Remove @mentions, expand #hashtags
        5. Remove special characters
        6. Tokenize
        7. Optionally remove stopwords
        8. Rejoin

    Args:
        text: Raw input string.
        remove_stops: Whether to remove stopwords (default True).

    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = expand_contractions(text)
    text = remove_urls(text)
    text = remove_mentions_hashtags(text)
    text = remove_special_characters(text)

    tokens = text.split()
    if remove_stops:
        tokens = remove_stopwords(tokens)

    # Remove very short tokens (1 char)
    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens).strip()


def extract_signals(raw_text: str) -> dict:
    """
    Extract linguistic and behavioral signals from raw (uncleaned) text.
    These are used as supplementary features alongside TF-IDF.

    Returns:
        Dictionary of named signal floats (all in [0, 1] or small int range).
    """
    if not isinstance(raw_text, str) or not raw_text.strip():
        return _empty_signals()

    text_lower = raw_text.lower()
    tokens = raw_text.split()
    word_count = max(len(tokens), 1)

    # ── Risk keyword density ──────────────────────────────────────────────────
    high_risk_words = {
        "depressed", "depression", "hopeless", "suicidal", "suicide",
        "worthless", "empty", "numb", "give up", "cant go on", "no reason",
        "no point", "end it", "kill myself", "hate myself", "self harm",
        "self-harm", "cutting", "overdose", "die", "death", "dead inside"
    }
    mod_risk_words = {
        "anxious", "anxiety", "panic", "stress", "stressed", "overwhelmed",
        "exhausted", "tired", "lonely", "alone", "sad", "crying", "cry",
        "miserable", "broken", "hurt", "pain", "scared", "fear",
        "worried", "struggling", "difficult", "hard time"
    }
    positive_words = {
        "happy", "joy", "excited", "grateful", "thankful", "love",
        "amazing", "wonderful", "great", "good", "better", "hopeful",
        "motivated", "inspired", "peaceful", "content", "proud"
    }

    high_risk_count = sum(1 for w in high_risk_words if w in text_lower)
    mod_risk_count  = sum(1 for w in mod_risk_words  if w in text_lower)
    positive_count  = sum(1 for w in positive_words   if w in text_lower)

    # ── Structural signals ────────────────────────────────────────────────────
    exclamation_density   = min(raw_text.count('!') / word_count, 1.0)
    question_density      = min(raw_text.count('?') / word_count, 1.0)
    ellipsis_density      = min(raw_text.count('...') / word_count, 1.0)
    caps_ratio            = sum(1 for c in raw_text if c.isupper()) / max(len(raw_text), 1)
    negation_count        = sum(1 for t in tokens if t.lower() in NEGATION_WORDS)
    avg_word_len          = sum(len(t) for t in tokens) / word_count

    return {
        "high_risk_density":    min(high_risk_count / word_count * 10, 1.0),
        "mod_risk_density":     min(mod_risk_count  / word_count * 5,  1.0),
        "positive_density":     min(positive_count  / word_count * 5,  1.0),
        "exclamation_density":  exclamation_density,
        "question_density":     question_density,
        "ellipsis_density":     ellipsis_density,
        "caps_ratio":           caps_ratio,
        "negation_count":       min(negation_count / word_count, 1.0),
        "avg_word_len":         min(avg_word_len / 10, 1.0),
        "text_length_norm":     min(word_count / 100, 1.0),
    }


def _empty_signals() -> dict:
    return {
        "high_risk_density": 0.0, "mod_risk_density": 0.0, "positive_density": 0.0,
        "exclamation_density": 0.0, "question_density": 0.0, "ellipsis_density": 0.0,
        "caps_ratio": 0.0, "negation_count": 0.0, "avg_word_len": 0.0,
        "text_length_norm": 0.0,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "I'm feeling really anxious today... http://example.com #sad @friend",
        "I can't go on anymore. Everything feels hopeless and I don't see the point.",
        "Just had an AMAZING day at the park! So grateful for life!",
        "So stressed about exams. I don't know what to do, I'm exhausted.",
    ]
    for s in samples:
        print(f"\nOriginal : {s}")
        print(f"Cleaned  : {clean_text(s)}")
        signals = extract_signals(s)
        print(f"Signals  : {signals}")
