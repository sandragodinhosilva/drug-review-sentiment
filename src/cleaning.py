"""
Reusable text cleaning utilities for drug review datasets.

What this module provides
- clean_text: Cleans a single string (decode HTML entities, remove tags,
  replace URLs/emails with placeholders, drop control chars, normalize spaces).
- clean_series: Vectorized wrapper over clean_text for pandas Series.
- filter_reviews: Filters a DataFrame to remove too-short/empty texts and duplicates.
- ReviewCleaner: Minimal sklearn-style transformer for composing pipelines.

Design notes
- Stateless functions so they are safe to reuse across training, EDA, and inference.
- Conservative defaults: we avoid lowercasing or accent stripping to keep signal
  for cased models and domain-specific tokens unless you explicitly add it.
"""
from __future__ import annotations

import html
import re
from typing import Iterable, List
import logging
try:
    import src.logging_config as _logcfg  # type: ignore
    _LOGGER = logging.getLogger("my_app")
    _INFO = getattr(_logcfg, "info", _LOGGER.info)
    _DEBUG = getattr(_logcfg, "debug", _LOGGER.debug)
except Exception:  # Fallback if logging not configured yet
    _LOGGER = logging.getLogger("my_app")
    _INFO = _LOGGER.info
    _DEBUG = _LOGGER.debug

import pandas as pd


# Precompiled regexes for speed and readability.
# Matches any HTML tag-like pattern, non-greedy inside angle brackets.
_RE_TAGS = re.compile(r"<.*?>")
# Collapses any run of whitespace (spaces, tabs, newlines) into a single space.
_RE_WS = re.compile(r"\s+")
# Rough URL detector (http/https/www) to replace with a placeholder token.
_RE_URL = re.compile(r"(?i)\b((?:https?://|www\.)\S+)")
# Simple email detector to replace with a placeholder token.
_RE_EMAIL = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b")
# Non-printable/control characters range (ASCII control and C1 control block).
_RE_CTRL = re.compile(r"[\x00-\x1F\x7F-\x9F]")


def clean_text(text: object) -> str:
    """Clean a single review string.

    Pipeline:
    1) Coerce to string (make NaN/None into empty string).
    2) Decode HTML entities (e.g., "&amp;" -> "&").
    3) Remove HTML tags like "<br>" or "<p>".
    4) Replace URLs and emails with placeholder tokens to retain their presence
       without leaking specific identities or noise.
    5) Remove control characters that can break tokenizers/logging.
    6) Collapse repeated whitespace and trim.
    """
    # 1) Robust string coercion: handle None/NaN safely
    if not isinstance(text, str):
        text = "" if (text is None or (isinstance(text, float) and pd.isna(text))) else str(text)
    # 2) Decode HTML entities like &amp; -> &
    text = html.unescape(text)
    # 3) Strip HTML tags
    text = _RE_TAGS.sub("", text)
    # 4) Normalize URLs/emails to placeholders
    text = _RE_URL.sub("[URL]", text)
    text = _RE_EMAIL.sub("[EMAIL]", text)
    # 5) Remove control characters
    text = _RE_CTRL.sub(" ", text)
    # 6) Normalize whitespace
    text = _RE_WS.sub(" ", text).strip()
    return text


def clean_series(series: pd.Series) -> pd.Series:
    """Vectorized cleaning for a pandas Series -> Not “vectorized” in the NumPy sense; 
        its a simple, readable per-row apply that fits custom Python/regex logic.

    Applies clean_text to every element in a pandas Series 
    and returns a new cleaned Series.

    Example:
        Input: pd.Series(["<p>Great!</p>", None, "Visit https://site"])
        Output: pd.Series(["Great!", "", "Visit [URL]"])
    """
    # map runs clean_text element-by-element
    result = series.map(clean_text)
    try:
        _INFO(f"clean_series: cleaned {len(series)} texts")
    except Exception:
        pass
    return result


def filter_reviews(df: pd.DataFrame, text_col: str = "review", min_len: int = 5) -> pd.DataFrame:
    """Filter a cleaned reviews DataFrame.

    - Drops rows where the text is shorter than ``min_len`` (after cleaning).
    - Removes exact duplicate rows based on the text column only (so the same
      text with different metadata is considered a duplicate and dropped).
    """
    before = len(df)
    s = df[text_col].fillna("").astype(str)

    # Keep rows meeting the minimum length requirement
    mask = s.str.len() >= int(min_len)
    after_len = mask.sum()
    dropped_short = before - after_len
    df = df.loc[mask]

    # Remove exact text duplicates to reduce leakage/bias
    df_final = df.drop_duplicates(subset=[text_col])
    dropped_dupes = after_len - len(df_final)
    
    kept = len(df_final)
    try:
        _INFO(
            f"filter_reviews: start={before}, dropped_short={dropped_short}, "
            f"dropped_dupes={dropped_dupes}, kept={kept}, min_len={min_len}"
        )
    except Exception:
        pass
    return df_final


class ReviewCleaner:
    """Lightweight sklearn-style transformer for integration in pipelines.

    Example:
        rc = ReviewCleaner(min_len=5)
        texts_clean = rc.transform(["<p>Great!</p>", "Visit http://x" ])
    """

    def __init__(self, min_len: int = 0):
        self.min_len = int(min_len)

    def fit(self, X: Iterable[str], y=None):  # no-op for compatibility
        return self

    def transform(self, X: Iterable[object]) -> List[str]:
        # Apply the same cleaning as clean_series/clean_text
        cleaned = [clean_text(x) for x in X]
        # Optionally filter out too-short strings
        if self.min_len > 0:
            cleaned = [t for t in cleaned if len(t) >= self.min_len]
        return cleaned
