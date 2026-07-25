"""
query_cache.py

In-memory cache for /query responses. Repeated questions (common during
demos, eval runs, and anyone re-asking the same thing) currently re-embed,
re-retrieve, and re-generate from scratch every time — wasting Groq's
scarce daily token quota on work already done.

Deliberately exact-match only (normalized whitespace/case), not semantic
near-duplicate matching — a similarity-threshold cache risks returning a
cached answer for a subtly different question (e.g. "Q2FY25" vs "Q3FY25"
figures), which is worse than a cache miss. Exact-match still covers the
common case: the same question asked twice.

Usage:
    from api.query_cache import QueryCache

    cache = QueryCache(max_entries=500, ttl_seconds=3600)
    key = cache.make_key(query, company, period, doc_type, top_k)
    hit = cache.get(key)
    if hit is None:
        result = ...  # do the real work
        cache.set(key, result)
"""

import re
import time
from collections import OrderedDict
from typing import Any, Optional


def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace so trivial formatting differences
    ("What was TCS revenue?" vs "what was tcs revenue?  ") still hit."""
    return re.sub(r"\s+", " ", text.strip().lower())


class QueryCache:
    def __init__(self, max_entries: int = 500, ttl_seconds: int = 3600):
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._store: "OrderedDict[tuple, tuple[float, Any]]" = OrderedDict()

    def make_key(
        self,
        query: str,
        company: Optional[str],
        period: Optional[str],
        doc_type: Optional[str],
        top_k: int,
    ) -> tuple:
        return (_normalize(query), company, period, doc_type, top_k)

    def get(self, key: tuple) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None

        stored_at, value = entry
        if time.time() - stored_at > self.ttl_seconds:
            del self._store[key]
            return None

        # LRU touch — move to the end so recently-used entries survive eviction
        self._store.move_to_end(key)
        return value

    def set(self, key: tuple, value: Any) -> None:
        self._store[key] = (time.time(), value)
        self._store.move_to_end(key)
        while len(self._store) > self.max_entries:
            self._store.popitem(last=False)  # evict least-recently-used

    def stats(self) -> dict:
        return {"entries": len(self._store), "max_entries": self.max_entries, "ttl_seconds": self.ttl_seconds}
