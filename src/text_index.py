"""Lightweight BM25 index over track metadata text (numpy implementation)."""
from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n = len(docs)
        tokenized = [tokenize(d) for d in docs]
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if self.n else 0.0
        df: Counter = Counter()
        for toks in tokenized:
            df.update(set(toks))
        self.vocab = {t: i for i, t in enumerate(df)}
        self.idf = np.zeros(len(self.vocab))
        for t, i in self.vocab.items():
            self.idf[i] = math.log(1 + (self.n - df[t] + 0.5) / (df[t] + 0.5))
        # term-frequency matrix in CSR-like dict-of-arrays form
        self.postings: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        tmp: dict[int, list[tuple[int, int]]] = {}
        for doc_id, toks in enumerate(tokenized):
            for term, cnt in Counter(toks).items():
                tmp.setdefault(self.vocab[term], []).append((doc_id, cnt))
        for term_id, lst in tmp.items():
            arr = np.array(lst)
            self.postings[term_id] = (arr[:, 0], arr[:, 1].astype(np.float64))

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float64)
        q_terms = [self.vocab[t] for t in tokenize(query) if t in self.vocab]
        if not q_terms:
            return scores
        for term_id in q_terms:
            docs, tfs = self.postings[term_id]
            denom = tfs + self.k1 * (1 - self.b + self.b * self.doc_len[docs] / max(self.avgdl, 1e-9))
            scores[docs] += self.idf[term_id] * (tfs * (self.k1 + 1) / denom)
        return scores
