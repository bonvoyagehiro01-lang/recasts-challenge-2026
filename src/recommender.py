"""Baseline conversational music recommender.

Pipeline (per turn), candidate generation over the FULL catalog, then ranking:

  Channels
  - bm25: BM25 over track text (name/artist/album/tags), queried with the
    conversation's user text (current request duplicated for weight, plus
    earlier user turns).
  - cf: cosine similarity between a context vector (user cf-bpr embedding +
    mean of already-played tracks' cf-bpr) and all track cf-bpr embeddings.
  - content: cosine similarity between the mean of already-played tracks'
    metadata (Qwen3) embeddings and all track metadata embeddings.
  - pop: popularity prior.

  Ranking = weighted fusion of per-channel min-max-normalized scores over the
  union of per-channel top candidates. Already-played tracks are excluded.
  Response = template-based, grounded in the top-1 pick.
"""
from __future__ import annotations

import random

import numpy as np

from text_index import BM25Index, tokenize

_STOP = set("the a an and or of to for with i me my please some more like want looking play song songs track tracks music".split())

DEFAULT_WEIGHTS = {"bm25": 1.0, "cf": 1.0, "content": 0.7, "pop": 0.15}


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _cosine_scores(mat_n: np.ndarray, vec: np.ndarray) -> np.ndarray:
    v = vec / max(np.linalg.norm(vec), 1e-9)
    return mat_n @ v


class BaselineRecommender:
    def __init__(self, track_ids: list[str], track_texts: list[str],
                 popularity: np.ndarray, track_meta: dict[str, dict],
                 cf_emb: np.ndarray | None = None, content_emb: np.ndarray | None = None,
                 cand_per_channel: int = 300):
        self.track_ids = track_ids
        self.id2idx = {t: i for i, t in enumerate(track_ids)}
        self.popularity = popularity.astype(np.float64)
        self.pop_norm = _norm01(self.popularity)
        self.bm25 = BM25Index(track_texts)
        self.meta = track_meta
        self.cand = cand_per_channel
        self.rng = random.Random(42)

        def _prep(mat):
            if mat is None:
                return None
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            return mat / np.maximum(norms, 1e-9)

        self.cf_raw = cf_emb
        self.content_raw = content_emb
        self.cf_n = _prep(cf_emb)
        self.content_n = _prep(content_emb)

    # ---------- channel scores (full catalog) ----------
    def channel_scores(self, user_emb: np.ndarray | None, played_track_ids: list[str],
                       user_texts: list[str], current_text: str) -> dict[str, np.ndarray]:
        scores: dict[str, np.ndarray] = {}
        q = current_text + " " + current_text + " " + " ".join(user_texts)
        scores["bm25"] = self.bm25.score(q)

        if self.cf_n is not None:
            parts = []
            if user_emb is not None and np.linalg.norm(user_emb) > 0:
                parts.append(user_emb)
            if self.cf_raw is not None:
                parts.extend(self.cf_raw[self.id2idx[t]] for t in played_track_ids if t in self.id2idx)
            if parts:
                scores["cf"] = _cosine_scores(self.cf_n, np.mean(parts, axis=0))

        if self.content_n is not None and self.content_raw is not None and played_track_ids:
            vecs = [self.content_raw[self.id2idx[t]] for t in played_track_ids if t in self.id2idx]
            if vecs:
                scores["content"] = _cosine_scores(self.content_n, np.mean(vecs, axis=0))
        return scores

    # ---------- fusion / ranking ----------
    def rank(self, channel_scores: dict[str, np.ndarray], played_track_ids: list[str],
             weights: dict | None = None, topk: int = 20) -> list[str]:
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            w.update(weights)
        n = len(self.track_ids)
        played_idx = {self.id2idx[t] for t in played_track_ids if t in self.id2idx}

        cand: set[int] = set()
        for name, scores in channel_scores.items():
            if w.get(name, 0) <= 0:
                continue
            k = min(self.cand, n - 1)
            top = np.argpartition(-scores, k)[:k]
            cand.update(int(i) for i in top if scores[i] > 0)
        pop_top = np.argpartition(-self.popularity, 99)[:100]
        cand.update(int(i) for i in pop_top)
        cand -= played_idx
        if not cand:
            cand = set(int(i) for i in pop_top) - played_idx

        cand_arr = np.fromiter(cand, dtype=int)
        fused = w.get("pop", 0.0) * self.pop_norm[cand_arr]
        for name, scores in channel_scores.items():
            if w.get(name, 0) > 0:
                fused = fused + w[name] * _norm01(scores[cand_arr])
        order = cand_arr[np.argsort(-fused)]
        return [self.track_ids[i] for i in order[:topk]]

    def recommend(self, user_emb, played_track_ids, user_texts, current_text,
                  weights: dict | None = None, topk: int = 20) -> list[str]:
        cs = self.channel_scores(user_emb, played_track_ids, user_texts, current_text)
        return self.rank(cs, played_track_ids, weights=weights, topk=topk)

    # ---------- simple baselines ----------
    def recommend_popularity(self, played_track_ids: list[str], topk: int = 20) -> list[str]:
        order = np.argsort(-self.popularity)
        out, played = [], set(played_track_ids)
        for i in order:
            t = self.track_ids[int(i)]
            if t not in played:
                out.append(t)
            if len(out) >= topk:
                break
        return out

    def recommend_random(self, played_track_ids: list[str], topk: int = 20) -> list[str]:
        pool = [t for t in self.track_ids if t not in set(played_track_ids)]
        return self.rng.sample(pool, topk)

    # ---------- template response ----------
    def make_response(self, top_track_id: str, current_text: str, turn_number: int) -> str:
        m = self.meta.get(top_track_id, {})
        name, artist = m.get("track_name", "this track"), m.get("artist_name", "")
        tags = [t for t in tokenize(" ".join(m.get("tags", []))) if t not in _STOP][:3]
        keywords = [t for t in tokenize(current_text) if t not in _STOP and len(t) > 3][:3]
        openers = [
            "Great pick coming up!",
            "Here's one I think you'll love.",
            "Based on what you've been enjoying, try this.",
            "This should fit the mood perfectly.",
            "Let me queue this one up for you.",
            "I found something right in that vein.",
            "Keeping the vibe going with this one.",
            "Here's another gem for you.",
        ]
        opener = openers[(turn_number - 1) % len(openers)]
        by = f" by {artist}" if artist else ""
        mid = f'"{name}"{by}'
        reasons = []
        if keywords:
            reasons.append(f"it matches your request for {' '.join(keywords)}")
        if tags:
            reasons.append(f"it has a {' / '.join(tags)} feel")
        tail = (" — " + " and ".join(reasons) + ".") if reasons else "."
        closers = ["Enjoy!", "Hope it hits the spot.", "Let me know what you think.",
                   "Happy listening!", "Give it a spin.", "Enjoy the track."]
        closer = self.rng.choice(closers)
        return f"{opener} {mid}{tail} {closer}"
