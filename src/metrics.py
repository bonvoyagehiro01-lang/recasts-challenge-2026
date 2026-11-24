"""Official-metric implementations for RecSys Challenge 2026 (Music-CRS).

Follows the official evaluator (nlp4musa/music-crs-evaluator):
- nDCG@k with binary relevance, DCG = sum_i (2^rel_i - 1) / log2(i + 1),
  macro-averaged first over turns within a session, then over sessions.
- Catalog Diversity = unique recommended tracks / total catalog size.
- Lexical Diversity = Distinct-2 over all generated responses.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def dcg_at_k(ranked_ids: list[str], gt_id: str, k: int) -> float:
    dcg = 0.0
    for i, tid in enumerate(ranked_ids[:k], start=1):
        if tid == gt_id:
            dcg += (2**1 - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked_ids: list[str], gt_id: str, k: int) -> float:
    idcg = 1.0 / math.log2(2)  # single relevant item at rank 1
    return dcg_at_k(ranked_ids, gt_id, k) / idcg


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def distinct_2(responses: list[str]) -> float:
    total = 0
    uniq: set[tuple[str, str]] = set()
    for resp in responses:
        toks = tokenize(resp)
        bigrams = list(zip(toks, toks[1:]))
        total += len(bigrams)
        uniq.update(bigrams)
    if total == 0:
        return 0.0
    return len(uniq) / total


def evaluate(predictions: list[dict], ground_truth: dict[tuple[str, int], str],
             catalog_size: int, ks: tuple[int, ...] = (1, 10, 20)) -> dict:
    """predictions: list of {session_id, turn_number, predicted_track_ids, predicted_response}.
    ground_truth: {(session_id, turn_number): gt_track_id}."""
    per_session: dict[str, dict[int, list[float]]] = defaultdict(lambda: {k: [] for k in ks})
    all_tracks: set[str] = set()
    responses: list[str] = []
    missing = 0
    for pred in predictions:
        key = (pred["session_id"], int(pred["turn_number"]))
        gt = ground_truth.get(key)
        if gt is None:
            missing += 1
            continue
        ranked = list(dict.fromkeys(pred["predicted_track_ids"]))  # dedupe, keep order
        for k in ks:
            per_session[pred["session_id"]][k].append(ndcg_at_k(ranked, gt, k))
        all_tracks.update(ranked)
        responses.append(pred.get("predicted_response", ""))

    result: dict = {"n_sessions": len(per_session), "missing_predictions": missing}
    for k in ks:
        session_means = [sum(v[k]) / len(v[k]) for v in per_session.values() if v[k]]
        result[f"ndcg@{k}"] = float(sum(session_means) / len(session_means)) if session_means else 0.0
    result["catalog_diversity"] = len(all_tracks) / catalog_size if catalog_size else 0.0
    result["n_unique_tracks"] = len(all_tracks)
    result["lexical_diversity_distinct2"] = distinct_2(responses)
    # Provisional composite: LLM-as-a-Judge (weight 0.30) is NOT included.
    ndcg20 = result.get("ndcg@20", 0.0)
    partial = 0.50 * ndcg20 + 0.10 * result["catalog_diversity"] + 0.10 * result["lexical_diversity_distinct2"]
    result["composite_partial_excl_judge"] = partial
    result["composite_partial_renorm_0.7"] = partial / 0.7
    return result
