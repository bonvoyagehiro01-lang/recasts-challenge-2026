"""End-to-end: build the baseline, predict on the dev set (HF 'test' split),
evaluate with the official metrics, and save predictions + scores.

Variants evaluated on identical per-turn channel scores:
  - random / popularity        (reference points; official dev numbers exist)
  - bm25_only                  (text channel only)
  - emb_only                   (cf + content channels)
  - fused                      (all channels; the proposed baseline)

Channel scores for the embedding channels are computed with batched GEMM
over all dev turns for speed on this small VM.
"""
from __future__ import annotations

import glob
import json
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from data_loading import load_conversations, load_track_metadata, session_turns  # noqa: E402
from metrics import evaluate  # noqa: E402
from recommender import BaselineRecommender  # noqa: E402

HF = "data/hf"


def _stack_valid(series: pd.Series, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack a column of variable-length vectors; return (matrix, valid_mask)."""
    arrs = series.to_numpy()
    valid = np.array([np.asarray(a).size == dim for a in arrs])
    mat = np.zeros((len(arrs), dim), dtype=np.float32)
    if valid.any():
        mat[valid] = np.stack([np.asarray(a, dtype=np.float32) for a in arrs[valid]])
    return mat, valid


def load_track_embeddings(track_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    idx = {t: i for i, t in enumerate(track_ids)}
    cf = np.zeros((len(track_ids), 128), dtype=np.float32)
    content = np.zeros((len(track_ids), 1024), dtype=np.float32)
    n_rows = n_cf = n_content = 0
    for f in sorted(glob.glob(f"{HF}/TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-*.parquet")):
        df = pd.read_parquet(f, columns=["track_id", "cf-bpr", "metadata-qwen3_embedding_0.6b"])
        pos = np.array([idx.get(t, -1) for t in df["track_id"]])
        keep = pos >= 0
        cf_mat, cf_ok = _stack_valid(df["cf-bpr"], 128)
        ct_mat, ct_ok = _stack_valid(df["metadata-qwen3_embedding_0.6b"], 1024)
        cf[pos[keep & cf_ok]] = cf_mat[keep & cf_ok]
        content[pos[keep & ct_ok]] = ct_mat[keep & ct_ok]
        n_rows += int(keep.sum()); n_cf += int((keep & cf_ok).sum()); n_content += int((keep & ct_ok).sum())
        print(f"  loaded {f.split('/')[-1]}")
    print(f"  embedding rows aligned: {n_rows:,}/{len(track_ids):,} catalog tracks "
          f"(cf-bpr present: {n_cf:,}, metadata-qwen3 present: {n_content:,})")
    return cf, content


def load_user_embeddings() -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for split in ["train", "test_warm"]:
        df = pd.read_parquet(f"{HF}/TalkPlayData-Challenge-User-Embeddings/data/{split}-00000-of-00001.parquet")
        for uid, v in zip(df["user_id"], df["cf-bpr"]):
            arr = np.asarray(v, dtype=np.float32)
            if arr.size == 128:
                out[uid] = arr
    return out


def batched_cosine(mat_n: np.ndarray, ctx: np.ndarray, chunk: int = 512) -> np.ndarray:
    """ctx: (T, d) context vectors -> scores (T, n_tracks). Rows with zero ctx stay zero."""
    out = np.zeros((ctx.shape[0], mat_n.shape[0]), dtype=np.float32)
    norms = np.linalg.norm(ctx, axis=1)
    for s in range(0, ctx.shape[0], chunk):
        block = ctx[s:s + chunk]
        bn = norms[s:s + chunk]
        safe = np.where(bn > 0, bn, 1.0)
        out[s:s + chunk] = (block / safe[:, None]) @ mat_n.T
        out[s:s + chunk][bn == 0] = 0.0
    return out


def main() -> None:
    t0 = time.time()
    print("[1/6] track metadata")
    tm = load_track_metadata()
    track_ids = tm["track_id"].tolist()
    id2idx = {t: i for i, t in enumerate(track_ids)}
    popularity = tm["popularity"].to_numpy(dtype=np.float64)

    def tags_of(x) -> list[str]:
        return [str(t) for t in x] if isinstance(x, (list, np.ndarray)) else []

    track_texts, track_meta = [], {}
    for tid, name, artist, album, tags in zip(
            tm["track_id"], tm["track_name"], tm["artist_name"], tm["album_name"], tm["tag_list"]):
        tag_list = tags_of(tags)
        track_texts.append(f"{name} {artist} {album} {' '.join(tag_list)}")
        track_meta[tid] = {"track_name": name, "artist_name": artist, "tags": tag_list}

    print("[2/6] track embeddings")
    cf_emb, content_emb = load_track_embeddings(track_ids)

    print("[3/6] user embeddings + dev conversations")
    user_embs = load_user_embeddings()
    print(f"  users with cf-bpr: {len(user_embs):,}")
    dev = load_conversations("test")
    print(f"  dev sessions: {len(dev):,}")

    print("[4/6] collect per-turn contexts")
    rec = BaselineRecommender(track_ids, track_texts, popularity, track_meta,
                              cf_emb=cf_emb, content_emb=content_emb)
    jobs = []  # per turn: dict with everything needed for scoring/ranking
    ground_truth: dict[tuple[str, int], str] = {}
    for _, row in dev.iterrows():
        sid, uid = row["session_id"], row["user_id"]
        turns = session_turns(row)
        user_emb = user_embs.get(uid)
        played: list[str] = []
        past_user_texts: list[str] = []
        for turn in turns:
            t = turn["turn_number"]
            ground_truth[(sid, t)] = turn["gt_track_id"]
            cf_parts, ct_parts = [], []
            if user_emb is not None:
                cf_parts.append(user_emb)
            for pt in played:
                j = id2idx.get(pt)
                if j is not None:
                    if np.any(cf_emb[j]):
                        cf_parts.append(cf_emb[j])
                    if np.any(content_emb[j]):
                        ct_parts.append(content_emb[j])
            jobs.append({
                "session_id": sid, "user_id": uid, "turn_number": t,
                "current_text": turn["user_text"], "past_user_texts": list(past_user_texts),
                "played": list(played),
                "cf_ctx": np.mean(cf_parts, axis=0) if cf_parts else np.zeros(128, dtype=np.float32),
                "ct_ctx": np.mean(ct_parts, axis=0) if ct_parts else np.zeros(1024, dtype=np.float32),
            })
            played.append(turn["gt_track_id"])
            past_user_texts.append(turn["user_text"])
    print(f"  turns: {len(jobs):,}  ({time.time() - t0:.0f}s)")

    print("[5/6] score + rank in chunks (embedding channels via batched GEMM)")
    variants = {
        "random": None,
        "popularity": None,
        "bm25_only": {"bm25": 1.0, "cf": 0.0, "content": 0.0, "pop": 0.0},
        "emb_only": {"bm25": 0.0, "cf": 1.0, "content": 0.7, "pop": 0.0},
        "fused": {"bm25": 1.0, "cf": 1.0, "content": 0.7, "pop": 0.15},
    }
    preds: dict[str, list[dict]] = {k: [] for k in variants}
    CH = 512
    for s in range(0, len(jobs), CH):
        block = jobs[s:s + CH]
        cf_s = batched_cosine(rec.cf_n, np.stack([j["cf_ctx"] for j in block]), chunk=CH)
        ct_s = batched_cosine(rec.content_n, np.stack([j["ct_ctx"] for j in block]), chunk=CH)
        for i, j in enumerate(block):
            q = j["current_text"] + " " + j["current_text"] + " " + " ".join(j["past_user_texts"])
            cs = {"bm25": rec.bm25.score(q), "cf": cf_s[i].astype(np.float64),
                  "content": ct_s[i].astype(np.float64)}
            if not np.any(j["ct_ctx"]):
                cs.pop("content")
            if not np.any(j["cf_ctx"]):
                cs.pop("cf")
            for name, w in variants.items():
                if name == "random":
                    ranked = rec.recommend_random(j["played"])
                elif name == "popularity":
                    ranked = rec.recommend_popularity(j["played"])
                else:
                    ranked = rec.rank(cs, j["played"], weights=w)
                resp = rec.make_response(ranked[0], j["current_text"], j["turn_number"]) if name == "fused" else ""
                preds[name].append({
                    "session_id": j["session_id"], "user_id": j["user_id"],
                    "turn_number": j["turn_number"], "predicted_track_ids": ranked,
                    "predicted_response": resp,
                })
        print(f"  scored turns {s + len(block):,}/{len(jobs):,} ({time.time() - t0:.0f}s)")

    print("[6/6] evaluate")

    results = {}
    for name, plist in preds.items():
        results[name] = evaluate(plist, ground_truth, catalog_size=len(track_ids))
        r = results[name]
        print(f"  {name:11s} nDCG@1={r['ndcg@1']:.4f} nDCG@10={r['ndcg@10']:.4f} "
              f"nDCG@20={r['ndcg@20']:.4f} catDiv={r['catalog_diversity']:.4f} "
              f"lexDiv={r['lexical_diversity_distinct2']:.4f} partial={r['composite_partial_excl_judge']:.4f}")

    with open("results/predictions_fused_dev.json", "w", encoding="utf-8") as f:
        json.dump(preds["fused"], f, ensure_ascii=False)
    with open("results/scores_dev.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved results/predictions_fused_dev.json, results/scores_dev.json  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
