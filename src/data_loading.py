"""Load TalkPlayData-Challenge components from the local HF snapshot."""
from __future__ import annotations

import ast
import glob

import numpy as np
import pandas as pd

HF_DIR = "data/hf"


def load_conversations(split: str) -> pd.DataFrame:
    path = f"{HF_DIR}/TalkPlayData-Challenge-Dataset/data/{split}-00000-of-00001.parquet"
    return pd.read_parquet(path)


def load_track_metadata() -> pd.DataFrame:
    path = f"{HF_DIR}/TalkPlayData-Challenge-Track-Metadata/data/all_tracks-00000-of-00001.parquet"
    df = pd.read_parquet(path)

    def first(x):
        """Fields like track_name are stored as single-element lists
        (either real arrays or stringified lists)."""
        if isinstance(x, (list, tuple, np.ndarray)):
            return str(x[0]) if len(x) else ""
        if isinstance(x, str):
            try:
                v = ast.literal_eval(x)
                if isinstance(v, (list, tuple, np.ndarray)):
                    return str(v[0]) if len(v) else ""
                return str(v)
            except (ValueError, SyntaxError):
                return x
        return str(x)

    for col in ["track_name", "artist_name", "album_name"]:
        df[col] = df[col].map(first)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0.0)
    return df


def load_user_metadata() -> pd.DataFrame:
    path = f"{HF_DIR}/TalkPlayData-Challenge-User-Metadata/data/all_users-00000-of-00001.parquet"
    return pd.read_parquet(path)


def _stack_embeddings(files: list[str], id_col: str) -> tuple[list[str], np.ndarray, list[str]]:
    """Read embedding parquet shards; return (ids, matrix, embedding column names)."""
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)
    emb_cols = [c for c in df.columns if c != id_col and df[c].dtype == object
                and isinstance(df[c].iloc[0], (list, np.ndarray))]
    return df, emb_cols


def session_turns(row: pd.Series) -> list[dict]:
    """Expand one session row into per-turn dicts.

    Each turn: {turn_number, user_text, gt_track_id, assistant_text}.
    gt/assistant are the *observed* values in the data (used as history for
    later turns and as ground truth for that turn's evaluation).
    """
    by_turn: dict[int, dict] = {}
    for msg in row["conversations"]:
        m = dict(msg)
        t = int(m["turn_number"])
        by_turn.setdefault(t, {"turn_number": t, "user_text": "", "gt_track_id": None, "assistant_text": ""})
        if m["role"] == "user":
            by_turn[t]["user_text"] = m["content"]
        elif m["role"] == "music":
            by_turn[t]["gt_track_id"] = m["content"]
        elif m["role"] == "assistant":
            by_turn[t]["assistant_text"] = m["content"]
    return [by_turn[t] for t in sorted(by_turn)]
