"""Explore the TalkPlayData-Challenge datasets: schema, sizes, samples."""
import glob
import json

import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)

BASE = "data/hf"


def show(path_pattern: str, n_rows: int = 2) -> None:
    files = sorted(glob.glob(path_pattern))
    print(f"\n===== {path_pattern} ({len(files)} files) =====")
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    print(f"rows={len(df):,}  cols={list(df.columns)}")
    for c in df.columns:
        s = df[c]
        sample = s.iloc[0]
        if isinstance(sample, (list, np.ndarray)):
            print(f"  - {c}: dtype={s.dtype}, elem example len={len(sample)}, first={np.asarray(sample).ravel()[:5]}")
        else:
            txt = str(sample)
            print(f"  - {c}: dtype={s.dtype}, example={txt[:150]!r}")
    print(df.head(n_rows).to_string()[:3000])


if __name__ == "__main__":
    show(f"{BASE}/TalkPlayData-Challenge-Dataset/data/train-*.parquet")
    show(f"{BASE}/TalkPlayData-Challenge-Dataset/data/test-*.parquet")
    show(f"{BASE}/TalkPlayData-Challenge-Track-Metadata/data/*.parquet")
    show(f"{BASE}/TalkPlayData-Challenge-User-Metadata/data/*.parquet")
    show(f"{BASE}/TalkPlayData-Challenge-Track-Embeddings/data/all_tracks-00000-of-00004.parquet")
    show(f"{BASE}/TalkPlayData-Challenge-User-Embeddings/data/train-*.parquet")
