# RecSys Challenge 2026 — Conversational Music Recommendation（練習実施）

ACM RecSys Challenge 2026「Music-CRS」（会話型音楽推薦）の **練習実施プロジェクト**。
コンペは 2026年4〜6月に開催・終了済みであり、本リポジトリは提出用ではなく、
推薦システム＋LLM＋評価設計の実証を目的に、公式データ・公式指標で
ベースライン構築とオフライン評価を一貫して再現したものである。

- 公式サイト: https://www.recsyschallenge.com/2026/
- チャレンジサイト: https://nlp4musa.github.io/music-crs-challenge/
- 公式 evaluator: https://github.com/nlp4musa/music-crs-evaluator
- データ: Hugging Face `talkpl-ai`（TalkPlayData-Challenge 系、CC BY-NC 4.0、研究・学習目的のみ。再配布不可のため本リポジトリにはデータを含めない）

## タスク

会話履歴とユーザー文脈から、そのターンでユーザーが求める楽曲を
カタログ全47,071曲からランキングし、上位20曲と応答文を出力する。

- 入力: セッション内の過去の会話（ユーザ発話・再生済み楽曲・応答）、ユーザープロファイル
- 出力（公式推論フォーマット）: `session_id`, `user_id`, `turn_number`,
  `predicted_track_ids`（重複なし・順序付き・最大20）, `predicted_response`
- 制約: 候補はカタログ全体から取得すること（`track_split_types` 等での絞り込み禁止）

## 評価指標（公式複合スコア）

```
Score = 0.50 × nDCG@20 + 0.10 × Catalog Diversity + 0.10 × Lexical Diversity + 0.30 × LLM-as-a-Judge
```

| 指標 | 定義 |
|---|---|
| nDCG@20 | 正解楽曲は各ターン1曲（バイナリ関連度）。DCG = Σ (2^rel − 1)/log2(i+1)。ターン→セッションの順にマクロ平均 |
| Catalog Diversity | 全予測で推薦されたユニーク楽曲数 ÷ カタログサイズ（47,071） |
| Lexical Diversity | 全応答文に対する Distinct-2（ユニークな bigram 数 ÷ 総 bigram 数） |
| LLM-as-a-Judge | Gemini 系モデルが Personalization / Explanation Quality を1–5で採点し [0,1] に正規化（Blind セットのみ・プロンプト非公開） |

本プロジェクトでは LLM API が利用できないため LLM-as-a-Judge は実施せず、
複合スコアは残り3指標の **部分スコア（満点0.7）** と **0.7で再正規化した暫定値** を併記する。

## データ概要（TalkPlayData-Challenge）

LLM エージェントパイプラインで生成された合成会話コーパス（LFM-2b 由来の実ユーザー属性・聴取履歴に接地）。

| コンポーネント | 内容 |
|---|---|
| TalkPlayData-Challenge-Dataset | 会話。train 15,199 セッション / dev（HF名 test）1,000 セッション。各8ターン、各ターンは user発話・music（正解 track_id）・assistant応答の3メッセージ |
| TalkPlayData-Challenge-Track-Metadata | カタログ 47,071 曲（曲名・アーティスト・アルバム・タグ・popularity・リリース日等） |
| TalkPlayData-Challenge-User-Metadata | ユーザー 8,772 人（年齢・性別・国） |
| TalkPlayData-Challenge-Track-Embeddings | 曲埋め込み: audio-laion_clap 512d / image-siglip2 768d / cf-bpr 128d / attributes・lyrics・metadata-qwen3 1024d |
| TalkPlayData-Challenge-User-Embeddings | ユーザー cf-bpr 128d（train 8,591人、test_warm 371人、test_cold 129人は空ベクトル） |

dev セッション内訳: warm 800 / cold 200。評価は dev セット（正解が公開されている唯一の評価用スプリット）で実施。

## 手法（ベースライン）

候補生成（全カタログ対象）→ ランキングの2段構成。詳細は `src/recommender.py`。

- **BM25 チャネル**: 曲テキスト（曲名・アーティスト・アルバム・タグ）に対する BM25。クエリは会話のユーザ発話（現ターンを重み付け）
- **CF チャネル**: ユーザー cf-bpr と再生済み曲 cf-bpr の平均ベクトルと全曲 cf-bpr のコサイン類似度
- **Content チャネル**: 再生済み曲の metadata 埋め込み平均と全曲 metadata 埋め込みのコサイン類似度
- **Popularity prior**: popularity の正規化スコア
- **融合**: チャネルごとの候補 union 上で min-max 正規化したスコアの重み付き和。再生済み曲は除外
- **応答文**: top-1 の曲名・アーティストとユーザ発話のキーワードを用いたテンプレート生成

アブレーションとして random / popularity / BM25のみ / 埋め込みのみ / 融合 の5条件を同一条件で評価する。

## 再現手順

```bash
python3 -m venv .venv && .venv/bin/pip install huggingface_hub datasets pyarrow numpy pandas scikit-learn

# データ取得（全て公開データセット。アカウント不要）
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download, hf_hub_download
for repo in ["TalkPlayData-Challenge-Dataset", "TalkPlayData-Challenge-Track-Metadata",
             "TalkPlayData-Challenge-User-Metadata", "TalkPlayData-Challenge-User-Embeddings"]:
    snapshot_download(f"talkpl-ai/{repo}", repo_type="dataset", local_dir=f"data/hf/{repo}")
# トラック埋め込み（約690MB）はファイル単位の取得が安定する
for i in range(4):
    hf_hub_download("talkpl-ai/TalkPlayData-Challenge-Track-Embeddings",
                    f"data/all_tracks-0000{i}-of-00004.parquet", repo_type="dataset",
                    local_dir="data/hf/TalkPlayData-Challenge-Track-Embeddings")
EOF

# 予測 + 評価（dev セット全1,000セッション×8ターン）
.venv/bin/python src/run_evaluation.py
# -> results/predictions_fused_dev.json（公式推論フォーマット）
# -> results/scores_dev.json
```

## ディレクトリ構成

```
src/
  data_loading.py   データ読み込み・セッション展開
  text_index.py     BM25 インデックス（numpy 実装）
  recommender.py    ベースライン（候補生成・融合・テンプレート応答）
  metrics.py        公式指標の実装（nDCG / Catalog Diversity / Distinct-2）
  run_evaluation.py 予測と評価のエンドツーエンド実行
  explore_data.py   データ構造の確認用
results/            予測 JSON とスコア
REPORT.md           実施結果の報告書
data/               ローカルデータ（git 管理外）
```

## ライセンス・注意

- データセットは CC BY-NC 4.0（非商用・研究目的）。再配布は禁止されているため、データ実体は本リポジトリに含めない。
- 本プロジェクトのコードは学習・研究目的で作成したもの。
