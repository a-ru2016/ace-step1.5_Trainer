# ACE-Step 1.5 - フォーク機能

このリポジトリは、[ACE-Step 1.5](https://github.com/ACE-Step/ACE-Step-1.5) をベースに、Gemini と Qwen を活用して拡張された機能を含んでいます。

## 追加機能

### 1. 音楽データセット前処理スクリプト (`preprocess_music_dataset.py`)

**概要**: 音楽ファイルからトレーニングデータを自動生成する包括的な前処理パイプライン

**主な機能**:

#### 自動メタデータ生成
- **Gemini API 統合**: 音声ファイルからキャプションと構造タグ（[Verse], [Chorus] など）を同時生成
- **ACE-Step 内蔵 LLM**: GUI と同じ方式でメタデータ・歌詞・コード進行を生成
- **Qwen2.5-Omni 4bit 推論**: 軽量量子化モデルによる効率的なバッチ処理

#### 音楽分析機能
- **コード進行推定**: librosa を使用したビート同期のコード進行分析
- **キー検出**: クロマ特徴量による調性判定（例：C Major, A Minor）
- **BPM 検出**: ビートトラッキングによるテンポ分析
- **和音テンプレートマッチング**: 30 種類以上の和音タイプを識別

#### 歌詞抽出
- **Faster-Whisper large-v3**: 高精度な歌詞トランスクリプション
- **Demucs 音声分離**: ボーカルと伴奏の分離による歌詞抽出精度向上
- **自動ブラックリスト**: 「ご視聴ありがとうございました」などの不要テキストをフィルタリング

**使用方法**:
```bash
python preprocess_music_dataset.py --music_dir /path/to/music --language ja
```

---

### 2. ComfyUI 統合 (`custom_nodes/ComfyUI-ACE-Step-LLM-LoKr`)

**概要**: ACE-Step 1.5 の LLM (5Hz-LM/Qwen3 ベース) 用 LoKr アダプターを ComfyUI で使用するためのカスタムノード

**ノード一覧**:

#### ACE-Step LLM LoKr Loader
学習済みの LoKr アダプターを LLM モデルに適用します。

**入力**:
- `clip`: ACE-Step 1.5 の CLIP（`Load CLIP` ノードから）
- `lokr_path`: loras ディレクトリからの LoKr ウェイトファイル（ドロップダウン選択）
- `strength`: アダプターの強度（0.0〜2.0、デフォルト：1.0）

**出力**:
- `clip`: LoKr アダプターが適用された CLIP

#### ACE-Step LLM LoKr Save
CLIP から LoKr アダプターを保存します。

**入力**:
- `clip`: LoKr アダプターを含む CLIP
- `output_path`: 出力パス

#### ACE-Step LLM LoKr Apply (Advanced)
詳細設定付きで LoKr アダプターを適用します。

**追加入力**:
- `target_modules`: 対象モジュール（カンマ区切り）
- `rank`: LoKr ランク（新規作成時）
- `alpha`: LoKr アルファ（新規作成時）

**ワークフロー例**:
```
[Load CLIP] → [ACE-Step LLM LoKr Loader] → [TextEncodeAceStepAudio1.5] → ...
                    ↑
            [lokr_path: path/to/weights.pt]
            [strength: 1.0]
```

**インストール**:
```bash
# ComfyUI の custom_nodes ディレクトリにコピー
cp -r custom_nodes/ComfyUI-ACE-Step-LLM-LoKr /path/to/ComfyUI/custom_nodes/
```

---

### 3. LLM 前処理スクリプト (`acestep/training/preprocess_llm_for_lokr.py`)

**概要**: 5Hz-LM (Qwen2.5 ベース) の LoKr 学習用トレーニングデータを作成する前処理スクリプト

**パイプライン**:

#### Phase 1: 音声→オーディオコード変換
- **DiT トークナイザー**: 音声ファイルを離散オーディオコードに変換
- **バッチ処理**: 全ファイルを効率的に処理し、中間結果を保存
- **VRAM 最適化**: 10 ファイルごとに GC・CUDA キャッシュクリア

#### Phase 2: メタデータ生成
- **LLM 推論**: オーディオコードからメタデータ（BPM、キー、言語、キャプション）を生成
- **既存メタデータ活用**: `.caption.txt`、`.lyrics.txt`、`.json` ファイルを自動検出
- **出力フォーマット**:
  ```
  <think>
  bpm: 120
  caption: A calm piano melody
  duration: 180
  keyscale: C major
  language: en
  timesignature: 4
  </think>
  [lyrics]
  <|audio_code_12345|><|audio_code_67890|>...
  ```

**使用方法**:
```bash
python acestep/training/preprocess_llm_for_lokr.py \
  --audio_dir /path/to/audio \
  --output_dir /path/to/output \
  --lm_model acestep-5Hz-lm-1.7B
```

**出力**:
- `*.json`: 各音声ファイルのトレーニングデータ
- `train.jsonl`: HuggingFace 形式のデータセット
- `manifest.json`: データセットの統計情報

---

### 4. LLM LoKr 学習スクリプト (`acestep/training/train_llm_lokr.py`)

**概要**: 5Hz-LM (Qwen2.5 ベース) の LoKr ファインチューニングを行う学習スクリプト

**主な機能**:

#### LoKr アダプター
- **低ランク適応**: 全パラメータの 1-2% のみを学習
- **ターゲットモジュール**: `q_proj`, `k_proj`, `v_proj`, `o_proj` など
- **FP8 対応**: 疑似 FP8 による VRAM 削減
- **勾配チェックポイント**: さらに VRAM を削減（オプション）

#### オプティマイザー
- **AdamW** (デフォルト)
- **AdamW 8bit**: bitsandbytes による量子化
- **SGD**, **Adam**, **Lion**, **Prodigy** (自動 LR 調整)

#### 学習スケジューラー
- **Cosine**: Linear Warmup + CosineAnnealing
- **Linear**: Linear Warmup + LinearDecay
- **Constant**: 固定学習率
- **Cosine Restarts**: CosineAnnealingWarmRestarts

#### チェックポイント
- エポックごとの自動保存
- 再開可能（`--resume_from`）
- 最終ウェイトと config.json を出力

**使用方法**:
```bash
python acestep/training/train_llm_lokr.py \
  --model_path /path/to/acestep-5Hz-lm-1.7B \
  --tensor_dir /path/to/preprocessed_data \
  --output_dir ./llm_lokr_output \
  --lokr_linear_dim 64 \
  --lokr_linear_alpha 64 \
  --learning_rate 1e-4 \
  --batch_size 1 \
  --gradient_accumulation 4 \
  --max_epochs 10 \
  --use_fp8 \
  --gradient_checkpointing
```

**出力**:
```
llm_lokr_output/
├── checkpoint_epoch_1/
│   ├── lokr_weights.pt
│   └── config.json
├── checkpoint_epoch_2/
│   └── ...
└── final/
    ├── lokr_weights.pt
    └── config.json
```

---

### 5. Gradio UI 起動スクリプト (`start_gradio_ui.bat`)

**概要**: ACE-Step Gradio Web UI を Windows で起動するためのバッチファイル

**主な機能**:

#### 環境自動検出
- **埋め込み Python**: `python_embeded/python.exe` を優先使用
- **uv パッケージマネージャー**: 見つからない場合は自動インストールを提案
- **仮想環境**: `.venv` を自動作成・管理

#### 起動時更新チェック
- **Git 統合**: リポジトリ更新を自動検出
- **対話的更新**: 更新の有無をユーザーに確認
- **PortableGit 対応**: 同梱の Git も使用可能

#### 設定管理
- **.env ファイル**: 永続的な設定保存
- **環境変数**: ポート、サーバー名、言語、モデルパスなど

**設定例 (.env)**:
```ini
PORT=7860
SERVER_NAME=127.0.0.1
LANGUAGE=ja
ACESTEP_CONFIG_PATH=acestep-v15-sft
ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-1.7B
ACESTEP_INIT_LLM=true
ACESTEP_DOWNLOAD_SOURCE=huggingface
CHECK_UPDATE=true
```

**使用方法**:
```bash
# 基本的な起動
start_gradio_ui.bat

# 設定カスタマイズ
# 1. .env.example を .env にコピー
# 2. .env を編集
# 3. 起動
```

**対応パラメータ**:
- `PORT`: サーバーポート（デフォルト：7860）
- `SERVER_NAME`: サーバーアドレス（デフォルト：127.0.0.1）
- `LANGUAGE`: UI 言語（en, zh, he, ja）
- `CONFIG_PATH`: DiT モデルパス
- `LM_MODEL_PATH`: LLM モデルパス
- `INIT_LLM`: LLM 初期化（auto/true/false）
- `DOWNLOAD_SOURCE`: ダウンロードソース（huggingface/modelscope）
- `ENABLE_API`: REST API 有効化
- `API_KEY`: API キー設定

---

## インストール

### 前提条件
- Python 3.11-3.12
- CUDA GPU 推奨（4GB 以上の VRAM）
- Git

### セットアップ

```bash
# 1. リポジトリのクローン
git clone https://github.com/YOUR_USERNAME/ACE-Step-1.5.git
cd ACE-Step-1.5

# 2. 依存関係のインストール
uv sync

# または pip
pip install -e .
```

### 追加依存関係

#### 前処理スクリプト用
```bash
pip install librosa soundfile langdetect faster-whisper demucs
```

#### ComfyUI ノード用
ComfyUI の `custom_nodes` ディレクトリにコピー後、ComfyUI を再起動してください。

---

## 使用例

### 音楽データセットの前処理

```bash
# 全機能を使用
python preprocess_music_dataset.py \
  --music_dir /path/to/music \
  --language ja \
  --use_gemini \
  --use_whisper

# Gemini API のみ
python preprocess_music_dataset.py \
  --music_dir /path/to/music \
  --gemini_api_key YOUR_API_KEY
```

### LoKr 学習パイプライン

```bash
# 1. 前処理
python acestep/training/preprocess_llm_for_lokr.py \
  --audio_dir /path/to/audio \
  --output_dir ./preprocessed_data

# 2. 学習
python acestep/training/train_llm_lokr.py \
  --model_path ./checkpoints/acestep-5Hz-lm-1.7B \
  --tensor_dir ./preprocessed_data \
  --output_dir ./output \
  --max_epochs 10

# 3. 結果を ComfyUI で使用
# 出力された lokr_weights.pt を ComfyUI の loras ディレクトリにコピー
```

---

## 謝辞

- [ACE-Step](https://github.com/ACE-Step/ACE-Step-1.5) - 元プロジェクト
- [Gemini](https://ai.google.dev/) - マルチモーダル API
- [Qwen](https://qwenlm.github.io/) - Qwen2.5-Omni モデル
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) - ノードベース UI

## ライセンス

元プロジェクトと同じライセンス（MIT）に従います。

## 貢献

GitHub と Reddit でフィードバックをお待ちしています！

- GitHub Issues: バグ報告・機能リクエスト
- Reddit: 使用例・生成作品の共有
