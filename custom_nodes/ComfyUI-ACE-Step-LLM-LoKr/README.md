# ComfyUI ACE-Step LLM LoKr Node

ACE-Step 1.5 の LLM (5Hz-LM/Qwen3 ベース) 用 LoKr アダプターを ComfyUI で読み込むためのカスタムノードです。

## 概要

このカスタムノードは、`acestep/training/train_llm_lokr.py` で学習した LoKr ウェイトを ComfyUI の ACE-Step 1.5 で使用できるようにします。

**CLIP 接続タイプ**: このノードは CLIP 入力/出力を使用し、`Load CLIP` → `LoKr Loader` → `TextEncodeAceStepAudio1.5` のように接続します。

## インストール

### 方法 1: 手動インストール

1. このリポジトリを ComfyUI の `custom_nodes` ディレクトリにクローン/コピーします：
   ```
   custom_nodes/ComfyUI-ACE-Step-LLM-LoKr/
   ```

2. ComfyUI を再起動します。

### 方法 2: ComfyUI Manager を使用

ComfyUI Manager から「ACE-Step LLM LoKr」を検索してインストールします。

## 使用方法

### ワークフロー例

1. **CLIP の読み込み**: `Load CLIP` ノードで ACE-Step 1.5 の CLIP を読み込み
2. **ACE-Step LLM LoKr Loader** ノードを配置
3. **CLIP 出力** を LoKr Loader の **clip 入力** に接続
4. **lokr_path** に学習した LoKr ウェイトへのパスを指定
5. **strength** でアダプターの強度を調整（デフォルト：1.0）
6. LoKr Loader の **clip 出力** を `TextEncodeAceStepAudio1.5` の **clip 入力** に接続

### ノード説明

### 1. ACE-Step LLM LoKr Loader

学習済みの LoKr アダプターを LLM モデルに適用する基本的なノードです。

**入力:**
- `clip`: ACE-Step 1.5 の CLIP（`Load CLIP` ノードから）
- `lokr_path`: loras ディレクトリからの LoKr ウェイトファイル（ドロップダウンから選択）
- `strength`: アダプターの強度（0.0〜2.0、デフォルト：1.0）

**出力:**
- `clip`: LoKr アダプターが適用された CLIP

**機能:**
- loras ディレクトリ内の `.pt` / `.safetensors` ファイルをドロップダウンから選択可能
- 同じディレクトリの `config.json` を自動検知・読み込み

#### ACE-Step LLM LoKr Save

CLIP から LoKr アダプターを保存します。

**入力:**
- `clip`: LoKr アダプターを含む CLIP
- `output_path`: 出力パス

#### ACE-Step LLM LoKr Apply (Advanced)

詳細設定付きで LoKr アダプターを適用します。

**入力:**
- `clip`: ACE-Step 1.5 の CLIP
- `lokr_path`: LoKr ウェイトファイルへのパス
- `strength`: アダプターの強度
- `target_modules`: 対象モジュール（カンマ区切り）
- `rank`: LoKr ランク（新規作成時）
- `alpha`: LoKr アルファ（新規作成時）

## ワークフロー例

### 基本的な LoKr 読み込みワークフロー

```
[Load CLIP] → [ACE-Step LLM LoKr Loader] → [TextEncodeAceStepAudio1.5] → ...
                    ↑
            [lokr_path: path/to/weights.pt]
            [strength: 1.0]
```

## 学習済みウェイトの互換性

このノードは、以下のスクリプトで学習された LoKr ウェイトと互換性があります：

- `acestep/training/train_llm_lokr.py`

保存形式：
- `.pt` (PyTorch): `{module_name: {param_name: tensor}}` 形式
- `.safetensors`: 平坦化された形式

## 対応デバイス

- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- CPU（フォールバック）

## トラブルシューティング

### 「Could not import custom LoKr implementation」エラー

`acestep` パッケージがインストールされているか確認してください：

```bash
pip install -e .
```

または、プロジェクトルートが Python パスに含まれているか確認してください。

### LoKr ウェイトが読み込めない

1. ファイルが loras ディレクトリにあるか確認
2. ファイル形式（`.pt` または `.safetensors`）を確認

### config.json が読み込まれない

1. `config.json` がウェイトファイルと同じディレクトリにあるか確認
2. JSON 形式が正しいか確認

## ライセンス

Apache 2.0

## 謝辞

- [ACE-Step](https://github.com/ace-step/ACE-Step) プロジェクト
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
