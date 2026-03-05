# ComfyUI ACE-Step LLM LoKr ノード 使用ガイド

## 概要

このカスタムノードは、`acestep/training/train_llm_lokr.py` で学習した LoKr アダプターを ComfyUI の ACE-Step 1.5 で使用できるようにします。

**重要**: このノードは **CLIP 入力/出力** を使用します。`Load CLIP` ノードと `TextEncodeAceStepAudio1.5` ノードの間に接続します。

## インストール方法

### 方法 1: 手動インストール（推奨）

1. このカスタムノードを ComfyUI の `custom_nodes` ディレクトリに配置します：
   ```
   ComfyUI/
   └── custom_nodes/
       └── ComfyUI-ACE-Step-LLM-LoKr/
   ```

2. ComfyUI を再起動します。

### 方法 2: git clone を使用

```bash
cd ComfyUI/custom_nodes
git clone <このリポジトリの URL> ComfyUI-ACE-Step-LLM-LoKr
```

3. ComfyUI を再起動します。

## ノードの説明

### 1. ACE-Step LLM LoKr Loader

学習済みの LoKr アダプターを LLM モデルに適用する基本的なノードです。

#### 入力
| 入力名 | タイプ | 説明 |
|--------|--------|------|
| `clip` | CLIP | ACE-Step 1.5 の CLIP（`Load CLIP` ノードから） |
| `lokr_path` | COMBO | loras ディレクトリ内の LoKr ウェイトファイル（ドロップダウンから選択） |
| `strength` | FLOAT | アダプターの強度（0.0〜2.0、デフォルト：1.0） |

#### 出力
| 出力名 | タイプ | 説明 |
|--------|--------|------|
| `clip` | CLIP | LoKr アダプターが適用された CLIP |

#### 機能
- **ドロップダウン選択**: loras ディレクトリ内の `.pt` / `.safetensors` ファイルを一覧から選択
- **自動 config 検知**: ウェイトファイルと同じディレクトリの `config.json` を自動で読み込み

#### 使用例
```
lokr_path: acestep_lokr/lokr_weights.pt
strength: 1.0
```

### 2. ACE-Step LLM LoKr Save

CLIP から LoKr アダプターを保存するノードです。

#### 入力
| 入力名 | タイプ | 説明 |
|--------|--------|------|
| `clip` | CLIP | LoKr アダプターを含む CLIP |
| `output_path` | STRING | 出力パス（デフォルト：`llm_lokr/lokr_weights.pt`） |
| `save_config` | BOOLEAN | config.json も保存する（デフォルト：True） |

#### 出力
なし（出力ノード）

#### 使用例
```
output_path: llm_lokr/my_custom_lokr.pt
save_config: True
```

### 3. ACE-Step LLM LoKr Apply (Advanced)

LoKr アダプターの適用を詳細に制御するためのアドバンスドノードです。

#### 入力
| 入力名 | タイプ | 説明 |
|--------|--------|------|
| `clip` | CLIP | ACE-Step 1.5 の CLIP |
| `lokr_path` | STRING | LoKr ウェイトファイルへのパス |
| `strength` | FLOAT | アダプターの強度（0.0〜2.0） |
| `target_modules` | STRING | 対象モジュール（カンマ区切り） |
| `rank` | INT | LoKr ランク（新規作成時） |
| `alpha` | INT | LoKr アルファ（新規作成時） |

## ワークフロー例

### 基本的な LoKr 読み込みワークフロー

```
[Load CLIP] → [ACE-Step LLM LoKr Loader] → [TextEncodeAceStepAudio1.5] → ...
                    ↑
            [lokr_path: path/to/weights.pt]
            [strength: 1.0]
```

### LoKr 付き生成ワークフロー

1. **Load CLIP** で ACE-Step 1.5 の CLIP を読み込み
2. **ACE-Step LLM LoKr Loader** で LoKr アダプターを適用
3. **TextEncodeAceStepAudio1.5** でプロンプトをエンコード
4. **Empty AceStep 1.5 Latent Audio** で空の latent を作成
5. **KSampler** でサンプリング
6. **VAE Decode** でオーディオをデコード
7. **Save Audio** で保存

## 学習済みウェイトの互換性

このノードは以下の形式の LoKr ウェイトをサポートしています：

### サポート形式
- **PyTorch (.pt)**: `{module_name: {param_name: tensor}}` 形式
- **Safetensors (.safetensors)**: 平坦化された形式

### 学習スクリプトとの互換性
- ✅ `acestep/training/train_llm_lokr.py` で学習されたウェイト
- ✅ カスタム LoKr 実装（`llm_lokr_custom.py`）を使用

## パスの指定方法

### 絶対パス
```
C:/Users/username/Desktop/ACE-Step-1.5/outputs/llm_lokr/lokr_weights.pt
```

### ComfyUI モデルディレクトリからの相対パス
```
loras/acestep/my_lokr.pt
```

### 出力ディレクトリからの相対パス
```
../output/llm_lokr/lokr_weights.pt
```

## トラブルシューティング

### エラー：「CLIP is required」

**原因**: CLIP が接続されていません

**解決方法**: ACE-Step 1.5 の CLIP 出力を `clip` 入力に接続してください。

### エラー：「LoKr weights not found in loras directory」

**原因**: 指定されたファイルが loras ディレクトリに存在しません

**解決方法**:
1. ファイルを `ComfyUI/models/loras/` ディレクトリにコピー
2. ComfyUI を再起動してファイル一覧を更新
3. ファイル形式（`.pt` または `.safetensors`）を確認

### config.json が読み込まれない

**原因**: `config.json` が同じディレクトリにないか、形式が正しくありません

**解決方法**:
1. `config.json` がウェイトファイルと同じディレクトリにあるか確認
2. JSON 形式が正しいか確認

### エラー：「Failed to load LoKr weights」

**原因**: ウェイトファイルの形式が正しくないか、破損しています

**解決方法**:
1. ファイルが PyTorch または Safetensors 形式か確認
2. 学習スクリプトで再度ウェイトを保存
3. `config.json` が存在するか確認（任意）

### LoKr の効果が感じられない

**原因**: `strength` 値が低すぎるか、ウェイトが正しく読み込まれていません

**解決方法**:
1. `strength` を 1.0〜1.5 に調整
2. ComfyUI のコンソールログで「Loaded X LoKr modules」を確認
3. ウェイトファイルが正しいか確認

### 「Could not find specific LLM model」警告

**原因**: ACE-Step 1.5 以外の CLIP が使用されています

**解決方法**: ACE-Step 1.5 用の CLIP を使用しているか確認してください。

## 注意事項

1. **メモリ使用量**: LoKr アダプターは追加のメモリを消費します。VRAM が不足する場合は、バッチサイズを減らしてください。

2. **互換性**: このノードは ACE-Step 1.5 の LLM モデル（Qwen3 ベース）専用です。他のモデルでは動作しません。

3. **パフォーマンス**: LoKr アダプターを適用すると、推論速度が若干低下する可能性があります。

## 更新履歴

### v1.0.0
- 初期リリース
- ACE-Step LLM LoKr Loader ノード
- ACE-Step LLM LoKr Save ノード
- ACE-Step LLM LoKr Apply (Advanced) ノード

## ライセンス

Apache 2.0

## サポート

問題や質問がある場合は、ACE-Step-1.5 プロジェクトの Issue で報告してください。
