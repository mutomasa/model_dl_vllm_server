# vLLM Server & Model Download

このリポジトリは、Hugging Faceモデルをダウンロードし、vLLMサーバーを起動するためのツールセットです。

## 📋 概要

- **モデルダウンロード**: PythonスクリプトでHugging Faceモデルを簡単にダウンロード
- **vLLMサーバー**: OpenAI互換APIでLLMサーバーを起動
- **8GB GPU対応**: 軽量モデルでGPUメモリ制限に対応

## 🚀 セットアップ

### 1. 環境準備

```bash
# リポジトリをクローン
git clone https://github.com/mutomasa/model_dl_vllm_server.git
cd model_dl_vllm_server

# uvでプロジェクトを初期化
uv init

# 必要なパッケージをインストール
uv add vllm transformers huggingface-hub accelerate

# 量子化を使用する場合（推奨）
uv add bitsandbytes autoawq
```

### 2. 依存関係

- **Python**: 3.9以上
- **GPU**: NVIDIA GPU（推奨8GB以上）
- **CUDA**: 対応バージョン
- **uv**: Pythonパッケージマネージャー

## 📥 モデルダウンロード

### 使用方法

```bash
# 基本的な使用方法
uv run python download_model.py <モデル名>

# 例：Qwen2-1.5B-Instruct（推奨）
uv run python download_model.py Qwen/Qwen2-1.5B-Instruct

# 例：DialoGPT-medium（軽量）
uv run python download_model.py microsoft/DialoGPT-medium

# ダウンロード先ディレクトリを指定
uv run python download_model.py Qwen/Qwen2-1.5B-Instruct --download-dir ./my_models
```

### 8GB GPU対応モデル

#### 非量子化モデル（推奨）

| モデル名 | サイズ | 説明 |
|---------|--------|------|
| `Qwen/Qwen2-1.5B-Instruct` | ~3GB | 推奨、高性能 |
| `microsoft/DialoGPT-medium` | ~1GB | 軽量、高速 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2GB | 軽量、多言語対応 |

#### 量子化済みモデル（推奨）

| モデル名 | サイズ | 量子化方法 | 説明 |
|---------|--------|------------|------|
| `Qwen/Qwen2.5-VL-3B-Instruct-AWQ` | ~1.35GB | AWQ | マルチモーダル、画像理解 |
| `Qwen/Qwen2-1.5B-Instruct-AWQ` | ~0.8GB | AWQ | 軽量、高性能 |
| `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` | ~3.5GB | AWQ | 高精度マルチモーダル |

#### 量子化で使用可能なモデル

| モデル名 | 元サイズ | 量子化後 | 量子化方法 | 説明 |
|---------|----------|----------|------------|------|
| `Qwen/Qwen2.5-VL-3B-Instruct` | ~7GB | ~3.5GB | BitsAndBytes | マルチモーダル |
| `Qwen/Qwen2-1.5B-Instruct` | ~3GB | ~1.5GB | BitsAndBytes | 軽量、高性能 |
| `Qwen/Qwen3-8B-Instruct` | ~16GB | ~8GB | BitsAndBytes | 高精度（8GB GPUで限界） |

### ダウンロード手順

1. **モデル情報取得**: 設定ファイルとトークナイザーをダウンロード
2. **モデルファイル**: 重いモデルファイルをダウンロード（時間がかかります）
3. **統計表示**: ダウンロードしたファイルのサイズと数を表示

### トラブルシューティング

**エラー**: `ModuleNotFoundError: No module named 'transformers'`
```bash
# 解決策
uv add transformers huggingface-hub accelerate
```

**エラー**: `accelerate`が必要
```bash
# 解決策
uv add accelerate
```

## 🖥️ vLLMサーバー起動

### 基本的な使用方法

```bash
# 基本的な使用方法
./run_vllm.sh <モデル名>

# 例
./run_vllm.sh Qwen/Qwen2-1.5B-Instruct
./run_vllm.sh microsoft/DialoGPT-medium
```

## 🔧 量子化vLLMサーバー起動

### 量子化とは

量子化は、モデルの精度を下げることでメモリ使用量を削減し、推論速度を向上させる技術です。8GB GPUでより大きなモデルを動作させるために重要です。

**主なメリット:**
- **メモリ削減**: 最大75%のメモリ使用量削減
- **推論速度向上**: 軽量化により高速推論
- **8GB GPU対応**: より大きなモデルを動作可能
- **コスト削減**: より少ないリソースで高性能モデル使用

### 量子化方法の種類

| 量子化方法 | 説明 | 推奨度 | メモリ削減 |
|-----------|------|--------|------------|
| **AWQ** | 事前量子化済みモデル | ⭐⭐⭐⭐⭐ | 最大75% |
| **GPTQ** | 汎用量子化 | ⭐⭐⭐⭐ | 最大75% |
| **BitsAndBytes** | 動的量子化 | ⭐⭐⭐⭐ | 最大75% |
| **SqueezeLLM** | 実験的 | ⭐⭐ | 最大75% |
| **FP4/NF4** | 実験的 | ⭐⭐ | 最大75% |

### 量子化サーバー起動

```bash
# 基本的な使用方法
./run_vllm_quantized.sh <モデル名> [量子化方法]

# 例：AWQ量子化済みモデル（推奨）
./run_vllm_quantized.sh Qwen/Qwen2.5-VL-3B-Instruct-AWQ awq

# 例：BitsAndBytes量子化
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct bitsandbytes

# 例：GPTQ量子化
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct gptq

# 例：実験的量子化（非推奨）
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct fp4
```

### 実際の使用例

#### 1. マルチモーダルモデル（画像理解）

```bash
# AWQ量子化済みマルチモーダルモデル
./run_vllm_quantized.sh Qwen/Qwen2.5-VL-3B-Instruct-AWQ awq

# 使用例：画像の説明生成
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "この画像を説明してください"}
    ],
    "max_tokens": 128,
    "temperature": 0.7
  }'
```

#### 2. 軽量テキストモデル

```bash
# BitsAndBytes量子化で軽量モデル
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct bitsandbytes

# 使用例：テキスト生成
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "AIの未来について説明してください"}
    ],
    "max_tokens": 256,
    "temperature": 0.7
  }'
```

### 量子化方法の詳細

#### 🔥 AWQ量子化（推奨・安定）
- **特徴**: 事前量子化済みモデル、最高の安定性
- **使用例**: `Qwen/Qwen2.5-VL-3B-Instruct-AWQ`
- **メリット**: ダウンロード後すぐ使用可能、安定動作
- **デメリット**: モデル選択が限定的

#### 🔥 GPTQ量子化（推奨・安定）
- **特徴**: 汎用的な量子化手法
- **使用例**: 多くのモデルで利用可能
- **メリット**: 幅広いモデルに対応
- **デメリット**: 初回起動時に時間がかかる場合がある

#### 🔥 BitsAndBytes量子化（推奨・4bit/8bit）
- **特徴**: 動的量子化、4bit/8bit精度
- **使用例**: 任意のモデルで利用可能
- **メリット**: 最も柔軟、メモリ効率が良い
- **デメリット**: 初回起動時に時間がかかる

#### ⚠️ 実験的量子化（非推奨）
- **SqueezeLLM**: 一部環境で未サポート
- **FP4/NF4**: 一部環境で未サポート
- **注意**: 動作しない場合は推奨方法を使用してください

### 量子化の選択ガイド

#### 🎯 推奨シナリオ

| 用途 | 推奨量子化 | モデル例 | 理由 |
|------|------------|----------|------|
| **マルチモーダル** | AWQ | `Qwen2.5-VL-3B-Instruct-AWQ` | 事前量子化済み、安定 |
| **軽量テキスト** | BitsAndBytes | `Qwen2-1.5B-Instruct` | 柔軟、メモリ効率 |
| **高精度** | GPTQ | 任意のモデル | 汎用性、安定性 |
| **実験・開発** | BitsAndBytes | 任意のモデル | 最も柔軟 |

#### ⚠️ 注意点

- **実験的量子化**: `sq`, `fp4`, `nf4`は一部環境で動作しない可能性
- **初回起動時間**: 量子化には時間がかかる場合があります
- **メモリ使用量**: 量子化後もGPUメモリを確認してください

### サーバー情報

- **URL**: http://localhost:8000
- **API形式**: OpenAI互換
- **エンドポイント**: `/v1/chat/completions`

### 起動手順

1. **サーバー起動**: vLLMサーバーをバックグラウンドで起動
2. **健全性チェック**: サーバーが正常に起動するまで待機（最大3分）
3. **テスト実行**: 簡単なテストクエリで動作確認
4. **使用方法表示**: 次のステップの案内

### サーバーテスト

```bash
# サーバーが起動したら、別のターミナルでテスト
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "富士山の高さは？"}
    ],
    "max_tokens": 64,
    "temperature": 0.7
  }'
```

## 🔧 設定ファイル

### pyproject.toml

```toml
[project]
name = "vllm-server"
version = "0.1.0"
description = "vLLM server with model download tools"
requires-python = ">=3.9"

dependencies = [
    "vllm",
    "transformers",
    "huggingface-hub",
    "accelerate",
]
```

### .gitignore

```
# Model files and cache
hf_models/
*.safetensors
*.bin
*.pt
*.pth

# Hugging Face cache
.cache/

# Virtual environments
.venv/
```

## 📁 ファイル構成

```
.
├── download_model.py           # モデルダウンロードスクリプト
├── run_vllm.sh                # 基本的なvLLMサーバー起動スクリプト
├── run_vllm_quantized.sh      # 量子化vLLMサーバー起動スクリプト
├── pyproject.toml             # プロジェクト設定
├── .gitignore                # Git除外設定
├── hf_models/                # ダウンロードしたモデル（Git除外）
└── README.md                 # このファイル
```

## 🚨 注意事項

### GPUメモリ制限

#### 非量子化モデル
- **8GB GPU**: Qwen2-1.5B-Instructなどの軽量モデルを推奨
- **Qwen2.5-VL-3B-Instruct**: 約7.5GBで8GB GPUでは厳しい可能性
- **Qwen3-8B**: 8GB GPUでは動作不可

#### 量子化モデル（推奨）
- **AWQ量子化**: メモリ使用量を最大75%削減
- **Qwen2.5-VL-3B-Instruct-AWQ**: 約1.35GBで8GB GPUで余裕
- **BitsAndBytes量子化**: 任意のモデルで4bit/8bit精度に変換可能

### ダウンロード時間

- **軽量モデル（1-3GB）**: 10-30分
- **中規模モデル（3-7GB）**: 30-60分
- **大規模モデル（7GB以上）**: 1時間以上

### ディスク容量

- モデルファイルは予想より大きくなる場合があります
- 十分な空き容量（最低10GB推奨）を確保してください

## 🔄 完全なワークフロー

### 基本的なワークフロー

```bash
# 1. 環境セットアップ
git clone https://github.com/mutomasa/model_dl_vllm_server.git
cd model_dl_vllm_server
uv add vllm transformers huggingface-hub accelerate

# 2. モデルダウンロード
uv run python download_model.py Qwen/Qwen2-1.5B-Instruct

# 3. vLLMサーバー起動
./run_vllm.sh Qwen/Qwen2-1.5B-Instruct

# 4. 別のターミナルでStreamlitアプリ起動（オプション）
cd ../dlt_generation_slide
uv run streamlit run streamlit_app.py
```

### 量子化ワークフロー（推奨）

```bash
# 1. 環境セットアップ
git clone https://github.com/mutomasa/model_dl_vllm_server.git
cd model_dl_vllm_server
uv add vllm transformers huggingface-hub accelerate bitsandbytes

# 2. AWQ量子化済みモデルダウンロード（推奨）
uv run python download_model.py Qwen/Qwen2.5-VL-3B-Instruct-AWQ

# 3. 量子化vLLMサーバー起動
./run_vllm_quantized.sh Qwen/Qwen2.5-VL-3B-Instruct-AWQ awq

# 4. 別のターミナルでStreamlitアプリ起動（オプション）
cd ../dlt_generation_slide
uv run streamlit run streamlit_app.py
```

### 動的量子化ワークフロー

```bash
# 1. 環境セットアップ
git clone https://github.com/mutomasa/model_dl_vllm_server.git
cd model_dl_vllm_server
uv add vllm transformers huggingface-hub accelerate bitsandbytes

# 2. 通常モデルダウンロード
uv run python download_model.py Qwen/Qwen2-1.5B-Instruct

# 3. BitsAndBytes量子化でvLLMサーバー起動
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct bitsandbytes

# 4. 別のターミナルでStreamlitアプリ起動（オプション）
cd ../dlt_generation_slide
uv run streamlit run streamlit_app.py
```

## 🆘 トラブルシューティング

### よくある問題

**Q: モデルダウンロードが途中で止まる**
A: インターネット接続を確認し、再実行してください

**Q: GPUメモリ不足エラー**
A: より小さいモデルを使用してください

**Q: サーバー起動タイムアウト**
A: モデルダウンロードが完了しているか確認してください

**Q: 権限エラー**
A: スクリプトに実行権限を付与してください
```bash
chmod +x run_vllm.sh run_vllm_quantized.sh download_model.py
```

**Q: 量子化エラー（torch.bfloat16 is not supported）**
A: 量子化方法を変更してください
```bash
# AWQ量子化済みモデルを使用
./run_vllm_quantized.sh Qwen/Qwen2.5-VL-3B-Instruct-AWQ awq

# またはBitsAndBytes量子化を使用
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct bitsandbytes
```

**Q: 量子化方法が無効**
A: サポートされている量子化方法を使用してください
- 推奨: `awq`, `gptq`, `bitsandbytes`
- 実験的: `sq`, `fp4`, `nf4`（動作しない場合があります）

**Q: 量子化サーバー起動が遅い**
A: 初回起動時は時間がかかります
```bash
# 起動状況を確認
nvidia-smi  # GPUメモリ使用量を確認
ps aux | grep vllm  # プロセス状況を確認
```

**Q: 量子化後もメモリ不足**
A: より軽量なモデルまたは量子化方法を試してください
```bash
# より軽量なモデル
./run_vllm_quantized.sh Qwen/Qwen2-1.5B-Instruct bitsandbytes

# または非量子化の軽量モデル
./run_vllm.sh microsoft/DialoGPT-medium
```

**Q: 量子化モデルの精度が低い**
A: 量子化は精度を下げる技術です
- より大きなモデルを使用
- 非量子化モデルを使用
- 量子化方法を変更（AWQ → GPTQ）

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. エラーメッセージの詳細
2. GPUメモリ使用量（`nvidia-smi`）
3. ディスク空き容量（`df -h`）
4. インターネット接続状況

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。 