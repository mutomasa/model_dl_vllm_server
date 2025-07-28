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

| モデル名 | サイズ | 説明 |
|---------|--------|------|
| `Qwen/Qwen2-1.5B-Instruct` | ~3GB | 推奨、高性能 |
| `microsoft/DialoGPT-medium` | ~1GB | 軽量、高速 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | ~2GB | 軽量、多言語対応 |

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

### 使用方法

```bash
# 基本的な使用方法
./run_vllm.sh <モデル名>

# 例
./run_vllm.sh Qwen/Qwen2-1.5B-Instruct
./run_vllm.sh microsoft/DialoGPT-medium
```

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
├── download_model.py      # モデルダウンロードスクリプト
├── run_vllm.sh           # vLLMサーバー起動スクリプト
├── pyproject.toml        # プロジェクト設定
├── .gitignore           # Git除外設定
├── hf_models/           # ダウンロードしたモデル（Git除外）
└── README.md            # このファイル
```

## 🚨 注意事項

### GPUメモリ制限

- **8GB GPU**: Qwen2-1.5B-Instructなどの軽量モデルを推奨
- **Qwen2.5-VL-3B-Instruct**: 約7.5GBで8GB GPUでは厳しい可能性
- **Qwen3-8B**: 8GB GPUでは動作不可

### ダウンロード時間

- **軽量モデル（1-3GB）**: 10-30分
- **中規模モデル（3-7GB）**: 30-60分
- **大規模モデル（7GB以上）**: 1時間以上

### ディスク容量

- モデルファイルは予想より大きくなる場合があります
- 十分な空き容量（最低10GB推奨）を確保してください

## 🔄 完全なワークフロー

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
chmod +x run_vllm.sh download_model.py
```

## 📞 サポート

問題が発生した場合は、以下を確認してください：

1. エラーメッセージの詳細
2. GPUメモリ使用量（`nvidia-smi`）
3. ディスク空き容量（`df -h`）
4. インターネット接続状況

## 📄 ライセンス

このプロジェクトはMITライセンスの下で公開されています。 