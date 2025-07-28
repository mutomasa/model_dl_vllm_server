#!/bin/bash

# ================================
# モデル名をコマンドライン引数から取得
# ================================
MODEL_NAME=$1

if [ -z "$MODEL_NAME" ]; then
  echo "モデル名を引数で指定してください"
  echo "使用例: ./run_vllm.sh Qwen/Qwen3-8B"
  echo "使用例: ./run_vllm.sh meta-llama/Llama-2-7b-chat-hf"
  exit 1
fi

PORT=8000
HOST="localhost"

# ================================
# HFトークン（必要なら）
# ================================
# export HUGGINGFACE_TOKEN="your_token"
# huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# ================================
# モデルAPIサーバを起動（OpenAI互換API）
# ================================
echo "🚀 vLLMサーバーを起動中..."
echo "モデル: $MODEL_NAME"
echo "URL: http://$HOST:$PORT"
echo "API形式: OpenAI互換"

# uvを使用してvLLMサーバーを起動
uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --trust-remote-code \
  --port $PORT \
  --host $HOST \
  --download-dir ./hf_models &

# ================================
# サーバが立ち上がるまで少し待機
# ================================
echo "📡 vLLM APIサーバを起動中..."
sleep 10

# ================================
# サーバーの健全性チェック
# ================================
echo "🔍 サーバーの健全性をチェック中..."

max_attempts=76  # 最大7分待機
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if curl -s http://$HOST:$PORT/v1/models > /dev/null; then
    echo "✅ vLLMサーバーが正常に起動しました！"
    echo "📡 OpenAI互換エンドポイント: http://$HOST:$PORT/v1"
    
    # テスト実行
    echo "🧪 サーバーテストを実行中..."
    curl -X POST http://$HOST:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "default",
        "messages": [
          {"role": "user", "content": "富士山の高さは？"}
        ],
        "max_tokens": 64,
        "temperature": 0.7
      }' 2>/dev/null | jq -r '.choices[0].message.content' 2>/dev/null || echo "テストレスポンスを取得できませんでした"
    
    echo ""
    echo "📋 使用方法:"
    echo "1. 別のターミナルでStreamlitアプリを起動:"
    echo "   uv run streamlit run streamlit_app.py"
    echo ""
    echo "2. サーバーを停止するには Ctrl+C を押してください"
    echo "   または: pkill -f 'vllm.entrypoints.openai.api_server'"
    
    # プロセスを待機
    wait
    break
  else
    attempt=$((attempt + 1))
    echo "⏳ 起動待機中... ($attempt/$max_attempts)"
    sleep 5
  fi
done

if [ $attempt -eq $max_attempts ]; then
  echo "❌ サーバー起動タイムアウト"
  pkill -f "vllm.entrypoints.openai.api_server"
  exit 1
fi 