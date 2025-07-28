#!/bin/bash

# ================================
# 量子化vLLMサーバー起動スクリプト
# ================================

# モデル名をコマンドライン引数から取得
MODEL_NAME=$1
QUANTIZATION=$2

if [ -z "$MODEL_NAME" ]; then
  echo "モデル名を引数で指定してください"
  echo "使用方法: $0 <モデル名> [量子化方法]"
  echo ""
  echo "量子化方法:"
  echo "  awq      - AWQ量子化（推奨・安定）"
  echo "  gptq     - GPTQ量子化（推奨・安定）"
  echo "  bitsandbytes - BitsAndBytes量子化（推奨・4bit/8bit）"
  echo "  sq       - SqueezeLLM量子化（実験的）"
  echo "  fp4      - FP4量子化（実験的）"
  echo "  nf4      - NF4量子化（実験的）"
  echo ""
  echo "例:"
  echo "  $0 Qwen/Qwen2.5-VL-3B-Instruct-AWQ awq"
  echo "  $0 Qwen/Qwen2-1.5B-Instruct bitsandbytes"
  exit 1
fi

PORT=8000
HOST="localhost"

# デフォルトの量子化方法
if [ -z "$QUANTIZATION" ]; then
  QUANTIZATION="awq"
fi

# ================================
# 量子化設定
# ================================
case $QUANTIZATION in
  "awq")
    QUANTIZATION_ARGS=""
    echo "🔧 AWQ量子化済みモデルを使用"
    ;;
  "gptq")
    QUANTIZATION_ARGS="--quantization gptq"
    echo "🔧 GPTQ量子化を使用"
    ;;
  "bitsandbytes")
    QUANTIZATION_ARGS="--quantization bitsandbytes"
    echo "🔧 BitsAndBytes量子化を使用"
    ;;
  "sq")
    QUANTIZATION_ARGS="--quantization sq"
    echo "🔧 SqueezeLLM量子化を使用"
    echo "⚠️ sq は一部環境で未サポートの可能性があります。動作しない場合は awq / gptq / bitsandbytes を推奨します。"
    ;;
  "fp4")
    QUANTIZATION_ARGS="--quantization fp4"
    echo "🔧 FP4量子化を使用"
    echo "⚠️ fp4 は一部環境で未サポートの可能性があります。動作しない場合は awq / gptq / bitsandbytes を推奨します。"
    ;;
  "nf4")
    QUANTIZATION_ARGS="--quantization nf4"
    echo "🔧 NF4量子化を使用"
    echo "⚠️ nf4 は一部環境で未サポートの可能性があります。動作しない場合は awq / gptq / bitsandbytes を推奨します。"
    ;;
  *)
    echo "❌ 無効な量子化方法: $QUANTIZATION"
    echo "有効なオプション: awq, gptq, bitsandbytes, sq, fp4, nf4"
    exit 1
    ;;
esac

# ================================
# メモリ管理設定
# ================================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLOAT16=1

# ================================
# モデルAPIサーバーを起動（量子化付き）
# ================================
echo "🚀 量子化vLLMサーバーを起動中..."
echo "モデル: $MODEL_NAME"
echo "量子化: $QUANTIZATION"
echo "URL: http://$HOST:$PORT"
echo "API形式: OpenAI互換"

# uvを使用してvLLMサーバーを起動
uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --trust-remote-code \
  --port $PORT \
  --host $HOST \
  --download-dir ./hf_models \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  $QUANTIZATION_ARGS &

# ================================
# サーバが立ち上がるまで少し待機
# ================================
echo "📡 量子化vLLM APIサーバを起動中..."
sleep 15

# ================================
# サーバーの健全性チェック
# ================================
echo "🔍 サーバーの健全性をチェック中..."

max_attempts=48  # 最大4分待機（量子化は時間がかかる）
attempt=0

while [ $attempt -lt $max_attempts ]; do
  if curl -s http://$HOST:$PORT/v1/models > /dev/null; then
    echo "✅ 量子化vLLMサーバーが正常に起動しました！"
    echo "📡 OpenAI互換エンドポイント: http://$HOST:$PORT/v1"
    echo "🔧 量子化方法: $QUANTIZATION"
    
    # テスト実行
    echo "🧪 サーバーテストを実行中..."
    curl -X POST http://$HOST:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [
          {"role": "user", "content": "量子化テスト：こんにちは"}
        ],
        "max_tokens": 64,
        "temperature": 0.7
      }' 2>/dev/null | jq -r '.choices[0].message.content' 2>/dev/null || echo "テストレスポンスを取得できませんでした"
    
    echo ""
    echo "📋 使用方法:"
    echo "1. 別のターミナルでStreamlitアプリを起動:"
    echo "   cd ../dlt_generation_slide && uv run streamlit run streamlit_app.py"
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
  echo "💡 ヒント: 量子化には時間がかかります。より軽量なモデルを試してください。"
  pkill -f "vllm.entrypoints.openai.api_server"
  exit 1
fi
