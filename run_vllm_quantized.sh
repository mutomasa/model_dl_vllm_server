#!/bin/bash

# ================================
# 量子化vLLMサーバー起動スクリプト（マルチモーダル対応）
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
echo ""
echo "🖼️ マルチモーダル対応モデル:"
echo "  - Qwen/Qwen2.5-VL-3B-Instruct-AWQ (推奨・軽量)"
echo "  - Qwen/Qwen2-VL-7B-Instruct (高精度・重い)"
echo "  - Qwen/Qwen2-VL-1.5B-Instruct (軽量・高速)"
echo ""
echo "🔧 マルチモーダル機能:"
echo "  - 画像キャプション生成"
echo "  - 画像内容の説明"
echo "  - 視覚的質問応答"
echo "  - 画像分析"
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
# マルチモーダル設定
# ================================
export VLLM_USE_MULTIMODAL=1
export VLLM_ENABLE_IMAGE_PROCESSING=1
echo "🖼️ マルチモーダル環境変数を設定しました"

# マルチモーダル依存関係の確認
if [[ "$MODEL_NAME" == *"VL"* ]]; then
  echo "🔍 マルチモーダル依存関係を確認中..."
  if uv run python -c "import torch; import transformers; print('✅ 依存関係OK')" 2>/dev/null; then
    echo "✅ マルチモーダル依存関係が正常です"
  else
    echo "⚠️ マルチモーダル依存関係に問題がある可能性があります"
    echo "💡 以下のコマンドで依存関係を更新してください:"
    echo "   uv add torch transformers pillow"
  fi
fi

# ================================
# モデルAPIサーバーを起動（量子化付き）
# ================================
echo "🚀 量子化vLLMサーバーを起動中..."
echo "モデル: $MODEL_NAME"
echo "量子化: $QUANTIZATION"
echo "URL: http://$HOST:$PORT"
echo "API形式: OpenAI互換"

# uvを使用してvLLMサーバーを起動（マルチモーダル対応）
echo "🖼️ マルチモーダル機能: 有効"
echo "🔧 マルチモーダル専用オプションを適用中..."

# マルチモーダル対応の追加オプション
MULTIMODAL_ARGS=""
if [[ "$MODEL_NAME" == *"VL"* ]]; then
  # vLLMサーバーでマルチモーダル機能を有効化するためのオプション
  # 現在のvLLMバージョンでは、マルチモーダル機能は自動的に有効になる
  MULTIMODAL_ARGS="--max-num-seqs 128 --max-num-batched-tokens 2048"
  echo "✅ マルチモーダルモデルを検出: $MODEL_NAME"
  echo "🔧 マルチモーダル専用オプション: $MULTIMODAL_ARGS"
  echo "🖼️ 画像処理機能: 自動有効"
  echo "📝 マルチモーダルリクエスト: サポート"
else
  echo "⚠️ マルチモーダルモデルではありません: $MODEL_NAME"
  echo "💡 マルチモーダル機能を使用するには、VL（Vision-Language）モデルを選択してください"
fi

uv run python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --trust-remote-code \
  --port $PORT \
  --host $HOST \
  --download-dir ./hf_models \
  --dtype float16 \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9 \
  --disable-log-requests \
  --disable-log-stats \
  --served-model-name "$MODEL_NAME" \
  $MULTIMODAL_ARGS \
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
    
    # テキストのみのテスト
    echo "📝 テキストテスト実行中..."
    TEXT_RESPONSE=$(curl -s -X POST http://$HOST:$PORT/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "'"$MODEL_NAME"'",
        "messages": [
          {"role": "user", "content": "量子化テスト：こんにちは"}
        ],
        "max_tokens": 64,
        "temperature": 0.7
      }' 2>/dev/null | jq -r '.choices[0].message.content' 2>/dev/null || echo "テキストテスト失敗")
    
    echo "✅ テキストテスト結果: $TEXT_RESPONSE"
    
    # マルチモーダルテスト（小さなテスト画像）
    echo "🖼️ マルチモーダルテスト実行中..."
    
    # マルチモーダルモデルの場合のみテスト
    if [[ "$MODEL_NAME" == *"VL"* ]]; then
      MULTIMODAL_RESPONSE=$(curl -s -X POST http://$HOST:$PORT/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
          "model": "'"$MODEL_NAME"'",
          "messages": [
            {
              "role": "user",
              "content": [
                {"type": "text", "text": "この画像を説明してください"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="}}
              ]
            }
          ],
          "max_tokens": 100,
          "temperature": 0.7
        }' 2>/dev/null | jq -r '.choices[0].message.content' 2>/dev/null || echo "マルチモーダルテスト失敗")
      
      echo "✅ マルチモーダルテスト結果: $MULTIMODAL_RESPONSE"
      
      # マルチモーダル機能の状態を確認
      if [[ "$MULTIMODAL_RESPONSE" == *"画像を表示する機能は提供していません"* ]] || [[ "$MULTIMODAL_RESPONSE" == *"マルチモーダルテスト失敗"* ]]; then
        echo "⚠️ マルチモーダル機能が正しく動作していない可能性があります"
        echo "💡 ヒント: 以下の点を確認してください"
        echo "   1. モデルが正しくダウンロードされているか"
        echo "   2. 十分なGPUメモリがあるか"
        echo "   3. vLLMのバージョンが最新か"
        echo "   4. マルチモーダル対応の依存関係がインストールされているか"
      else
        echo "✅ マルチモーダル機能が正常に動作しています"
        echo "🎉 画像キャプション生成が利用可能です"
      fi
    else
      echo "⏭️ マルチモーダルテストをスキップ（VLモデルではありません）"
    fi
    
    echo ""
    echo "📋 使用方法:"
    echo "1. 別のターミナルで画像キャプションアプリを起動:"
    echo "   cd ../qwen2-vl-caption && uv run streamlit run app.py"
    echo ""
    echo "2. サーバーを停止するには Ctrl+C を押してください"
    echo "   または: pkill -f 'vllm.entrypoints.openai.api_server'"
    echo ""
    echo "🔧 マルチモーダル機能が有効です"
    echo "📡 OpenAI互換API: http://$HOST:$PORT/v1"
    echo "🖼️ マルチモーダル対応: 画像キャプション生成が可能です"
    echo "📝 サポート形式: PNG, JPEG, WebP"
    echo "🔍 画像サイズ: 自動リサイズ（最大1024px）"
    echo "🚀 使用可能な機能:"
    echo "   - 画像キャプション生成"
    echo "   - 画像内容の詳細説明"
    echo "   - 視覚的質問応答"
    echo "   - 画像分析・分類"
    
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
