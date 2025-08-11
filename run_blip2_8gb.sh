#!/usr/bin/env bash

# ==============================================
# BLIP-2 起動スクリプト（8GB VRAM向け・VLM専用）
# transformers + bitsandbytes による 8bit/4bit 量子化対応
# 例: bash run_blip2_8gb.sh [モデル名] [8bit|4bit] [画像パス] [質問]
# ==============================================

set -euo pipefail

MODEL_ID=${1:-"Salesforce/blip2-opt-2.7b"}
QUANT=${2:-"8bit"}           # 8bit | 4bit
IMAGE_PATH=${3:-""}          # 画像パス（省略可）
QUESTION=${4:-"Describe the image."}

HOST="localhost"
PORT=0  # サーバは立てず、CLI実行のみ（必要なら将来拡張）

echo "🚀 BLIP-2 起動 (VLM専用)"
echo "  🔹 MODEL_ID : $MODEL_ID"
echo "  🔹 QUANT    : $QUANT"
if [[ -n "$IMAGE_PATH" ]]; then
  echo "  🔹 IMAGE    : $IMAGE_PATH"
else
  echo "  🔹 IMAGE    : (未指定)"
fi
echo "  🔹 QUESTION : $QUESTION"

# ========= メモリ/実行環境 設定 =========
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# ========= 依存関係チェック =========
echo "🔍 依存関係チェック: torch, transformers, bitsandbytes, pillow"
uv run python - <<'PY' 2>/dev/null || { echo "❌ 依存関係チェックに失敗しました。上記メッセージを参照してください。"; exit 1; }
import importlib, sys
required = ["torch", "transformers", "bitsandbytes", "PIL"]
missing = []
for m in required:
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("❌ 依存関係が不足しています:", ", ".join(missing))
    print("💡 以下のコマンドで追加してください:")
    print("   uv add torch torchvision pillow transformers bitsandbytes accelerate")
    sys.exit(1)
print("✅ 依存関係OK")
PY

# ========= 実行 =========
echo "🧠 モデルをロードし、CLIで画像キャプション/VQAを実行します..."

UV_CACHE_DIR=${UV_CACHE_DIR:-"./hf_models"}
OFFLOAD_DIR=${OFFLOAD_DIR:-"./offload"}
mkdir -p "$UV_CACHE_DIR" "$OFFLOAD_DIR"

UV_MODEL_ID="$MODEL_ID" \
UV_QUANT="$QUANT" \
UV_IMAGE_PATH="$IMAGE_PATH" \
UV_QUESTION="$QUESTION" \
UV_CACHE_DIR="$UV_CACHE_DIR" \
UV_OFFLOAD_DIR="$OFFLOAD_DIR" \
uv run python - <<'PY'
import os
import sys
import base64
from io import BytesIO

import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig


def print_banner(message: str) -> None:
    print(message, flush=True)


def load_image(path: str | None) -> Image.Image:
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB")
    # 最小のプレースホルダー画像 (1x1 PNG, white)
    tiny_png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQAB"
        "J1N6WQAAAABJRU5ErkJggg=="
    )
    data = base64.b64decode(tiny_png_base64)
    return Image.open(BytesIO(data)).convert("RGB")


model_id = os.environ.get("UV_MODEL_ID", "Salesforce/blip2-opt-2.7b")
quant_mode = os.environ.get("UV_QUANT", "8bit").lower()
image_path = os.environ.get("UV_IMAGE_PATH", "")
question = os.environ.get("UV_QUESTION", "Describe the image.")
cache_dir = os.environ.get("UV_CACHE_DIR", "./hf_models")
offload_dir = os.environ.get("UV_OFFLOAD_DIR", "./offload")

if quant_mode not in {"8bit", "4bit"}:
    print(f"[ERROR] 無効な量子化指定: {quant_mode}. 8bit か 4bit を指定してください。", file=sys.stderr)
    sys.exit(2)

print_banner("🔧 量子化設定を適用中...")
if quant_mode == "8bit":
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

print_banner("📦 モデル/プロセッサを読み込み中...")
processor = Blip2Processor.from_pretrained(model_id, cache_dir=cache_dir)
model = Blip2ForConditionalGeneration.from_pretrained(
    model_id,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=bnb_config,
    offload_folder=offload_dir,
)

image = load_image(image_path)

print_banner("📝 推論を実行中...")
inputs = processor(images=image, text=question, return_tensors="pt")

# device_map="auto" を用いているため、入力はCPUのままでOK（Accelerateが自動で割当）
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=64)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print("\n================ RESULT ================")
print(f"Question: {question}")
if image_path:
    print(f"Image   : {image_path}")
else:
    print("Image   : (placeholder 1x1)")
print(f"Answer  : {generated_text}")
print("=======================================\n")

print("✅ 完了。別の入力で再実行する場合は引数を変えてコマンドを再度実行してください。")
PY

exit_code=$?
if [[ $exit_code -ne 0 ]]; then
  echo "❌ 実行中にエラーが発生しました (exit: $exit_code)"
  exit $exit_code
fi

echo "🎉 正常終了"

echo "\n📋 使い方の例:"
echo "  1) 8bit量子化・キャプション生成:"
echo "     bash run_blip2_8gb.sh 'Salesforce/blip2-opt-2.7b' 8bit ./path/to/image.jpg 'Describe the image.'"
echo ""
echo "  2) 4bit量子化(NF4)・VQA:"
echo "     bash run_blip2_8gb.sh 'Salesforce/blip2-opt-2.7b' 4bit ./path/to/image.jpg 'What is in the picture?'"
echo ""
echo "  3) 画像省略（プレースホルダー画像で動作確認のみ）:"
echo "     bash run_blip2_8gb.sh 'Salesforce/blip2-opt-2.7b' 8bit '' 'Hello'"


