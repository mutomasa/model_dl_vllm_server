#!/bin/bash

# ================================
# モデルダウンロードスクリプト
# ================================

# 使用方法の表示
show_usage() {
    echo "使用方法: $0 <モデル名>"
    echo ""
    echo "例:"
    echo "  $0 Qwen/Qwen2.5-VL-3B-Instruct"
    echo "  $0 Qwen/Qwen2-1.5B-Instruct"
    echo "  $0 microsoft/DialoGPT-medium"
    echo ""
    echo "注意: モデル名は正しいHugging FaceのリポジトリIDである必要があります"
}

# 引数チェック
if [ $# -eq 0 ]; then
    echo "❌ エラー: モデル名を指定してください"
    show_usage
    exit 1
fi

MODEL_NAME=$1
DOWNLOAD_DIR="./hf_models"

# ================================
# ディレクトリ作成
# ================================
echo "📁 ダウンロードディレクトリを作成中..."
mkdir -p "$DOWNLOAD_DIR"
echo "✅ ディレクトリ: $DOWNLOAD_DIR"

# ================================
# モデル情報の表示
# ================================
echo ""
echo "🚀 モデルダウンロードを開始します"
echo "モデル: $MODEL_NAME"
echo "保存先: $DOWNLOAD_DIR"
echo ""

# ================================
# Pythonスクリプトでダウンロード実行
# ================================
echo "📥 ダウンロード中..."
echo "⏳ 初回ダウンロードは時間がかかる場合があります..."
echo ""

# Pythonスクリプトを一時的に作成して実行
python3 -c "
import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download
import time

def download_model_with_progress(model_name, download_dir):
    print(f'🔍 モデル情報を取得中: {model_name}')
    
    try:
        # 設定ファイルをダウンロード
        print('📋 設定ファイルをダウンロード中...')
        config = AutoConfig.from_pretrained(model_name, cache_dir=download_dir)
        print(f'✅ 設定完了: {config.model_type}')
        
        # トークナイザーをダウンロード
        print('🔤 トークナイザーをダウンロード中...')
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
        print('✅ トークナイザー完了')
        
        # モデルファイルをダウンロード
        print('🧠 モデルファイルをダウンロード中...')
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=download_dir,
            torch_dtype='auto',
            device_map='auto'
        )
        print('✅ モデルファイル完了')
        
        # ファイルサイズを確認
        model_path = os.path.join(download_dir, 'models--' + model_name.replace('/', '--'))
        if os.path.exists(model_path):
            total_size = 0
            file_count = 0
            for root, dirs, files in os.walk(model_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        total_size += os.path.getsize(file_path)
                        file_count += 1
            
            print(f'📊 ダウンロード統計:')
            print(f'   - ファイル数: {file_count}')
            print(f'   - 総サイズ: {total_size / (1024**3):.2f} GB')
        
        print('🎉 ダウンロード完了！')
        return True
        
    except Exception as e:
        print(f'❌ エラー: {e}')
        return False

# メイン実行
if __name__ == '__main__':
    model_name = '$MODEL_NAME'
    download_dir = '$DOWNLOAD_DIR'
    
    success = download_model_with_progress(model_name, download_dir)
    sys.exit(0 if success else 1)
"

# ================================
# 結果確認
# ================================
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ダウンロードが正常に完了しました！"
    echo ""
    echo "📋 次のステップ:"
    echo "1. vLLMサーバーを起動:"
    echo "   ./run_vllm.sh $MODEL_NAME"
    echo ""
    echo "2. または、別のモデルをダウンロード:"
    echo "   ./download_model.sh <別のモデル名>"
else
    echo ""
    echo "❌ ダウンロード中にエラーが発生しました"
    echo "以下を確認してください:"
    echo "1. モデル名が正しいか"
    echo "2. インターネット接続が安定しているか"
    echo "3. 十分なディスク容量があるか"
    exit 1
fi
