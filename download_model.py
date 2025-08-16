#!/usr/bin/env python3
"""
モデルダウンロードスクリプト
使用方法: python download_model.py <モデル名>
例: python download_model.py Qwen/Qwen2.5-VL-3B-Instruct
"""

import os
import sys
import argparse
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq, AutoModel, AutoConfig
from huggingface_hub import snapshot_download
import torch

def show_usage():
    """使用方法を表示"""
    print("使用方法: python download_model.py <モデル名>")
    print("")
    print("例:")
    print("🔤 テキストのみモデル（軽量）:")
    print("  python download_model.py Qwen/Qwen2.5-0.5B-Instruct")
    print("  python download_model.py Qwen/Qwen2.5-1.5B-Instruct")
    print("  python download_model.py Qwen/Qwen2.5-3B-Instruct")
    print("")
    print("🖼️ マルチモーダルモデル:")
    print("  python download_model.py Qwen/Qwen2.5-VL-3B-Instruct-AWQ")
    print("  python download_model.py Qwen/Qwen2-VL-1.5B-Instruct")
    print("")
    print("📝 その他:")
    print("  python download_model.py microsoft/DialoGPT-medium")
    print("")
    print("注意: モデル名は正しいHugging FaceのリポジトリIDである必要があります")

def get_directory_size(directory):
    """ディレクトリの総サイズを計算"""
    total_size = 0
    file_count = 0
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)
                file_count += 1
    
    return total_size, file_count

def download_model_with_progress(model_name, download_dir):
    """モデルをダウンロードして進捗を表示"""
    print(f"🔍 モデル情報を取得中: {model_name}")
    
    try:
        # ディレクトリ作成
        print("📁 ダウンロードディレクトリを作成中...")
        os.makedirs(download_dir, exist_ok=True)
        print(f"✅ ディレクトリ: {download_dir}")
        
        # 設定ファイルをダウンロード
        print("📋 設定ファイルをダウンロード中...")
        config = AutoConfig.from_pretrained(model_name, cache_dir=download_dir)
        print(f"✅ 設定完了: {config.model_type}")
        
        # トークナイザーをダウンロード
        print("🔤 トークナイザーをダウンロード中...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
        print("✅ トークナイザー完了")
        
        # モデルファイルをダウンロード
        print("🧠 モデルファイルをダウンロード中...")
        print("⏳ 初回ダウンロードは時間がかかる場合があります...")
        
        # 汎用的なAutoModelを使用してモデルを自動判定
        print("🔍 モデルタイプを自動判定中...")
        model = AutoModel.from_pretrained(
            model_name, 
            cache_dir=download_dir,
            torch_dtype='auto',
            device_map='auto'
        )
        print("✅ モデルファイル完了")
        
        # ファイルサイズを確認
        model_path = os.path.join(download_dir, 'models--' + model_name.replace('/', '--'))
        if os.path.exists(model_path):
            total_size, file_count = get_directory_size(model_path)
            
            print(f"📊 ダウンロード統計:")
            print(f"   - ファイル数: {file_count}")
            print(f"   - 総サイズ: {total_size / (1024**3):.2f} GB")
        
        print("🎉 ダウンロード完了！")
        return True
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(description='Hugging Faceモデルをダウンロード')
    parser.add_argument('model_name', nargs='?', help='ダウンロードするモデル名（例: Qwen/Qwen2.5-VL-3B-Instruct）')
    parser.add_argument('--download-dir', default='./hf_models', help='ダウンロード先ディレクトリ（デフォルト: ./hf_models）')
    
    args = parser.parse_args()
    
    # モデル名が指定されていない場合
    if not args.model_name:
        print("❌ エラー: モデル名を指定してください")
        show_usage()
        sys.exit(1)
    
    print("🚀 モデルダウンロードを開始します")
    print(f"モデル: {args.model_name}")
    print(f"保存先: {args.download_dir}")
    print("")
    
    # ダウンロード実行
    success = download_model_with_progress(args.model_name, args.download_dir)
    
    if success:
        print("")
        print("✅ ダウンロードが正常に完了しました！")
        print("")
        print("📋 次のステップ:")
        print(f"1. vLLMサーバーを起動:")
        print(f"   ./run_vllm_quantized.sh {args.model_name} none")
        print("")
        print("2. 量子化を使用する場合:")
        print(f"   ./run_vllm_quantized.sh {args.model_name} bitsandbytes")
        print("")
        print("3. または、別のモデルをダウンロード:")
        print("   python download_model.py <別のモデル名>")
    else:
        print("")
        print("❌ ダウンロード中にエラーが発生しました")
        print("以下を確認してください:")
        print("1. モデル名が正しいか")
        print("2. インターネット接続が安定しているか")
        print("3. 十分なディスク容量があるか")
        sys.exit(1)

if __name__ == "__main__":
    main() 