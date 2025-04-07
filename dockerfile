# CUDA対応のTensorFlow GPUバージョンを使用
FROM nvidia/cuda:11.0-base

# Pythonをインストール
RUN apt-get update && apt-get install -y python3 python3-pip

# 作業ディレクトリを設定
WORKDIR /app

# requirements.txtをコピー
COPY requirements.txt .

# 依存関係をインストール
RUN pip3 install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# 学習スクリプトを実行
CMD ["python3"]
