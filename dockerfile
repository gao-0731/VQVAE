FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip uninstall -y opencv && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2/ && \
    pip install opencv-python-headless==4.9.0.80

# ※ 開発時は「COPY . .」不要（volumeマウント推奨）

CMD ["/bin/bash"]
