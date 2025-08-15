# # 使用NVIDIA官方PyTorch镜像作为基础（已包含CUDA 12.6支持）
# FROM nvcr.io/nvidia/pytorch:24.02-py3

# # 系统更新和必要依赖安装
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     && rm -rf /var/lib/apt/lists/*

# # 设置工作目录
# WORKDIR /app

# # 先复制requirements.txt（为了利用Docker缓存层）
# COPY requirements.txt .

# # 安装依赖（使用国内镜像源加速）
# RUN pip install --no-cache-dir --upgrade pip && \
#     pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# # 复制应用代码
# COPY . .

# # 容器启动命令
# CMD ["python3", "main.py"]

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
