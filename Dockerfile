# 使用 NVIDIA 提供的 CUDA 11.8 基础镜像（基于 Ubuntu 22.04）
FROM registry.cn-hangzhou.aliyuncs.com/serverless_devs/cuda:11.8.0-devel-ubuntu22.04

# 安装基础依赖
RUN apt-get update && \
    apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        libgl1 \
        libgomp1 \
        ffmpeg \
        python3.10 \
        python3-pip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置 Python 环境
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# 安装 PyTorch 2.3 和 Transformers
RUN pip install torch==2.3.0+cu118 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com transformers
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com ms-swift
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com librosa
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com pydantic==1.10.22
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com fastapi
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com uvicorn
RUN pip install --no-cache-dir --index-url https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com accelerate==0.28.0

# 设置工作目录
WORKDIR /app

# 将当前目录下的 app 目录复制到镜像中
COPY main.py /app/main.py


# 设置启动命令（假设 app 的入口文件为 run.py）
CMD ["uvicorn", "main:app"]