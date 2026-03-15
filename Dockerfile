#===============================================================================
# LTuner Dockerfile
# 用于在 Docker 容器中快速部署 LTuner
#
# 使用方法:
#   1. 构建镜像: docker build -t ltuner .
#   2. 运行容器: docker run -d -p 8501:8501 --name ltuner ltuner
#   3. 访问界面:  http://localhost:8501
#===============================================================================

# 基础镜像
FROM ubuntu:22.04

# 维护者
LABEL maintainer="LTuner Project"

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PROJECT_ROOT=/app
ENV PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    net-tools \
    openjdk-21-jdk \
    python3.10 \
    python3-pip \
    python3-venv \
    postgresql-14 \
    postgresql-client-14 \
    && rm -rf /var/lib/apt/lists/*

# 设置 Java 环境
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

# 创建项目目录
RUN mkdir -p $PROJECT_ROOT
WORKDIR $PROJECT_ROOT

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# 复制项目代码
COPY . .

# 设置 HuggingFace 离线模式（避免联网检查）
ENV TRANSFORMERS_OFFLINE=1
ENV HF_HUB_OFFLINE=1

# 创建必要的目录
RUN mkdir -p optimization_results/postgres/{coarse,fine,ltuner,tuner} \
             optimization_results/temp_results \
             optimization_results/comparison \
             optimization_results/comparison_real

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["/bin/bash", "-c", "service postgresql start && . venv/bin/activate && streamlit run src/web/app.py --server.port=8501 --server.address=0.0.0.0"]
