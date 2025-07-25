# PyTorch 공식 베이스 이미지 사용 (Python 3.11, CUDA 12.1 지원)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 작업 디렉토리 설정
WORKDIR /workspace

# 시스템 패키지 설치
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드 (python3.11 사용 명시)
RUN python3.11 -m pip install --upgrade pip

# requirements.txt 복사 및 설치 (python3.11 사용 명시)
COPY requirements.txt /workspace/
RUN python3.11 -m pip install -r requirements.txt

# 소스 코드 복사
COPY . /workspace/

# 포트 노출 (TensorBoard용)
EXPOSE 6006

# 기본 실행 명령어 (python3.11 사용 명시)
CMD ["python3.11", "train.py"]