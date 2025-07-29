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
# NVIDIA repo 문제 해결을 위해 모든 CUDA repo를 완전히 제거하고 비활성화
RUN rm -f /etc/apt/sources.list.d/cuda.list \
    /etc/apt/sources.list.d/nvidia-ml.list \
    /etc/apt/sources.list.d/cuda*.list \
    /etc/apt/sources.list.d/nvidia*.list && \
    echo 'Package: *\nPin: origin "developer.download.nvidia.com"\nPin-Priority: -1' > /etc/apt/preferences.d/nvidia && \
    apt-get update && \
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

# pip 업그레이드
RUN python -m pip install --upgrade pip

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . /workspace/

# 포트 노출 (TensorBoard용)
EXPOSE 6006

# 기본 명령어 (python3.10 사용 명시)
CMD ["python", "train.py"]