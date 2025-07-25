#!/bin/bash
# ArcFace-PyTorch Docker 컨테이너 실행 스크립트

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 프로젝트 정보
IMAGE_NAME="arcface-pytorch"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="arcface-training"

echo -e "${CYAN}🐳 ArcFace-PyTorch Docker 컨테이너 실행${NC}"
echo "=============================================="

# 이미지 존재 확인
if ! docker images ${FULL_IMAGE_NAME} | grep -q ${IMAGE_NAME}; then
    echo -e "${RED}❌ Docker 이미지 '${FULL_IMAGE_NAME}'를 찾을 수 없습니다.${NC}"
    echo -e "${YELLOW}먼저 이미지를 빌드해주세요: ./build_docker.sh${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker 이미지 확인 완료${NC}"

# GPU 지원 확인
GPU_SUPPORT=""
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_SUPPORT="--gpus all"
        echo -e "${GREEN}✓ NVIDIA GPU 감지됨 - GPU 가속 활성화${NC}"
        
        # GPU 정보 출력
        echo -e "${BLUE}📊 GPU 정보:${NC}"
        nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits | \
        while IFS=, read -r name memory_total memory_free; do
            echo -e "${BLUE}  - ${name}: ${memory_free}MB/${memory_total}MB 사용 가능${NC}"
        done
    else
        echo -e "${YELLOW}⚠️  NVIDIA GPU가 있지만 nvidia-smi 실행 실패${NC}"
        echo -e "${YELLOW}   GPU 없이 실행합니다.${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  GPU를 감지할 수 없습니다. CPU 모드로 실행합니다.${NC}"
fi

# 기존 컨테이너 확인 및 정리
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}⚠️  기존 컨테이너 '${CONTAINER_NAME}'가 존재합니다.${NC}"
    
    # 실행 중인지 확인
    if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
        echo -e "${YELLOW}컨테이너가 실행 중입니다. 중지 중...${NC}"
        docker stop ${CONTAINER_NAME}
    fi
    
    echo -e "${YELLOW}기존 컨테이너를 제거합니다...${NC}"
    docker rm ${CONTAINER_NAME}
fi

# 디렉토리 설정
CURRENT_DIR=$(pwd)
DATA_DIR="${CURRENT_DIR}/data"
CHECKPOINTS_DIR="${CURRENT_DIR}/checkpoints"
LOGS_DIR="${CURRENT_DIR}/logs"

# 필요한 디렉토리 생성
mkdir -p "${DATA_DIR}"
mkdir -p "${CHECKPOINTS_DIR}"
mkdir -p "${LOGS_DIR}"

echo -e "${BLUE}📂 마운트 디렉토리:${NC}"
echo -e "${BLUE}  - 소스 코드: ${CURRENT_DIR} -> /workspace${NC}"
echo -e "${BLUE}  - 데이터: ${DATA_DIR} -> /workspace/data${NC}"
echo -e "${BLUE}  - 체크포인트: ${CHECKPOINTS_DIR} -> /workspace/checkpoints${NC}"
echo -e "${BLUE}  - 로그: ${LOGS_DIR} -> /workspace/logs${NC}"

# 실행 모드 선택
echo ""
echo -e "${CYAN}🎯 실행 모드를 선택하세요:${NC}"
echo "1) 인터랙티브 모드 (bash 셸)"
echo "2) 훈련 시작 (train.py 실행)"
echo "3) TensorBoard 서버"
echo "4) 추론 모드 (inference.py)"
echo "5) 커스텀 명령어 입력"

read -p "선택 (1-5): " -n 1 -r mode
echo

case $mode in
    1)
        echo -e "${GREEN}🖥️  인터랙티브 모드로 실행합니다...${NC}"
        COMMAND="/bin/bash"
        INTERACTIVE_FLAGS="-it"
        ;;
    2)
        echo -e "${GREEN}🏃 훈련 모드로 실행합니다...${NC}"
        COMMAND="python train.py"
        INTERACTIVE_FLAGS="-it"
        ;;
    3)
        echo -e "${GREEN}📊 TensorBoard 서버를 시작합니다...${NC}"
        COMMAND="tensorboard --logdir=/workspace/runs --host=0.0.0.0 --port=6006"
        INTERACTIVE_FLAGS="-d"
        echo -e "${YELLOW}TensorBoard는 http://localhost:6006 에서 접속 가능합니다.${NC}"
        ;;
    4)
        echo -e "${GREEN}🔍 추론 모드로 실행합니다...${NC}"
        COMMAND="python inference.py"
        INTERACTIVE_FLAGS="-it"
        ;;
    5)
        read -p "실행할 명령어를 입력하세요: " custom_command
        COMMAND="$custom_command"
        INTERACTIVE_FLAGS="-it"
        ;;
    *)
        echo -e "${RED}❌ 잘못된 선택입니다. 인터랙티브 모드로 실행합니다.${NC}"
        COMMAND="/bin/bash"
        INTERACTIVE_FLAGS="-it"
        ;;
esac

# 포트 설정
PORT_MAPPING="-p 6006:6006 -p 8888:8888"

# 환경 변수 설정
ENV_VARS=""
ENV_VARS="${ENV_VARS} -e CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
ENV_VARS="${ENV_VARS} -e PYTHONPATH=/workspace"

# 추가 볼륨 마운트 (선택적)
EXTRA_VOLUMES=""
if [ -d "/tmp/.X11-unix" ]; then
    EXTRA_VOLUMES="${EXTRA_VOLUMES} -v /tmp/.X11-unix:/tmp/.X11-unix"
    ENV_VARS="${ENV_VARS} -e DISPLAY=${DISPLAY}"
fi

echo ""
echo -e "${CYAN}🚀 Docker 컨테이너 실행 중...${NC}"
echo -e "${BLUE}이미지: ${FULL_IMAGE_NAME}${NC}"
echo -e "${BLUE}컨테이너: ${CONTAINER_NAME}${NC}"
echo -e "${BLUE}명령어: ${COMMAND}${NC}"

# Docker 실행
docker run \
    ${INTERACTIVE_FLAGS} \
    --name ${CONTAINER_NAME} \
    --rm \
    ${GPU_SUPPORT} \
    ${PORT_MAPPING} \
    ${ENV_VARS} \
    -v "${CURRENT_DIR}:/workspace" \
    -v "${DATA_DIR}:/workspace/data" \
    -v "${CHECKPOINTS_DIR}:/workspace/checkpoints" \
    -v "${LOGS_DIR}:/workspace/logs" \
    ${EXTRA_VOLUMES} \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    ${FULL_IMAGE_NAME} \
    ${COMMAND}

echo ""
echo -e "${GREEN}✅ 컨테이너 실행 완료${NC}"
echo "=============================================="
