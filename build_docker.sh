#!/bin/bash
# ArcFace-PyTorch Docker 이미지 빌드 스크립트

set -e  # 에러 발생시 스크립트 중단

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 프로젝트 정보
PROJECT_NAME="arcface-pytorch"
IMAGE_NAME="arcface-pytorch"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${CYAN}🐳 ArcFace-PyTorch Docker 이미지 빌드 시작${NC}"
echo "=============================================="

# 현재 디렉토리 확인
CURRENT_DIR=$(pwd)
echo -e "${BLUE}📂 현재 디렉토리: ${CURRENT_DIR}${NC}"

# 필수 파일 존재 확인
echo -e "${YELLOW}📋 필수 파일 확인 중...${NC}"

required_files=(
    "Dockerfile"
    "train.py"
    "config"
    "models"
    "data"
    "utils"
    "requirements.txt"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        missing_files+=("$file")
    else
        echo -e "${GREEN}✓ ${file} 존재${NC}"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo -e "${RED}❌ 다음 필수 파일/폴더가 없습니다:${NC}"
    for file in "${missing_files[@]}"; do
        echo -e "${RED}  - $file${NC}"
    done
    exit 1
fi

# Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker가 설치되어 있지 않습니다.${NC}"
    echo "Docker를 설치한 후 다시 시도해주세요."
    exit 1
fi

echo -e "${GREEN}✓ Docker 설치 확인 완료${NC}"

# NVIDIA Docker 런타임 확인
if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
    echo -e "${GREEN}✓ NVIDIA Docker 런타임 사용 가능${NC}"
else
    echo -e "${YELLOW}⚠️  NVIDIA Docker 런타임을 감지할 수 없습니다.${NC}"
    echo -e "${YELLOW}   GPU 기능이 제한될 수 있습니다.${NC}"
fi

# .dockerignore 생성/업데이트
echo -e "${YELLOW}📝 .dockerignore 파일 생성 중...${NC}"
cat > .dockerignore << 'EOF'
# Git 관련
.git
.gitignore
*.md
README.md

# Python 관련
.venv
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# 로그 및 캐시
*.log
.cache
.coverage
.coverage.*
.pytest_cache/
.mypy_cache/

# IDE 관련
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# 사용하지 않는 패키지 폴더 (명시적으로 제외)
insight_face_package_model/

# 임시 파일들
temp/
tmp/

# 훈련 관련 (선택적)
checkpoints/*.pth
logs/
runs/
wandb/

# 데이터셋 (용량이 큰 경우 주석 해제)
# dataset/
# data/celebA/
# pair/
# pair_test/

# 기타
*.zip
*.tar.gz
*.tar
EOF

echo -e "${GREEN}✓ .dockerignore 파일 생성 완료${NC}"

# 빌드 컨텍스트 크기 확인
echo -e "${YELLOW}📊 빌드 컨텍스트 크기 확인 중...${NC}"
context_size=$(du -sh . --exclude='.git' 2>/dev/null | cut -f1)
echo -e "${BLUE}빌드 컨텍스트 크기: ${context_size}${NC}"

# 기존 이미지 확인
if docker images ${FULL_IMAGE_NAME} | grep -q ${IMAGE_NAME}; then
    echo -e "${YELLOW}⚠️  기존 이미지 '${FULL_IMAGE_NAME}'가 존재합니다.${NC}"
    read -p "기존 이미지를 덮어쓰시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}빌드를 취소합니다.${NC}"
        exit 0
    fi
fi

# Docker 빌드 시작
echo -e "${CYAN}🔨 Docker 이미지 빌드 시작...${NC}"
echo "이미지 이름: ${FULL_IMAGE_NAME}"
echo "빌드 시작 시간: $(date '+%Y-%m-%d %H:%M:%S')"

# 빌드 시작 시간 기록
start_time=$(date +%s)

# Docker 빌드 실행 (진행률 표시)
echo -e "${BLUE}============== 빌드 로그 ==============${NC}"

if docker build \
    --progress=plain \
    --no-cache \
    -t ${FULL_IMAGE_NAME} \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .; then
    
    # 빌드 완료 시간 계산
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    minutes=$((duration / 60))
    seconds=$((duration % 60))
    
    echo -e "${BLUE}============== 빌드 완료 ==============${NC}"
    echo -e "${GREEN}✅ Docker 이미지 빌드 성공!${NC}"
    echo -e "${GREEN}⏱️  총 빌드 시간: ${minutes}분 ${seconds}초${NC}"
    echo -e "${GREEN}🏷️  이미지 이름: ${FULL_IMAGE_NAME}${NC}"
    
    # 이미지 정보 출력
    echo -e "${CYAN}📊 이미지 상세 정보:${NC}"
    docker images ${FULL_IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.ID}}\t{{.CreatedAt}}\t{{.Size}}"
    
    # 이미지 레이어 정보
    echo -e "${CYAN}📋 이미지 히스토리 (최근 5개):${NC}"
    docker history ${FULL_IMAGE_NAME} --format "table {{.CreatedBy}}\t{{.Size}}" | head -6
    
    # 빌드된 이미지 검증
    echo -e "${YELLOW}🔍 이미지 검증 중...${NC}"
    if docker run --rm ${FULL_IMAGE_NAME} python --version; then
        echo -e "${GREEN}✓ Python 버전 확인 완료${NC}"
    else
        echo -e "${RED}❌ Python 실행 실패${NC}"
    fi
    
    if docker run --rm ${FULL_IMAGE_NAME} pip show torch | grep -q "Version:"; then
        echo -e "${GREEN}✓ PyTorch 설치 확인 완료${NC}"
    else
        echo -e "${RED}❌ PyTorch 설치 확인 실패${NC}"
    fi
    
    echo ""
    echo -e "${CYAN}🚀 사용 방법:${NC}"
    echo -e "${YELLOW}  컨테이너 실행: ./run_docker.sh${NC}"
    echo -e "${YELLOW}  수동 실행:     docker run --gpus all -it --rm -v \$(pwd):/workspace/data ${FULL_IMAGE_NAME}${NC}"
    echo -e "${YELLOW}  TensorBoard:   http://localhost:6006${NC}"
    echo ""
    
    # 정리 옵션
    read -p "빌드 캐시를 정리하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}🧹 Docker 빌드 캐시 정리 중...${NC}"
        docker system prune -f
        echo -e "${GREEN}✓ 빌드 캐시 정리 완료${NC}"
    fi
    
else
    echo -e "${RED}❌ Docker 이미지 빌드 실패!${NC}"
    echo -e "${RED}빌드 로그를 확인하고 문제를 해결해주세요.${NC}"
    echo ""
    echo -e "${YELLOW}💡 문제 해결 팁:${NC}"
    echo -e "${YELLOW}  1. 네트워크 연결 확인${NC}"
    echo -e "${YELLOW}  2. 디스크 공간 확인 (최소 10GB 필요)${NC}"
    echo -e "${YELLOW}  3. Docker 버전 확인 (최신 버전 권장)${NC}"
    echo -e "${YELLOW}  4. requirements.txt 파일 확인${NC}"
    exit 1
fi

echo "=============================================="
echo -e "${CYAN}🎉 빌드 프로세스 완료!${NC}"
echo "빌드 완료 시간: $(date '+%Y-%m-%d %H:%M:%S')"
