#!/bin/bash
# Docker 환경 검증 및 설정 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}🔍 Docker 환경 검증 및 설정${NC}"
echo "=============================================="

# 1. Docker 설치 확인
echo -e "${YELLOW}1. Docker 설치 확인...${NC}"
if command -v docker &> /dev/null; then
    docker_version=$(docker --version)
    echo -e "${GREEN}✓ Docker 설치됨: ${docker_version}${NC}"
else
    echo -e "${RED}❌ Docker가 설치되어 있지 않습니다.${NC}"
    echo -e "${YELLOW}Docker 설치 방법:${NC}"
    echo "curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "sudo sh get-docker.sh"
    echo "sudo usermod -aG docker \$USER"
    exit 1
fi

# 2. Docker 서비스 상태 확인
echo -e "${YELLOW}2. Docker 서비스 상태 확인...${NC}"
if systemctl is-active --quiet docker; then
    echo -e "${GREEN}✓ Docker 서비스 실행 중${NC}"
else
    echo -e "${RED}❌ Docker 서비스가 실행되고 있지 않습니다.${NC}"
    echo "sudo systemctl start docker"
    exit 1
fi

# 3. 사용자 권한 확인
echo -e "${YELLOW}3. Docker 사용자 권한 확인...${NC}"
if docker ps &> /dev/null; then
    echo -e "${GREEN}✓ Docker 명령어 실행 권한 있음${NC}"
else
    echo -e "${RED}❌ Docker 명령어 실행 권한이 없습니다.${NC}"
    echo "sudo usermod -aG docker \$USER"
    echo "로그아웃 후 다시 로그인하세요."
    exit 1
fi

# 4. NVIDIA Docker 확인 (선택적)
echo -e "${YELLOW}4. NVIDIA Docker 지원 확인...${NC}"
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        echo -e "${GREEN}✓ NVIDIA GPU 감지됨${NC}"
        
        # nvidia-docker2 또는 nvidia-container-toolkit 확인
        if docker info | grep -q nvidia || command -v nvidia-docker &> /dev/null; then
            echo -e "${GREEN}✓ NVIDIA Docker 런타임 사용 가능${NC}"
        else
            echo -e "${YELLOW}⚠️  NVIDIA Docker 런타임이 설치되어 있지 않습니다.${NC}"
            echo -e "${YELLOW}NVIDIA Docker 설치 방법:${NC}"
            echo "# NVIDIA Container Toolkit 저장소 추가"
            echo "distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
            echo "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -"
            echo "curl -s -L https://nvidia.github.io/nvidia-docker/\$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list"
            echo "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
            echo "sudo systemctl restart docker"
        fi
    else
        echo -e "${YELLOW}⚠️  NVIDIA GPU가 있지만 nvidia-smi 실행 실패${NC}"
    fi
else
    echo -e "${BLUE}ℹ️  NVIDIA GPU를 감지할 수 없습니다. CPU 모드로 실행됩니다.${NC}"
fi

# 5. 디스크 공간 확인
echo -e "${YELLOW}5. 디스크 공간 확인...${NC}"
available_space=$(df . | awk 'NR==2 {print $4}')
available_gb=$((available_space / 1024 / 1024))

if [ $available_gb -ge 10 ]; then
    echo -e "${GREEN}✓ 충분한 디스크 공간: ${available_gb}GB 사용 가능${NC}"
else
    echo -e "${RED}❌ 디스크 공간 부족: ${available_gb}GB 사용 가능 (최소 10GB 필요)${NC}"
    echo "디스크 공간을 확보한 후 다시 시도하세요."
    exit 1
fi

# 6. 메모리 확인
echo -e "${YELLOW}6. 시스템 메모리 확인...${NC}"
total_mem=$(free -g | awk 'NR==2{print $2}')
available_mem=$(free -g | awk 'NR==2{print $7}')

echo -e "${BLUE}ℹ️  총 메모리: ${total_mem}GB, 사용 가능: ${available_mem}GB${NC}"

if [ $available_mem -ge 4 ]; then
    echo -e "${GREEN}✓ 충분한 메모리${NC}"
else
    echo -e "${YELLOW}⚠️  메모리가 부족할 수 있습니다 (4GB 이상 권장)${NC}"
fi

# 7. 네트워크 연결 확인
echo -e "${YELLOW}7. 네트워크 연결 확인...${NC}"
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo -e "${GREEN}✓ 인터넷 연결 정상${NC}"
else
    echo -e "${RED}❌ 인터넷 연결을 확인할 수 없습니다.${NC}"
    echo "Docker 이미지 다운로드를 위해 인터넷 연결이 필요합니다."
fi

# 8. Docker Hub 연결 확인
echo -e "${YELLOW}8. Docker Hub 연결 확인...${NC}"
if docker pull hello-world &> /dev/null; then
    docker rmi hello-world &> /dev/null
    echo -e "${GREEN}✓ Docker Hub 연결 정상${NC}"
else
    echo -e "${RED}❌ Docker Hub에 연결할 수 없습니다.${NC}"
fi

# 9. 필수 파일 확인
echo -e "${YELLOW}9. 프로젝트 필수 파일 확인...${NC}"
required_files=(
    "Dockerfile"
    "requirements.txt"
    "train.py"
    "config"
    "models"
    "data"
    "utils"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -e "$file" ]; then
        missing_files+=("$file")
    else
        echo -e "${GREEN}✓ ${file}${NC}"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo -e "${RED}❌ 다음 필수 파일이 없습니다:${NC}"
    for file in "${missing_files[@]}"; do
        echo -e "${RED}  - $file${NC}"
    done
    exit 1
fi

# 10. 스크립트 실행 권한 확인
echo -e "${YELLOW}10. 스크립트 실행 권한 설정...${NC}"
scripts=("build_docker.sh" "run_docker.sh" "verify_docker.sh")
for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo -e "${GREEN}✓ ${script} 실행 권한 설정${NC}"
    fi
done

# 검증 완료
echo ""
echo -e "${GREEN}✅ Docker 환경 검증 완료!${NC}"
echo "=============================================="
echo -e "${CYAN}🚀 이제 다음 명령어로 이미지를 빌드할 수 있습니다:${NC}"
echo -e "${YELLOW}  ./build_docker.sh${NC}"
echo ""
echo -e "${CYAN}📊 시스템 정보 요약:${NC}"
echo -e "${BLUE}  - Docker: $(docker --version | cut -d' ' -f3 | sed 's/,//')${NC}"
echo -e "${BLUE}  - 사용 가능 디스크: ${available_gb}GB${NC}"
echo -e "${BLUE}  - 사용 가능 메모리: ${available_mem}GB${NC}"
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    echo -e "${BLUE}  - GPU: ${gpu_info}${NC}"
fi
