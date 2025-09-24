# 🚀 HuggingFace GUI 설치 가이드

이 문서는 HuggingFace GUI를 다양한 환경에서 설치하고 실행하는 방법을 설명합니다.

## 📋 시스템 요구사항

### 최소 요구사항
- **OS**: Linux, macOS, Windows 10+
- **Python**: 3.9 이상 (3.11+ 권장)
- **메모리**: 4GB RAM (8GB+ 권장)
- **저장공간**: 10GB 이상 (모델 캐시용)
- **네트워크**: 인터넷 연결 (모델 다운로드용)

### 권장 요구사항
- **GPU**: CUDA 지원 GPU 또는 Apple Silicon (선택사항)
- **메모리**: 16GB+ RAM
- **저장공간**: 50GB+ SSD

## ⚡ 빠른 설치 (권장)

### 1. 자동 설치 스크립트 사용

**Linux/macOS:**
```bash
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
./setup.sh
```

**Windows:**
```cmd
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
setup.bat
```

**개발자용:**
```bash
make quick-start
```

### 2. 실행
설치가 완료되면 자동으로 실행 안내가 표시됩니다. 브라우저에서 `http://localhost:8501`에 접속하세요.

## 🔧 수동 설치

자동 설치가 실패하거나 사용자 정의 설정이 필요한 경우 수동으로 설치할 수 있습니다.

### Step 1: 저장소 클론
```bash
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
```

### Step 2: Python 가상환경 설정 (권장)
```bash
# Python 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate  # Linux/Mac
# 또는
.venv\\Scripts\\activate  # Windows
```

### Step 3: 패키지 관리자 선택

**Option A: uv 사용 (빠르고 권장)**
```bash
# uv 설치 (처음 한 번만)
curl -LsSf https://astral.sh/uv/install.sh | sh
# 또는 Windows의 경우:
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 의존성 설치
uv sync
```

**Option B: pip 사용 (전통적인 방법)**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: 환경 설정
```bash
# .env 파일 생성
cp .env.example .env

# 필요한 디렉토리 생성
mkdir -p logs
```

### Step 5: 설치 확인
```bash
# 시스템 호환성 체크
python scripts/compatibility_check.py

# 또는 간단한 import 테스트
python -c "import streamlit, transformers, torch; print('Installation successful!')"
```

## 🐳 Docker 설치

Docker를 사용하여 격리된 환경에서 실행할 수 있습니다.

### 단일 컨테이너
```bash
# 이미지 빌드
docker build -t huggingface-gui .

# 컨테이너 실행
docker run -p 8501:8501 -p 8000:8000 \
  -v $(pwd)/model_cache:/app/model_cache \
  -v $(pwd)/logs:/app/logs \
  huggingface-gui
```

### Docker Compose (권장)
```bash
# 서비스 시작
docker-compose up -d

# 로그 확인
docker-compose logs -f

# 서비스 중지
docker-compose down
```

## 🎯 특정 환경별 설치

### Ubuntu/Debian
```bash
# 시스템 패키지 업데이트
sudo apt update

# 필수 패키지 설치
sudo apt install -y python3 python3-pip python3-venv git build-essential curl

# 프로젝트 설치
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
./setup.sh
```

### CentOS/RHEL/Fedora
```bash
# 필수 패키지 설치
sudo yum install -y python3 python3-pip git gcc gcc-c++ make curl
# 또는 Fedora의 경우: sudo dnf install -y python3 python3-pip git gcc gcc-c++ make curl

# 프로젝트 설치
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
./setup.sh
```

### macOS
```bash
# Homebrew가 없는 경우 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치 (시스템 Python 사용 가능하면 생략)
brew install python3 git

# Xcode Command Line Tools 설치 (필요시)
xcode-select --install

# 프로젝트 설치
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
./setup.sh
```

### Windows (WSL 사용)
```bash
# WSL2 Ubuntu 환경에서
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git build-essential curl

# 프로젝트 설치
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
./setup.sh
```

### Windows (네이티브)
1. Python 3.9+ 설치: https://python.org/downloads/
2. Git 설치: https://git-scm.com/download/win
3. PowerShell에서 실행:
```powershell
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
.\setup.bat
```

## 🔍 설치 문제 해결

### 일반적인 문제들

#### 1. Python 버전 문제
```bash
# Python 버전 확인
python --version
python3 --version

# 올바른 Python 사용하도록 설정
alias python=python3  # 임시
# 또는 ~/.bashrc에 추가하여 영구적으로 설정
```

#### 2. 권한 문제 (Linux/macOS)
```bash
# 스크립트 실행 권한 부여
chmod +x setup.sh

# pip 사용자 설치 (시스템 패키지와 충돌 방지)
pip install --user -r requirements.txt
```

#### 3. 네트워크 문제
```bash
# 중국 사용자의 경우 미러 사용
pip install -i https://pypi.douban.com/simple/ -r requirements.txt

# 또는 국내 미러
pip install -i https://mirror.kakao.com/pypi/simple/ -r requirements.txt
```

#### 4. GPU 관련 문제
```bash
# CUDA 버전 확인
nvidia-smi

# CPU 전용 PyTorch 설치 (GPU 없는 경우)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 5. 메모리 부족
```bash
# 스왑 메모리 추가 (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 고급 문제 해결

#### 의존성 충돌
```bash
# 새로운 가상환경에서 재설치
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -tlnp | grep 8501
lsof -i :8501

# 다른 포트 사용
streamlit run app.py --server.port=8502
```

#### 캐시 문제
```bash
# Python 캐시 정리
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# pip 캐시 정리
pip cache purge

# 또는 Makefile 사용
make clean
```

## ✅ 설치 검증

설치가 완료되면 다음 단계로 검증하세요:

### 1. 자동 검증
```bash
# 종합적인 호환성 체크
python scripts/compatibility_check.py

# 애플리케이션 상태 확인 (실행 후)
python scripts/health_check.py
```

### 2. 수동 검증
```bash
# 모든 핵심 라이브러리 임포트 테스트
python -c "
import streamlit
import transformers
import torch
import fastapi
import uvicorn
print('✅ All core dependencies imported successfully')
"

# 애플리케이션 실행 테스트
streamlit run app.py --server.headless=true &
sleep 5
curl -f http://localhost:8501/_stcore/health && echo "✅ Streamlit is running"
pkill -f streamlit
```

### 3. 성능 테스트
```bash
# GPU 감지 테스트
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
else:
    print('Using CPU')
"
```

## 🎉 설치 완료

설치가 성공적으로 완료되었다면:

1. **애플리케이션 실행**: `streamlit run app.py` 또는 `make run`
2. **브라우저 접속**: `http://localhost:8501`
3. **문서 확인**: README.md의 사용법 섹션 참조
4. **도움말**: `make help`로 사용 가능한 명령어 확인

## 📞 지원

문제가 발생하면:

1. **호환성 리포트 생성**: `python scripts/compatibility_check.py --save-report`
2. **로그 확인**: `tail -f app_debug.log`
3. **GitHub Issues**: 리포트와 함께 이슈 등록
4. **문서**: README.md와 TROUBLESHOOTING_GUIDE.md 참조

즐거운 개발되세요! 🚀