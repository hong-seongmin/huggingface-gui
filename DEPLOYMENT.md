# HuggingFace GUI 배포 가이드

## 목차
- [요구사항](#요구사항)
- [로컬 배포](#로컬-배포)
- [Docker 배포](#docker-배포)
- [프로덕션 배포](#프로덕션-배포)
- [환경 변수](#환경-변수)
- [문제 해결](#문제-해결)

## 요구사항

### 시스템 요구사항
- **Python**: 3.12+ 권장
- **메모리**: 최소 4GB, 권장 8GB+
- **저장공간**: 최소 10GB (모델 캐시용)
- **네트워크**: 인터넷 연결 (모델 다운로드용)

### 선택적 요구사항
- **GPU**: CUDA 지원 GPU (선택사항, 성능 향상용)
- **Docker**: 컨테이너 배포용

## 로컬 배포

### 1. 저장소 클론
```bash
git clone https://github.com/hong-seongmin/huggingface-gui.git
cd huggingface-gui
```

### 2. 가상환경 설정
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\\Scripts\\activate  # Windows
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정 (선택사항)
```bash
cp .env.example .env
# .env 파일을 편집하여 필요한 설정 변경
```

### 5. 애플리케이션 실행
```bash
streamlit run app.py
```

웹 브라우저에서 `http://localhost:8501` 접속

## Docker 배포

### 1. Docker 이미지 빌드
```bash
docker build -t huggingface-gui .
```

### 2. 단일 컨테이너 실행
```bash
docker run -p 8501:8501 -p 8000:8000 \
  -v $(pwd)/model_cache:/app/model_cache \
  -v $(pwd)/logs:/app/logs \
  --name huggingface-gui \
  huggingface-gui
```

### 3. Docker Compose 사용 (권장)
```bash
docker-compose up -d
```

### 4. 컨테이너 상태 확인
```bash
docker-compose ps
docker-compose logs -f huggingface-gui
```

## 프로덕션 배포

### 1. 환경 변수 설정
프로덕션 환경에서는 반드시 환경 변수를 설정하세요:

```bash
# 필수 설정
export HOST=0.0.0.0
export PORT=8501
export HF_MODEL_CACHE_DIR=/app/model_cache
export LOG_LEVEL=WARNING

# 보안 설정
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### 2. 리버스 프록시 설정 (Nginx 예시)
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 3. 시스템 서비스 설정 (systemd 예시)
```ini
[Unit]
Description=HuggingFace GUI
After=network.target

[Service]
Type=simple
User=app
WorkingDirectory=/opt/huggingface-gui
Environment=PATH=/opt/huggingface-gui/venv/bin
ExecStart=/opt/huggingface-gui/venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
```

## 환경 변수

### 서버 설정
- `HOST`: 서버 바인딩 주소 (기본값: 127.0.0.1)
- `PORT`: Streamlit 포트 (기본값: 8501)
- `FASTAPI_HOST`: FastAPI 서버 주소 (기본값: 127.0.0.1)
- `FASTAPI_PORT`: FastAPI 포트 (기본값: 8000)

### 모델 설정
- `HF_MODEL_CACHE_DIR`: 모델 캐시 디렉토리
- `DEFAULT_DEVICE`: 기본 디바이스 (auto, cpu, cuda, mps)
- `MAX_CONCURRENT_MODELS`: 동시 로드 가능 모델 수

### HuggingFace Hub 설정
- `HF_TOKEN`: HuggingFace Hub 토큰 (비공개 모델용)
- `HF_HUB_OFFLINE`: 오프라인 모드 (true/false)
- `TRANSFORMERS_OFFLINE`: Transformers 오프라인 모드
- `TOKENIZERS_PARALLELISM`: 토크나이저 병렬처리

### 성능 설정
- `MAX_MODEL_MEMORY`: 모델 메모리 제한 (GB)
- `GPU_MEMORY_FRACTION`: GPU 메모리 사용률 (0.0-1.0)
- `ENABLE_QUANTIZATION`: 모델 양자화 활성화

## 문제 해결

### 일반적인 문제들

#### 1. 메모리 부족 오류
```bash
# 스왑 메모리 설정 (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 2. CUDA 관련 오류
```bash
# CPU 전용 PyTorch 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 3. 포트 충돌
```bash
# 사용 중인 포트 확인
netstat -tlnp | grep :8501
# 다른 포트 사용
streamlit run app.py --server.port=8502
```

#### 4. 모델 다운로드 실패
```bash
# 오프라인 모드 비활성화
export HF_HUB_OFFLINE=false
export TRANSFORMERS_OFFLINE=false
```

### 로그 확인
```bash
# Docker 컨테이너 로그
docker-compose logs -f huggingface-gui

# 로컬 실행 로그
tail -f app_debug.log
```

### 성능 모니터링
```bash
# 시스템 리소스 모니터링
htop
nvidia-smi  # GPU 사용 시

# 애플리케이션 메트릭
curl http://localhost:8501/_stcore/health
```

## 보안 고려사항

1. **방화벽 설정**: 필요한 포트만 열기
2. **HTTPS 설정**: 프로덕션에서는 반드시 HTTPS 사용
3. **인증**: 필요시 Streamlit 인증 설정
4. **토큰 관리**: HF_TOKEN 등 민감 정보 안전하게 관리

## 업데이트

```bash
# 코드 업데이트
git pull origin main

# 의존성 업데이트
pip install -r requirements.txt --upgrade

# Docker 이미지 재빌드
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## 지원

문제가 발생하면 다음을 확인하세요:
1. [GitHub Issues](https://github.com/hong-seongmin/huggingface-gui/issues)
2. 로그 파일 (`app_debug.log`)
3. 시스템 리소스 상태
4. 네트워크 연결 상태