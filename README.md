# Hugging Face GUI

[huggingface-cli](https://huggingface.co/docs/huggingface_hub/ko/guides/cli)를 GUI로 만드는 것이 목적입니다.

## 🚀 새로운 기능들

### 1. 실시간 시스템 모니터링
- **CPU, GPU, 메모리 사용률 실시간 모니터링**
- 시각화된 차트로 리소스 사용 현황 확인
- 시스템 알림 및 경고 기능
- 히스토리 데이터 저장 및 내보내기

### 2. 다중 모델 관리
- **여러 모델 동시 로드/언로드 지원**
- 모델별 메모리 사용량 추적
- 비동기 모델 로딩으로 UI 블로킹 방지
- 모델 상태 실시간 업데이트

### 3. 포괄적인 모델 분석
- **config.json, tokenizer.json, vocab.txt 등 모든 모델 파일 분석**
- 모델 타입, 파라미터 수, 지원 태스크 자동 추론
- 모델 크기 및 메모리 요구사항 예측
- 권장사항 및 최적화 제안

### 4. FastAPI 자동 호스팅
- **로드된 모델을 즉시 API 서버로 호스팅**
- RESTful API 엔드포인트 자동 생성
- 모델별 지원 태스크에 따른 추론 파이프라인 구성
- Swagger UI를 통한 API 문서 자동 생성

## 📁 파일 구조

```
huggingface-gui/
├── app.py                 # Streamlit 앱
├── run.py                 # CustomTkinter 앱
├── model_analyzer.py      # 모델 분석 모듈
├── model_manager.py       # 다중 모델 관리 모듈
├── system_monitor.py      # 시스템 모니터링 모듈
├── fastapi_server.py      # FastAPI 서버 모듈
├── demo.py               # 기능 데모 스크립트
├── requirements.txt       # 의존성 패키지
└── README.md             # 이 파일
```

## 🛠️ 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 실행 방법

#### Streamlit 버전
```bash
streamlit run app.py
```

#### CustomTkinter 버전
```bash
python run.py
```

#### 기능 데모 실행
```bash
python demo.py
```

## 🎯 주요 기능

### 시스템 모니터링
- CPU, GPU, 메모리, 디스크 사용률 실시간 추적
- 시각화된 차트와 히스토리 데이터
- 시스템 알림 및 경고 메시지
- 모니터링 데이터 내보내기

### 모델 관리
- 여러 모델 동시 로드 및 관리
- 모델별 메모리 사용량 추적
- 비동기 로딩으로 UI 반응성 향상
- 모델 상태 실시간 업데이트

### 모델 분석
- 모든 모델 파일 포괄적 분석
- 지원 태스크 자동 추론
- 모델 크기 및 파라미터 수 계산
- 최적화 권장사항 제공

### FastAPI 서버
- 로드된 모델 즉시 API 서버화
- RESTful API 엔드포인트 자동 생성
- Swagger UI 문서 자동 생성
- 모델별 추론 파이프라인 구성

## 🔧 API 엔드포인트

FastAPI 서버가 실행되면 다음 엔드포인트들을 사용할 수 있습니다:

- `GET /` - 서버 정보
- `GET /models` - 로드된 모델 목록
- `POST /models/load` - 모델 로드
- `POST /models/{model_name}/predict` - 모델 예측
- `POST /models/{model_name}/unload` - 모델 언로드
- `GET /system/status` - 시스템 상태
- `GET /docs` - API 문서 (Swagger UI)

## 📊 지원하는 모델 태스크

- Text Classification (텍스트 분류)
- Token Classification (토큰 분류)
- Question Answering (질문 답변)
- Text Generation (텍스트 생성)
- Fill Mask (빈칸 채우기)
- Text2Text Generation (텍스트 변환)

## ⚠️ 주의사항

- GPU 메모리 사용량을 모니터링하여 OOM 에러 방지
- 대용량 모델 로드 시 충분한 시스템 메모리 확보
- FastAPI 서버는 개발/테스트 용도로 설계됨
- 프로덕션 환경에서는 별도의 배포 설정 필요

## 🐛 알려진 문제점

- FastAPI 서버 중지 기능 제한 (uvicorn 특성)
- 매우 큰 모델(>10GB)의 경우 로딩 시간 지연
- GPU 메모리 부족 시 자동 CPU 폴백 미구현

