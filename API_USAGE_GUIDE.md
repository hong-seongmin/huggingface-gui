# HuggingFace FastAPI 서버 종합 사용 가이드

## 목차
1. [개요](#개요)
2. [시작하기](#시작하기)
3. [API 엔드포인트](#api-엔드포인트)
4. [지원 모델 타입](#지원-모델-타입)
5. [모델별 API 예제](#모델별-api-예제)
6. [에러 핸들링](#에러-핸들링)
7. [성능 팁](#성능-팁)

## 개요

이 FastAPI 서버는 HuggingFace 트랜스포머 모델들을 REST API로 제공하는 서비스입니다. 다양한 NLP 태스크(텍스트 분류, 토큰 분류, 텍스트 생성, 질문 답변 등)를 지원하며, 멀티포트 서버 실행과 동적 모델 로딩을 지원합니다.

### 주요 기능
- **멀티 모델 지원**: 다양한 HuggingFace 모델을 동시에 로드 및 서빙
- **멀티포트 서버**: 모델별로 다른 포트에서 독립적으로 실행 가능
- **동적 모델 로딩**: 런타임에 모델을 로드/언로드
- **자동 모델 타입 감지**: 모델의 아키텍처를 자동으로 분석하여 적절한 태스크 결정
- **메모리 관리**: 시스템 메모리 사용량 모니터링 및 최적화

## 시작하기

### 서버 실행
```bash
# 기본 서버 실행 (포트 8000)
uvicorn fastapi_server:app --host 0.0.0.0 --port 8000

# 또는 Python 스크립트로 실행
python fastapi_server.py
```

### 서버 상태 확인
```bash
curl http://localhost:8000/health
```

## API 엔드포인트

### 기본 엔드포인트

#### 1. 루트 엔드포인트
```http
GET /
```
서버 기본 정보를 반환합니다.

**응답 예시:**
```json
{
  "message": "HuggingFace Model API",
  "version": "1.0.0",
  "docs": "/docs",
  "models_loaded": 3
}
```

#### 2. 상태 확인
```http
GET /health
```
서버 상태와 로드된 모델 수를 확인합니다.

**응답 예시:**
```json
{
  "status": "healthy",
  "timestamp": "2024-07-17T10:30:00.123456",
  "models_loaded": 3
}
```

### 모델 관리 엔드포인트

#### 3. 모델 목록 조회
```http
GET /models
```
현재 로드된 모든 모델의 목록을 반환합니다.

**응답 예시:**
```json
{
  "loaded_models": ["sentiment-model", "ner-model", "embedding-model"],
  "all_models": ["sentiment-model", "ner-model", "embedding-model"],
  "models_status": {
    "sentiment-model": "loaded",
    "ner-model": "loaded",
    "embedding-model": "loading"
  },
  "target_models": null
}
```

#### 4. 특정 모델 정보 조회
```http
GET /models/{model_name}
```
특정 모델의 상세 정보를 반환합니다.

**응답 예시:**
```json
{
  "name": "sentiment-model",
  "path": "/path/to/model",
  "status": "loaded",
  "memory_usage": 1024.5,
  "load_time": "2024-07-17T10:25:00.123456",
  "config_analysis": {
    "model_type": "distilbert",
    "task_type": "text-classification",
    "num_labels": 2
  },
  "available_tasks": ["text-classification"]
}
```

#### 5. 모델 로드
```http
POST /models/load
```
새로운 모델을 로드합니다.

**요청 형식:**
```json
{
  "model_name": "my-sentiment-model",
  "model_path": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

**응답 예시:**
```json
{
  "message": "Model my-sentiment-model loading started",
  "model_name": "my-sentiment-model",
  "model_path": "cardiffnlp/twitter-roberta-base-sentiment-latest"
}
```

#### 6. 모델 언로드
```http
POST /models/{model_name}/unload
```
특정 모델을 언로드합니다.

**응답 예시:**
```json
{
  "message": "Model my-sentiment-model unloaded successfully"
}
```

### 추론 엔드포인트

#### 7. 모델 예측
```http
POST /models/{model_name}/predict
```
특정 모델을 사용하여 텍스트를 예측합니다.

**요청 형식:**
```json
{
  "text": "이 영화는 정말 재미있었다!",
  "max_length": 512,
  "temperature": 1.0,
  "top_p": 1.0,
  "top_k": 50,
  "do_sample": false
}
```

**응답 예시:**
```json
{
  "model_name": "sentiment-model",
  "task": "text-classification",
  "input": "이 영화는 정말 재미있었다!",
  "result": [
    {
      "label": "POSITIVE",
      "score": 0.9542
    }
  ],
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

#### 8. 모델 지원 태스크 조회
```http
GET /models/{model_name}/tasks
```
특정 모델이 지원하는 태스크 목록을 반환합니다.

**응답 예시:**
```json
{
  "model_name": "sentiment-model",
  "supported_tasks": ["text-classification"]
}
```

### 시스템 모니터링 엔드포인트

#### 9. 시스템 상태 조회
```http
GET /system/status
```
전체 시스템 상태를 반환합니다.

#### 10. 메모리 정보 조회
```http
GET /system/memory
```
시스템 메모리 사용량을 반환합니다.

**응답 예시:**
```json
{
  "system_memory": {
    "total": 16777216000,
    "available": 8388608000,
    "used": 8388608000,
    "percent": 50.0
  },
  "gpu_memory": [
    {
      "device": 0,
      "name": "NVIDIA RTX 4090",
      "total": 25769803776,
      "allocated": 2147483648,
      "reserved": 4294967296
    }
  ]
}
```

#### 11. 시스템 정리
```http
POST /system/cleanup
```
GPU 캐시와 파이프라인 캐시를 정리합니다.

**응답 예시:**
```json
{
  "message": "System cleanup completed",
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

## 지원 모델 타입

### 텍스트 분류 (Text Classification)
- **지원 아키텍처**: BERT, DistilBERT, RoBERTa, DeBERTa, ELECTRA, XLM-RoBERTa
- **주요 태스크**: 감정 분석, 스팸 분류, 주제 분류, 독성 탐지
- **예시 모델**: `cardiffnlp/twitter-roberta-base-sentiment-latest`

### 토큰 분류 (Token Classification)
- **지원 아키텍처**: BERT, DistilBERT, RoBERTa, DeBERTa, ELECTRA
- **주요 태스크**: 개체명 인식(NER), 품사 태깅(POS), 청크 분석
- **예시 모델**: `skimb22/koelectra-ner-klue-test1`

### 텍스트 생성 (Text Generation)
- **지원 아키텍처**: GPT-2, GPT-Neo, GPT-J, LLaMA, Mistral, Qwen
- **주요 태스크**: 텍스트 생성, 대화, 창작
- **예시 모델**: `microsoft/DialoGPT-medium`

### 질문 답변 (Question Answering)
- **지원 아키텍처**: BERT, DistilBERT, RoBERTa, DeBERTa
- **주요 태스크**: 독해 기반 질문 답변
- **예시 모델**: `deepset/roberta-base-squad2`

### 특징 추출 (Feature Extraction)
- **지원 아키텍처**: BERT, RoBERTa, sentence-transformers
- **주요 태스크**: 문장 임베딩, 의미 검색, 유사도 계산
- **예시 모델**: `BAAI/bge-m3`

### 시퀀스 투 시퀀스 (Seq2Seq)
- **지원 아키텍처**: T5, BART, Pegasus, Marian
- **주요 태스크**: 번역, 요약, 텍스트 변환
- **예시 모델**: `google/flan-t5-base`

## 모델별 API 예제

### 1. 텍스트 분류 (감정 분석)

#### 모델 로드
```bash
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "sentiment-analyzer",
    "model_path": "cardiffnlp/twitter-roberta-base-sentiment-latest"
  }'
```

#### 예측 요청
```bash
curl -X POST "http://localhost:8000/models/sentiment-analyzer/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "이 영화는 정말 재미있었다! 강력 추천합니다."
  }'
```

#### 응답 예시
```json
{
  "model_name": "sentiment-analyzer",
  "task": "text-classification",
  "input": "이 영화는 정말 재미있었다! 강력 추천합니다.",
  "result": [
    {
      "label": "POSITIVE",
      "score": 0.9542
    }
  ],
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

### 2. 토큰 분류 (개체명 인식)

#### 모델 로드
```bash
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "ner-model",
    "model_path": "skimb22/koelectra-ner-klue-test1"
  }'
```

#### 예측 요청
```bash
curl -X POST "http://localhost:8000/models/ner-model/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "김철수는 서울대학교에서 컴퓨터과학을 전공했다."
  }'
```

#### 응답 예시
```json
{
  "model_name": "ner-model",
  "task": "token-classification",
  "input": "김철수는 서울대학교에서 컴퓨터과학을 전공했다.",
  "result": [
    {
      "label": "B-PER",
      "score": 0.9995
    },
    {
      "label": "O",
      "score": 0.9999
    },
    {
      "label": "B-ORG",
      "score": 0.9987
    }
  ],
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

### 3. 특징 추출 (임베딩)

#### 모델 로드
```bash
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "embedding-model",
    "model_path": "BAAI/bge-m3"
  }'
```

#### 예측 요청
```bash
curl -X POST "http://localhost:8000/models/embedding-model/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "안녕하세요, 반갑습니다!"
  }'
```

#### 응답 예시
```json
{
  "model_name": "embedding-model",
  "task": "feature-extraction",
  "input": "안녕하세요, 반갑습니다!",
  "result": [
    [-0.1234, 0.5678, -0.9012, 0.3456, ...]
  ],
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

### 4. 텍스트 생성

#### 모델 로드
```bash
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "text-generator",
    "model_path": "microsoft/DialoGPT-medium"
  }'
```

#### 예측 요청
```bash
curl -X POST "http://localhost:8000/models/text-generator/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "오늘 날씨가 정말 좋네요.",
    "max_length": 100,
    "temperature": 0.8,
    "do_sample": true
  }'
```

#### 응답 예시
```json
{
  "model_name": "text-generator",
  "task": "text-generation",
  "input": "오늘 날씨가 정말 좋네요.",
  "result": [
    {
      "generated_text": "네, 정말 산책하기 좋은 날씨입니다!"
    }
  ],
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

### 5. 멀티태스크 모델

#### 모델 로드
```bash
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "multitask-model",
    "model_path": "upskyy/kf-deberta-multitask"
  }'
```

#### 예측 요청
```bash
curl -X POST "http://localhost:8000/models/multitask-model/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "이 제품은 품질이 우수하고 가격도 합리적입니다."
  }'
```

#### 응답 예시
```json
{
  "model_name": "multitask-model",
  "task": "text-classification",
  "input": "이 제품은 품질이 우수하고 가격도 합리적입니다.",
  "result": [
    {
      "label": "POSITIVE",
      "score": 0.9234
    }
  ],
  "timestamp": "2024-07-17T10:30:00.123456"
}
```

## 에러 핸들링

### 일반적인 에러 코드

#### 404 - 모델을 찾을 수 없음
```json
{
  "detail": "Model not found or not loaded"
}
```

#### 500 - 예측 실패
```json
{
  "detail": "Prediction failed: CUDA out of memory"
}
```

#### 400 - 잘못된 요청
```json
{
  "detail": "No supported tasks found for this model"
}
```

### 에러 상황별 대응

#### 1. 모델 로드 실패
```bash
# 모델 상태 확인
curl http://localhost:8000/models/your-model

# 시스템 메모리 확인
curl http://localhost:8000/system/memory

# 시스템 정리
curl -X POST http://localhost:8000/system/cleanup
```

#### 2. 메모리 부족
```bash
# 사용하지 않는 모델 언로드
curl -X POST http://localhost:8000/models/unused-model/unload

# GPU 캐시 정리
curl -X POST http://localhost:8000/system/cleanup
```

#### 3. 예측 실패
```bash
# 모델 지원 태스크 확인
curl http://localhost:8000/models/your-model/tasks

# 모델 재로드
curl -X POST http://localhost:8000/models/your-model/unload
curl -X POST http://localhost:8000/models/load -d '{"model_name": "your-model", "model_path": "path/to/model"}'
```

## 성능 팁

### 1. 메모리 최적화

#### 모델 로드 순서 최적화
```python
# 큰 모델부터 로드하여 메모리 단편화 방지
models_to_load = [
    ("large-model", "path/to/large/model"),
    ("medium-model", "path/to/medium/model"),
    ("small-model", "path/to/small/model")
]
```

#### 정기적인 메모리 정리
```bash
# 정기적으로 시스템 정리 실행
curl -X POST http://localhost:8000/system/cleanup
```

### 2. 멀티포트 서버 활용

#### 모델별 포트 분리
```python
# 각 모델을 다른 포트에서 실행
model_ports = {
    "sentiment-model": 8001,
    "ner-model": 8002,
    "embedding-model": 8003
}
```

### 3. 배치 처리

#### 여러 텍스트 동시 처리
```python
# 여러 요청을 동시에 처리하여 처리량 증가
import asyncio
import aiohttp

async def batch_predict(texts, model_name):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for text in texts:
            task = session.post(
                f"http://localhost:8000/models/{model_name}/predict",
                json={"text": text}
            )
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
```

### 4. 모델 캐싱

#### 자주 사용하는 모델 사전 로드
```python
# 자주 사용하는 모델들을 시작할 때 미리 로드
essential_models = [
    ("sentiment", "cardiffnlp/twitter-roberta-base-sentiment-latest"),
    ("ner", "skimb22/koelectra-ner-klue-test1"),
    ("embedding", "BAAI/bge-m3")
]
```

### 5. 하드웨어 최적화

#### GPU 메모리 관리
```python
# GPU 메모리 사용량 모니터링
import torch

def monitor_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.memory_allocated(i)} / {torch.cuda.max_memory_allocated(i)}")
```

#### CPU 최적화
```python
# 멀티스레딩 활용
import torch

torch.set_num_threads(4)  # CPU 코어 수에 맞게 조정
```

### 6. 모니터링 및 로깅

#### 성능 모니터링
```bash
# 정기적인 시스템 상태 확인
watch -n 30 "curl -s http://localhost:8000/system/status"

# 메모리 사용량 모니터링
watch -n 10 "curl -s http://localhost:8000/system/memory"
```

#### 로그 분석
```python
import logging

# 상세한 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

## 추가 참고사항

### 지원 모델 확장
새로운 모델 타입을 추가하려면 `model_type_detector.py`의 `architecture_to_task` 딕셔너리를 업데이트하세요.

### 커스텀 토크나이저
특별한 토크나이저가 필요한 경우 `model_manager.py`를 수정하여 지원할 수 있습니다.

### 보안 고려사항
- 프로덕션 환경에서는 적절한 인증과 권한 부여를 구현하세요
- 입력 텍스트의 길이와 내용을 검증하세요
- 모델 로드 경로에 대한 접근 권한을 제한하세요

이 가이드는 HuggingFace FastAPI 서버의 모든 주요 기능을 다룹니다. 추가 질문이나 특정 사용 사례에 대한 도움이 필요하시면 언제든지 문의해 주세요.