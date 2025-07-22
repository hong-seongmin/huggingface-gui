# HuggingFace FastAPI 서버 문제 해결 가이드

## 목차
1. [일반적인 문제들](#일반적인-문제들)
2. [모델 로딩 관련 문제](#모델-로딩-관련-문제)
3. [메모리 관련 문제](#메모리-관련-문제)
4. [성능 관련 문제](#성능-관련-문제)
5. [네트워크 및 연결 문제](#네트워크-및-연결-문제)
6. [디버깅 도구](#디버깅-도구)
7. [로그 분석](#로그-분석)

## 일반적인 문제들

### 1. 서버 시작 오류

#### 문제: 포트가 이미 사용 중
```bash
Error: [Errno 98] Address already in use
```

**해결 방법:**
```bash
# 포트 사용 중인 프로세스 확인
lsof -i :8000

# 프로세스 종료
kill -9 <PID>

# 또는 다른 포트 사용
uvicorn fastapi_server:app --port 8001
```

#### 문제: 모듈을 찾을 수 없음
```bash
ModuleNotFoundError: No module named 'transformers'
```

**해결 방법:**
```bash
# 필요한 패키지 설치
pip install transformers torch fastapi uvicorn

# 가상환경 확인
which python
pip list | grep transformers
```

### 2. 모델 감지 실패

#### 문제: 모델 타입을 감지할 수 없음
```json
{
  "detail": "No supported tasks found for this model"
}
```

**해결 방법:**
```bash
# 모델 config.json 확인
curl http://localhost:8000/models/your-model

# 모델 분석 요청
curl -X POST "http://localhost:8000/models/your-model/analyze" \
  -H "Content-Type: application/json" \
  -d '{"model_path": "/path/to/model"}'
```

**Python 스크립트로 모델 정보 확인:**
```python
from transformers import AutoConfig

# 모델 설정 확인
config = AutoConfig.from_pretrained("model_name_or_path")
print(f"Model type: {config.model_type}")
print(f"Architectures: {config.architectures}")
print(f"Task specific params: {getattr(config, 'task_specific_params', {})}")
```

## 모델 로딩 관련 문제

### 1. 모델 로딩 실패

#### 문제: HuggingFace 모델 다운로드 실패
```bash
OSError: Model not found
```

**해결 방법:**
```bash
# 모델 ID 확인
curl -s "https://huggingface.co/api/models/model-name"

# 로컬 캐시 확인
ls -la ~/.cache/huggingface/hub/

# 수동 다운로드
from transformers import AutoModel
model = AutoModel.from_pretrained("model-name")
```

#### 문제: 로컬 모델 경로 오류
```bash
OSError: [Errno 2] No such file or directory
```

**해결 방법:**
```bash
# 경로 확인
ls -la /path/to/model/
find /path/to/model/ -name "*.json"

# 절대 경로 사용
{
  "model_name": "my-model",
  "model_path": "/home/user/models/my-model"
}
```

### 2. 모델 호환성 문제

#### 문제: 토크나이저 호환성 오류
```bash
ValueError: Can't load tokenizer for 'model-name'
```

**해결 방법:**
```python
# 토크나이저 수동 지정
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model-name", trust_remote_code=True)

# 또는 호환되는 토크나이저 사용
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
```

#### 문제: 모델 아키텍처 불일치
```bash
RuntimeError: Error(s) in loading state_dict
```

**해결 방법:**
```python
# 모델 구조 확인
import torch
checkpoint = torch.load("pytorch_model.bin", map_location='cpu')
print(checkpoint.keys())

# 부분 로딩 허용
model.load_state_dict(checkpoint, strict=False)
```

## 메모리 관련 문제

### 1. GPU 메모리 부족

#### 문제: CUDA out of memory
```bash
RuntimeError: CUDA out of memory. Tried to allocate 1.95 GiB
```

**해결 방법:**
```bash
# GPU 메모리 사용량 확인
nvidia-smi

# 시스템 정리
curl -X POST http://localhost:8000/system/cleanup
```

**Python 스크립트로 메모리 정리:**
```python
import torch
import gc

# GPU 캐시 정리
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 메모리 가비지 컬렉션
gc.collect()

# 메모리 사용량 확인
print(f"GPU 메모리 사용량: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### 2. 시스템 메모리 부족

#### 문제: 시스템 메모리 부족으로 모델 로딩 실패
```bash
MemoryError: Unable to allocate memory
```

**해결 방법:**
```bash
# 메모리 사용량 확인
curl http://localhost:8000/system/memory

# 사용하지 않는 모델 언로드
curl -X POST http://localhost:8000/models/unused-model/unload
```

**메모리 최적화 설정:**
```python
# 모델 로딩 시 메모리 최적화
import torch

# CPU 메모리 사용 제한
torch.set_num_threads(2)

# 혼합 정밀도 사용
model = model.half()  # FP16 사용
```

### 3. 메모리 누수

#### 문제: 시간이 지남에 따라 메모리 사용량 증가
```python
# 메모리 모니터링 스크립트
import psutil
import time

def monitor_memory():
    while True:
        memory = psutil.virtual_memory()
        print(f"메모리 사용률: {memory.percent}%")
        time.sleep(60)

monitor_memory()
```

**해결 방법:**
```python
# 정기적인 가비지 컬렉션
import gc
import schedule

def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 매 30분마다 정리
schedule.every(30).minutes.do(cleanup_memory)
```

## 성능 관련 문제

### 1. 응답 시간 지연

#### 문제: 모델 예측 응답이 느림
```bash
# 응답 시간 측정
curl -w "@curl-format.txt" -o /dev/null -s \
  "http://localhost:8000/models/model-name/predict" \
  -d '{"text": "test"}'
```

**해결 방법:**
```python
# 모델 최적화
import torch

# 컴파일 최적화 (PyTorch 2.0+)
model = torch.compile(model)

# 추론 모드 설정
model.eval()
with torch.inference_mode():
    output = model(**inputs)
```

### 2. 동시 요청 처리 성능

#### 문제: 동시 요청 시 성능 저하
```python
# 동시 요청 테스트
import concurrent.futures
import requests

def test_concurrent_requests(num_requests=10):
    def single_request():
        return requests.post("http://localhost:8000/models/model-name/predict",
                           json={"text": "test"})
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(single_request) for _ in range(num_requests)]
        results = [future.result() for future in futures]
    
    return results
```

**해결 방법:**
```python
# 비동기 처리 설정
import asyncio
from fastapi import FastAPI

app = FastAPI()

@app.post("/models/{model_name}/predict")
async def predict(model_name: str, request: PredictionRequest):
    # 비동기 처리 구현
    result = await async_model_inference(model_name, request.text)
    return result
```

### 3. 배치 처리 최적화

#### 문제: 개별 요청 처리로 인한 비효율성
```python
# 배치 처리 구현
def batch_predict(model, tokenizer, texts, batch_size=8):
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        # 배치 토큰화
        inputs = tokenizer(batch_texts, return_tensors="pt", 
                          padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 결과 처리
        batch_results = process_batch_output(outputs, len(batch_texts))
        results.extend(batch_results)
    
    return results
```

## 네트워크 및 연결 문제

### 1. 연결 시간 초과

#### 문제: Request timeout
```bash
curl: (28) Operation timed out after 30000 milliseconds
```

**해결 방법:**
```python
# 클라이언트 타임아웃 설정
import requests

response = requests.post(
    "http://localhost:8000/models/model-name/predict",
    json={"text": "test"},
    timeout=60  # 60초 타임아웃
)
```

### 2. 멀티포트 서버 연결 문제

#### 문제: 특정 포트에서 모델에 접근할 수 없음
```bash
# 포트별 서버 상태 확인
for port in 8000 8001 8002; do
    echo "포트 $port:"
    curl -s "http://localhost:$port/models" | jq '.loaded_models'
done
```

**해결 방법:**
```python
# 포트 매핑 확인
def check_model_ports():
    ports = [8000, 8001, 8002, 8003]
    
    for port in ports:
        try:
            response = requests.get(f"http://localhost:{port}/models")
            if response.status_code == 200:
                models = response.json()['loaded_models']
                print(f"포트 {port}: {models}")
        except requests.exceptions.ConnectionError:
            print(f"포트 {port}: 연결 실패")
```

### 3. 로드 밸런싱 문제

#### 문제: 특정 서버에 요청 집중
```python
# 라운드로빈 로드 밸런싱
class RoundRobinBalancer:
    def __init__(self, servers):
        self.servers = servers
        self.current = 0
    
    def get_next_server(self):
        server = self.servers[self.current]
        self.current = (self.current + 1) % len(self.servers)
        return server

# 사용 예제
balancer = RoundRobinBalancer([
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8003"
])

server_url = balancer.get_next_server()
```

## 디버깅 도구

### 1. 로그 레벨 설정

```python
import logging

# 상세한 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('debug.log'),
        logging.StreamHandler()
    ]
)

# 모듈별 로그 레벨
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
```

### 2. 성능 프로파일링

```python
import cProfile
import pstats
from functools import wraps

def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        
        result = func(*args, **kwargs)
        
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # 상위 20개 함수
        
        return result
    return wrapper

# 사용 예제
@profile_function
def slow_function():
    # 프로파일링할 함수
    pass
```

### 3. 메모리 프로파일링

```python
import tracemalloc
import gc

def memory_profile():
    tracemalloc.start()
    
    # 메모리 사용량 측정할 코드
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
    
    tracemalloc.stop()
```

### 4. 모델 분석 도구

```python
def analyze_model_structure(model_path):
    from transformers import AutoConfig, AutoModel
    
    try:
        config = AutoConfig.from_pretrained(model_path)
        print(f"Model type: {config.model_type}")
        print(f"Architectures: {config.architectures}")
        print(f"Hidden size: {config.hidden_size}")
        print(f"Number of layers: {config.num_hidden_layers}")
        print(f"Number of attention heads: {config.num_attention_heads}")
        
        # 모델 크기 추정
        model = AutoModel.from_pretrained(model_path)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        print(f"Estimated size: {total_params * 4 / 1024**2:.2f} MB")
        
    except Exception as e:
        print(f"분석 실패: {e}")
```

## 로그 분석

### 1. 일반적인 로그 패턴

```bash
# 에러 로그 검색
grep -i "error\|exception\|failed" server.log

# 성능 관련 로그
grep -i "timeout\|slow\|memory" server.log

# 모델 로딩 로그
grep -i "load\|model" server.log
```

### 2. 구조화된 로그 분석

```python
import json
import re
from datetime import datetime

def analyze_logs(log_file):
    error_count = 0
    warning_count = 0
    response_times = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # 에러 카운트
            if 'ERROR' in line:
                error_count += 1
            elif 'WARNING' in line:
                warning_count += 1
            
            # 응답 시간 추출
            time_match = re.search(r'response_time: (\d+\.?\d*)ms', line)
            if time_match:
                response_times.append(float(time_match.group(1)))
    
    print(f"에러 수: {error_count}")
    print(f"경고 수: {warning_count}")
    if response_times:
        print(f"평균 응답 시간: {sum(response_times)/len(response_times):.2f}ms")
```

### 3. 실시간 로그 모니터링

```python
import subprocess
import re

def monitor_logs():
    # tail -f 명령어로 실시간 로그 모니터링
    process = subprocess.Popen(['tail', '-f', 'server.log'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
    
    try:
        for line in iter(process.stdout.readline, b''):
            line = line.decode('utf-8').strip()
            
            # 에러 패턴 감지
            if 'ERROR' in line:
                print(f"🚨 에러 감지: {line}")
            elif 'CUDA out of memory' in line:
                print(f"⚠️ GPU 메모리 부족: {line}")
            elif 'timeout' in line.lower():
                print(f"⏰ 타임아웃 발생: {line}")
    
    except KeyboardInterrupt:
        process.terminate()
```

## 자주 발생하는 문제와 해결책

### 1. 모델 타입 감지 오류

**문제:** 새로운 모델 아키텍처를 인식하지 못함

**해결책:**
```python
# model_type_detector.py에 새로운 아키텍처 추가
self.architecture_to_task.update({
    "NewModelForSequenceClassification": "text-classification",
    "NewModelForTokenClassification": "token-classification"
})
```

### 2. 한국어 텍스트 처리 문제

**문제:** 한국어 텍스트가 제대로 인코딩되지 않음

**해결책:**
```python
# UTF-8 인코딩 확인
import json

def safe_json_encode(data):
    return json.dumps(data, ensure_ascii=False, indent=2)

# 요청 시 인코딩 확인
response = requests.post(
    url,
    json={"text": "한국어 텍스트"},
    headers={"Content-Type": "application/json; charset=utf-8"}
)
```

### 3. 토큰 길이 초과 오류

**문제:** 입력 텍스트가 모델의 최대 토큰 길이를 초과

**해결책:**
```python
def truncate_text(text, tokenizer, max_length=512):
    tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
    return tokenizer.decode(tokens, skip_special_tokens=True)

# 사용 예제
truncated_text = truncate_text(long_text, tokenizer)
```

### 4. 배치 처리 시 메모리 오류

**문제:** 배치 크기가 너무 커서 메모리 부족

**해결책:**
```python
def adaptive_batch_size(texts, model, tokenizer, initial_batch_size=32):
    batch_size = initial_batch_size
    
    while batch_size > 0:
        try:
            # 배치 처리 시도
            batch = texts[:batch_size]
            result = process_batch(batch, model, tokenizer)
            return result, batch_size
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                batch_size //= 2
                print(f"메모리 부족으로 배치 크기를 {batch_size}로 감소")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            else:
                raise e
    
    raise RuntimeError("배치 크기를 줄여도 메모리 부족")
```

이 문제 해결 가이드를 참고하여 HuggingFace FastAPI 서버 운영 시 발생할 수 있는 다양한 문제들을 효과적으로 해결할 수 있습니다.