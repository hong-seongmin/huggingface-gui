# HuggingFace FastAPI 서버 성능 최적화 가이드

## 목차
1. [하드웨어 최적화](#하드웨어-최적화)
2. [모델 최적화](#모델-최적화)
3. [메모리 최적화](#메모리-최적화)
4. [네트워크 최적화](#네트워크-최적화)
5. [배치 처리 최적화](#배치-처리-최적화)
6. [캐싱 전략](#캐싱-전략)
7. [모니터링 및 프로파일링](#모니터링-및-프로파일링)
8. [배포 최적화](#배포-최적화)

## 하드웨어 최적화

### 1. GPU 최적화

#### CUDA 설정 최적화
```python
import torch
import os

# GPU 설정
if torch.cuda.is_available():
    # 최적의 GPU 선택
    torch.cuda.set_device(0)
    
    # 메모리 할당 최적화
    torch.cuda.empty_cache()
    
    # 다중 GPU 사용 (가능한 경우)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # CUDA 커널 캐싱 활성화
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # 메모리 할당 전략 설정
    torch.cuda.set_per_process_memory_fraction(0.8)
```

#### 메모리 사용량 모니터링
```python
def monitor_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            
            print(f"GPU {i}: {allocated:.2f}GB / {reserved:.2f}GB / {total:.2f}GB")
```

### 2. CPU 최적화

#### 멀티스레딩 설정
```python
import torch
import os

# CPU 코어 활용 최적화
num_cores = os.cpu_count()
torch.set_num_threads(num_cores // 2)  # 하이퍼스레딩 고려

# OMP 설정
os.environ['OMP_NUM_THREADS'] = str(num_cores // 2)
os.environ['MKL_NUM_THREADS'] = str(num_cores // 2)

# 인텔 MKL 최적화
if 'mkl' in torch.__config__.show():
    torch.set_num_interop_threads(2)
```

## 모델 최적화

### 1. 모델 양자화 (Quantization)

#### 동적 양자화
```python
import torch
from torch.quantization import quantize_dynamic

def quantize_model(model):
    """모델을 INT8로 양자화"""
    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

# 사용 예제
optimized_model = quantize_model(model)
```

#### 정적 양자화
```python
import torch
from torch.quantization import prepare, convert

def static_quantize_model(model, calibration_data):
    """정적 양자화 수행"""
    model.eval()
    
    # 양자화 준비
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    
    # 캘리브레이션
    with torch.no_grad():
        for data in calibration_data:
            model(data)
    
    # 양자화 적용
    torch.quantization.convert(model, inplace=True)
    return model
```

### 2. 모델 압축

#### 지식 증류 (Knowledge Distillation)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationLoss(nn.Module):
    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
    
    def forward(self, student_logits, teacher_logits, labels):
        # 소프트 타겟 손실
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=1),
            F.softmax(teacher_logits / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 하드 타겟 손실
        hard_loss = F.cross_entropy(student_logits, labels)
        
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
```

### 3. 모델 컴파일 최적화

#### PyTorch 2.0+ 컴파일
```python
import torch

# 모델 컴파일 (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    compiled_model = torch.compile(
        model,
        mode="max-autotune",  # 최대 성능 모드
        dynamic=True  # 동적 형태 지원
    )
else:
    compiled_model = model
```

#### TorchScript 최적화
```python
import torch

def optimize_with_torchscript(model, sample_input):
    """TorchScript로 모델 최적화"""
    model.eval()
    
    # 모델 트레이싱
    traced_model = torch.jit.trace(model, sample_input)
    
    # 최적화 적용
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    return optimized_model
```

## 메모리 최적화

### 1. 메모리 풀링

#### 커스텀 메모리 풀
```python
import torch
from typing import Dict, Any

class MemoryPool:
    def __init__(self, max_size_gb: float = 4.0):
        self.max_size = max_size_gb * 1024**3
        self.allocated_memory = 0
        self.tensor_cache = {}
    
    def get_tensor(self, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
        """메모리 풀에서 텐서 획득"""
        key = (shape, dtype)
        
        if key in self.tensor_cache:
            return self.tensor_cache[key]
        
        # 메모리 한계 확인
        tensor_size = torch.tensor(shape).prod().item() * dtype.itemsize
        if self.allocated_memory + tensor_size > self.max_size:
            self.cleanup_cache()
        
        # 새 텐서 생성
        tensor = torch.empty(shape, dtype=dtype)
        self.tensor_cache[key] = tensor
        self.allocated_memory += tensor_size
        
        return tensor
    
    def cleanup_cache(self):
        """캐시 정리"""
        self.tensor_cache.clear()
        self.allocated_memory = 0
        torch.cuda.empty_cache()

# 전역 메모리 풀
memory_pool = MemoryPool()
```

### 2. 그래디언트 체크포인팅

```python
import torch
from torch.utils.checkpoint import checkpoint

def memory_efficient_forward(model, inputs):
    """메모리 효율적인 순전파"""
    def custom_forward(*inputs):
        return model(*inputs)
    
    # 체크포인팅 사용
    return checkpoint(custom_forward, *inputs)
```

### 3. 메모리 모니터링

```python
import psutil
import torch
import gc
from typing import Dict, Any

class MemoryMonitor:
    def __init__(self):
        self.initial_memory = self.get_memory_usage()
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """현재 메모리 사용량 반환"""
        memory = psutil.virtual_memory()
        gpu_memory = {}
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory[f'gpu_{i}'] = {
                    'allocated': torch.cuda.memory_allocated(i),
                    'reserved': torch.cuda.memory_reserved(i),
                    'total': torch.cuda.get_device_properties(i).total_memory
                }
        
        return {
            'system': {
                'total': memory.total,
                'used': memory.used,
                'percent': memory.percent
            },
            'gpu': gpu_memory
        }
    
    def log_memory_usage(self, label: str = ""):
        """메모리 사용량 로깅"""
        current = self.get_memory_usage()
        print(f"[{label}] 시스템 메모리: {current['system']['percent']:.1f}%")
        
        for gpu_id, gpu_info in current['gpu'].items():
            allocated_gb = gpu_info['allocated'] / 1024**3
            total_gb = gpu_info['total'] / 1024**3
            print(f"[{label}] {gpu_id}: {allocated_gb:.2f}GB / {total_gb:.2f}GB")
    
    def force_cleanup(self):
        """강제 메모리 정리"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

# 사용 예제
monitor = MemoryMonitor()
monitor.log_memory_usage("시작")

# 모델 로딩 후
monitor.log_memory_usage("모델 로딩 후")
```

## 네트워크 최적화

### 1. 연결 풀링

```python
import aiohttp
import asyncio
from typing import List, Dict, Any

class ConnectionPoolManager:
    def __init__(self, max_connections: int = 100):
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self.session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()
        await self.connector.close()
```

### 2. 요청 압축

```python
import gzip
import json
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware

class CompressionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # 요청 처리
        response = await call_next(request)
        
        # 응답 압축
        if (response.status_code == 200 and 
            'gzip' in request.headers.get('accept-encoding', '')):
            
            # 응답 본문 압축
            if hasattr(response, 'body'):
                compressed_body = gzip.compress(response.body)
                
                # 압축된 응답 반환
                return Response(
                    content=compressed_body,
                    status_code=response.status_code,
                    headers={
                        **response.headers,
                        'content-encoding': 'gzip',
                        'content-length': str(len(compressed_body))
                    }
                )
        
        return response
```

### 3. 비동기 처리

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Any

class AsyncModelManager:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def predict_async(self, model, tokenizer, text: str) -> Any:
        """비동기 예측"""
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.executor, self._predict_sync, model, tokenizer, text
            )
    
    def _predict_sync(self, model, tokenizer, text: str) -> Any:
        """동기 예측 (스레드에서 실행)"""
        inputs = tokenizer(text, return_tensors="pt", truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return self._process_outputs(outputs)
    
    async def batch_predict_async(self, model, tokenizer, texts: List[str]) -> List[Any]:
        """배치 비동기 예측"""
        tasks = [self.predict_async(model, tokenizer, text) for text in texts]
        return await asyncio.gather(*tasks)
```

## 배치 처리 최적화

### 1. 동적 배치 크기

```python
import torch
from typing import List, Tuple, Any

class AdaptiveBatchProcessor:
    def __init__(self, initial_batch_size: int = 32, min_batch_size: int = 1):
        self.batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        self.success_count = 0
        self.failure_count = 0
    
    def process_batch(self, model, tokenizer, texts: List[str]) -> List[Any]:
        """적응형 배치 처리"""
        results = []
        i = 0
        
        while i < len(texts):
            batch_texts = texts[i:i + self.batch_size]
            
            try:
                # 배치 처리 시도
                batch_results = self._process_single_batch(model, tokenizer, batch_texts)
                results.extend(batch_results)
                
                i += self.batch_size
                self.success_count += 1
                
                # 성공 시 배치 크기 증가 고려
                if self.success_count >= 10:
                    self._increase_batch_size()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    self._decrease_batch_size()
                    torch.cuda.empty_cache()
                    
                    # 배치 크기가 최소값에 도달하면 개별 처리
                    if self.batch_size < self.min_batch_size:
                        result = self._process_single_item(model, tokenizer, texts[i])
                        results.append(result)
                        i += 1
                else:
                    raise e
        
        return results
    
    def _process_single_batch(self, model, tokenizer, texts: List[str]) -> List[Any]:
        """단일 배치 처리"""
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        return self._process_batch_outputs(outputs, len(texts))
    
    def _increase_batch_size(self):
        """배치 크기 증가"""
        self.batch_size = min(self.batch_size * 2, 64)
        self.success_count = 0
    
    def _decrease_batch_size(self):
        """배치 크기 감소"""
        self.batch_size = max(self.batch_size // 2, self.min_batch_size)
        self.failure_count += 1
```

### 2. 배치 큐 시스템

```python
import asyncio
from collections import deque
from typing import List, Dict, Any, Optional
import time

class BatchQueue:
    def __init__(self, max_batch_size: int = 16, max_wait_time: float = 0.1):
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.queue = deque()
        self.processing = False
        self.results = {}
    
    async def add_request(self, request_id: str, text: str) -> Any:
        """요청을 큐에 추가하고 결과 대기"""
        future = asyncio.Future()
        
        self.queue.append({
            'id': request_id,
            'text': text,
            'future': future,
            'timestamp': time.time()
        })
        
        # 배치 처리 시작
        if not self.processing:
            asyncio.create_task(self._process_queue())
        
        return await future
    
    async def _process_queue(self):
        """큐 처리"""
        self.processing = True
        
        while self.queue:
            batch = []
            current_time = time.time()
            
            # 배치 수집
            while (len(batch) < self.max_batch_size and 
                   self.queue and 
                   (len(batch) == 0 or 
                    current_time - batch[0]['timestamp'] < self.max_wait_time)):
                
                batch.append(self.queue.popleft())
            
            if batch:
                await self._process_batch(batch)
            
            # 큐가 비어있으면 짧은 대기
            if not self.queue:
                await asyncio.sleep(0.01)
        
        self.processing = False
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """배치 처리 실행"""
        texts = [item['text'] for item in batch]
        
        try:
            # 모델 추론 (실제 구현 필요)
            results = await self._model_inference(texts)
            
            # 결과 반환
            for item, result in zip(batch, results):
                item['future'].set_result(result)
                
        except Exception as e:
            # 에러 시 모든 요청에 에러 반환
            for item in batch:
                item['future'].set_exception(e)
```

## 캐싱 전략

### 1. 결과 캐싱

```python
import hashlib
import json
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

class ResultCache:
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = {}
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _generate_key(self, model_name: str, text: str, params: Dict[str, Any] = None) -> str:
        """캐시 키 생성"""
        data = {
            'model': model_name,
            'text': text,
            'params': params or {}
        }
        
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def get(self, model_name: str, text: str, params: Dict[str, Any] = None) -> Optional[Any]:
        """캐시에서 결과 조회"""
        key = self._generate_key(model_name, text, params)
        
        if key in self.cache:
            item = self.cache[key]
            
            # TTL 확인
            if datetime.now() - item['timestamp'] < self.ttl:
                return item['result']
            else:
                del self.cache[key]
        
        return None
    
    def set(self, model_name: str, text: str, result: Any, params: Dict[str, Any] = None):
        """캐시에 결과 저장"""
        key = self._generate_key(model_name, text, params)
        
        # 캐시 크기 제한
        if len(self.cache) >= self.max_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        self.cache[key] = {
            'result': result,
            'timestamp': datetime.now()
        }
    
    def clear(self):
        """캐시 정리"""
        self.cache.clear()
    
    def cleanup_expired(self):
        """만료된 항목 정리"""
        now = datetime.now()
        expired_keys = [
            key for key, item in self.cache.items()
            if now - item['timestamp'] >= self.ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]

# 전역 캐시 인스턴스
result_cache = ResultCache()
```

### 2. 모델 캐싱

```python
import torch
from typing import Dict, Tuple, Any
import pickle
import os

class ModelCache:
    def __init__(self, cache_dir: str = "./model_cache"):
        self.cache_dir = cache_dir
        self.loaded_models = {}
        os.makedirs(cache_dir, exist_ok=True)
    
    def get_model(self, model_name: str, model_path: str) -> Tuple[Any, Any]:
        """모델 캐시에서 로드"""
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        # 디스크 캐시 확인
        cache_path = os.path.join(self.cache_dir, f"{model_name}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    model, tokenizer = pickle.load(f)
                    self.loaded_models[model_name] = (model, tokenizer)
                    return model, tokenizer
            except Exception as e:
                print(f"디스크 캐시 로드 실패: {e}")
        
        # 모델 로드
        model, tokenizer = self._load_model(model_path)
        
        # 캐시에 저장
        self.loaded_models[model_name] = (model, tokenizer)
        self._save_to_disk(model_name, model, tokenizer)
        
        return model, tokenizer
    
    def _load_model(self, model_path: str) -> Tuple[Any, Any]:
        """실제 모델 로드"""
        from transformers import AutoModel, AutoTokenizer
        
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        return model, tokenizer
    
    def _save_to_disk(self, model_name: str, model: Any, tokenizer: Any):
        """디스크에 모델 캐시 저장"""
        cache_path = os.path.join(self.cache_dir, f"{model_name}.pkl")
        
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump((model, tokenizer), f)
        except Exception as e:
            print(f"디스크 캐시 저장 실패: {e}")
    
    def unload_model(self, model_name: str):
        """모델 언로드"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            torch.cuda.empty_cache()
```

## 모니터링 및 프로파일링

### 1. 성능 메트릭 수집

```python
import time
import statistics
from typing import Dict, List, Any
from collections import defaultdict, deque

class PerformanceMetrics:
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.counters = defaultdict(int)
    
    def record_response_time(self, model_name: str, response_time: float):
        """응답 시간 기록"""
        self.metrics[f"{model_name}_response_time"].append(response_time)
    
    def record_memory_usage(self, usage_bytes: int):
        """메모리 사용량 기록"""
        self.metrics["memory_usage"].append(usage_bytes)
    
    def increment_counter(self, metric_name: str):
        """카운터 증가"""
        self.counters[metric_name] += 1
    
    def get_stats(self, model_name: str) -> Dict[str, Any]:
        """통계 정보 반환"""
        response_times = self.metrics[f"{model_name}_response_time"]
        
        if not response_times:
            return {}
        
        return {
            "avg_response_time": statistics.mean(response_times),
            "median_response_time": statistics.median(response_times),
            "p95_response_time": sorted(response_times)[int(len(response_times) * 0.95)],
            "total_requests": len(response_times),
            "requests_per_second": len(response_times) / sum(response_times) if sum(response_times) > 0 else 0
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """시스템 전체 통계"""
        memory_usage = self.metrics["memory_usage"]
        
        stats = {
            "total_requests": sum(self.counters.values()),
            "error_count": self.counters["errors"],
            "success_rate": (self.counters["success"] / max(sum(self.counters.values()), 1)) * 100
        }
        
        if memory_usage:
            stats.update({
                "avg_memory_usage": statistics.mean(memory_usage),
                "max_memory_usage": max(memory_usage)
            })
        
        return stats

# 전역 메트릭 수집기
metrics = PerformanceMetrics()
```

### 2. 실시간 대시보드

```python
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import time

class RealTimeDashboard:
    def __init__(self, metrics: PerformanceMetrics):
        self.metrics = metrics
        self.connected_clients = set()
    
    async def websocket_endpoint(self, websocket: WebSocket):
        """WebSocket 엔드포인트"""
        await websocket.accept()
        self.connected_clients.add(websocket)
        
        try:
            while True:
                # 메트릭 데이터 수집
                stats = self._collect_current_stats()
                
                # 모든 클라이언트에게 브로드캐스트
                await self._broadcast_stats(stats)
                
                await asyncio.sleep(1)  # 1초마다 업데이트
                
        except Exception as e:
            print(f"WebSocket 오류: {e}")
        finally:
            self.connected_clients.discard(websocket)
    
    def _collect_current_stats(self) -> Dict[str, Any]:
        """현재 통계 수집"""
        return {
            "timestamp": time.time(),
            "system_stats": self.metrics.get_system_stats(),
            "memory_usage": self._get_memory_usage(),
            "active_connections": len(self.connected_clients)
        }
    
    async def _broadcast_stats(self, stats: Dict[str, Any]):
        """통계 브로드캐스트"""
        if self.connected_clients:
            message = json.dumps(stats)
            
            # 연결이 끊어진 클라이언트 제거
            disconnected = set()
            
            for client in self.connected_clients:
                try:
                    await client.send_text(message)
                except:
                    disconnected.add(client)
            
            self.connected_clients -= disconnected
    
    def _get_memory_usage(self) -> Dict[str, Any]:
        """메모리 사용량 조회"""
        import psutil
        
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "used": memory.used,
            "percent": memory.percent
        }
    
    def generate_html_dashboard(self) -> str:
        """대시보드 HTML 생성"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HuggingFace API Dashboard</title>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        </head>
        <body>
            <h1>HuggingFace API 실시간 대시보드</h1>
            
            <div id="stats">
                <div>총 요청 수: <span id="total-requests">0</span></div>
                <div>성공률: <span id="success-rate">0%</span></div>
                <div>메모리 사용률: <span id="memory-usage">0%</span></div>
            </div>
            
            <canvas id="responseTimeChart" width="400" height="200"></canvas>
            
            <script>
                const ws = new WebSocket('ws://localhost:8000/ws');
                const ctx = document.getElementById('responseTimeChart').getContext('2d');
                
                const chart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: '응답 시간',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: true
                            }
                        }
                    }
                });
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    
                    // 통계 업데이트
                    document.getElementById('total-requests').textContent = data.system_stats.total_requests;
                    document.getElementById('success-rate').textContent = data.system_stats.success_rate.toFixed(1) + '%';
                    document.getElementById('memory-usage').textContent = data.memory_usage.percent.toFixed(1) + '%';
                    
                    // 차트 업데이트
                    const time = new Date(data.timestamp * 1000).toLocaleTimeString();
                    chart.data.labels.push(time);
                    chart.data.datasets[0].data.push(data.system_stats.avg_response_time || 0);
                    
                    // 최대 50개 포인트만 유지
                    if (chart.data.labels.length > 50) {
                        chart.data.labels.shift();
                        chart.data.datasets[0].data.shift();
                    }
                    
                    chart.update();
                };
            </script>
        </body>
        </html>
        """
```

## 배포 최적화

### 1. Docker 최적화

```dockerfile
# Dockerfile
FROM python:3.10-slim

# 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . /app
WORKDIR /app

# 환경 변수
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

# 포트 설정
EXPOSE 8000

# 실행 명령
CMD ["uvicorn", "fastapi_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### 2. 로드 밸런싱

```python
import random
from typing import List, Dict, Any
import requests
import time

class LoadBalancer:
    def __init__(self, servers: List[str]):
        self.servers = servers
        self.server_weights = {server: 1.0 for server in servers}
        self.server_stats = {server: {'requests': 0, 'errors': 0, 'response_times': []} 
                           for server in servers}
    
    def get_server(self) -> str:
        """가중치 기반 서버 선택"""
        total_weight = sum(self.server_weights.values())
        
        if total_weight == 0:
            return random.choice(self.servers)
        
        # 가중치 기반 선택
        weights = [self.server_weights[server] / total_weight for server in self.servers]
        return random.choices(self.servers, weights=weights)[0]
    
    def update_server_stats(self, server: str, response_time: float, success: bool):
        """서버 통계 업데이트"""
        stats = self.server_stats[server]
        stats['requests'] += 1
        stats['response_times'].append(response_time)
        
        if not success:
            stats['errors'] += 1
        
        # 가중치 업데이트 (성능 기반)
        if len(stats['response_times']) >= 10:
            avg_response_time = sum(stats['response_times'][-10:]) / 10
            error_rate = stats['errors'] / stats['requests']
            
            # 가중치 계산 (응답 시간과 에러율 고려)
            self.server_weights[server] = max(0.1, 1.0 / (avg_response_time + error_rate))
    
    def health_check(self):
        """서버 상태 확인"""
        for server in self.servers:
            try:
                response = requests.get(f"{server}/health", timeout=5)
                if response.status_code != 200:
                    self.server_weights[server] *= 0.5  # 가중치 감소
            except:
                self.server_weights[server] = 0.1  # 거의 사용하지 않음
```

### 3. 오토스케일링

```python
import time
import threading
from typing import Dict, Any
import subprocess

class AutoScaler:
    def __init__(self, min_instances: int = 2, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.current_instances = min_instances
        self.monitoring = False
        self.metrics_history = []
    
    def start_monitoring(self, metrics: PerformanceMetrics):
        """모니터링 시작"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                # 메트릭 수집
                stats = metrics.get_system_stats()
                self.metrics_history.append(stats)
                
                # 최근 5분간 데이터만 유지
                if len(self.metrics_history) > 300:  # 5분 * 60초
                    self.metrics_history.pop(0)
                
                # 스케일링 결정
                self._evaluate_scaling()
                
                time.sleep(1)
        
        thread = threading.Thread(target=monitor_loop)
        thread.daemon = True
        thread.start()
    
    def _evaluate_scaling(self):
        """스케일링 평가"""
        if len(self.metrics_history) < 60:  # 1분간 데이터 필요
            return
        
        recent_stats = self.metrics_history[-60:]  # 최근 1분
        
        # 평균 메트릭 계산
        avg_response_time = sum(stat.get('avg_response_time', 0) for stat in recent_stats) / len(recent_stats)
        avg_memory_usage = sum(stat.get('avg_memory_usage', 0) for stat in recent_stats) / len(recent_stats)
        
        # 스케일 업 조건
        if (avg_response_time > 1000 or  # 1초 초과
            avg_memory_usage > 0.8 * (8 * 1024**3)):  # 메모리 80% 초과
            
            if self.current_instances < self.max_instances:
                self._scale_up()
        
        # 스케일 다운 조건
        elif (avg_response_time < 200 and  # 200ms 미만
              avg_memory_usage < 0.4 * (8 * 1024**3)):  # 메모리 40% 미만
            
            if self.current_instances > self.min_instances:
                self._scale_down()
    
    def _scale_up(self):
        """인스턴스 추가"""
        new_port = 8000 + self.current_instances
        
        # 새 인스턴스 시작
        subprocess.Popen([
            'uvicorn', 'fastapi_server:app',
            '--host', '0.0.0.0',
            '--port', str(new_port),
            '--workers', '1'
        ])
        
        self.current_instances += 1
        print(f"인스턴스 추가: 현재 {self.current_instances}개")
    
    def _scale_down(self):
        """인스턴스 제거"""
        if self.current_instances > self.min_instances:
            # 마지막 인스턴스 종료 (실제 구현 필요)
            port_to_stop = 8000 + self.current_instances - 1
            self._stop_instance(port_to_stop)
            
            self.current_instances -= 1
            print(f"인스턴스 제거: 현재 {self.current_instances}개")
    
    def _stop_instance(self, port: int):
        """특정 포트의 인스턴스 중지"""
        try:
            subprocess.run(['pkill', '-f', f'--port {port}'])
        except:
            pass
```

이 성능 최적화 가이드를 통해 HuggingFace FastAPI 서버의 성능을 크게 향상시킬 수 있습니다. 각 최적화 기법은 독립적으로 적용할 수 있으며, 시스템의 특성에 맞게 조합하여 사용하면 됩니다.