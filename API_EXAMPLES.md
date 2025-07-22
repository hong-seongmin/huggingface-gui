# HuggingFace FastAPI 서버 고급 사용 예제

## 목차
1. [Python 클라이언트 예제](#python-클라이언트-예제)
2. [JavaScript 클라이언트 예제](#javascript-클라이언트-예제)
3. [배치 처리 예제](#배치-처리-예제)
4. [멀티포트 서버 설정](#멀티포트-서버-설정)
5. [실시간 모니터링](#실시간-모니터링)
6. [성능 벤치마킹](#성능-벤치마킹)

## Python 클라이언트 예제

### 1. 기본 클라이언트 클래스

```python
import requests
import json
import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PredictionResult:
    """예측 결과 데이터 클래스"""
    model_name: str
    task: str
    input_text: str
    result: List[Dict[str, Any]]
    timestamp: str
    processing_time: float = 0.0

class HuggingFaceAPIClient:
    """HuggingFace FastAPI 서버 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def load_model(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """모델 로드"""
        url = f"{self.base_url}/models/load"
        data = {
            "model_name": model_name,
            "model_path": model_path
        }
        
        response = self.session.post(url, json=data)
        response.raise_for_status()
        return response.json()
    
    def predict(self, model_name: str, text: str, **kwargs) -> PredictionResult:
        """텍스트 예측"""
        url = f"{self.base_url}/models/{model_name}/predict"
        data = {"text": text, **kwargs}
        
        start_time = datetime.now()
        response = self.session.post(url, json=data)
        end_time = datetime.now()
        
        response.raise_for_status()
        result_data = response.json()
        
        processing_time = (end_time - start_time).total_seconds()
        
        return PredictionResult(
            model_name=result_data["model_name"],
            task=result_data["task"],
            input_text=result_data["input"],
            result=result_data["result"],
            timestamp=result_data["timestamp"],
            processing_time=processing_time
        )
    
    def get_models(self) -> List[str]:
        """로드된 모델 목록 조회"""
        url = f"{self.base_url}/models"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()["loaded_models"]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """모델 정보 조회"""
        url = f"{self.base_url}/models/{model_name}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        """시스템 상태 조회"""
        url = f"{self.base_url}/system/status"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
    
    def cleanup_system(self) -> Dict[str, Any]:
        """시스템 정리"""
        url = f"{self.base_url}/system/cleanup"
        response = self.session.post(url)
        response.raise_for_status()
        return response.json()

# 사용 예제
def main():
    client = HuggingFaceAPIClient()
    
    # 모델 로드
    print("감정 분석 모델 로드 중...")
    client.load_model(
        model_name="sentiment-analyzer",
        model_path="cardiffnlp/twitter-roberta-base-sentiment-latest"
    )
    
    # 예측 수행
    result = client.predict(
        model_name="sentiment-analyzer",
        text="이 영화는 정말 재미있었다!"
    )
    
    print(f"예측 결과: {result.result}")
    print(f"처리 시간: {result.processing_time:.2f}초")

if __name__ == "__main__":
    main()
```

### 2. 비동기 클라이언트

```python
import asyncio
import aiohttp
from typing import List, Dict, Any
import time

class AsyncHuggingFaceAPIClient:
    """비동기 HuggingFace FastAPI 서버 클라이언트"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        
    async def predict_async(self, model_name: str, text: str, **kwargs) -> Dict[str, Any]:
        """비동기 예측"""
        url = f"{self.base_url}/models/{model_name}/predict"
        data = {"text": text, **kwargs}
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                return await response.json()
    
    async def batch_predict(self, model_name: str, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """배치 예측"""
        tasks = []
        for text in texts:
            task = self.predict_async(model_name, text, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        return results
    
    async def load_model_async(self, model_name: str, model_path: str) -> Dict[str, Any]:
        """비동기 모델 로드"""
        url = f"{self.base_url}/models/load"
        data = {
            "model_name": model_name,
            "model_path": model_path
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=data) as response:
                response.raise_for_status()
                return await response.json()

# 사용 예제
async def async_main():
    client = AsyncHuggingFaceAPIClient()
    
    # 여러 텍스트 동시 처리
    texts = [
        "이 영화는 정말 재미있었다!",
        "서비스가 너무 별로였다.",
        "품질이 우수하고 가격도 합리적이다.",
        "배송이 빠르고 포장도 깔끔했다."
    ]
    
    start_time = time.time()
    results = await client.batch_predict("sentiment-analyzer", texts)
    end_time = time.time()
    
    print(f"배치 처리 시간: {end_time - start_time:.2f}초")
    for i, result in enumerate(results):
        print(f"텍스트 {i+1}: {result['result']}")

if __name__ == "__main__":
    asyncio.run(async_main())
```

## JavaScript 클라이언트 예제

### 1. Node.js 클라이언트

```javascript
const axios = require('axios');

class HuggingFaceAPIClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
        this.client = axios.create({
            baseURL: baseUrl,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json'
            }
        });
    }
    
    async loadModel(modelName, modelPath) {
        try {
            const response = await this.client.post('/models/load', {
                model_name: modelName,
                model_path: modelPath
            });
            return response.data;
        } catch (error) {
            console.error('모델 로드 실패:', error.response?.data || error.message);
            throw error;
        }
    }
    
    async predict(modelName, text, options = {}) {
        try {
            const response = await this.client.post(`/models/${modelName}/predict`, {
                text: text,
                ...options
            });
            return response.data;
        } catch (error) {
            console.error('예측 실패:', error.response?.data || error.message);
            throw error;
        }
    }
    
    async getModels() {
        try {
            const response = await this.client.get('/models');
            return response.data.loaded_models;
        } catch (error) {
            console.error('모델 목록 조회 실패:', error.response?.data || error.message);
            throw error;
        }
    }
    
    async getSystemStatus() {
        try {
            const response = await this.client.get('/system/status');
            return response.data;
        } catch (error) {
            console.error('시스템 상태 조회 실패:', error.response?.data || error.message);
            throw error;
        }
    }
}

// 사용 예제
async function main() {
    const client = new HuggingFaceAPIClient();
    
    try {
        // 모델 로드
        console.log('감정 분석 모델 로드 중...');
        await client.loadModel(
            'sentiment-analyzer',
            'cardiffnlp/twitter-roberta-base-sentiment-latest'
        );
        
        // 예측 수행
        const result = await client.predict(
            'sentiment-analyzer',
            '이 영화는 정말 재미있었다!'
        );
        
        console.log('예측 결과:', result.result);
        console.log('처리 시간:', result.timestamp);
        
    } catch (error) {
        console.error('오류 발생:', error);
    }
}

main();
```

### 2. React 컴포넌트

```jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';

const HuggingFaceAPIDemo = () => {
    const [models, setModels] = useState([]);
    const [selectedModel, setSelectedModel] = useState('');
    const [inputText, setInputText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    
    const API_BASE_URL = 'http://localhost:8000';
    
    useEffect(() => {
        loadModels();
    }, []);
    
    const loadModels = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/models`);
            setModels(response.data.loaded_models);
        } catch (err) {
            setError('모델 목록을 불러오는데 실패했습니다.');
        }
    };
    
    const predict = async () => {
        if (!selectedModel || !inputText) {
            setError('모델과 텍스트를 모두 선택해주세요.');
            return;
        }
        
        setLoading(true);
        setError(null);
        
        try {
            const response = await axios.post(
                `${API_BASE_URL}/models/${selectedModel}/predict`,
                { text: inputText }
            );
            setResult(response.data);
        } catch (err) {
            setError('예측에 실패했습니다: ' + (err.response?.data?.detail || err.message));
        } finally {
            setLoading(false);
        }
    };
    
    return (
        <div className="container mx-auto p-4">
            <h1 className="text-2xl font-bold mb-4">HuggingFace API 데모</h1>
            
            <div className="mb-4">
                <label className="block text-sm font-medium mb-2">모델 선택:</label>
                <select 
                    value={selectedModel} 
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="w-full p-2 border rounded"
                >
                    <option value="">모델을 선택하세요</option>
                    {models.map(model => (
                        <option key={model} value={model}>{model}</option>
                    ))}
                </select>
            </div>
            
            <div className="mb-4">
                <label className="block text-sm font-medium mb-2">입력 텍스트:</label>
                <textarea
                    value={inputText}
                    onChange={(e) => setInputText(e.target.value)}
                    className="w-full p-2 border rounded"
                    rows="4"
                    placeholder="분석할 텍스트를 입력하세요..."
                />
            </div>
            
            <button
                onClick={predict}
                disabled={loading}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
            >
                {loading ? '처리 중...' : '예측하기'}
            </button>
            
            {error && (
                <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
                    {error}
                </div>
            )}
            
            {result && (
                <div className="mt-4 p-3 bg-green-100 border border-green-400 rounded">
                    <h3 className="font-bold mb-2">예측 결과:</h3>
                    <pre className="text-sm">{JSON.stringify(result, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default HuggingFaceAPIDemo;
```

## 배치 처리 예제

### 1. 대용량 텍스트 처리

```python
import asyncio
import aiohttp
import pandas as pd
from typing import List, Dict, Any
import time
import json

class BatchProcessor:
    """배치 처리 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000", max_concurrent: int = 10):
        self.base_url = base_url
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_single_text(self, session: aiohttp.ClientSession, 
                                 model_name: str, text: str, index: int) -> Dict[str, Any]:
        """단일 텍스트 처리"""
        async with self.semaphore:
            url = f"{self.base_url}/models/{model_name}/predict"
            data = {"text": text}
            
            try:
                async with session.post(url, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return {
                        "index": index,
                        "text": text,
                        "success": True,
                        "result": result["result"],
                        "processing_time": result.get("timestamp")
                    }
            except Exception as e:
                return {
                    "index": index,
                    "text": text,
                    "success": False,
                    "error": str(e)
                }
    
    async def process_batch(self, model_name: str, texts: List[str]) -> List[Dict[str, Any]]:
        """배치 처리"""
        connector = aiohttp.TCPConnector(limit=self.max_concurrent)
        timeout = aiohttp.ClientTimeout(total=300)  # 5분 타임아웃
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = []
            for i, text in enumerate(texts):
                task = self.process_single_text(session, model_name, text, i)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 예외 처리
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        "success": False,
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
    
    def process_csv(self, csv_path: str, model_name: str, text_column: str, 
                   output_path: str = None) -> pd.DataFrame:
        """CSV 파일 배치 처리"""
        # CSV 파일 읽기
        df = pd.read_csv(csv_path)
        texts = df[text_column].astype(str).tolist()
        
        print(f"총 {len(texts)}개의 텍스트 처리 시작...")
        
        # 배치 처리 실행
        start_time = time.time()
        results = asyncio.run(self.process_batch(model_name, texts))
        end_time = time.time()
        
        print(f"배치 처리 완료: {end_time - start_time:.2f}초")
        
        # 결과를 DataFrame에 추가
        df['prediction_result'] = None
        df['prediction_success'] = False
        
        for result in results:
            if result['success']:
                df.loc[result['index'], 'prediction_result'] = json.dumps(result['result'])
                df.loc[result['index'], 'prediction_success'] = True
            else:
                df.loc[result['index'], 'prediction_result'] = result.get('error', 'Unknown error')
                df.loc[result['index'], 'prediction_success'] = False
        
        # 결과 저장
        if output_path:
            df.to_csv(output_path, index=False)
            print(f"결과를 {output_path}에 저장했습니다.")
        
        return df

# 사용 예제
async def batch_example():
    processor = BatchProcessor(max_concurrent=5)
    
    # 샘플 텍스트
    sample_texts = [
        "이 영화는 정말 재미있었다!",
        "서비스가 너무 별로였다.",
        "품질이 우수하고 가격도 합리적이다.",
        "배송이 빠르고 포장도 깔끔했다.",
        "추천하고 싶지 않은 제품이다."
    ] * 20  # 100개 텍스트
    
    results = await processor.process_batch("sentiment-analyzer", sample_texts)
    
    # 성공/실패 통계
    success_count = sum(1 for r in results if r['success'])
    print(f"성공: {success_count}/{len(results)}")
    
    # 결과 출력
    for result in results[:5]:  # 처음 5개만 출력
        if result['success']:
            print(f"텍스트: {result['text'][:50]}...")
            print(f"결과: {result['result']}")
        else:
            print(f"실패: {result['error']}")

if __name__ == "__main__":
    asyncio.run(batch_example())
```

## 멀티포트 서버 설정

### 1. 서버 설정 스크립트

```python
import threading
import uvicorn
from fastapi_server import FastAPIServer
from model_manager import MultiModelManager
import time

class MultiPortServerManager:
    """멀티포트 서버 관리자"""
    
    def __init__(self):
        self.model_manager = MultiModelManager()
        self.servers = {}
        self.server_threads = {}
    
    def start_model_server(self, port: int, model_config: dict):
        """특정 포트에서 모델 서버 시작"""
        def run_server():
            server = FastAPIServer(self.model_manager, port=port)
            
            # 모델 로드
            for model_name, model_path in model_config.items():
                print(f"포트 {port}에서 모델 {model_name} 로드 중...")
                thread = self.model_manager.load_model_async(model_name, model_path)
                thread.join()  # 모델 로드 완료 대기
            
            # 서버 시작
            server.start_server()
            
        thread = threading.Thread(target=run_server)
        thread.daemon = True
        thread.start()
        
        self.servers[port] = {'config': model_config, 'running': True}
        self.server_threads[port] = thread
        
        return f"서버가 포트 {port}에서 시작되었습니다."
    
    def stop_server(self, port: int):
        """특정 포트의 서버 중지"""
        if port in self.servers:
            # 서버 중지 로직 (실제 구현 필요)
            self.servers[port]['running'] = False
            return f"포트 {port}의 서버가 중지되었습니다."
        return f"포트 {port}에서 실행 중인 서버가 없습니다."
    
    def get_server_status(self):
        """모든 서버 상태 반환"""
        return {
            port: {
                'running': info['running'],
                'models': list(info['config'].keys())
            }
            for port, info in self.servers.items()
        }

# 사용 예제
def main():
    manager = MultiPortServerManager()
    
    # 감정 분석 서버 (포트 8001)
    sentiment_config = {
        "sentiment-roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "sentiment-bert": "nlptown/bert-base-multilingual-uncased-sentiment"
    }
    
    # NER 서버 (포트 8002)
    ner_config = {
        "ner-electra": "skimb22/koelectra-ner-klue-test1",
        "ner-bert": "dbmdz/bert-large-cased-finetuned-conll03-english"
    }
    
    # 임베딩 서버 (포트 8003)
    embedding_config = {
        "embedding-bge": "BAAI/bge-m3",
        "embedding-sentence": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # 서버 시작
    print(manager.start_model_server(8001, sentiment_config))
    print(manager.start_model_server(8002, ner_config))
    print(manager.start_model_server(8003, embedding_config))
    
    # 서버 상태 확인
    time.sleep(5)
    print("\n서버 상태:")
    print(manager.get_server_status())
    
    # 서버 계속 실행
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n서버 종료 중...")

if __name__ == "__main__":
    main()
```

### 2. 로드 밸런싱 클라이언트

```python
import random
import requests
from typing import List, Dict, Any
import time

class LoadBalancedClient:
    """로드 밸런싱 클라이언트"""
    
    def __init__(self, server_urls: List[str]):
        self.server_urls = server_urls
        self.server_stats = {url: {'requests': 0, 'errors': 0, 'avg_time': 0} for url in server_urls}
    
    def get_best_server(self) -> str:
        """가장 성능이 좋은 서버 선택"""
        best_server = min(self.server_stats.items(), 
                         key=lambda x: x[1]['avg_time'] + x[1]['errors'] * 0.1)
        return best_server[0]
    
    def predict_with_fallback(self, model_name: str, text: str, **kwargs) -> Dict[str, Any]:
        """장애 복구를 지원하는 예측"""
        servers = self.server_urls.copy()
        random.shuffle(servers)  # 서버 순서 무작위화
        
        for server_url in servers:
            try:
                start_time = time.time()
                
                url = f"{server_url}/models/{model_name}/predict"
                response = requests.post(url, json={"text": text, **kwargs}, timeout=30)
                
                end_time = time.time()
                response_time = end_time - start_time
                
                if response.status_code == 200:
                    # 성공 통계 업데이트
                    stats = self.server_stats[server_url]
                    stats['requests'] += 1
                    stats['avg_time'] = (stats['avg_time'] * (stats['requests'] - 1) + response_time) / stats['requests']
                    
                    result = response.json()
                    result['server_url'] = server_url
                    result['response_time'] = response_time
                    return result
                else:
                    self.server_stats[server_url]['errors'] += 1
                    
            except Exception as e:
                self.server_stats[server_url]['errors'] += 1
                print(f"서버 {server_url}에서 오류 발생: {e}")
                continue
        
        raise Exception("모든 서버에서 요청 실패")
    
    def get_server_stats(self) -> Dict[str, Any]:
        """서버 통계 반환"""
        return self.server_stats

# 사용 예제
def main():
    # 멀티포트 서버 URLs
    servers = [
        "http://localhost:8001",  # 감정 분석 서버
        "http://localhost:8002",  # NER 서버
        "http://localhost:8003"   # 임베딩 서버
    ]
    
    client = LoadBalancedClient(servers)
    
    # 여러 요청 테스트
    texts = [
        "이 영화는 정말 재미있었다!",
        "서비스가 너무 별로였다.",
        "품질이 우수하고 가격도 합리적이다."
    ]
    
    for text in texts:
        try:
            result = client.predict_with_fallback("sentiment-analyzer", text)
            print(f"예측 결과: {result['result']}")
            print(f"서버: {result['server_url']}")
            print(f"응답 시간: {result['response_time']:.2f}초")
            print("-" * 50)
        except Exception as e:
            print(f"예측 실패: {e}")
    
    # 서버 통계 출력
    print("\n서버 통계:")
    for server, stats in client.get_server_stats().items():
        print(f"{server}: {stats}")

if __name__ == "__main__":
    main()
```

## 실시간 모니터링

### 1. 실시간 성능 모니터링

```python
import time
import threading
import requests
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import json

class RealTimeMonitor:
    """실시간 모니터링 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.monitoring = False
        self.data = {
            'timestamps': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'response_times': deque(maxlen=100),
            'active_models': deque(maxlen=100)
        }
    
    def start_monitoring(self, interval: int = 5):
        """모니터링 시작"""
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                try:
                    # 시스템 상태 수집
                    memory_info = self._get_memory_info()
                    system_status = self._get_system_status()
                    
                    # 응답 시간 측정
                    response_time = self._measure_response_time()
                    
                    # 데이터 저장
                    now = datetime.now()
                    self.data['timestamps'].append(now)
                    self.data['memory_usage'].append(memory_info['system_memory']['percent'])
                    self.data['response_times'].append(response_time)
                    self.data['active_models'].append(len(system_status.get('loaded_models', [])))
                    
                    # 로그 출력
                    self._log_status(now, memory_info, response_time)
                    
                except Exception as e:
                    print(f"모니터링 오류: {e}")
                
                time.sleep(interval)
        
        thread = threading.Thread(target=monitor_loop)
        thread.daemon = True
        thread.start()
        
        print("실시간 모니터링이 시작되었습니다.")
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        print("모니터링이 중지되었습니다.")
    
    def _get_memory_info(self):
        """메모리 정보 수집"""
        try:
            response = requests.get(f"{self.base_url}/system/memory", timeout=5)
            response.raise_for_status()
            return response.json()
        except:
            return {'system_memory': {'percent': 0}}
    
    def _get_system_status(self):
        """시스템 상태 수집"""
        try:
            response = requests.get(f"{self.base_url}/system/status", timeout=5)
            response.raise_for_status()
            return response.json()
        except:
            return {}
    
    def _measure_response_time(self):
        """응답 시간 측정"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/health", timeout=5)
            end_time = time.time()
            
            if response.status_code == 200:
                return (end_time - start_time) * 1000  # 밀리초
            else:
                return -1
        except:
            return -1
    
    def _log_status(self, timestamp, memory_info, response_time):
        """상태 로깅"""
        memory_percent = memory_info['system_memory']['percent']
        
        print(f"[{timestamp.strftime('%H:%M:%S')}] "
              f"메모리: {memory_percent:.1f}%, "
              f"응답시간: {response_time:.1f}ms")
    
    def generate_report(self, save_path: str = None):
        """모니터링 리포트 생성"""
        if not self.data['timestamps']:
            print("모니터링 데이터가 없습니다.")
            return
        
        # 그래프 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 메모리 사용량
        axes[0, 0].plot(self.data['timestamps'], self.data['memory_usage'])
        axes[0, 0].set_title('메모리 사용량 (%)')
        axes[0, 0].set_ylabel('사용률 (%)')
        
        # 응답 시간
        axes[0, 1].plot(self.data['timestamps'], self.data['response_times'])
        axes[0, 1].set_title('응답 시간 (ms)')
        axes[0, 1].set_ylabel('시간 (ms)')
        
        # 활성 모델 수
        axes[1, 0].plot(self.data['timestamps'], self.data['active_models'])
        axes[1, 0].set_title('활성 모델 수')
        axes[1, 0].set_ylabel('모델 수')
        
        # 통계 정보
        axes[1, 1].axis('off')
        stats_text = f"""
        통계 정보:
        
        평균 메모리 사용률: {sum(self.data['memory_usage'])/len(self.data['memory_usage']):.1f}%
        평균 응답 시간: {sum(self.data['response_times'])/len(self.data['response_times']):.1f}ms
        최대 응답 시간: {max(self.data['response_times']):.1f}ms
        평균 활성 모델 수: {sum(self.data['active_models'])/len(self.data['active_models']):.1f}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"리포트를 {save_path}에 저장했습니다.")
        
        plt.show()

# 사용 예제
def main():
    monitor = RealTimeMonitor()
    
    # 모니터링 시작
    monitor.start_monitoring(interval=2)
    
    try:
        # 30초간 모니터링
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    
    # 모니터링 중지
    monitor.stop_monitoring()
    
    # 리포트 생성
    monitor.generate_report("monitoring_report.png")

if __name__ == "__main__":
    main()
```

## 성능 벤치마킹

### 1. 성능 벤치마크 스크립트

```python
import time
import statistics
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import json

class PerformanceBenchmark:
    """성능 벤치마크 클래스"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    def single_request_benchmark(self, model_name: str, text: str, 
                                num_requests: int = 100) -> Dict[str, Any]:
        """단일 요청 벤치마크"""
        response_times = []
        errors = 0
        
        print(f"단일 요청 벤치마크 시작: {num_requests}회 요청")
        
        for i in range(num_requests):
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/models/{model_name}/predict",
                    json={"text": text},
                    timeout=30
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # 밀리초
                
                if response.status_code == 200:
                    response_times.append(response_time)
                else:
                    errors += 1
                    
            except Exception as e:
                errors += 1
                print(f"요청 {i+1} 실패: {e}")
            
            # 진행률 표시
            if (i + 1) % 10 == 0:
                print(f"진행률: {i+1}/{num_requests}")
        
        return {
            "test_type": "single_request",
            "model_name": model_name,
            "num_requests": num_requests,
            "successful_requests": len(response_times),
            "errors": errors,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    def concurrent_request_benchmark(self, model_name: str, text: str, 
                                   num_requests: int = 100, 
                                   num_threads: int = 10) -> Dict[str, Any]:
        """동시 요청 벤치마크"""
        response_times = []
        errors = 0
        
        print(f"동시 요청 벤치마크 시작: {num_requests}회 요청, {num_threads}개 스레드")
        
        def single_request():
            try:
                start_time = time.time()
                
                response = requests.post(
                    f"{self.base_url}/models/{model_name}/predict",
                    json={"text": text},
                    timeout=30
                )
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000  # 밀리초
                
                if response.status_code == 200:
                    return response_time
                else:
                    return None
                    
            except Exception as e:
                return None
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(single_request) for _ in range(num_requests)]
            
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result is not None:
                    response_times.append(result)
                else:
                    errors += 1
                
                # 진행률 표시
                if (i + 1) % 10 == 0:
                    print(f"진행률: {i+1}/{num_requests}")
        
        total_time = time.time() - start_time
        
        return {
            "test_type": "concurrent_request",
            "model_name": model_name,
            "num_requests": num_requests,
            "num_threads": num_threads,
            "total_time": total_time,
            "successful_requests": len(response_times),
            "errors": errors,
            "requests_per_second": len(response_times) / total_time if total_time > 0 else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "std_response_time": statistics.stdev(response_times) if len(response_times) > 1 else 0
        }
    
    def throughput_benchmark(self, model_name: str, texts: List[str], 
                           duration: int = 60) -> Dict[str, Any]:
        """처리량 벤치마크"""
        print(f"처리량 벤치마크 시작: {duration}초간 실행")
        
        start_time = time.time()
        end_time = start_time + duration
        
        completed_requests = 0
        errors = 0
        response_times = []
        
        def worker():
            nonlocal completed_requests, errors, response_times
            
            while time.time() < end_time:
                text = texts[completed_requests % len(texts)]
                
                try:
                    request_start = time.time()
                    
                    response = requests.post(
                        f"{self.base_url}/models/{model_name}/predict",
                        json={"text": text},
                        timeout=30
                    )
                    
                    request_end = time.time()
                    response_time = (request_end - request_start) * 1000
                    
                    if response.status_code == 200:
                        completed_requests += 1
                        response_times.append(response_time)
                    else:
                        errors += 1
                        
                except Exception as e:
                    errors += 1
        
        # 여러 스레드로 요청 실행
        threads = []
        for _ in range(5):  # 5개 스레드
            thread = threading.Thread(target=worker)
            thread.start()
            threads.append(thread)
        
        # 모든 스레드 완료 대기
        for thread in threads:
            thread.join()
        
        actual_duration = time.time() - start_time
        
        return {
            "test_type": "throughput",
            "model_name": model_name,
            "duration": actual_duration,
            "completed_requests": completed_requests,
            "errors": errors,
            "requests_per_second": completed_requests / actual_duration if actual_duration > 0 else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "median_response_time": statistics.median(response_times) if response_times else 0
        }
    
    def run_full_benchmark(self, model_name: str, sample_text: str = "테스트 텍스트입니다."):
        """전체 벤치마크 실행"""
        print(f"모델 {model_name}에 대한 전체 벤치마크 실행")
        
        # 1. 단일 요청 벤치마크
        single_result = self.single_request_benchmark(model_name, sample_text, 50)
        self.results.append(single_result)
        
        # 2. 동시 요청 벤치마크
        concurrent_result = self.concurrent_request_benchmark(model_name, sample_text, 50, 5)
        self.results.append(concurrent_result)
        
        # 3. 처리량 벤치마크
        sample_texts = [
            "긍정적인 리뷰입니다.",
            "부정적인 피드백입니다.",
            "중립적인 의견입니다.",
            sample_text
        ]
        throughput_result = self.throughput_benchmark(model_name, sample_texts, 30)
        self.results.append(throughput_result)
    
    def generate_report(self, save_path: str = None):
        """벤치마크 리포트 생성"""
        if not self.results:
            print("벤치마크 결과가 없습니다.")
            return
        
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": self.results,
            "summary": {}
        }
        
        # 요약 정보 생성
        for result in self.results:
            model_name = result["model_name"]
            test_type = result["test_type"]
            
            if model_name not in report["summary"]:
                report["summary"][model_name] = {}
            
            report["summary"][model_name][test_type] = {
                "avg_response_time": result.get("avg_response_time", 0),
                "requests_per_second": result.get("requests_per_second", 0),
                "success_rate": result.get("successful_requests", 0) / result.get("num_requests", 1) * 100
            }
        
        # 리포트 출력
        print("\n=== 벤치마크 리포트 ===")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        
        # 파일 저장
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n리포트를 {save_path}에 저장했습니다.")

# 사용 예제
def main():
    benchmark = PerformanceBenchmark()
    
    # 벤치마크 실행
    benchmark.run_full_benchmark("sentiment-analyzer", "이 영화는 정말 재미있었다!")
    
    # 리포트 생성
    benchmark.generate_report("benchmark_report.json")

if __name__ == "__main__":
    main()
```

이 고급 예제들은 HuggingFace FastAPI 서버를 실제 운영 환경에서 효과적으로 사용할 수 있도록 도와줍니다. 각 예제는 독립적으로 사용할 수 있으며, 필요에 따라 수정하여 사용할 수 있습니다.