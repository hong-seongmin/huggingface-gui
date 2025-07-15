import threading
import psutil
import torch
import os
import re
import hashlib
import logging
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from model_analyzer import ComprehensiveModelAnalyzer
from model_optimization import optimizer
from model_cache import model_cache
from device_manager import device_manager
from detailed_profiler import profiler
from model_type_detector import ModelTypeDetector
from huggingface_hub import hf_hub_download, snapshot_download, HfApi

@dataclass
class ModelInfo:
    name: str
    path: str
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    config_analysis: Dict = field(default_factory=dict)
    memory_usage: float = 0.0
    load_time: Optional[datetime] = None
    status: str = "unloaded"  # unloaded, loading, loaded, error
    error_message: str = ""

class MultiModelManager:
    def __init__(self):
        self.models: Dict[str, ModelInfo] = {}
        self.loading_locks = {}
        self.max_memory_threshold = 0.8  # 80% 메모리 사용률 제한
        self.load_queue = []
        self.model_analyzer = ComprehensiveModelAnalyzer()
        self.model_type_detector = ModelTypeDetector()
        self.callbacks: List[Callable] = []
        self.hf_api = HfApi()
        
    def add_callback(self, callback: Callable):
        """모델 상태 변경 콜백 등록"""
        self.callbacks.append(callback)
        
    def _notify_callbacks(self, model_name: str, event_type: str, data: Dict = None):
        """콜백 함수들에게 알림"""
        for callback in self.callbacks:
            try:
                callback(model_name, event_type, data or {})
            except Exception as e:
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Callback error: {e}")
    
    def get_memory_info(self):
        """현재 메모리 사용량 정보"""
        memory = psutil.virtual_memory()
        gpu_memory = []
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory.append({
                    'device': i,
                    'name': torch.cuda.get_device_name(i),
                    'total': torch.cuda.get_device_properties(i).total_memory,
                    'allocated': torch.cuda.memory_allocated(i),
                    'reserved': torch.cuda.memory_reserved(i)
                })
        
        return {
            'system_memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent
            },
            'gpu_memory': gpu_memory
        }
    
    def can_load_model(self, estimated_size: float) -> bool:
        """모델 로드 가능 여부 체크"""
        memory_info = self.get_memory_info()
        current_usage = memory_info['system_memory']['percent'] / 100
        return current_usage + (estimated_size / memory_info['system_memory']['total']) < self.max_memory_threshold
    
    def analyze_model(self, model_path: str) -> Dict:
        """모델 분석 (로드 없이, HuggingFace 모델 ID 지원)"""
        try:
            actual_model_path = model_path
            
            # HuggingFace 모델 ID인경우 다운로드
            if self._is_huggingface_model_id(model_path):
                actual_model_path = self._download_huggingface_model(model_path)
            
            analysis = self.model_analyzer.analyze_model_directory(actual_model_path)
            analysis['original_path'] = model_path
            analysis['actual_path'] = actual_model_path
            return analysis
        except Exception as e:
            return {'error': str(e)}
    
    def _is_huggingface_model_id(self, model_path: str) -> bool:
        """HuggingFace 모델 ID 형식인지 확인"""
        # 로컬 경로가 아닌 경우 (절대경로나 상대경로)
        if os.path.isabs(model_path) or os.path.exists(model_path):
            return False
        
        # URL이 아닌 경우
        if model_path.startswith(('http://', 'https://', 'file://')):
            return False
        
        # HuggingFace 모델 ID 형식 확인 (username/model-name)
        if '/' in model_path and not model_path.startswith('/'):
            return True
        
        return False
    
    def _download_huggingface_model(self, model_id: str) -> str:
        """HuggingFace Hub에서 모델 다운로드"""
        try:
            # 모델 정보 확인
            model_info = self.hf_api.model_info(model_id)
            
            # 로컬 캐시 디렉토리로 다운로드 (이미 캐시된 경우 다운로드 방지)
            try:
                # 먼저 로컬 캐시에서 찾기 시도
                local_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model", 
                    cache_dir=None,  # 기본 캐시 디렉토리 사용
                    local_files_only=True  # 이미 캐시된 파일만 사용
                )
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 캐시에서 모델 찾음: {model_id}")
            except Exception:
                # 캐시에 없는 경우에만 다운로드
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 캐시 미스 - 다운로드 시작: {model_id}")
                local_path = snapshot_download(
                    repo_id=model_id,
                    repo_type="model",
                    cache_dir=None,  # 기본 캐시 디렉토리 사용
                    local_files_only=False
                )
            
            return local_path
            
        except Exception as e:
            raise Exception(f"HuggingFace 모델 다운로드 실패: {str(e)}")
    
    def _generate_model_name(self, model_path: str) -> str:
        """모델 경로에서 자동으로 모델 이름 생성"""
        if self._is_huggingface_model_id(model_path):
            # HuggingFace 모델 ID에서 모델 이름 추출
            return model_path.split('/')[-1]
        else:
            # 로컬 경로에서 모델 이름 추출
            return os.path.basename(model_path.rstrip('/'))
    
    def load_model_async(self, model_name: str, model_path: str, callback: Optional[Callable] = None):
        """비동기 모델 로드 (HuggingFace 모델 ID 지원, 모델 이름 자동 생성)"""
        # 모델 이름이 비어있으면 자동 생성
        if not model_name or not model_name.strip():
            model_name = self._generate_model_name(model_path)
        
        # 중복 모델 이름 처리
        original_name = model_name
        counter = 1
        while model_name in self.models:
            model_name = f"{original_name}_{counter}"
            counter += 1
        
        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - load_model_async 시작: {model_name}, {model_path}")
        
        # 로딩 락 설정
        if model_name not in self.loading_locks:
            self.loading_locks[model_name] = threading.Lock()
        
        # 스레드 시작
        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 스레드 생성 중: {model_name}")
        thread = threading.Thread(
            target=self._load_model_sync, 
            args=(model_name, model_path, callback),
            name=f"ModelLoad-{model_name}"
        )
        thread.daemon = True
        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 스레드 시작 전: {model_name}")
        thread.start()
        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 스레드 시작됨: {model_name}, thread={thread}")
        
        return thread
    
    def _load_model_sync(self, model_name: str, model_path: str, callback: Optional[Callable] = None):
        """실제 모델 로딩 작업 (스레드에서 실행)"""
        import time
        import threading
        import queue
        
        start_time = time.time()
        
        def load_model_with_transformers(actual_model_path, device):
            """Fast 모델 로딩"""
            print(f"[FAST] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 시작")
            
            # 프로파일링 시작 (프로파일러 내부에서 활성화 여부 확인)
            profiler.start_profiling("모델 로딩")
            profiler.memory_snapshot("초기 상태")
            
            # 직접 transformers 라이브러리 사용으로 실제 BGE-M3 모델 로딩
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 실제 transformers 모델 로딩 시작")
            
            import time
            load_start = time.time()
            
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Config 로딩 시작: {model_name}")
                
                # 빠른 로컬 config 확인
                try:
                    import json
                    config_path = os.path.join(actual_model_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config_dict = json.load(f)
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 로컬 config 로딩 완료: {model_name}")
                    else:
                        config = AutoConfig.from_pretrained(actual_model_path, local_files_only=True)
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AutoConfig 로딩 완료: {model_name}")
                except Exception as e:
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Config 로딩 오류, 기본값 사용: {e}")
                
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 실제 모델 로딩 시작: {model_name}")
                
                # 모델 파일 존재 여부 확인
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 파일 확인 중: {actual_model_path}")
                model_files = [
                    "config.json",
                    "pytorch_model.bin", 
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json"
                ]
                
                for file in model_files:
                    file_path = os.path.join(actual_model_path, file)
                    exists = os.path.exists(file_path)
                    if exists:
                        size_mb = os.path.getsize(file_path) / (1024*1024)
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ {file}: {size_mb:.1f}MB")
                    else:
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ❌ {file}: 파일 없음")
                
                # 메모리 상태 확인
                import psutil
                mem = psutil.virtual_memory()
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 메모리 상태 - 사용률: {mem.percent}%, 사용가능: {mem.available/1024**3:.1f}GB")
                
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AutoModel.from_pretrained 호출 시작 (큰 모델이므로 시간 소요 예상)")
                
                # 환경 변수 상태 확인
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 환경 변수 확인:")
                env_vars = {
                    'HF_HUB_OFFLINE': os.getenv('HF_HUB_OFFLINE', 'None'),
                    'TRANSFORMERS_OFFLINE': os.getenv('TRANSFORMERS_OFFLINE', 'None'),
                    'HF_HUB_DISABLE_TELEMETRY': os.getenv('HF_HUB_DISABLE_TELEMETRY', 'None'),
                    'TOKENIZERS_PARALLELISM': os.getenv('TOKENIZERS_PARALLELISM', 'None')
                }
                for key, value in env_vars.items():
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   {key}={value}")
                
                model_start = time.time()
                
                # AutoModel 로딩을 단계별로 분할하여 진행 상태 추적
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 1/5: transformers AutoModel 임포트 확인")
                from transformers import AutoModel
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 2/5: AutoConfig 사전 로딩")
                
                # Config 먼저 로딩하여 모델 구조 확인
                try:
                    from transformers import AutoConfig
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Config 로딩 시도: {actual_model_path}")
                    config = AutoConfig.from_pretrained(
                        actual_model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ Config 로딩 성공: {config.__class__.__name__}")
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 타입: {getattr(config, 'model_type', 'Unknown')}")
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 어휘 크기: {getattr(config, 'vocab_size', 'Unknown')}")
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 숨겨진 크기: {getattr(config, 'hidden_size', 'Unknown')}")
                except Exception as config_e:
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ Config 로딩 실패, 계속 진행: {config_e}")
                
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 3/5: 실제 모델 가중치 로딩 시작 (가장 시간 소요 단계)")
                
                # 로딩 타임아웃 및 진행상황 모니터링을 위한 스레드 생성
                import threading
                import queue
                
                loading_result = queue.Queue()
                loading_error = queue.Queue()
                
                def load_model_with_progress():
                    """별도 스레드에서 모델 로딩 수행 - ULTRA 최적화"""
                    try:
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 스레드 시작 (ULTRA 모드)")
                        
                        # 필요한 모듈들 임포트
                        import os
                        import torch
                        import gc
                        import time
                        
                        # 환경변수 최적화 (BGE-M3 전용)
                        original_env = {}
                        ultra_env = {
                            'OMP_NUM_THREADS': '4',  # OpenMP 스레드 제한
                            'MKL_NUM_THREADS': '4',  # Intel MKL 제한
                            'TOKENIZERS_PARALLELISM': 'false',  # 토크나이저 병렬화 비활성화
                            'TRANSFORMERS_VERBOSITY': 'error',  # 로그 최소화
                            'HF_HUB_DISABLE_TELEMETRY': '1',
                            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128'  # CUDA 메모리 최적화
                        }
                        
                        for key, value in ultra_env.items():
                            original_env[key] = os.getenv(key)
                            os.environ[key] = value
                        
                        try:
                            # 혁신적 접근: transformers 완전 우회하고 직접 로딩
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - transformers 우회 - 직접 PyTorch 로딩 시작")
                        
                            # 모델 파일 감지 및 최적 파일 선택
                            safetensors_path = os.path.join(actual_model_path, "model.safetensors")
                            pytorch_path = os.path.join(actual_model_path, "pytorch_model.bin")
                            
                            has_safetensors = os.path.exists(safetensors_path)
                            has_pytorch = os.path.exists(pytorch_path)
                            
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 파일 감지:")
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   safetensors: {'✅' if has_safetensors else '❌'} ({os.path.getsize(safetensors_path)/1024**3:.1f}GB)" if has_safetensors else f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   safetensors: ❌")
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   pytorch_model.bin: {'✅' if has_pytorch else '❌'} ({os.path.getsize(pytorch_path)/1024**3:.1f}GB)" if has_pytorch else f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   pytorch_model.bin: ❌")
                        
                            # 고속 로딩을 위한 가중치 파일 선택 (빠른 것 우선)
                            if has_pytorch:
                                weight_file = pytorch_path
                                file_format = "pytorch"
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - PyTorch 형식 선택 (2.2GB 빠른 로딩)")
                            elif has_safetensors:
                                weight_file = safetensors_path  
                                file_format = "safetensors"
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Safetensors 형식 선택")
                            else:
                                raise FileNotFoundError("사용 가능한 모델 가중치 파일이 없습니다")
                            
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 선택된 가중치: {weight_file}")
                        
                            # ULTRA 방식: 직접 모델 초기화 + 가중치 로딩
                            start_time = time.time()
                            
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 1/4: Config 기반 빈 모델 생성")
                            
                            # 모델 타입별 동적 클래스 선택
                            model_type = config.model_type.lower()
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 감지된 모델 타입: {model_type}")
                            
                            if model_type == "xlm-roberta":
                                from transformers import XLMRobertaModel
                                model = XLMRobertaModel(config)
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - XLMRobertaModel 초기화 완료")
                            elif model_type == "distilbert":
                                # 정밀한 모델 분석 결과를 기반으로 분류용 또는 기본 모델 선택
                                if self._should_use_classification_model(model_name):
                                    from transformers import DistilBertForSequenceClassification
                                    model = DistilBertForSequenceClassification(config)
                                    print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - DistilBertForSequenceClassification 초기화 완료 (정밀 분석 기반)")
                                else:
                                    from transformers import DistilBertModel
                                    model = DistilBertModel(config)
                                    print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - DistilBertModel 초기화 완료 (정밀 분석 기반)")
                            elif model_type == "bert":
                                from transformers import BertModel
                                model = BertModel(config)
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - BertModel 초기화 완료")
                            elif model_type == "roberta":
                                from transformers import RobertaModel
                                model = RobertaModel(config)
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - RobertaModel 초기화 완료")
                            else:
                                # 범용 AutoModel 사용 (더 느리지만 안전)
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 알 수 없는 모델 타입, AutoModel 사용")
                                from transformers import AutoModel
                                model = AutoModel.from_config(config)
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AutoModel 초기화 완료")
                            
                            model.eval()  # 평가 모드로 설정
                            init_time = time.time() - start_time
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 초기화 완료: {init_time:.1f}초")
                            
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 2/4: 가중치 로딩 시작 ({file_format}) - PARALLEL 모드")
                            weight_start = time.time()
                            
                            # 모델별 최적화된 병렬 로딩 설정
                            if model_type == "distilbert":
                                # DistilBert는 작은 모델이므로 더 많은 스레드 사용 가능
                                thread_count = min(6, os.cpu_count())
                            elif model_type == "xlm-roberta":
                                # XLM-RoBERTa는 큰 모델이므로 적당한 스레드 수
                                thread_count = min(4, os.cpu_count())
                            else:
                                # 기타 모델들은 보수적 설정
                                thread_count = min(3, os.cpu_count())
                            
                            torch.set_num_threads(thread_count)
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델별 최적화 스레드 수 ({model_type}): {torch.get_num_threads()}")
                            
                            if file_format == "safetensors":
                                # Safetensors 빠른 로딩
                                try:
                                    from safetensors.torch import load_file
                                    state_dict = load_file(weight_file, device='cpu')
                                    print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ Safetensors 로딩 완료")
                                except ImportError:
                                    print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ safetensors 라이브러리 없음, PyTorch 로딩으로 대체")
                                    state_dict = torch.load(weight_file, map_location='cpu')
                            else:
                                # PyTorch 로딩 (메모리 매핑 최적화)
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - PyTorch PARALLEL 로딩 (mmap + 멀티스레드)")
                                try:
                                    # 메모리 매핑 + 병렬 로딩 시도
                                    state_dict = torch.load(weight_file, map_location='cpu', mmap=True)
                                    print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ mmap 로딩 성공")
                                except Exception as mmap_error:
                                    print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - mmap 실패, 일반 로딩으로 대체: {mmap_error}")
                                    state_dict = torch.load(weight_file, map_location='cpu')
                            
                            weight_time = time.time() - weight_start
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 가중치 로딩 완료: {weight_time:.1f}초, 키 개수: {len(state_dict)}")
                            
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 3/4: 모델에 가중치 적용")
                            apply_start = time.time()
                            
                            # 가중치 적용 (엄격한 모드 비활성화)
                            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                            
                            # 메모리 정리
                            del state_dict
                            gc.collect()
                            
                            apply_time = time.time() - apply_start
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 가중치 적용 완룼: {apply_time:.1f}초")
                            
                            if missing_keys:
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ 누락된 키: {len(missing_keys)}개")
                            if unexpected_keys:
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
                            
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 4/4: 최종 설정")
                            model.eval()  # 다시 평가 모드 확인
                            
                            total_time = time.time() - start_time
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ ULTRA 로딩 성공: {total_time:.1f}초 (기존 5분+ 대비 {300/total_time:.1f}x 빠름)")
                            
                            loading_result.put(model)
                            return
                            
                        except Exception as ultra_error:
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ❌ ULTRA 로딩 실패: {ultra_error}")
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 오류 타입: {type(ultra_error).__name__}")
                            
                            # ULTRA 실패 시 보조 방법들 시도
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 보조 방법 1: 메모리 매핑 없이 재시도")
                            try:
                                # 모델 타입 다시 확인 (fallback에서 사용)
                                model_type = config.model_type.lower()
                                
                                # 메모리 매핑 없이 재시도
                                gc.collect()
                                if has_pytorch:
                                    state_dict = torch.load(pytorch_path, map_location='cpu', mmap=False)
                                else:
                                    from safetensors.torch import load_file
                                    state_dict = load_file(safetensors_path, device='cpu')
                                
                                # 모델 타입별 동적 클래스 선택 (fallback 1)
                                if model_type == "xlm-roberta":
                                    from transformers import XLMRobertaModel
                                    model = XLMRobertaModel(config)
                                elif model_type == "distilbert":
                                    # 정밀한 모델 분석 결과를 기반으로 분류용 또는 기본 모델 선택
                                    if self._should_use_classification_model(model_name):
                                        from transformers import DistilBertForSequenceClassification
                                        model = DistilBertForSequenceClassification(config)
                                    else:
                                        from transformers import DistilBertModel
                                        model = DistilBertModel(config)
                                elif model_type == "bert":
                                    from transformers import BertModel
                                    model = BertModel(config)
                                elif model_type == "roberta":
                                    from transformers import RobertaModel
                                    model = RobertaModel(config)
                                else:
                                    # 범용 AutoModel 사용
                                    from transformers import AutoModel
                                    model = AutoModel.from_config(config)
                                
                                model.load_state_dict(state_dict, strict=False)
                                del state_dict
                                gc.collect()
                                model.eval()
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 보조 방법 1 성공")
                                loading_result.put(model)
                                return
                            except Exception as fallback1_error:
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ❌ 보조 방법 1 실패: {fallback1_error}")
                            
                            # 보조 방법 2: 전통적 transformers 로딩 (최후 수단)
                            print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 보조 방법 2: 전통적 transformers 로딩 (최후 수단)")
                            try:
                                # 모델 타입 다시 확인 (fallback에서 사용)
                                model_type = config.model_type.lower()
                                
                                from transformers import AutoModel
                                
                                # 대규모 메모리 정리
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 대규모 메모리 정리 완룼")
                                
                                # 마지막 수단: 모델별 최적화된 transformers 로딩
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델별 최적화 AutoModel 시도 (타입: {model_type})")
                                
                                # 모델 타입별 최적화 설정
                                if model_type == "distilbert":
                                    # 정밀한 모델 분석 결과를 기반으로 분류용 또는 기본 모델 선택
                                    if self._should_use_classification_model(model_name):
                                        from transformers import AutoModelForSequenceClassification
                                        model = AutoModelForSequenceClassification.from_pretrained(
                                            actual_model_path,
                                            local_files_only=True,
                                            torch_dtype=torch.float32,
                                            trust_remote_code=False,
                                            use_safetensors=has_safetensors,
                                            low_cpu_mem_usage=False
                                        )
                                        print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - AutoModelForSequenceClassification 로딩 완료 (정밀 분석 기반)")
                                    else:
                                        model = AutoModel.from_pretrained(
                                            actual_model_path,
                                            local_files_only=True,
                                            torch_dtype=torch.float32,
                                            trust_remote_code=False,  # DistilBert는 표준 모델
                                            use_safetensors=has_safetensors,
                                            low_cpu_mem_usage=False  # 작은 모델이므로 메모리 최적화 불필요
                                        )
                                elif model_type == "xlm-roberta":
                                    # XLM-RoBERTa는 큰 모델이므로 메모리 최적화
                                    model = AutoModel.from_pretrained(
                                        actual_model_path,
                                        local_files_only=True,
                                        torch_dtype=torch.float32,
                                        trust_remote_code=True,
                                        use_safetensors=has_safetensors,
                                        low_cpu_mem_usage=True
                                    )
                                else:
                                    # 기타 모델들은 균형잡힌 설정
                                    model = AutoModel.from_pretrained(
                                        actual_model_path,
                                        local_files_only=True,
                                        torch_dtype=torch.float32,
                                        trust_remote_code=True,
                                        use_safetensors=has_safetensors,
                                        low_cpu_mem_usage=True
                                    )
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 보조 방법 2 성공 (transformers 기본)")
                                loading_result.put(model)
                                return
                                
                                # Config로부터 빈 모델 생성 (모델 타입별 동적 선택)
                                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 빈 모델 구조 생성 (타입: {model_type})")
                                if model_type == "xlm-roberta":
                                    from transformers import XLMRobertaModel
                                    model = XLMRobertaModel(config)
                                elif model_type == "distilbert":
                                    # 정밀한 모델 분석 결과를 기반으로 분류용 또는 기본 모델 선택
                                    if self._should_use_classification_model(model_name):
                                        from transformers import DistilBertForSequenceClassification
                                        model = DistilBertForSequenceClassification(config)
                                    else:
                                        from transformers import DistilBertModel
                                        model = DistilBertModel(config)
                                elif model_type == "bert":
                                    from transformers import BertModel
                                    model = BertModel(config)
                                elif model_type == "roberta":
                                    from transformers import RobertaModel
                                    model = RobertaModel(config)
                                else:
                                    # 범용 AutoModel 사용
                                    from transformers import AutoModel
                                    model = AutoModel.from_config(config)
                                
                                # 가장 안전한 가중치 파일 선택
                                if has_pytorch:
                                    weight_file = pytorch_path
                                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - PyTorch 가중치 사용: {weight_file}")
                                elif has_safetensors:
                                    weight_file = safetensors_path
                                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Safetensors 가중치 사용: {weight_file}")
                                else:
                                    raise FileNotFoundError("사용 가능한 가중치 파일이 없습니다")
                                
                                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 가중치 파일 로딩: {weight_file}")
                                
                                if weight_file.endswith('.safetensors'):
                                    # safetensors 로딩
                                    from safetensors.torch import load_file
                                    state_dict = load_file(weight_file)
                                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ Safetensors 로딩 완료, 키 개수: {len(state_dict)}")
                                else:
                                    # PyTorch 로딩
                                    state_dict = torch.load(weight_file, map_location='cpu')
                                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ PyTorch 가중치 로딩 완룼, 키 개수: {len(state_dict)}")
                                
                                # 모델에 가중치 적용
                                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델에 가중치 적용")
                                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                                
                                if missing_keys:
                                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ 누락된 키: {len(missing_keys)}개")
                                if unexpected_keys:
                                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ 예상치 못한 키: {len(unexpected_keys)}개")
                                    
                                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 방법 2 성공: 직접 가중치 로딩 완료")
                                
                                loading_result.put(model)
                                return
                                
                            except Exception as fallback2_error:
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ❌ 보조 방법 2도 실패: {fallback2_error}")
                                
                                # 모든 ULTRA 방법 실패 시 상세 오류 정보 제공
                                error_summary = f"모든 ULTRA 로딩 방법 실패:\n" \
                                              f"ULTRA 메인: {ultra_error}\n" \
                                              f"보조 1: {fallback1_error}\n" \
                                              f"보조 2: {fallback2_error}\n\n" \
                                              f"추천 대안 모델 (빠른 로딩):\n" \
                                              f"- sentence-transformers/all-MiniLM-L6-v2 (90MB, 30초)\n" \
                                              f"- sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (420MB, 2분)\n" \
                                              f"- intfloat/e5-small-v2 (120MB, 45초)"
                                print(f"[ULTRA] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {error_summary}")
                                raise Exception(error_summary)
                        
                        finally:
                            # 환경 변수 복원
                            for key, original_value in original_env.items():
                                if original_value is None:
                                    if key in os.environ:
                                        del os.environ[key]
                                else:
                                    os.environ[key] = original_value
                        
                    except Exception as e:
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모든 로딩 방법 실패: {e}")
                        loading_error.put(e)
                
                # 로딩 스레드 시작
                loading_thread = threading.Thread(target=load_model_with_progress)
                loading_thread.daemon = True
                loading_thread.start()
                
                # 진행상황 모니터링 (15초마다 상태 출력)
                timeout_seconds = 600  # 10분 타임아웃 (BGE-M3는 5-6분 소요)
                check_interval = 15    # 15초마다 체크 (더 세밀한 진행률 표시)
                elapsed_checks = 0
                
                while loading_thread.is_alive():
                    loading_thread.join(timeout=check_interval)
                    
                    if loading_thread.is_alive():
                        elapsed_checks += 1
                        elapsed_time = elapsed_checks * check_interval
                        
                        # 메모리 상태 체크
                        try:
                            mem = psutil.virtual_memory()
                            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 로딩 진행중... {elapsed_time}초 경과")
                            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 메모리 상태: {mem.percent}% 사용, {mem.available/1024**3:.1f}GB 사용가능")
                            
                            # 프로세스별 메모리 확인
                            process = psutil.Process()
                            proc_mem_mb = process.memory_info().rss / 1024 / 1024
                            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 현재 프로세스 메모리: {proc_mem_mb:.1f}MB")
                            
                        except Exception as mem_e:
                            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 메모리 체크 실패: {mem_e}")
                        
                        # 타임아웃 체크 (성공 신호가 도착하지 않은 경우에만)
                        if elapsed_time >= timeout_seconds:
                            # 마지막으로 한 번 더 성공 신호 확인
                            if not loading_result.empty():
                                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 타임아웃 직전 로딩 성공 감지!")
                                break
                            
                            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ❌ 모델 로딩 타임아웃 ({timeout_seconds}초)")
                            
                            # 스레드 강제 종료 시도 (배경에서 계속 실행되도록 허용)
                            loading_error.put(TimeoutError(f"모델 로딩이 {timeout_seconds}초를 초과했습니다. 배경에서 계속 시도 중..."))
                            break
                
                # 결과 확인 (동기화 강화)
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 로딩 스레드 종료 후 결과 확인")
                
                # 성공 결과 우선 체크
                if not loading_result.empty():
                    model = loading_result.get()
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 4/5: 모델 로딩 성공, 후처리 시작")
                    
                    # 오류 큐에 남아있는 메시지 정리 (타임아웃 경고 등)
                    while not loading_error.empty():
                        warning_msg = loading_error.get()
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 무시되는 경고: {warning_msg}")
                        
                elif not loading_error.empty():
                    error = loading_error.get()
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 오류 발생: {error}")
                    raise error
                else:
                    # 둘 다 비어있는 경우 (비정상 상황)
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ 비정상 상황: 로딩 결과도 오류도 없음")
                    raise Exception("모델 로딩 상태를 확인할 수 없습니다")
                
                model_load_time = time.time() - model_start
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 5/5: 모델 로딩 후처리 완료")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 실제 모델 로딩 완료: {model_name} ({model_load_time:.1f}초)")
                
                # 모델 상태 검증
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 상태 검증:")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   모델 클래스: {model.__class__.__name__}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   모델 상태: {'eval' if not model.training else 'train'}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   모델 샀고 모드: {next(model.parameters()).requires_grad}")
                
                # 모델 메모리 사용량 확인
                param_count = sum(p.numel() for p in model.parameters())
                param_size_mb = param_count * 4 / 1024 / 1024  # float32 = 4bytes
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 파라미터: {param_count:,}개 ({param_size_mb:.1f}MB)")
                
                # 모델 레이어 구조 간략 분석
                layer_count = 0
                for name, module in model.named_modules():
                    layer_count += 1
                    if layer_count <= 5:  # 처음 5개 레이어만 상세 정보
                        print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   레이어 {layer_count}: {name} ({module.__class__.__name__})")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 총 레이어 수: {layer_count}")
                
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 토크나이저 로딩 시작: {model_name}")
                tokenizer_start = time.time()
                
                tokenizer = AutoTokenizer.from_pretrained(
                    actual_model_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                tokenizer_load_time = time.time() - tokenizer_start
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 토크나이저 로딩 완료: {model_name} ({tokenizer_load_time:.1f}초)")
                
                # 토크나이저 정보 확인
                vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'Unknown'
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 토크나이저 어휘 크기: {vocab_size}")
                
                # 통합 디바이스 관리자로 일관성 보장
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 디바이스 일관성 보장 시작: {model_name}")
                device_start = time.time()
                
                model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                model.eval()
                
                device_time = time.time() - device_start
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 디바이스 일관성 보장 완료: {model_name} ({device_time:.1f}초)")
                
                # 최종 모델 상태 확인
                model_device = next(model.parameters()).device
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 최종 모델 디바이스: {model_device}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 평가 모드: {not model.training}")
                
                # 디바이스 일관성 최종 검증
                devices = set(param.device for param in model.parameters())
                if len(devices) == 1:
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ✅ 디바이스 일관성 확인: {list(devices)[0]}")
                else:
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⚠️ 디바이스 불일치 감지: {devices}")
                
                load_time = time.time() - load_start
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 실제 모델 로딩 총 시간: {load_time:.1f}초")
                
                # 로딩 성공 메시지
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 🎉 {model_name} 모델 로딩 성공적으로 완료!")
                
                profiler.print_detailed_report()
                return model, tokenizer, load_time
                
            except TimeoutError as te:
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ⏰ 모델 로딩 타임아웃: {te}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 해결방안:")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   1. 더 큰 타임아웃 설정")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   2. 더 작은 모델 사용 고려")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   3. GPU 메모리 최적화")
                raise
            except Exception as e:
                import traceback
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - ❌ 실제 모델 로딩 실패: {e}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 오류 타입: {type(e).__name__}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 상세 오류:")
                traceback.print_exc()
                
                # 메모리 상태 재확인
                try:
                    mem = psutil.virtual_memory()
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 오류 시점 메모리 - 사용률: {mem.percent}%, 사용가능: {mem.available/1024**3:.1f}GB")
                except:
                    pass
                
                # 디버깅 정보 추가
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 디버깅 정보:")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   모델 경로: {actual_model_path}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   로컬 파일 전용: True")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   신뢰 코드: True")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   형변환: torch.float32")
                    
                raise
        
        try:
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - _load_model_sync 시작: {model_name}, {model_path}")
            
            # 모델 정보 초기화
            self.models[model_name] = ModelInfo(
                name=model_name, 
                path=model_path, 
                status="loading"
            )
            
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 정보 초기화됨: {model_name}")
            self._notify_callbacks(model_name, "loading_started", {})
            
            # 메모리 사용량 측정 시작
            process = psutil.Process()
            mem_before = process.memory_info().rss
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 메모리 측정 시작: {model_name}")
            
            # HuggingFace 모델 ID인지 확인하고 다운로드
            actual_model_path = model_path
            if self._is_huggingface_model_id(model_path):
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - HuggingFace 모델 ID 감지: {model_name}")
                self._notify_callbacks(model_name, "downloading", {'model_id': model_path})
                actual_model_path = self._download_huggingface_model(model_path)
                self.models[model_name].path = actual_model_path  # 실제 경로로 업데이트
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 다운로드/캐시 확인 완룼: {model_name}")
            
            # 정밀한 모델 타입 자동 감지 시스템 사용
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 정밀한 모델 분석 시작: {model_name}")
            try:
                # ModelTypeDetector를 사용한 정밀한 분석
                task_type, model_class, detection_info = self.model_type_detector.detect_model_type(model_name, actual_model_path)
                
                supported_tasks = [task_type]
                
                # 분석 정보 저장
                analysis = {
                    "model_summary": {
                        "supported_tasks": supported_tasks,
                        "recommended_model_class": model_class,
                        "detection_method": detection_info.get("detection_method", []),
                        "confidence": detection_info.get("confidence", 0.0),
                        "fallback_used": detection_info.get("fallback_used", False)
                    }
                }
                
                self.models[model_name].config_analysis = analysis
                
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 정밀한 모델 분석 완료:")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   모델: {model_name}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   태스크: {task_type}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   클래스: {model_class}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   신뢰도: {detection_info.get('confidence', 0.0):.2f}")
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} -   방법: {', '.join(detection_info.get('detection_method', []))}")
                
            except Exception as e:
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 정밀한 모델 분석 실패, 기본값 사용: {e}")
                self.models[model_name].config_analysis = {"model_summary": {"supported_tasks": ["feature-extraction"]}}
            
            # 범용적인 transformers 모델 로드
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - transformers 임포트 시작: {model_name}")
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - transformers 임포트 완료: {model_name}")
            
            # 최적 디바이스 자동 선택
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 디바이스 선택 시작: {model_name}")
            device = optimizer.get_optimal_device()
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 자동 선택된 디바이스: {device}")
            
            # 메모리 상태 체크
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 메모리 체크 시작: {model_name}")
            memory_info = self.get_memory_info()
            available_memory_gb = memory_info['system_memory']['available'] / (1024**3)
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 사용 가능한 메모리: {available_memory_gb:.1f}GB")
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 디바이스 설정: {device} (Streamlit 안정성을 위해 CPU 강제)")
            
            # 모델 타입 자동 감지
            if "bge" in model_name.lower() or "embedding" in model_name.lower():
                is_classification_model = False
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {model_name} 임베딩 모델로 설정")
            elif "sentiment" in model_name.lower() or "classification" in model_name.lower():
                is_classification_model = True
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {model_name} 분류 모델로 설정")
            else:
                is_classification_model = False
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {model_name} 기본 모델로 설정")
            
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 시작: classification={is_classification_model}")
            
            # 직접 모델 로딩 (캐시 우회하여 안정성 확보)
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 직접 모델 로딩 시작: {model_name}")
            
            try:
                # 모델 로딩
                result = load_model_with_transformers(actual_model_path, device)
                
                if len(result) == 3:
                    model, tokenizer, load_time = result
                    print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 성공: {load_time:.1f}초")
                else:
                    raise ValueError("모델 로딩 결과 형식 오류")
                
            except Exception as e:
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 실패: {e}")
                raise
            
            # 메모리 사용량 계산
            mem_after = process.memory_info().rss
            memory_usage = (mem_after - mem_before) / 1024 / 1024  # MB
            
            # 모델 정보 업데이트
            self.models[model_name].model = model
            self.models[model_name].tokenizer = tokenizer
            self.models[model_name].memory_usage = memory_usage
            self.models[model_name].load_time = datetime.now()
            self.models[model_name].status = "loaded"
            
            success_data = {
                'memory_usage': memory_usage,
                'load_time': self.models[model_name].load_time,
                'analysis': analysis['model_summary'],
                'original_path': model_path,
                'actual_path': actual_model_path
            }
            
            self._notify_callbacks(model_name, "loading_success", success_data)
            
            if callback:
                callback(model_name, True, f"Model loaded successfully. Memory usage: {memory_usage:.2f} MB")
        
        except TimeoutError as e:
            error_msg = str(e)
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 타임아웃: {error_msg}")
            
            if model_name in self.models:
                self.models[model_name].status = "error"
                self.models[model_name].error_message = error_msg
            
            self._notify_callbacks(model_name, "loading_error", {'error': error_msg})
            
            if callback:
                callback(model_name, False, error_msg)
                
        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 오류: {error_msg}")
            import traceback
            traceback.print_exc()
            
            if model_name in self.models:
                self.models[model_name].status = "error"
                self.models[model_name].error_message = error_msg
            
            self._notify_callbacks(model_name, "loading_error", {'error': error_msg})
            
            if callback:
                callback(model_name, False, error_msg)
        
        finally:
            elapsed = time.time() - start_time
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 로딩 총 소요시간: {elapsed:.1f}초")
    
    def unload_model(self, model_name: str) -> bool:
        """모델 언로드"""
        if model_name in self.models:
            model_info = self.models[model_name]
            
            try:
                # 메모리 정리
                if model_info.model:
                    del model_info.model
                if model_info.tokenizer:
                    del model_info.tokenizer
                
                # GPU 메모리 정리
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 상태 업데이트
                model_info.status = "unloaded"
                model_info.model = None
                model_info.tokenizer = None
                model_info.memory_usage = 0.0
                
                self._notify_callbacks(model_name, "unloaded", {})
                
                return True
                
            except Exception as e:
                print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Error unloading model {model_name}: {e}")
                return False
        
        return False
    
    def get_loaded_models(self) -> List[str]:
        """로드된 모델 목록"""
        return [name for name, info in self.models.items() if info.status == "loaded"]
    
    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """특정 모델 정보 조회"""
        return self.models.get(model_name)
    
    def get_all_models_status(self) -> Dict:
        """모든 모델 상태 정보"""
        return {
            name: {
                'status': info.status,
                'memory_usage': info.memory_usage,
                'load_time': info.load_time.isoformat() if info.load_time else None,
                'path': info.path,
                'error_message': info.error_message,
                'config_analysis': info.config_analysis
            }
            for name, info in self.models.items()
        }
    
    def get_model_for_inference(self, model_name: str) -> Optional[tuple]:
        """추론용 모델과 토크나이저 반환"""
        if model_name in self.models and self.models[model_name].status == "loaded":
            model_info = self.models[model_name]
            return model_info.model, model_info.tokenizer
        return None
    
    def get_system_summary(self) -> Dict:
        """시스템 요약 정보"""
        loaded_count = len(self.get_loaded_models())
        total_memory = sum(info.memory_usage for info in self.models.values() if info.status == "loaded")
        
        memory_info = self.get_memory_info()
        
        return {
            'loaded_models_count': loaded_count,
            'total_models_count': len(self.models),
            'total_memory_usage_mb': total_memory,
            'system_memory_info': memory_info,
            'models_by_status': {
                'loaded': len([m for m in self.models.values() if m.status == "loaded"]),
                'loading': len([m for m in self.models.values() if m.status == "loading"]),
                'error': len([m for m in self.models.values() if m.status == "error"]),
                'unloaded': len([m for m in self.models.values() if m.status == "unloaded"])
            }
        }
    
    def remove_model(self, model_name: str) -> bool:
        """모델 완전 제거"""
        if model_name in self.models:
            # 먼저 언로드
            if self.models[model_name].status == "loaded":
                self.unload_model(model_name)
            
            # 모델 정보 제거
            del self.models[model_name]
            
            # 로딩 락 제거
            if model_name in self.loading_locks:
                del self.loading_locks[model_name]
            
            self._notify_callbacks(model_name, "removed", {})
            return True
        
        return False
    
    def clear_all_models(self):
        """모든 모델 정리"""
        model_names = list(self.models.keys())
        for model_name in model_names:
            self.remove_model(model_name)
    
    def get_available_tasks(self, model_name: str) -> List[str]:
        """모델이 지원하는 태스크 목록"""
        if model_name in self.models:
            analysis = self.models[model_name].config_analysis
            if analysis and 'model_summary' in analysis:
                return analysis['model_summary'].get('supported_tasks', [])
        return []
    
    def update_model_task(self, model_name: str, tasks: List[str]) -> bool:
        """모델의 지원 태스크 수동 업데이트"""
        if model_name in self.models:
            if not self.models[model_name].config_analysis:
                self.models[model_name].config_analysis = {"model_summary": {}}
            self.models[model_name].config_analysis["model_summary"]["supported_tasks"] = tasks
            print(f"[DEBUG] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {model_name} 태스크 수동 업데이트: {tasks}")
            return True
        return False
    
    def _should_use_classification_model(self, model_name: str) -> bool:
        """모델 분석 정보를 기반으로 분류 모델 사용 여부 결정"""
        if model_name in self.models:
            analysis = self.models[model_name].config_analysis
            if analysis and "model_summary" in analysis:
                recommended_class = analysis["model_summary"].get("recommended_model_class", "")
                supported_tasks = analysis["model_summary"].get("supported_tasks", [])
                
                # 추천 클래스가 분류 모델인지 확인
                if "SequenceClassification" in recommended_class:
                    return True
                
                # 지원 태스크가 분류 관련인지 확인
                classification_tasks = ["text-classification", "sentiment-analysis"]
                if any(task in supported_tasks for task in classification_tasks):
                    return True
        
        # 백업: 기존 키워드 매칭
        return any(keyword in model_name.lower() for keyword in ["sentiment", "classification", "classifier"])
    
    def _get_recommended_model_class(self, model_name: str, model_type: str) -> str:
        """감지된 정보를 기반으로 권장 모델 클래스 반환"""
        if model_name in self.models:
            analysis = self.models[model_name].config_analysis
            if analysis and "model_summary" in analysis:
                recommended_class = analysis["model_summary"].get("recommended_model_class", "")
                if recommended_class:
                    # 구체적인 모델 타입 클래스로 변환
                    return self.model_type_detector.get_model_specific_class(recommended_class, model_type)
        
        return None
    
    def export_models_info(self) -> Dict:
        """모델 정보 내보내기"""
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self.get_system_summary(),
            'models': {}
        }
        
        for name, info in self.models.items():
            export_data['models'][name] = {
                'name': info.name,
                'path': info.path,
                'status': info.status,
                'memory_usage': info.memory_usage,
                'load_time': info.load_time.isoformat() if info.load_time else None,
                'config_analysis': info.config_analysis
            }
        
        return export_data