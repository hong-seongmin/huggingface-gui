import threading
import psutil
import torch
import os
import re
import hashlib
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from model_analyzer import ComprehensiveModelAnalyzer
from model_optimization import optimizer
from model_cache import model_cache
from device_manager import device_manager
from detailed_profiler import profiler
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
                print(f"Callback error: {e}")
    
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
                print(f"[DEBUG] 캐시에서 모델 찾음: {model_id}")
            except Exception:
                # 캐시에 없는 경우에만 다운로드
                print(f"[DEBUG] 캐시 미스 - 다운로드 시작: {model_id}")
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
        
        print(f"[DEBUG] load_model_async 시작: {model_name}, {model_path}")
        
        # 로딩 락 설정
        if model_name not in self.loading_locks:
            self.loading_locks[model_name] = threading.Lock()
        
        # 스레드 시작
        print(f"[DEBUG] 스레드 생성 중: {model_name}")
        thread = threading.Thread(
            target=self._load_model_sync, 
            args=(model_name, model_path, callback),
            name=f"ModelLoad-{model_name}"
        )
        thread.daemon = True
        print(f"[DEBUG] 스레드 시작 전: {model_name}")
        thread.start()
        print(f"[DEBUG] 스레드 시작됨: {model_name}, thread={thread}")
        
        return thread
    
    def _load_model_sync(self, model_name: str, model_path: str, callback: Optional[Callable] = None):
        """실제 모델 로딩 작업 (스레드에서 실행)"""
        import time
        import threading
        import queue
        
        start_time = time.time()
        
        def load_model_with_transformers(actual_model_path, device):
            """Fast 모델 로딩"""
            print(f"[FAST] 모델 로딩 시작")
            
            # 프로파일링 시작 (프로파일러 내부에서 활성화 여부 확인)
            profiler.start_profiling("모델 로딩")
            profiler.memory_snapshot("초기 상태")
            
            # 직접 transformers 라이브러리 사용으로 실제 BGE-M3 모델 로딩
            print("[DEBUG] 실제 transformers 모델 로딩 시작")
            
            import time
            load_start = time.time()
            
            try:
                from transformers import AutoModel, AutoTokenizer, AutoConfig
                
                print(f"[DEBUG] Config 로딩 시작: {model_name}")
                
                # 빠른 로컬 config 확인
                try:
                    import json
                    config_path = os.path.join(actual_model_path, "config.json")
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as f:
                            config_dict = json.load(f)
                        print(f"[DEBUG] 로컬 config 로딩 완료: {model_name}")
                    else:
                        config = AutoConfig.from_pretrained(actual_model_path, local_files_only=True)
                        print(f"[DEBUG] AutoConfig 로딩 완료: {model_name}")
                except Exception as e:
                    print(f"[DEBUG] Config 로딩 오류, 기본값 사용: {e}")
                
                print(f"[DEBUG] 실제 모델 로딩 시작: {model_name}")
                
                # 모델 파일 존재 여부 확인
                print(f"[DEBUG] 모델 파일 확인 중: {actual_model_path}")
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
                        print(f"[DEBUG] ✅ {file}: {size_mb:.1f}MB")
                    else:
                        print(f"[DEBUG] ❌ {file}: 파일 없음")
                
                # 메모리 상태 확인
                import psutil
                mem = psutil.virtual_memory()
                print(f"[DEBUG] 메모리 상태 - 사용률: {mem.percent}%, 사용가능: {mem.available/1024**3:.1f}GB")
                
                print(f"[DEBUG] AutoModel.from_pretrained 호출 시작 (큰 모델이므로 시간 소요 예상)")
                
                # 환경 변수 상태 확인
                print(f"[DEBUG] 환경 변수 확인:")
                env_vars = {
                    'HF_HUB_OFFLINE': os.getenv('HF_HUB_OFFLINE', 'None'),
                    'TRANSFORMERS_OFFLINE': os.getenv('TRANSFORMERS_OFFLINE', 'None'),
                    'HF_HUB_DISABLE_TELEMETRY': os.getenv('HF_HUB_DISABLE_TELEMETRY', 'None'),
                    'TOKENIZERS_PARALLELISM': os.getenv('TOKENIZERS_PARALLELISM', 'None')
                }
                for key, value in env_vars.items():
                    print(f"[DEBUG]   {key}={value}")
                
                model_start = time.time()
                
                # AutoModel 로딩을 단계별로 분할하여 진행 상태 추적
                print(f"[DEBUG] 1/5: transformers AutoModel 임포트 확인")
                from transformers import AutoModel
                print(f"[DEBUG] 2/5: AutoConfig 사전 로딩")
                
                # Config 먼저 로딩하여 모델 구조 확인
                try:
                    from transformers import AutoConfig
                    print(f"[DEBUG] Config 로딩 시도: {actual_model_path}")
                    config = AutoConfig.from_pretrained(
                        actual_model_path,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                    print(f"[DEBUG] ✅ Config 로딩 성공: {config.__class__.__name__}")
                    print(f"[DEBUG] 모델 타입: {getattr(config, 'model_type', 'Unknown')}")
                    print(f"[DEBUG] 어휘 크기: {getattr(config, 'vocab_size', 'Unknown')}")
                    print(f"[DEBUG] 숨겨진 크기: {getattr(config, 'hidden_size', 'Unknown')}")
                except Exception as config_e:
                    print(f"[DEBUG] ⚠️ Config 로딩 실패, 계속 진행: {config_e}")
                
                print(f"[DEBUG] 3/5: 실제 모델 가중치 로딩 시작 (가장 시간 소요 단계)")
                
                # 로딩 타임아웃 및 진행상황 모니터링을 위한 스레드 생성
                import threading
                import queue
                
                loading_result = queue.Queue()
                loading_error = queue.Queue()
                
                def load_model_with_progress():
                    """별도 스레드에서 모델 로딩 수행"""
                    try:
                        print(f"[DEBUG] 모델 로딩 스레드 시작")
                        
                        # BGE-M3는 embedding 모델이므로 AutoModel 사용
                        model = AutoModel.from_pretrained(
                            actual_model_path, 
                            local_files_only=True,
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                        
                        print(f"[DEBUG] 모델 로딩 스레드 완료")
                        loading_result.put(model)
                        
                    except Exception as e:
                        print(f"[DEBUG] 모델 로딩 스레드 오류: {e}")
                        loading_error.put(e)
                
                # 로딩 스레드 시작
                loading_thread = threading.Thread(target=load_model_with_progress)
                loading_thread.daemon = True
                loading_thread.start()
                
                # 진행상황 모니터링 (30초마다 상태 출력)
                timeout_seconds = 300  # 5분 타임아웃
                check_interval = 30    # 30초마다 체크
                elapsed_checks = 0
                
                while loading_thread.is_alive():
                    loading_thread.join(timeout=check_interval)
                    
                    if loading_thread.is_alive():
                        elapsed_checks += 1
                        elapsed_time = elapsed_checks * check_interval
                        
                        # 메모리 상태 체크
                        try:
                            mem = psutil.virtual_memory()
                            print(f"[DEBUG] 로딩 진행중... {elapsed_time}초 경과")
                            print(f"[DEBUG] 메모리 상태: {mem.percent}% 사용, {mem.available/1024**3:.1f}GB 사용가능")
                            
                            # 프로세스별 메모리 확인
                            process = psutil.Process()
                            proc_mem_mb = process.memory_info().rss / 1024 / 1024
                            print(f"[DEBUG] 현재 프로세스 메모리: {proc_mem_mb:.1f}MB")
                            
                        except Exception as mem_e:
                            print(f"[DEBUG] 메모리 체크 실패: {mem_e}")
                        
                        # 타임아웃 체크
                        if elapsed_time >= timeout_seconds:
                            print(f"[DEBUG] ❌ 모델 로딩 타임아웃 ({timeout_seconds}초)")
                            loading_error.put(TimeoutError(f"모델 로딩이 {timeout_seconds}초를 초과했습니다"))
                            break
                
                # 결과 확인
                if not loading_error.empty():
                    error = loading_error.get()
                    raise error
                
                if not loading_result.empty():
                    model = loading_result.get()
                    print(f"[DEBUG] 4/5: 모델 로딩 완료, 후처리 시작")
                else:
                    raise Exception("모델 로딩이 완료되지 않았습니다")
                
                model_load_time = time.time() - model_start
                print(f"[DEBUG] 5/5: 모델 로딩 후처리 완료")
                print(f"[DEBUG] ✅ 실제 모델 로딩 완료: {model_name} ({model_load_time:.1f}초)")
                
                # 모델 상태 검증
                print(f"[DEBUG] 모델 상태 검증:")
                print(f"[DEBUG]   모델 클래스: {model.__class__.__name__}")
                print(f"[DEBUG]   모델 상태: {'eval' if not model.training else 'train'}")
                print(f"[DEBUG]   모델 샀고 모드: {next(model.parameters()).requires_grad}")
                
                # 모델 메모리 사용량 확인
                param_count = sum(p.numel() for p in model.parameters())
                param_size_mb = param_count * 4 / 1024 / 1024  # float32 = 4bytes
                print(f"[DEBUG] 모델 파라미터: {param_count:,}개 ({param_size_mb:.1f}MB)")
                
                # 모델 레이어 구조 간략 분석
                layer_count = 0
                for name, module in model.named_modules():
                    layer_count += 1
                    if layer_count <= 5:  # 처음 5개 레이어만 상세 정보
                        print(f"[DEBUG]   레이어 {layer_count}: {name} ({module.__class__.__name__})")
                print(f"[DEBUG] 총 레이어 수: {layer_count}")
                
                print(f"[DEBUG] 토크나이저 로딩 시작: {model_name}")
                tokenizer_start = time.time()
                
                tokenizer = AutoTokenizer.from_pretrained(
                    actual_model_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
                
                tokenizer_load_time = time.time() - tokenizer_start
                print(f"[DEBUG] ✅ 토크나이저 로딩 완료: {model_name} ({tokenizer_load_time:.1f}초)")
                
                # 토크나이저 정보 확인
                vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'Unknown'
                print(f"[DEBUG] 토크나이저 어휘 크기: {vocab_size}")
                
                # 통합 디바이스 관리자로 일관성 보장
                print(f"[DEBUG] 디바이스 일관성 보장 시작: {model_name}")
                device_start = time.time()
                
                model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                model.eval()
                
                device_time = time.time() - device_start
                print(f"[DEBUG] ✅ 디바이스 일관성 보장 완료: {model_name} ({device_time:.1f}초)")
                
                # 최종 모델 상태 확인
                model_device = next(model.parameters()).device
                print(f"[DEBUG] 최종 모델 디바이스: {model_device}")
                print(f"[DEBUG] 모델 평가 모드: {not model.training}")
                
                # 디바이스 일관성 최종 검증
                devices = set(param.device for param in model.parameters())
                if len(devices) == 1:
                    print(f"[DEBUG] ✅ 디바이스 일관성 확인: {list(devices)[0]}")
                else:
                    print(f"[DEBUG] ⚠️ 디바이스 불일치 감지: {devices}")
                
                load_time = time.time() - load_start
                print(f"[DEBUG] 실제 모델 로딩 총 시간: {load_time:.1f}초")
                
                # 로딩 성공 메시지
                print(f"[DEBUG] 🎉 BGE-M3 모델 로딩 성공적으로 완료!")
                
                profiler.print_detailed_report()
                return model, tokenizer, load_time
                
            except TimeoutError as te:
                print(f"[DEBUG] ⏰ 모델 로딩 타임아웃: {te}")
                print(f"[DEBUG] 해결방안:")
                print(f"[DEBUG]   1. 더 큰 타임아웃 설정")
                print(f"[DEBUG]   2. 더 작은 모델 사용 고려")
                print(f"[DEBUG]   3. GPU 메모리 최적화")
                raise
            except Exception as e:
                import traceback
                print(f"[DEBUG] ❌ 실제 모델 로딩 실패: {e}")
                print(f"[DEBUG] 오류 타입: {type(e).__name__}")
                print(f"[DEBUG] 상세 오류:")
                traceback.print_exc()
                
                # 메모리 상태 재확인
                try:
                    mem = psutil.virtual_memory()
                    print(f"[DEBUG] 오류 시점 메모리 - 사용률: {mem.percent}%, 사용가능: {mem.available/1024**3:.1f}GB")
                except:
                    pass
                
                # 디버깅 정보 추가
                print(f"[DEBUG] 디버깅 정보:")
                print(f"[DEBUG]   모델 경로: {actual_model_path}")
                print(f"[DEBUG]   로컬 파일 전용: True")
                print(f"[DEBUG]   신뢰 코드: True")
                print(f"[DEBUG]   형변환: torch.float32")
                    
                raise
        
        try:
            print(f"[DEBUG] _load_model_sync 시작: {model_name}, {model_path}")
            
            # 모델 정보 초기화
            self.models[model_name] = ModelInfo(
                name=model_name, 
                path=model_path, 
                status="loading"
            )
            
            print(f"[DEBUG] 모델 정보 초기화됨: {model_name}")
            self._notify_callbacks(model_name, "loading_started", {})
            
            # 메모리 사용량 측정 시작
            process = psutil.Process()
            mem_before = process.memory_info().rss
            print(f"[DEBUG] 메모리 측정 시작: {model_name}")
            
            # HuggingFace 모델 ID인지 확인하고 다운로드
            actual_model_path = model_path
            if self._is_huggingface_model_id(model_path):
                print(f"[DEBUG] HuggingFace 모델 ID 감지: {model_name}")
                self._notify_callbacks(model_name, "downloading", {'model_id': model_path})
                actual_model_path = self._download_huggingface_model(model_path)
                self.models[model_name].path = actual_model_path  # 실제 경로로 업데이트
                print(f"[DEBUG] 모델 다운로드/캐시 확인 완료: {model_name}")
            
            # 모델 분석 - 성능상 이유로 간소화
            print(f"[DEBUG] 모델 분석 시작: {model_name}")
            try:
                # 빠른 기본 분석만 수행 (전체 분석은 스킵)
                analysis = {"model_summary": {"supported_tasks": ["feature-extraction"]}}
                self.models[model_name].config_analysis = analysis
                print(f"[DEBUG] 모델 분석 완료 (간소화): {model_name}")
            except Exception as e:
                print(f"[DEBUG] 모델 분석 실패, 기본값 사용: {e}")
                self.models[model_name].config_analysis = {"model_summary": {"supported_tasks": ["feature-extraction"]}}
            
            # 범용적인 transformers 모델 로드
            print(f"[DEBUG] transformers 임포트 시작: {model_name}")
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
            print(f"[DEBUG] transformers 임포트 완료: {model_name}")
            
            # 최적 디바이스 자동 선택
            print(f"[DEBUG] 디바이스 선택 시작: {model_name}")
            device = optimizer.get_optimal_device()
            print(f"[DEBUG] 자동 선택된 디바이스: {device}")
            
            # 메모리 상태 체크
            print(f"[DEBUG] 메모리 체크 시작: {model_name}")
            memory_info = self.get_memory_info()
            available_memory_gb = memory_info['system_memory']['available'] / (1024**3)
            print(f"[DEBUG] 사용 가능한 메모리: {available_memory_gb:.1f}GB")
            print(f"[DEBUG] 디바이스 설정: {device} (Streamlit 안정성을 위해 CPU 강제)")
            
            # BGE-M3는 임베딩 모델이므로 분류 모델이 아님
            is_classification_model = False
            print(f"[DEBUG] BGE-M3 임베딩 모델로 설정")
            
            print(f"[DEBUG] 모델 로딩 시작: classification={is_classification_model}")
            
            # 직접 모델 로딩 (캐시 우회하여 안정성 확보)
            print(f"[DEBUG] 직접 모델 로딩 시작: {model_name}")
            
            try:
                # 모델 로딩
                result = load_model_with_transformers(actual_model_path, device)
                
                if len(result) == 3:
                    model, tokenizer, load_time = result
                    print(f"[DEBUG] 모델 로딩 성공: {load_time:.1f}초")
                else:
                    raise ValueError("모델 로딩 결과 형식 오류")
                
            except Exception as e:
                print(f"[DEBUG] 모델 로딩 실패: {e}")
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
            print(f"[DEBUG] 모델 로딩 타임아웃: {error_msg}")
            
            if model_name in self.models:
                self.models[model_name].status = "error"
                self.models[model_name].error_message = error_msg
            
            self._notify_callbacks(model_name, "loading_error", {'error': error_msg})
            
            if callback:
                callback(model_name, False, error_msg)
                
        except Exception as e:
            error_msg = str(e)
            print(f"[DEBUG] 모델 로딩 오류: {error_msg}")
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
            print(f"[DEBUG] 모델 로딩 총 소요시간: {elapsed:.1f}초")
    
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
                print(f"Error unloading model {model_name}: {e}")
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