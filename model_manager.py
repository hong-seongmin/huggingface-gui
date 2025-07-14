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
from fast_tensor_loader import fast_loader
from parallel_model_loader import parallel_loader
from cpu_optimizer import cpu_optimizer
from extreme_optimizer import extreme_optimizer
from lightning_loader import lightning_loader
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
        
        def load_model_ultra_fast(actual_model_path, device):
            """Ultra-Fast 모델 로딩 - 상세한 병목 분석 포함"""
            print(f"[ULTRA-FAST] 혁신적인 모델 로딩 시작")
            
            # 상세한 프로파일링 시작 (프로파일러 내부에서 활성화 여부 확인)
            profiler.start_profiling("Ultra-Fast 모델 로딩")
            profiler.profile_transformers_loading()
            profiler.profile_safetensors_loading()
            profiler.memory_snapshot("초기 상태")
            
            # 0단계: Lightning 로딩 시도 (초고속)
            try:
                profiler.checkpoint("0단계: Lightning 로딩 시도")
                model, tokenizer, load_time = lightning_loader.lightning_load(actual_model_path, device)
                
                if model and tokenizer:
                    profiler.checkpoint(f"Lightning 로딩 성공: {load_time:.1f}초")
                    
                    # 통합 디바이스 관리자로 일관성 보장
                    model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                    profiler.checkpoint("디바이스 일관성 보장 완료")
                    profiler.memory_snapshot("Lightning 완료")
                    
                    # 분석 리포트 출력
                    profiler.print_detailed_report()
                    return model, tokenizer, load_time
                else:
                    profiler.checkpoint("Lightning 로딩 실패, 1단계로 전환")
                    
            except Exception as e:
                profiler.checkpoint(f"Lightning 로딩 오류: {e}")
            
            # 1단계: 병렬 로딩 시도 (가장 빠름)
            try:
                profiler.checkpoint("1단계: 병렬 로딩 시도")
                profiler.memory_snapshot("병렬 로딩 시작")
                
                model, tokenizer, load_time = parallel_loader.load_model_and_tokenizer_parallel(
                    actual_model_path, device
                )
                
                if model and tokenizer:
                    profiler.checkpoint(f"병렬 로딩 성공: {load_time:.1f}초")
                    profiler.memory_snapshot("병렬 로딩 완료")
                    
                    # 통합 디바이스 관리자로 일관성 보장
                    model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                    profiler.checkpoint("디바이스 일관성 보장 완료")
                    
                    # CPU 최적화 적용
                    profiler.checkpoint("CPU 최적화 시작")
                    model = cpu_optimizer.optimize_model_for_cpu(model, optimize_level=3)
                    profiler.checkpoint("CPU 최적화 완료")
                    profiler.memory_snapshot("최적화 완료")
                    
                    # 분석 리포트 출력
                    profiler.print_detailed_report()
                    return model, tokenizer, load_time
                else:
                    profiler.checkpoint("병렬 로딩 실패, 2단계로 전환")
                    
            except Exception as e:
                profiler.checkpoint(f"병렬 로딩 오류: {e}")
            
            # 2단계: 직접 텐서 로딩 시도 (매우 빠름)
            try:
                profiler.checkpoint("2단계: 직접 텐서 로딩 시도")
                profiler.memory_snapshot("직접 로딩 시작")
                
                # 모델 직접 로딩
                model, model_time = fast_loader.load_model_ultra_fast(actual_model_path, device)
                
                if model:
                    profiler.checkpoint(f"직접 텐서 로딩 성공: {model_time:.1f}초")
                    profiler.memory_snapshot("모델 로딩 완료")
                    
                    # 토크나이저 로딩
                    tokenizer, tokenizer_time = fast_loader.load_tokenizer_fast(actual_model_path)
                    
                    total_time = model_time + tokenizer_time
                    profiler.checkpoint(f"전체 로딩 시간: {total_time:.1f}초")
                    profiler.memory_snapshot("토크나이저 완료")
                    
                    # 통합 디바이스 관리자로 일관성 보장
                    model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                    profiler.checkpoint("디바이스 일관성 보장 완료")
                    
                    # CPU 최적화 적용
                    profiler.checkpoint("CPU 최적화 시작")
                    model = cpu_optimizer.optimize_model_for_cpu(model, optimize_level=2)
                    profiler.checkpoint("CPU 최적화 완료")
                    profiler.memory_snapshot("최적화 완료")
                    
                    # 분석 리포트 출력
                    profiler.print_detailed_report()
                    return model, tokenizer, total_time
                else:
                    profiler.checkpoint("직접 텐서 로딩 실패, 3단계로 전환")
                    
            except Exception as e:
                profiler.checkpoint(f"직접 텐서 로딩 오류: {e}")
            
            # 3단계: EXTREME 최적화 로딩 (최후의 무기)
            try:
                profiler.checkpoint("3단계: EXTREME 최적화 로딩")
                profiler.memory_snapshot("EXTREME 시작")
                
                # EXTREME 최적화 적용
                model, tokenizer, load_time = extreme_optimizer.ultra_fast_model_loading(
                    actual_model_path, device
                )
                
                if model and tokenizer:
                    profiler.checkpoint(f"EXTREME 로딩 성공: {load_time:.1f}초")
                    profiler.memory_snapshot("EXTREME 완료")
                    
                    # 통합 디바이스 관리자로 일관성 보장
                    model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                    profiler.checkpoint("디바이스 일관성 보장 완료")
                    
                    # CPU 최적화 적용
                    profiler.checkpoint("CPU 최적화 시작")
                    model = cpu_optimizer.optimize_model_for_cpu(model, optimize_level=1)
                    profiler.checkpoint("CPU 최적화 완료")
                    profiler.memory_snapshot("최적화 완료")
                    
                    # 분석 리포트 출력
                    profiler.print_detailed_report()
                    return model, tokenizer, load_time
                else:
                    profiler.checkpoint("EXTREME 로딩 실패, 최후 폴백")
                    raise ValueError("EXTREME 로딩 실패")
                
            except Exception as e:
                profiler.checkpoint(f"EXTREME 로딩도 실패: {e}")
                
                # 최후 폴백: 기본 로딩
                profiler.checkpoint("최후 폴백: 기본 로딩")
                profiler.memory_snapshot("폴백 시작")
                
                import time
                fallback_start = time.time()
                
                try:
                    from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
                    
                    profiler.checkpoint("Config 로딩 시작")
                    config = AutoConfig.from_pretrained(actual_model_path, local_files_only=True)
                    profiler.checkpoint("Config 로딩 완료")
                    
                    is_classification = (
                        hasattr(config, 'architectures') and 
                        config.architectures and
                        any('Classification' in arch for arch in config.architectures)
                    )
                    
                    profiler.checkpoint("모델 로딩 시작 (폴백)")
                    profiler.memory_snapshot("모델 로딩 전")
                    
                    if is_classification:
                        model = AutoModelForSequenceClassification.from_pretrained(
                            actual_model_path, local_files_only=True
                        )
                    else:
                        model = AutoModel.from_pretrained(
                            actual_model_path, local_files_only=True
                        )
                    
                    profiler.checkpoint("모델 로딩 완료 (폴백)")
                    profiler.memory_snapshot("모델 로딩 후")
                    
                    profiler.checkpoint("토크나이저 로딩 시작 (폴백)")
                    tokenizer = AutoTokenizer.from_pretrained(
                        actual_model_path, local_files_only=True
                    )
                    profiler.checkpoint("토크나이저 로딩 완료 (폴백)")
                    
                    # 통합 디바이스 관리자로 일관성 보장
                    model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
                    model.eval()
                    profiler.checkpoint("모델 최종 설정 완료 (통합 디바이스 관리)")
                    profiler.memory_snapshot("폴백 완료")
                    
                    load_time = time.time() - fallback_start
                    profiler.checkpoint(f"최후 폴백 완료: {load_time:.1f}초")
                    
                    # 분석 리포트 출력
                    profiler.print_detailed_report()
                    return model, tokenizer, load_time
                    
                except Exception as final_e:
                    profiler.checkpoint(f"모든 로딩 방법 실패: {final_e}")
                    profiler.print_detailed_report()
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
            
            # HuggingFace 모델 ID인지 확인하고 다운로드
            actual_model_path = model_path
            if self._is_huggingface_model_id(model_path):
                self._notify_callbacks(model_name, "downloading", {'model_id': model_path})
                actual_model_path = self._download_huggingface_model(model_path)
                self.models[model_name].path = actual_model_path  # 실제 경로로 업데이트
            
            # 모델 분석
            analysis = self.model_analyzer.analyze_model_directory(actual_model_path)
            self.models[model_name].config_analysis = analysis
            
            # 범용적인 transformers 모델 로드
            from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
            
            # 최적 디바이스 자동 선택
            device = optimizer.get_optimal_device()
            print(f"[DEBUG] 자동 선택된 디바이스: {device}")
            
            # 메모리 상태 체크
            memory_info = self.get_memory_info()
            available_memory_gb = memory_info['system_memory']['available'] / (1024**3)
            print(f"[DEBUG] 사용 가능한 메모리: {available_memory_gb:.1f}GB")
            print(f"[DEBUG] 디바이스 설정: {device} (Streamlit 안정성을 위해 CPU 강제)")
            
            # accelerate 사용 가능 여부 확인 (단순화)
            try:
                import accelerate
                use_device_map = device == "cuda"
            except ImportError:
                use_device_map = False
            
            # 설정에서 architecture 확인
            config = AutoConfig.from_pretrained(actual_model_path)
            is_classification_model = (
                hasattr(config, 'architectures') and 
                config.architectures and
                any('Classification' in arch for arch in config.architectures)
            )
            
            print(f"[DEBUG] 모델 로딩 시작: classification={is_classification_model}")
            
            # 캐시 키 생성
            cache_key = f"{model_name}_{actual_model_path}_{device}_{is_classification_model}"
            cache_key_hash = hashlib.md5(cache_key.encode()).hexdigest()
            
            # 캐시에서 모델 확인
            cached_result = model_cache.get_cached_model(cache_key_hash)
            if cached_result:
                model, tokenizer = cached_result
                print(f"[DEBUG] 캐시에서 모델 로드 완료 (즉시)")
            else:
                # 캐시 미스 - Ultra-Fast 로딩 시작
                print(f"[DEBUG] 캐시 미스 - Ultra-Fast 로딩 시작")
                
                try:
                    # 혁신적인 Ultra-Fast 로딩
                    result = load_model_ultra_fast(actual_model_path, device)
                    
                    if len(result) == 3:
                        model, tokenizer, load_time = result
                        print(f"[DEBUG] Ultra-Fast 로딩 성공: {load_time:.1f}초")
                    else:
                        raise ValueError("Ultra-Fast 로딩 결과 형식 오류")
                    
                    # 캐시에 저장
                    cache_config = {
                        'device': device,
                        'load_time': load_time,
                        'model_path': actual_model_path,
                        'ultra_fast': True
                    }
                    model_cache.cache_model(cache_key_hash, model, tokenizer, actual_model_path, cache_config)
                    
                except Exception as e:
                    print(f"[DEBUG] Ultra-Fast 로딩 실패: {e}")
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