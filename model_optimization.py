"""
모델 로딩 성능 최적화 유틸리티
"""
import os
import torch
import psutil
from typing import Dict, Optional
import logging

class ModelLoadingOptimizer:
    """모델 로딩 성능 최적화 클래스"""
    
    def __init__(self):
        self.setup_environment()
        self.logger = self._setup_logger()
    
    def setup_environment(self):
        """극한 최적화된 환경 변수 설정"""
        # HuggingFace 최적화
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        
        # CPU 극한 최적화
        cpu_count = os.cpu_count()
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)  # 모든 코어 사용
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
        
        # PyTorch CPU 최적화 (안전하게 시도)
        try:
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
        except RuntimeError as e:
            self.logger.warning(f"PyTorch 스레드 설정 실패 (이미 초기화됨): {e}")
        
        # 메모리 최적화
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # CUDA 최적화 (사용 가능시)
        if torch.cuda.is_available():
            os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
            torch.backends.cudnn.benchmark = True
        else:
            # CPU 전용 최적화
            torch.backends.mkldnn.enabled = True
    
    def _setup_logger(self):
        """로거 설정"""
        logger = logging.getLogger("ModelOptimizer")
        logger.setLevel(logging.INFO)
        return logger
    
    def get_optimal_device(self) -> str:
        """최적 디바이스 선택"""
        if not torch.cuda.is_available():
            return "cpu"
        
        # GPU 메모리 체크
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_gb = gpu_memory / (1024**3)
            
            # 시스템 메모리 체크
            system_memory = psutil.virtual_memory()
            available_memory_gb = system_memory.available / (1024**3)
            
            # GPU가 4GB 이상이고 시스템 메모리가 8GB 이상이면 GPU 사용
            if gpu_memory_gb >= 4.0 and available_memory_gb >= 8.0:
                return "cuda"
            else:
                return "cpu"
        except:
            return "cpu"
    
    def get_optimal_loading_config(self, device: str, model_size_estimate: Optional[float] = None) -> Dict:
        """극한 최적화된 모델 로딩 설정 반환"""
        config = {
            "trust_remote_code": True,
            "local_files_only": True,
            "low_cpu_mem_usage": True,
            "use_safetensors": True
        }
        
        if device == "cuda":
            # GPU 사용시 추가 최적화
            config["device_map"] = "auto"
            config["torch_dtype"] = torch.float16
            
            # 모델 크기에 따른 메모리 최적화
            if model_size_estimate and model_size_estimate > 1000:  # 1GB 이상
                config["load_in_8bit"] = True
        else:
            # CPU 전용 극한 최적화
            config["torch_dtype"] = torch.float32  # CPU에서는 float32가 더 최적화됨
            
            # CPU에서 양자화 시도 (사용 가능한 경우)
            try:
                import bitsandbytes as bnb
                if model_size_estimate and model_size_estimate > 500:  # 500MB 이상
                    config["load_in_8bit"] = True
                    config["llm_int8_enable_fp32_cpu_offload"] = True
            except ImportError:
                pass
        
        return config
    
    def get_optimal_tokenizer_config(self) -> Dict:
        """최적화된 토크나이저 설정 반환"""
        return {
            "use_fast": True,
            "local_files_only": True,
            "padding_side": "right",
            "truncation_side": "right"
        }
    
    def preload_optimizations(self):
        """모델 로딩 전 최적화 작업"""
        # 가비지 컬렉터 실행
        import gc
        gc.collect()
        
        # GPU 캐시 정리 (사용 가능시)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def post_load_optimizations(self, model, device: str):
        """모델 로딩 후 최적화 작업"""
        try:
            # 모델 평가 모드 설정
            model.eval()
            
            # GPU 사용시 추가 최적화
            if device == "cuda" and hasattr(model, 'half'):
                # Half precision으로 변환 (메모리 절약)
                model = model.half()
            
            # 불필요한 그래디언트 비활성화
            for param in model.parameters():
                param.requires_grad = False
            
            return model
        except Exception as e:
            self.logger.warning(f"Post-load optimization failed: {e}")
            return model
    
    def estimate_model_size(self, model_path: str) -> float:
        """모델 크기 추정 (MB 단위)"""
        try:
            import glob
            
            # 모델 파일들 크기 계산
            model_files = glob.glob(os.path.join(model_path, "*.bin")) + \
                         glob.glob(os.path.join(model_path, "*.safetensors"))
            
            total_size = 0
            for file_path in model_files:
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            
            return total_size / (1024 * 1024)  # MB 단위
        except:
            return 0.0
    
    def monitor_loading_progress(self, start_time: float, current_time: float, stage: str):
        """로딩 진행상황 모니터링"""
        elapsed = current_time - start_time
        self.logger.info(f"[{stage}] 경과시간: {elapsed:.1f}초")
        
        # 메모리 사용량 체크
        memory = psutil.virtual_memory()
        self.logger.info(f"[{stage}] 메모리 사용률: {memory.percent:.1f}%")
    
    def optimize_file_access(self, model_path: str):
        """파일 접근 최적화 (프리로딩 대신 경량화된 최적화)"""
        try:
            # 디스크 캐시 워밍업 (매우 가벼운 작업)
            import glob
            safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
            
            for file_path in safetensors_files:
                # 파일 존재 확인만 (실제 읽기 없음)
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path) / (1024*1024)  # MB
                    self.logger.info(f"파일 확인: {os.path.basename(file_path)} ({file_size:.1f}MB)")
                    
        except Exception as e:
            self.logger.warning(f"파일 접근 최적화 실패: {e}")
    
    def use_cpu_optimizations(self):
        """CPU 전용 극한 최적화 적용"""
        try:
            # Intel MKL 최적화 (설치된 경우)
            try:
                import intel_extension_for_pytorch as ipex
                self.logger.info("Intel Extension for PyTorch 활성화")
            except ImportError:
                pass
            
            # OpenMP 최적화
            torch.backends.openmp.is_available()
            
            # CPU 텐서 최적화
            torch.backends.mkl.is_available()
            torch.backends.mkldnn.enabled = True
            
            self.logger.info("CPU 극한 최적화 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"CPU 최적화 실패: {e}")
    
    def optimized_model_loading(self, model_path: str, model_class, **kwargs):
        """최적화된 모델 로딩 (프리로딩 제거)"""
        try:
            # 경량 파일 접근 최적화
            self.optimize_file_access(model_path)
            
            # CPU 최적화 적용
            self.use_cpu_optimizations()
            
            # 메모리 최적화된 로딩
            with torch.no_grad():  # 그래디언트 계산 비활성화
                model = model_class.from_pretrained(model_path, **kwargs)
            
            return model
            
        except Exception as e:
            self.logger.error(f"최적화된 로딩 실패, 일반 로딩으로 전환: {e}")
            # 실패시 일반 로딩으로 폴백
            return model_class.from_pretrained(model_path, **kwargs)

# 전역 인스턴스
optimizer = ModelLoadingOptimizer()