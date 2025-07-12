"""
CPU 특화 극한 최적화 엔진
"""
import os
import torch
import logging
from typing import Any, Optional, Dict
import warnings

class CPUOptimizer:
    """CPU 환경에서 극한 성능을 위한 최적화 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger("CPUOptimizer")
        self.logger.setLevel(logging.INFO)
        self.intel_extension_available = False
        self.torch_compile_available = False
        
        # 최적화 엔진 초기화
        self._initialize_optimizers()
    
    def _initialize_optimizers(self):
        """사용 가능한 최적화 엔진들 초기화"""
        
        # Intel Extension for PyTorch 확인
        try:
            import intel_extension_for_pytorch as ipex
            self.intel_extension_available = True
            self.logger.info("[CPU-OPT] Intel Extension for PyTorch 사용 가능")
        except ImportError:
            self.logger.info("[CPU-OPT] Intel Extension for PyTorch 미설치")
        
        # torch.compile 확인 (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
                self.torch_compile_available = True
                self.logger.info("[CPU-OPT] torch.compile 사용 가능")
            else:
                self.logger.info("[CPU-OPT] torch.compile 사용 불가 (PyTorch 2.0+ 필요)")
        except:
            self.torch_compile_available = False
        
        # CPU 환경 최적화
        self._setup_cpu_environment()
    
    def _setup_cpu_environment(self):
        """CPU 환경 극한 최적화"""
        try:
            # CPU 코어 최대 활용
            cpu_count = os.cpu_count()
            
            # OpenMP 설정
            os.environ["OMP_NUM_THREADS"] = str(cpu_count)
            os.environ["MKL_NUM_THREADS"] = str(cpu_count)
            os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_count)
            os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_count)
            os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_count)
            
            # PyTorch 스레드 설정 (안전하게 시도)
            try:
                torch.set_num_threads(cpu_count)
                torch.set_num_interop_threads(cpu_count)
            except RuntimeError as e:
                self.logger.warning(f"[CPU-OPT] PyTorch 스레드 설정 실패 (이미 초기화됨): {e}")
            
            # CPU 백엔드 최적화
            if torch.backends.mkl.is_available():
                torch.backends.mkl.enabled = True
                self.logger.info("[CPU-OPT] Intel MKL 활성화")
            
            if torch.backends.mkldnn.is_available():
                torch.backends.mkldnn.enabled = True
                self.logger.info("[CPU-OPT] MKL-DNN 활성화")
            
            # JIT 최적화
            torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
            
            self.logger.info(f"[CPU-OPT] CPU 환경 최적화 완료 ({cpu_count} 코어)")
            
        except Exception as e:
            self.logger.warning(f"[CPU-OPT] CPU 환경 최적화 실패: {e}")
    
    def optimize_model_for_cpu(self, model: Any, optimize_level: int = 3) -> Any:
        """
        모델을 CPU에 특화하여 최적화
        
        Args:
            model: 최적화할 모델
            optimize_level: 최적화 레벨 (1-3, 3이 가장 공격적)
        
        Returns:
            최적화된 모델
        """
        try:
            optimized_model = model
            
            # Level 1: 기본 최적화
            if optimize_level >= 1:
                optimized_model = self._apply_basic_optimizations(optimized_model)
            
            # Level 2: Intel Extension 최적화
            if optimize_level >= 2 and self.intel_extension_available:
                optimized_model = self._apply_intel_optimizations(optimized_model)
            
            # Level 3: torch.compile 최적화
            if optimize_level >= 3 and self.torch_compile_available:
                optimized_model = self._apply_torch_compile(optimized_model)
            
            self.logger.info(f"[CPU-OPT] 모델 최적화 완료 (레벨 {optimize_level})")
            return optimized_model
            
        except Exception as e:
            self.logger.warning(f"[CPU-OPT] 모델 최적화 실패: {e}")
            return model
    
    def _apply_basic_optimizations(self, model: Any) -> Any:
        """기본 CPU 최적화 적용"""
        try:
            # 평가 모드 설정
            model.eval()
            
            # 그래디언트 비활성화
            for param in model.parameters():
                param.requires_grad = False
            
            # JIT 스크립팅 시도
            try:
                if hasattr(model, 'forward'):
                    # 간단한 입력으로 JIT 추적
                    example_input = self._create_example_input(model)
                    if example_input is not None:
                        model = torch.jit.trace(model, example_input)
                        self.logger.info("[CPU-OPT] JIT 추적 적용됨")
            except Exception as e:
                self.logger.warning(f"[CPU-OPT] JIT 추적 실패: {e}")
            
            # 메모리 최적화
            if hasattr(model, 'half') and torch.cuda.is_available() == False:
                # CPU에서는 half precision 사용 안 함 (성능 저하)
                pass
            
            self.logger.info("[CPU-OPT] 기본 최적화 적용됨")
            return model
            
        except Exception as e:
            self.logger.warning(f"[CPU-OPT] 기본 최적화 실패: {e}")
            return model
    
    def _apply_intel_optimizations(self, model: Any) -> Any:
        """Intel Extension 최적화 적용"""
        try:
            import intel_extension_for_pytorch as ipex
            
            # Intel Extension 최적화
            model = ipex.optimize(model, dtype=torch.float32)
            
            # Intel 특화 최적화 설정
            ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.BF16, device="cpu")
            
            self.logger.info("[CPU-OPT] Intel Extension 최적화 적용됨")
            return model
            
        except Exception as e:
            self.logger.warning(f"[CPU-OPT] Intel Extension 최적화 실패: {e}")
            return model
    
    def _apply_torch_compile(self, model: Any) -> Any:
        """torch.compile 최적화 적용"""
        try:
            # torch.compile로 모델 컴파일
            compiled_model = torch.compile(
                model,
                mode="max-autotune",  # 최대 최적화
                dynamic=False,        # 정적 컴파일
                backend="inductor"    # 기본 백엔드
            )
            
            self.logger.info("[CPU-OPT] torch.compile 최적화 적용됨")
            return compiled_model
            
        except Exception as e:
            self.logger.warning(f"[CPU-OPT] torch.compile 최적화 실패: {e}")
            return model
    
    def _create_example_input(self, model: Any) -> Optional[torch.Tensor]:
        """JIT 추적용 예제 입력 생성"""
        try:
            # 모델 타입에 따른 예제 입력 생성
            if hasattr(model, 'config'):
                config = model.config
                
                # 시퀀스 모델인 경우
                if hasattr(config, 'max_position_embeddings'):
                    seq_len = min(config.max_position_embeddings, 128)
                    batch_size = 1
                    
                    if hasattr(config, 'vocab_size'):
                        # 토큰 ID 입력
                        input_ids = torch.randint(0, min(config.vocab_size, 1000), (batch_size, seq_len))
                        return {"input_ids": input_ids}
                    else:
                        # 일반적인 텐서 입력
                        hidden_size = getattr(config, 'hidden_size', 768)
                        return torch.randn(batch_size, seq_len, hidden_size)
            
            # 기본 입력
            return torch.randn(1, 128)  # 배치 크기 1, 시퀀스 길이 128
            
        except Exception as e:
            self.logger.warning(f"[CPU-OPT] 예제 입력 생성 실패: {e}")
            return None
    
    def benchmark_model_performance(self, model: Any, num_runs: int = 10) -> Dict[str, float]:
        """모델 성능 벤치마크"""
        try:
            import time
            
            model.eval()
            example_input = self._create_example_input(model)
            
            if example_input is None:
                return {"error": "예제 입력 생성 실패"}
            
            # 워밍업
            with torch.no_grad():
                for _ in range(3):
                    if isinstance(example_input, dict):
                        _ = model(**example_input)
                    else:
                        _ = model(example_input)
            
            # 실제 벤치마크
            times = []
            with torch.no_grad():
                for _ in range(num_runs):
                    start_time = time.time()
                    
                    if isinstance(example_input, dict):
                        _ = model(**example_input)
                    else:
                        _ = model(example_input)
                    
                    end_time = time.time()
                    times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            self.logger.info(f"[CPU-OPT] 벤치마크 완료: 평균 {avg_time:.4f}초")
            
            return {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "throughput": 1.0 / avg_time  # 초당 추론 수
            }
            
        except Exception as e:
            self.logger.error(f"[CPU-OPT] 벤치마크 실패: {e}")
            return {"error": str(e)}
    
    def get_optimization_info(self) -> Dict[str, Any]:
        """최적화 정보 반환"""
        return {
            "intel_extension_available": self.intel_extension_available,
            "torch_compile_available": self.torch_compile_available,
            "torch_version": torch.__version__,
            "mkl_available": torch.backends.mkl.is_available(),
            "mkldnn_available": torch.backends.mkldnn.is_available(),
            "cpu_count": os.cpu_count(),
            "num_threads": torch.get_num_threads()
        }

# 전역 인스턴스
cpu_optimizer = CPUOptimizer()