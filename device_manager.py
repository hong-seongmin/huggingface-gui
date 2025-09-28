"""
통합 디바이스 관리 유틸리티
모든 모델 타입에 대한 일관된 디바이스 처리를 제공합니다.
"""
import torch
import logging
from typing import Any, Dict, Optional, Tuple, Union

# Import adaptive GPU detector
try:
    from utils.gpu_detector import gpu_detector
    GPU_DETECTOR_AVAILABLE = True
except ImportError:
    GPU_DETECTOR_AVAILABLE = False

class UniversalDeviceManager:
    """모든 모델에 대한 통합 디바이스 관리"""
    
    def __init__(self):
        self.logger = logging.getLogger("DeviceManager")
        self.preferred_device = self._detect_optimal_device()
        
    def _detect_optimal_device(self) -> torch.device:
        """최적 디바이스 자동 감지 (adaptive GPU detection 사용)"""
        if GPU_DETECTOR_AVAILABLE:
            try:
                # GPU detector를 사용하여 지능적 디바이스 선택
                if gpu_detector.is_gpu_recommended():
                    device_name = gpu_detector.get_device_recommendation()
                    status_msg = gpu_detector.get_status_message()
                    self.logger.info(f"GPU 사용 권장: {device_name} - {status_msg}")
                    return torch.device(device_name)
                else:
                    status_msg = gpu_detector.get_status_message()
                    self.logger.info(f"CPU 모드 사용: {status_msg}")
                    return torch.device('cpu')
            except Exception as e:
                self.logger.warning(f"GPU 감지 실패, CPU로 폴백: {e}")
                return torch.device('cpu')
        else:
            # Fallback: 기존 로직
            self.logger.info("GPU detector 없음 - 안정성을 위해 CPU 디바이스 사용")
            return torch.device('cpu')
    
    def ensure_device_consistency(self, model: Any, tokenizer: Any = None) -> Tuple[Any, Any]:
        """모델과 토크나이저의 디바이스 일관성 보장"""
        try:
            target_device = self.preferred_device
            
            # 모델 디바이스 통일
            if hasattr(model, 'to'):
                model = model.to(target_device)
                self.logger.info(f"모델을 {target_device}로 이동")
            
            # 모델의 모든 파라미터가 같은 디바이스에 있는지 확인
            devices = set()
            for param in model.parameters():
                devices.add(param.device)
            
            if len(devices) > 1:
                self.logger.warning(f"모델 파라미터가 여러 디바이스에 분산: {devices}")
                # 강제로 모든 파라미터를 target_device로 이동
                self._force_model_to_device(model, target_device)
            
            # 토크나이저는 별도 디바이스 처리 불필요 (CPU에서 동작)
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"디바이스 일관성 확보 실패: {e}")
            # 폴백: CPU로 강제 이동
            return self._fallback_to_cpu(model, tokenizer)
    
    def _force_model_to_device(self, model: Any, device: torch.device):
        """모델의 모든 구성요소를 강제로 지정 디바이스로 이동"""
        try:
            # 1. 기본 to() 메서드 사용
            model.to(device)
            
            # 2. 명시적으로 모든 파라미터 이동
            for param in model.parameters():
                param.data = param.data.to(device)
                if param.grad is not None:
                    param.grad = param.grad.to(device)
            
            # 3. 버퍼 이동 (예: running_mean, running_var 등)
            for buffer in model.buffers():
                buffer.data = buffer.data.to(device)
            
            self.logger.info(f"모델의 모든 구성요소를 {device}로 강제 이동")
            
        except Exception as e:
            self.logger.error(f"강제 디바이스 이동 실패: {e}")
            raise
    
    def _fallback_to_cpu(self, model: Any, tokenizer: Any) -> Tuple[Any, Any]:
        """CPU로 폴백"""
        try:
            self.logger.info("CPU로 폴백 실행")
            cpu_device = torch.device('cpu')
            
            if hasattr(model, 'to'):
                model = model.to(cpu_device)
            
            self._force_model_to_device(model, cpu_device)
            self.preferred_device = cpu_device
            
            return model, tokenizer
            
        except Exception as e:
            self.logger.error(f"CPU 폴백 실패: {e}")
            raise
    
    def prepare_inputs(self, inputs: Dict[str, torch.Tensor], model: Any) -> Dict[str, torch.Tensor]:
        """입력 텐서를 모델과 같은 디바이스로 이동"""
        try:
            # 모델의 디바이스 확인
            model_device = next(model.parameters()).device
            
            # 모든 입력을 모델 디바이스로 이동
            prepared_inputs = {}
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    prepared_inputs[key] = value.to(model_device)
                else:
                    prepared_inputs[key] = value
            
            return prepared_inputs
            
        except StopIteration:
            # 모델에 파라미터가 없는 경우
            self.logger.warning("모델에 파라미터가 없음, 입력을 CPU로 유지")
            return inputs
        except Exception as e:
            self.logger.error(f"입력 준비 실패: {e}")
            return inputs
    
    def validate_device_consistency(self, model: Any) -> bool:
        """모델의 디바이스 일관성 검증"""
        try:
            devices = set()
            for param in model.parameters():
                devices.add(param.device)
            
            for buffer in model.buffers():
                devices.add(buffer.device)
            
            if len(devices) <= 1:
                self.logger.info(f"디바이스 일관성 확인: {devices}")
                return True
            else:
                self.logger.error(f"디바이스 불일치 감지: {devices}")
                return False
                
        except Exception as e:
            self.logger.error(f"디바이스 검증 실패: {e}")
            return False
    
    def get_device_info(self) -> Dict[str, Any]:
        """현재 디바이스 상태 정보 (adaptive GPU detection 포함)"""
        info = {
            "preferred_device": str(self.preferred_device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }

        # Adaptive GPU detector 정보 추가
        if GPU_DETECTOR_AVAILABLE:
            try:
                gpu_status = gpu_detector.get_status()
                info.update({
                    "gpu_detector_available": True,
                    "gpu_hardware_detected": gpu_status.nvidia_smi_available,
                    "gpu_compatibility": gpu_status.torch_cuda_available,
                    "gpu_status_message": gpu_detector.get_status_message(),
                    "gpu_count_detected": len(gpu_status.gpus),
                    "compatibility_issues": gpu_status.compatibility_issues,
                    "recommended_action": gpu_status.recommended_action
                })
            except Exception as e:
                info["gpu_detector_error"] = str(e)
        else:
            info["gpu_detector_available"] = False

        if torch.cuda.is_available():
            try:
                info.update({
                    "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                    "gpu_memory_allocated": torch.cuda.memory_allocated(0),
                    "gpu_memory_cached": torch.cuda.memory_reserved(0)
                })
            except:
                pass

        return info

# 전역 인스턴스
device_manager = UniversalDeviceManager()