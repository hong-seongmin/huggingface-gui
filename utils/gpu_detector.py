"""
GPU Detection and Compatibility Utility
Provides adaptive GPU detection that works with or without GPUtil and handles PyTorch compatibility.
"""
import warnings
import subprocess
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """GPU information structure"""
    id: int
    name: str
    memory_total: int
    memory_used: int = 0
    memory_free: int = 0
    temperature: Optional[float] = None
    load: float = 0.0
    memory_util: float = 0.0
    uuid: Optional[str] = None

@dataclass
class GPUStatus:
    """Overall GPU status information"""
    gpu_available: bool = False
    torch_cuda_available: bool = False
    gputil_available: bool = False
    nvidia_smi_available: bool = False
    gpu_count: int = 0
    gpus: List[GPUInfo] = None
    compatibility_issues: List[str] = None
    recommended_action: str = ""

class AdaptiveGPUDetector:
    """
    Adaptive GPU detector that gracefully handles various GPU configurations:
    - No GPU hardware
    - GPU hardware but no drivers
    - GPU hardware with incompatible PyTorch
    - GPU hardware with full compatibility
    """

    def __init__(self):
        self.status = GPUStatus()
        self.status.gpus = []
        self.status.compatibility_issues = []
        self._detect_capabilities()

    def _detect_capabilities(self):
        """Detect available GPU capabilities"""
        # Check if GPUtil is available
        try:
            import GPUtil
            self.status.gputil_available = True
            logger.info("GPUtil 사용 가능")
        except ImportError:
            self.status.gputil_available = False
            logger.info("GPUtil 사용 불가 - 선택적 GPU 모니터링")

        # Check if nvidia-smi is available
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.status.nvidia_smi_available = True
                logger.info("nvidia-smi 사용 가능")
            else:
                self.status.nvidia_smi_available = False
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            self.status.nvidia_smi_available = False
            logger.info("nvidia-smi 사용 불가")

        # Check PyTorch CUDA availability
        try:
            import torch
            self.status.torch_cuda_available = torch.cuda.is_available()
            self.status.gpu_count = torch.cuda.device_count()

            if self.status.torch_cuda_available:
                logger.info(f"PyTorch CUDA 사용 가능: {self.status.gpu_count}개 GPU")
            else:
                logger.info("PyTorch CUDA 사용 불가")

            # Check for compatibility warnings
            if self.status.nvidia_smi_available and not self.status.torch_cuda_available:
                self.status.compatibility_issues.append("GPU 하드웨어는 있지만 PyTorch와 호환되지 않음")

        except ImportError:
            logger.warning("PyTorch가 설치되지 않음")
            self.status.compatibility_issues.append("PyTorch가 설치되지 않음")

        # Determine overall GPU availability
        self.status.gpu_available = (
            self.status.torch_cuda_available or
            self.status.nvidia_smi_available or
            self.status.gputil_available
        )

        # Set recommended action
        self._set_recommended_action()

    def _set_recommended_action(self):
        """Set recommended action based on detected capabilities"""
        if self.status.torch_cuda_available:
            self.status.recommended_action = "GPU 사용 권장"
        elif self.status.nvidia_smi_available and not self.status.torch_cuda_available:
            self.status.recommended_action = "GPU 호환성 문제 - CPU 모드 사용"
        elif not self.status.nvidia_smi_available:
            self.status.recommended_action = "GPU 없음 - CPU 모드 사용"
        else:
            self.status.recommended_action = "CPU 모드 사용"

    def get_gpu_info(self) -> List[GPUInfo]:
        """Get detailed GPU information using available methods"""
        gpu_list = []

        # Try GPUtil first (most detailed)
        if self.status.gputil_available:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    gpu_info = GPUInfo(
                        id=gpu.id,
                        name=gpu.name,
                        memory_total=gpu.memoryTotal,
                        memory_used=gpu.memoryUsed,
                        memory_free=gpu.memoryFree,
                        temperature=gpu.temperature,
                        load=gpu.load * 100,
                        memory_util=gpu.memoryUtil * 100,
                        uuid=gpu.uuid
                    )
                    gpu_list.append(gpu_info)
                logger.info(f"GPUtil로 {len(gpu_list)}개 GPU 정보 수집")
                return gpu_list
            except Exception as e:
                logger.warning(f"GPUtil GPU 정보 수집 실패: {e}")

        # Fallback to nvidia-smi
        if self.status.nvidia_smi_available:
            try:
                gpu_list = self._get_gpu_info_nvidia_smi()
                logger.info(f"nvidia-smi로 {len(gpu_list)}개 GPU 정보 수집")
                return gpu_list
            except Exception as e:
                logger.warning(f"nvidia-smi GPU 정보 수집 실패: {e}")

        # Fallback to PyTorch basic info
        if self.status.torch_cuda_available:
            try:
                import torch
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info = GPUInfo(
                        id=i,
                        name=props.name,
                        memory_total=props.total_memory // (1024**2)  # Convert to MB
                    )
                    gpu_list.append(gpu_info)
                logger.info(f"PyTorch로 {len(gpu_list)}개 GPU 정보 수집")
                return gpu_list
            except Exception as e:
                logger.warning(f"PyTorch GPU 정보 수집 실패: {e}")

        return gpu_list

    def _get_gpu_info_nvidia_smi(self) -> List[GPUInfo]:
        """Get GPU info using nvidia-smi"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=index,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,utilization.memory,uuid',
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                return []

            gpu_list = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 7:
                        try:
                            gpu_info = GPUInfo(
                                id=int(parts[0]),
                                name=parts[1],
                                memory_total=int(parts[2]),
                                memory_used=int(parts[3]),
                                memory_free=int(parts[4]),
                                temperature=float(parts[5]) if parts[5] != '[Not Supported]' else None,
                                load=float(parts[6]),
                                memory_util=float(parts[7]) if len(parts) > 7 else 0.0,
                                uuid=parts[8] if len(parts) > 8 else None
                            )
                            gpu_list.append(gpu_info)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"nvidia-smi 파싱 오류: {e}")
                            continue

            return gpu_list
        except Exception as e:
            logger.error(f"nvidia-smi 실행 오류: {e}")
            return []

    def get_status(self) -> GPUStatus:
        """Get current GPU status"""
        # Update GPU list
        self.status.gpus = self.get_gpu_info()
        return self.status

    def is_gpu_recommended(self) -> bool:
        """Check if GPU usage is recommended"""
        return self.status.torch_cuda_available

    def get_device_recommendation(self) -> str:
        """Get recommended device (cuda/cpu) for PyTorch"""
        if self.status.torch_cuda_available:
            return "cuda"
        else:
            return "cpu"

    def get_status_message(self) -> str:
        """Get human-readable status message"""
        if self.status.torch_cuda_available:
            return f"✅ GPU 사용 가능 ({self.status.gpu_count}개)"
        elif self.status.nvidia_smi_available:
            return "⚠️ GPU 있음, 하지만 PyTorch 호환성 문제"
        else:
            return "ℹ️ CPU 전용 모드"

# Global instance for easy access
gpu_detector = AdaptiveGPUDetector()