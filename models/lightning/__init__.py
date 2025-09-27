"""
Lightning 모델 로딩 패키지

이 패키지는 고속 모델 로딩을 위한 모듈화된 컴포넌트들을 제공합니다:
- LightningModelLoader: 메인 로더 클래스 (backward compatibility maintained)
- CheckpointLoader: 체크포인트 처리 전문 클래스
- ModelConverter: 경량 모델/토크나이저 생성 클래스
"""

from .checkpoint_loader import CheckpointLoader
from .model_converter import ModelConverter

# For backward compatibility, expose the main lightning_loader from parent directory
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
try:
    from lightning_loader import LightningModelLoader
except ImportError:
    # If the main lightning_loader isn't available, create a stub
    class LightningModelLoader:
        def __init__(self):
            self.checkpoint_loader = CheckpointLoader()
            self.model_converter = ModelConverter()

        def lightning_load(self, model_path: str, device: str = "cpu"):
            return self.checkpoint_loader.load_checkpoint(model_path), None, 1.0

__all__ = [
    'CheckpointLoader',
    'ModelConverter',
    'LightningModelLoader',  # For backward compatibility
]

# 버전 정보
__version__ = '1.0.0'

# 패키지 정보
__author__ = 'HuggingFace GUI Team'
__description__ = 'Lightning fast model loading components'