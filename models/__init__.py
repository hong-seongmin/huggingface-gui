"""
Models package for model management, loading, and memory optimization.
"""

from .model_info import (
    ModelInfo,
    ModelStatus,
    ModelEventType
)
from .memory_manager import (
    MemoryManager,
    memory_manager
)
from .model_loader import (
    ModelLoader,
    model_loader
)
from .model_manager import (
    ModelManager,
    ModelManagerSingleton,
    MultiModelManager,
    get_model_manager
)

# Import analyzer and type detector if available
try:
    from .model_analyzer import ComprehensiveModelAnalyzer
    from .model_type_detector import ModelTypeDetector
    _analyzer_available = True
except ImportError as e:
    # Create placeholder classes if import fails
    class ComprehensiveModelAnalyzer:
        def __init__(self):
            pass
        def analyze_model_directory(self, path, name):
            return {'error': 'Model analyzer not available'}

    class ModelTypeDetector:
        def __init__(self):
            pass
        def detect_model_type(self, path):
            return 'unknown'

    _analyzer_available = False

__all__ = [
    # Model Info
    'ModelInfo',
    'ModelStatus',
    'ModelEventType',

    # Memory Manager
    'MemoryManager',
    'memory_manager',

    # Model Loader
    'ModelLoader',
    'model_loader',

    # Model Manager
    'ModelManager',
    'ModelManagerSingleton',
    'MultiModelManager',
    'get_model_manager',

    # Model Analysis
    'ComprehensiveModelAnalyzer',
    'ModelTypeDetector'
]