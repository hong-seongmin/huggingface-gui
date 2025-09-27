"""
Model information data classes and related utilities.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional, Any


@dataclass
class ModelInfo:
    """Model information container."""
    name: str
    path: str
    model: Optional[object] = None
    tokenizer: Optional[object] = None
    config_analysis: Dict = field(default_factory=dict)
    memory_usage: float = 0.0
    load_time: Optional[datetime] = None
    status: str = "unloaded"  # unloaded, loading, loaded, error
    error_message: str = ""

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.status == "loaded" and self.model is not None

    def is_loading(self) -> bool:
        """Check if model is currently loading."""
        return self.status == "loading"

    def is_error(self) -> bool:
        """Check if model has an error."""
        return self.status == "error"

    def get_memory_mb(self) -> float:
        """Get memory usage in MB."""
        return self.memory_usage / (1024 ** 2)

    def get_memory_gb(self) -> float:
        """Get memory usage in GB."""
        return self.memory_usage / (1024 ** 3)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'path': self.path,
            'config_analysis': self.config_analysis,
            'memory_usage': self.memory_usage,
            'memory_mb': self.get_memory_mb(),
            'memory_gb': self.get_memory_gb(),
            'load_time': self.load_time.isoformat() if self.load_time else None,
            'status': self.status,
            'error_message': self.error_message,
            'is_loaded': self.is_loaded(),
            'is_loading': self.is_loading(),
            'is_error': self.is_error()
        }

    def __repr__(self) -> str:
        return f"ModelInfo(name='{self.name}', status='{self.status}', memory={self.get_memory_mb():.1f}MB)"


class ModelStatus:
    """Model status constants."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNLOADING = "unloading"


class ModelEventType:
    """Model event type constants."""
    LOAD_STARTED = "load_started"
    LOAD_PROGRESS = "load_progress"
    LOAD_COMPLETED = "load_completed"
    LOAD_FAILED = "load_failed"
    UNLOAD_STARTED = "unload_started"
    UNLOAD_COMPLETED = "unload_completed"
    MEMORY_WARNING = "memory_warning"
    STATUS_CHANGED = "status_changed"