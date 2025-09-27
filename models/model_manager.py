"""
Main model manager - refactored and modularized version.
"""
import threading
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any

from .model_info import ModelInfo, ModelStatus, ModelEventType
from .memory_manager import memory_manager
from .model_loader import model_loader
from core.logging_config import get_logger

logger = get_logger(__name__)


class ModelManager:
    """
    Main model management class - refactored and modularized.

    This class coordinates between different components:
    - ModelInfo: Data structures
    - MemoryManager: Memory monitoring and optimization
    - ModelLoader: Model loading operations
    """

    def __init__(self, max_memory_threshold: float = 0.8):
        """
        Initialize model manager.

        Args:
            max_memory_threshold: Maximum memory usage threshold (0-1)
        """
        self.models: Dict[str, ModelInfo] = {}
        self.callbacks: List[Callable] = []
        self.max_memory_threshold = max_memory_threshold

        # Set memory threshold
        memory_manager.max_memory_threshold = max_memory_threshold

        logger.info("ModelManager initialized")

    def add_callback(self, callback: Callable):
        """
        Register callback for model state changes.

        Args:
            callback: Callback function (model_name, event_type, data)
        """
        self.callbacks.append(callback)
        logger.debug(f"Added callback: {callback}")

    def _notify_callbacks(self, model_name: str, event_type: str, data: Dict = None):
        """
        Notify all registered callbacks.

        Args:
            model_name: Name of the model
            event_type: Type of event
            data: Additional event data
        """
        for callback in self.callbacks:
            try:
                callback(model_name, event_type, data or {})
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory information."""
        return memory_manager.get_memory_info()

    def can_load_model(self, estimated_size: float) -> bool:
        """
        Check if model can be loaded based on memory constraints.

        Args:
            estimated_size: Estimated model size in bytes

        Returns:
            True if model can be loaded, False otherwise
        """
        return memory_manager.can_load_model(estimated_size)

    def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """
        Analyze model without loading it.

        Args:
            model_path: Path or HuggingFace model ID

        Returns:
            Analysis results dictionary
        """
        try:
            from model_analyzer import ComprehensiveModelAnalyzer
            from model_type_detector import ModelTypeDetector

            analyzer = ComprehensiveModelAnalyzer()
            type_detector = ModelTypeDetector()

            actual_model_path = model_path

            # Handle HuggingFace model ID
            if model_loader.is_huggingface_model_id(model_path):
                actual_model_path = model_loader.download_huggingface_model(model_path)

            # Generate model name for analysis
            model_name = model_loader.generate_model_name(model_path)

            # Perform analysis
            analysis = analyzer.analyze_model_directory(actual_model_path, model_name)
            analysis['original_path'] = model_path
            analysis['actual_path'] = actual_model_path

            return analysis

        except Exception as e:
            logger.error(f"Model analysis failed: {e}")
            return {'error': str(e)}

    def load_model_async(self, model_name: str, model_path: str) -> threading.Thread:
        """
        Load model asynchronously.

        Args:
            model_name: Name for the model
            model_path: Path or HuggingFace model ID

        Returns:
            Thread handling the loading operation
        """
        # Generate unique model name if needed
        original_name = model_name
        if not model_name or not model_name.strip():
            model_name = model_loader.generate_model_name(model_path)

        # Handle duplicate names
        counter = 1
        while model_name in self.models:
            model_name = f"{original_name}_{counter}"
            counter += 1

        # Create model info placeholder
        model_info = ModelInfo(
            name=model_name,
            path=model_path,
            status=ModelStatus.LOADING
        )
        self.models[model_name] = model_info

        # Define callback for loading progress
        def loading_callback(name: str, event_type: str, data: Dict):
            if event_type == ModelEventType.LOAD_COMPLETED:
                # Update model info with loaded data
                loaded_model_info = data.get('model_info')
                if loaded_model_info:
                    self.models[name] = loaded_model_info
            elif event_type == ModelEventType.LOAD_FAILED:
                # Update status on failure
                self.models[name].status = ModelStatus.ERROR
                self.models[name].error_message = data.get('error', 'Unknown error')

            # Forward to registered callbacks
            self._notify_callbacks(name, event_type, data)

        # Start loading
        thread = model_loader.load_model_async(model_name, model_path, loading_callback)

        logger.info(f"Started async loading for model: {model_name}")
        return thread

    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model from memory.

        Args:
            model_name: Name of the model to unload

        Returns:
            True if unloaded successfully, False otherwise
        """
        if model_name not in self.models:
            logger.warning(f"Model not found: {model_name}")
            return False

        model_info = self.models[model_name]

        # Notify unloading started
        self._notify_callbacks(model_name, ModelEventType.UNLOAD_STARTED, {
            'model_name': model_name
        })

        # Unload using model loader
        success = model_loader.unload_model(model_info)

        if success:
            # Remove from models dict
            del self.models[model_name]

            # Notify unloading completed
            self._notify_callbacks(model_name, ModelEventType.UNLOAD_COMPLETED, {
                'model_name': model_name
            })

            logger.info(f"Model {model_name} unloaded successfully")
        else:
            logger.error(f"Failed to unload model: {model_name}")

        return success

    def get_loaded_models(self) -> List[str]:
        """Get list of loaded model names."""
        return [
            name for name, info in self.models.items()
            if info.is_loaded()
        ]

    def get_loading_models(self) -> List[str]:
        """Get list of currently loading model names."""
        return [
            name for name, info in self.models.items()
            if info.is_loading()
        ]

    def get_failed_models(self) -> Dict[str, str]:
        """Get dictionary of failed models and their error messages."""
        return {
            name: info.error_message
            for name, info in self.models.items()
            if info.is_error()
        }

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        Get model information.

        Args:
            model_name: Name of the model

        Returns:
            ModelInfo object or None if not found
        """
        return self.models.get(model_name)

    def get_all_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information for all models."""
        return {
            name: info.to_dict()
            for name, info in self.models.items()
        }

    def predict(self, text: str, model_name: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Make prediction using a loaded model.

        Args:
            text: Input text
            model_name: Name of model to use (uses first loaded if not specified)
            **kwargs: Additional prediction parameters

        Returns:
            Prediction results dictionary
        """
        try:
            # Find model to use
            if model_name is None:
                loaded_models = self.get_loaded_models()
                if not loaded_models:
                    return {'error': 'No models loaded'}
                model_name = loaded_models[0]

            if model_name not in self.models:
                return {'error': f'Model not found: {model_name}'}

            model_info = self.models[model_name]
            if not model_info.is_loaded():
                return {'error': f'Model not loaded: {model_name}'}

            # Perform prediction (simplified - actual implementation depends on model type)
            model = model_info.model
            tokenizer = model_info.tokenizer

            if model is None or tokenizer is None:
                return {'error': 'Model or tokenizer not available'}

            # Basic prediction logic (this should be expanded based on model type)
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            outputs = model(**inputs)

            # Process outputs based on model type
            # This is a simplified version - actual processing depends on the specific model
            result = {
                'model_name': model_name,
                'input_text': text,
                'raw_outputs': str(outputs),
                'processed': True
            }

            logger.debug(f"Prediction completed for model: {model_name}")
            return result

        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            logger.error(error_msg)
            return {'error': error_msg}

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory usage summary."""
        summary = memory_manager.get_memory_summary()

        # Add model-specific memory info
        summary['models'] = {}
        for name, info in self.models.items():
            if info.is_loaded():
                summary['models'][name] = {
                    'memory_mb': info.get_memory_mb(),
                    'memory_gb': info.get_memory_gb(),
                    'status': info.status
                }

        return summary

    def cleanup_memory(self):
        """Clean up system and GPU memory."""
        memory_manager.cleanup_gpu_memory()
        logger.info("Memory cleanup completed")

    def get_model_count_by_status(self) -> Dict[str, int]:
        """Get count of models by status."""
        status_counts = {
            ModelStatus.LOADED: 0,
            ModelStatus.LOADING: 0,
            ModelStatus.ERROR: 0,
            ModelStatus.UNLOADED: 0
        }

        for info in self.models.values():
            status_counts[info.status] = status_counts.get(info.status, 0) + 1

        return status_counts

    def load_model(self, model_path: str, model_name: str) -> bool:
        """
        Synchronous model loading for backward compatibility.

        Args:
            model_path: Path or HuggingFace model ID
            model_name: Name for the model

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            # Start async loading
            thread = self.load_model_async(model_name, model_path)

            # Wait for completion
            thread.join()

            # Check if model was loaded successfully
            model_info = self.get_model_info(model_name)
            if model_info and model_info.is_loaded():
                logger.info(f"Model {model_name} loaded successfully (sync)")
                return True
            else:
                logger.error(f"Model {model_name} failed to load (sync)")
                return False

        except Exception as e:
            logger.error(f"Synchronous model loading failed: {e}")
            return False

    @property
    def loaded_models(self) -> Dict[str, Any]:
        """
        Backward compatibility property for loaded models.

        Returns:
            Dictionary mapping model names to model info objects
        """
        return {
            name: info for name, info in self.models.items()
            if info.is_loaded()
        }

    @property
    def loading_models(self) -> List[str]:
        """
        Backward compatibility property for loading models.

        Returns:
            List of model names currently loading
        """
        return self.get_loading_models()

    @property
    def failed_models(self) -> Dict[str, str]:
        """
        Backward compatibility property for failed models.

        Returns:
            Dictionary mapping failed model names to error messages
        """
        return self.get_failed_models()

    def shutdown(self):
        """Shutdown model manager and clean up resources."""
        logger.info("Shutting down ModelManager...")

        # Unload all models
        model_names = list(self.models.keys())
        for model_name in model_names:
            try:
                self.unload_model(model_name)
            except Exception as e:
                logger.error(f"Error unloading model {model_name} during shutdown: {e}")

        # Clear callbacks
        self.callbacks.clear()

        # Clean up memory
        self.cleanup_memory()

        logger.info("ModelManager shutdown completed")


# Singleton pattern for global access
class ModelManagerSingleton:
    """Singleton wrapper for ModelManager."""
    _instance: Optional[ModelManager] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> ModelManager:
        """Get or create ModelManager instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = ModelManager()
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.shutdown()
            cls._instance = None


# Global access function
def get_model_manager() -> ModelManager:
    """Get the global ModelManager instance."""
    return ModelManagerSingleton.get_instance()


# Backward compatibility alias
MultiModelManager = ModelManager