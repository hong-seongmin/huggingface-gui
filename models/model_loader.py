"""
Model loading utilities and asynchronous loading management.
"""
import os
import time
import threading
import json
from datetime import datetime
from typing import Optional, Callable, Dict, Any, Tuple
from huggingface_hub import HfApi, snapshot_download

from .model_info import ModelInfo, ModelStatus, ModelEventType
from .memory_manager import memory_manager
from core.logging_config import get_logger

logger = get_logger(__name__)


class ModelLoader:
    """Handles model loading operations."""

    def __init__(self):
        """Initialize model loader."""
        self.hf_api = HfApi()
        self.loading_locks = {}

    def is_huggingface_model_id(self, model_path: str) -> bool:
        """
        Check if the path is a HuggingFace model ID.

        Args:
            model_path: Path or model ID to check

        Returns:
            True if it's a HuggingFace model ID, False otherwise
        """
        try:
            # Simple heuristic: if it contains '/' but doesn't start with '/' or contain '\\'
            # and doesn't exist as a local path, assume it's a HF model ID
            if '/' in model_path and not model_path.startswith('/') and '\\' not in model_path:
                if not os.path.exists(model_path):
                    return True
            return False
        except Exception as e:
            logger.error(f"Error checking model ID: {e}")
            return False

    def download_huggingface_model(self, model_id: str) -> str:
        """
        Download HuggingFace model to local cache.

        Args:
            model_id: HuggingFace model ID

        Returns:
            Local path to downloaded model
        """
        try:
            logger.info(f"Downloading HuggingFace model: {model_id}")

            # Use snapshot_download to get the complete model
            local_path = snapshot_download(
                repo_id=model_id,
                cache_dir=os.path.expanduser("~/.cache/huggingface/transformers"),
                local_files_only=False,
                use_auth_token=True  # Use saved token if available
            )

            logger.info(f"Model downloaded to: {local_path}")
            return local_path

        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            raise

    def generate_model_name(self, model_path: str) -> str:
        """
        Generate a model name from path or ID.

        Args:
            model_path: Path or HuggingFace model ID

        Returns:
            Generated model name
        """
        if self.is_huggingface_model_id(model_path):
            # Extract model name from HuggingFace model ID
            return model_path.split('/')[-1]
        else:
            # Extract model name from local path
            return os.path.basename(model_path.rstrip('/'))

    def load_model_async(self, model_name: str, model_path: str,
                        callback: Optional[Callable] = None) -> threading.Thread:
        """
        Load model asynchronously.

        Args:
            model_name: Name for the model
            model_path: Path or HuggingFace model ID
            callback: Progress callback function

        Returns:
            Thread handling the loading operation
        """
        # Generate model name if empty
        if not model_name or not model_name.strip():
            model_name = self.generate_model_name(model_path)

        logger.info(f"Starting async model load: {model_name} from {model_path}")

        # Set up loading lock
        if model_name not in self.loading_locks:
            self.loading_locks[model_name] = threading.Lock()

        # Start loading thread
        thread = threading.Thread(
            target=self._load_model_sync,
            args=(model_name, model_path, callback),
            name=f"ModelLoad-{model_name}"
        )
        thread.daemon = True
        thread.start()

        logger.info(f"Loading thread started for {model_name}")
        return thread

    def _load_model_sync(self, model_name: str, model_path: str,
                        callback: Optional[Callable] = None):
        """
        Synchronous model loading (runs in thread).

        Args:
            model_name: Name for the model
            model_path: Path or HuggingFace model ID
            callback: Progress callback function
        """
        start_time = time.time()

        try:
            with self.loading_locks[model_name]:
                logger.info(f"Loading model {model_name} synchronously")

                # Notify loading started
                if callback:
                    callback(model_name, ModelEventType.LOAD_STARTED, {
                        'path': model_path,
                        'start_time': start_time
                    })

                # Resolve actual model path
                actual_model_path = model_path
                if self.is_huggingface_model_id(model_path):
                    actual_model_path = self.download_huggingface_model(model_path)

                # Load model with transformers
                model_info = self._load_with_transformers(
                    model_name, actual_model_path, callback
                )

                load_time = time.time() - start_time
                model_info.load_time = datetime.now()

                logger.info(f"Model {model_name} loaded successfully in {load_time:.2f}s")

                # Notify loading completed
                if callback:
                    callback(model_name, ModelEventType.LOAD_COMPLETED, {
                        'model_info': model_info,
                        'load_time': load_time
                    })

                return model_info

        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {e}"
            logger.error(error_msg)

            # Notify loading failed
            if callback:
                callback(model_name, ModelEventType.LOAD_FAILED, {
                    'error': str(e),
                    'load_time': time.time() - start_time
                })

            raise

    def _load_with_transformers(self, model_name: str, model_path: str,
                               callback: Optional[Callable] = None) -> ModelInfo:
        """
        Load model using transformers library.

        Args:
            model_name: Name for the model
            model_path: Local path to model
            callback: Progress callback function

        Returns:
            ModelInfo object with loaded model
        """
        try:
            from transformers import AutoModel, AutoTokenizer, AutoConfig

            logger.info(f"Loading model with transformers: {model_name}")

            # Monitor memory before loading
            memory_before = memory_manager.monitor_memory_during_loading()

            # Progress update
            if callback:
                callback(model_name, ModelEventType.LOAD_PROGRESS, {
                    'stage': 'loading_config',
                    'progress': 0.1
                })

            # Load configuration
            config = self._load_model_config(model_path)

            # Progress update
            if callback:
                callback(model_name, ModelEventType.LOAD_PROGRESS, {
                    'stage': 'loading_tokenizer',
                    'progress': 0.3
                })

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False
            )

            # Progress update
            if callback:
                callback(model_name, ModelEventType.LOAD_PROGRESS, {
                    'stage': 'loading_model',
                    'progress': 0.5
                })

            # Determine optimal device
            device = memory_manager.get_optimal_device()
            logger.info(f"Loading model on device: {device}")

            # Load model
            model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=False,
                torch_dtype='auto',
                device_map=device if device != 'cpu' else None
            )

            # Move to device if needed
            if device != 'cpu' and not hasattr(model, 'device_map'):
                model = model.to(device)

            # Progress update
            if callback:
                callback(model_name, ModelEventType.LOAD_PROGRESS, {
                    'stage': 'finalizing',
                    'progress': 0.9
                })

            # Calculate memory usage
            memory_after = memory_manager.get_memory_info()
            memory_delta = memory_manager.calculate_memory_delta(memory_before)

            # Create ModelInfo
            model_info = ModelInfo(
                name=model_name,
                path=model_path,
                model=model,
                tokenizer=tokenizer,
                config_analysis=config,
                memory_usage=memory_delta.get('system_delta', 0) * 1024**2,  # Convert to bytes
                status=ModelStatus.LOADED
            )

            logger.info(f"Model {model_name} loaded successfully on {device}")
            return model_info

        except Exception as e:
            logger.error(f"Transformers loading failed for {model_name}: {e}")
            raise

    def _load_model_config(self, model_path: str) -> Dict[str, Any]:
        """
        Load model configuration.

        Args:
            model_path: Path to model directory

        Returns:
            Configuration dictionary
        """
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                logger.debug(f"Loaded config from {config_path}")
                return config
            else:
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_path, local_files_only=True)
                return config.to_dict()

        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def verify_model_files(self, model_path: str) -> Dict[str, bool]:
        """
        Verify required model files exist.

        Args:
            model_path: Path to model directory

        Returns:
            Dictionary of file existence status
        """
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json"
        ]

        model_files = [
            "pytorch_model.bin",
            "model.safetensors",
            "pytorch_model-00001-of-00001.bin"
        ]

        verification = {}

        # Check required files
        for file in required_files:
            file_path = os.path.join(model_path, file)
            verification[file] = os.path.exists(file_path)

        # Check at least one model file exists
        model_file_exists = False
        for file in model_files:
            file_path = os.path.join(model_path, file)
            if os.path.exists(file_path):
                model_file_exists = True
                verification[file] = True
                break

        verification['has_model_file'] = model_file_exists

        return verification

    def unload_model(self, model_info: ModelInfo) -> bool:
        """
        Unload model from memory.

        Args:
            model_info: ModelInfo object to unload

        Returns:
            True if unloaded successfully, False otherwise
        """
        try:
            logger.info(f"Unloading model: {model_info.name}")

            # Update status
            model_info.status = ModelStatus.UNLOADING

            # Clear model and tokenizer references
            if model_info.model is not None:
                del model_info.model
                model_info.model = None

            if model_info.tokenizer is not None:
                del model_info.tokenizer
                model_info.tokenizer = None

            # Clean up GPU memory
            memory_manager.cleanup_gpu_memory()

            # Update status
            model_info.status = ModelStatus.UNLOADED
            model_info.memory_usage = 0.0

            logger.info(f"Model {model_info.name} unloaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to unload model {model_info.name}: {e}")
            model_info.status = ModelStatus.ERROR
            model_info.error_message = str(e)
            return False


# Global model loader instance
model_loader = ModelLoader()