"""
Memory management utilities for model loading and monitoring.
"""
import psutil
import torch
from typing import Dict, List, Optional, Any
from core.logging_config import get_logger

logger = get_logger(__name__)


class MemoryManager:
    """Manages system and GPU memory for model operations."""

    def __init__(self, max_memory_threshold: float = 0.8):
        """
        Initialize memory manager.

        Args:
            max_memory_threshold: Maximum memory usage threshold (0-1)
        """
        self.max_memory_threshold = max_memory_threshold

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Get current memory usage information.

        Returns:
            Dict containing system and GPU memory information
        """
        memory = psutil.virtual_memory()
        gpu_memory = []

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    gpu_memory.append({
                        'device': i,
                        'name': torch.cuda.get_device_name(i),
                        'total': torch.cuda.get_device_properties(i).total_memory,
                        'allocated': torch.cuda.memory_allocated(i),
                        'reserved': torch.cuda.memory_reserved(i),
                        'free': torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)
                    })
                except Exception as e:
                    logger.warning(f"Failed to get GPU {i} memory info: {e}")

        return {
            'system_memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'free': memory.available
            },
            'gpu_memory': gpu_memory
        }

    def can_load_model(self, estimated_size: float) -> bool:
        """
        Check if model can be loaded based on available memory.

        Args:
            estimated_size: Estimated model size in bytes

        Returns:
            True if model can be loaded, False otherwise
        """
        try:
            memory_info = self.get_memory_info()
            current_usage = memory_info['system_memory']['percent'] / 100

            # Calculate estimated usage after loading
            total_memory = memory_info['system_memory']['total']
            estimated_usage = current_usage + (estimated_size / total_memory)

            can_load = estimated_usage < self.max_memory_threshold

            logger.info(f"Memory check - Current: {current_usage:.2%}, "
                       f"Estimated after load: {estimated_usage:.2%}, "
                       f"Threshold: {self.max_memory_threshold:.2%}, "
                       f"Can load: {can_load}")

            return can_load

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return True  # Allow loading if check fails

    def get_available_memory_bytes(self) -> int:
        """Get available system memory in bytes."""
        try:
            memory_info = self.get_memory_info()
            return memory_info['system_memory']['available']
        except Exception as e:
            logger.error(f"Failed to get available memory: {e}")
            return 0

    def get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage."""
        try:
            memory_info = self.get_memory_info()
            return memory_info['system_memory']['percent']
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return 0.0

    def cleanup_gpu_memory(self):
        """Clean up GPU memory caches."""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU memory cache cleaned up")
        except Exception as e:
            logger.error(f"GPU memory cleanup failed: {e}")

    def get_optimal_device(self) -> str:
        """
        Get optimal device for model loading.

        Returns:
            Device string ('cuda:0', 'cuda:1', 'cpu', etc.)
        """
        try:
            if not torch.cuda.is_available():
                return "cpu"

            # Find GPU with most available memory
            memory_info = self.get_memory_info()
            gpu_memory = memory_info['gpu_memory']

            if not gpu_memory:
                return "cpu"

            # Sort by available memory (descending)
            sorted_gpus = sorted(gpu_memory, key=lambda x: x['free'], reverse=True)
            best_gpu = sorted_gpus[0]

            # Use GPU if it has more than 2GB free memory
            if best_gpu['free'] > 2 * 1024**3:  # 2GB
                return f"cuda:{best_gpu['device']}"
            else:
                return "cpu"

        except Exception as e:
            logger.error(f"Failed to determine optimal device: {e}")
            return "cpu"

    def estimate_model_size(self, config: Dict[str, Any]) -> float:
        """
        Estimate model size based on configuration.

        Args:
            config: Model configuration dictionary

        Returns:
            Estimated size in bytes
        """
        try:
            # Basic estimation based on common model parameters
            vocab_size = config.get('vocab_size', 50000)
            hidden_size = config.get('hidden_size', 768)
            num_layers = config.get('num_hidden_layers', 12)

            # Rough estimation (this can be improved)
            # Each parameter is typically 4 bytes (float32)
            estimated_params = vocab_size * hidden_size + num_layers * hidden_size * hidden_size * 4
            estimated_size = estimated_params * 4  # 4 bytes per parameter

            logger.debug(f"Estimated model size: {estimated_size / (1024**2):.1f} MB")
            return estimated_size

        except Exception as e:
            logger.error(f"Model size estimation failed: {e}")
            return 1024**3  # Default to 1GB if estimation fails

    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary."""
        try:
            memory_info = self.get_memory_info()

            summary = {
                'system': {
                    'total_gb': memory_info['system_memory']['total'] / (1024**3),
                    'used_gb': memory_info['system_memory']['used'] / (1024**3),
                    'free_gb': memory_info['system_memory']['free'] / (1024**3),
                    'usage_percent': memory_info['system_memory']['percent']
                },
                'gpu': []
            }

            for gpu in memory_info['gpu_memory']:
                summary['gpu'].append({
                    'device': gpu['device'],
                    'name': gpu['name'],
                    'total_gb': gpu['total'] / (1024**3),
                    'allocated_gb': gpu['allocated'] / (1024**3),
                    'free_gb': gpu['free'] / (1024**3),
                    'utilization_percent': (gpu['allocated'] / gpu['total']) * 100 if gpu['total'] > 0 else 0
                })

            return summary

        except Exception as e:
            logger.error(f"Failed to get memory summary: {e}")
            return {'system': {}, 'gpu': []}

    def monitor_memory_during_loading(self) -> Dict[str, float]:
        """Monitor memory usage during model loading."""
        try:
            before = self.get_memory_info()
            return {
                'system_before': before['system_memory']['percent'],
                'gpu_before': [gpu['allocated'] for gpu in before['gpu_memory']]
            }
        except Exception as e:
            logger.error(f"Memory monitoring failed: {e}")
            return {}

    def calculate_memory_delta(self, before_stats: Dict[str, float]) -> Dict[str, Any]:
        """Calculate memory usage delta."""
        try:
            after = self.get_memory_info()

            delta = {
                'system_delta': after['system_memory']['percent'] - before_stats.get('system_before', 0),
                'gpu_delta': []
            }

            gpu_before = before_stats.get('gpu_before', [])
            for i, gpu in enumerate(after['gpu_memory']):
                before_allocated = gpu_before[i] if i < len(gpu_before) else 0
                delta['gpu_delta'].append(gpu['allocated'] - before_allocated)

            return delta

        except Exception as e:
            logger.error(f"Memory delta calculation failed: {e}")
            return {'system_delta': 0, 'gpu_delta': []}


# Global memory manager instance
memory_manager = MemoryManager()