"""
Performance optimization utilities for model inference and memory management.
"""
import time
import torch
from typing import Tuple, Any, List, Dict
from core.logging_config import get_logger

logger = get_logger(__name__)


def optimize_memory():
    """Optimize memory usage settings for model inference."""
    try:
        # Disable gradient computation
        torch.set_grad_enabled(False)

        # Enable optimizations if CUDA is available
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True

            # Clear CUDA cache
            torch.cuda.empty_cache()

        logger.info("Memory optimization settings applied")

    except Exception as e:
        logger.error(f"Memory optimization failed: {e}")


def benchmark_inference(model, tokenizer, text: str, num_runs: int = 100) -> Tuple[float, Any]:
    """
    Benchmark model inference performance.

    Args:
        model: Model to benchmark
        tokenizer: Tokenizer for the model
        text: Input text for testing
        num_runs: Number of runs for benchmarking

    Returns:
        Tuple of (average_time, sample_output)
    """
    try:
        logger.info(f"Starting inference benchmark with {num_runs} runs")

        # Warm-up run
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            _ = model(**inputs)

        # Benchmark runs
        times = []
        sample_output = None

        for i in range(num_runs):
            start_time = time.time()

            with torch.no_grad():
                outputs = model(**inputs)

            end_time = time.time()
            times.append(end_time - start_time)

            if i == 0:
                sample_output = outputs

        avg_time = sum(times) / len(times)
        logger.info(f"Benchmark completed - Average time: {avg_time*1000:.2f}ms")

        return avg_time, sample_output

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 0.0, None


@torch.no_grad()
def optimized_inference(model, tokenizer, text: str, max_length: int = 512) -> Any:
    """
    Perform optimized model inference.

    Args:
        model: Model for inference
        tokenizer: Tokenizer for the model
        text: Input text
        max_length: Maximum sequence length

    Returns:
        Model outputs
    """
    try:
        # Tokenize input
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )

        # Move to GPU if available
        if torch.cuda.is_available() and next(model.parameters()).is_cuda:
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Use automatic mixed precision if available
        if torch.cuda.is_available():
            with torch.cuda.amp.autocast():
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

        return outputs

    except Exception as e:
        logger.error(f"Optimized inference failed: {e}")
        return None


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage information
    """
    try:
        import psutil

        # System memory
        system_memory = psutil.virtual_memory()
        memory_info = {
            'system_total_gb': system_memory.total / (1024**3),
            'system_used_gb': system_memory.used / (1024**3),
            'system_available_gb': system_memory.available / (1024**3),
            'system_percent': system_memory.percent
        }

        # GPU memory
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = f'gpu_{i}'
                memory_info[f'{device_name}_allocated_gb'] = torch.cuda.memory_allocated(i) / (1024**3)
                memory_info[f'{device_name}_reserved_gb'] = torch.cuda.memory_reserved(i) / (1024**3)
                memory_info[f'{device_name}_max_memory_gb'] = torch.cuda.get_device_properties(i).total_memory / (1024**3)

        return memory_info

    except Exception as e:
        logger.error(f"Failed to get memory usage: {e}")
        return {}


def cleanup_memory():
    """Clean up memory and GPU cache."""
    try:
        # Clear Python garbage collection
        import gc
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        logger.info("Memory cleanup completed")

    except Exception as e:
        logger.error(f"Memory cleanup failed: {e}")


def optimize_model_for_inference(model):
    """
    Optimize model for inference performance.

    Args:
        model: Model to optimize

    Returns:
        Optimized model
    """
    try:
        # Set to evaluation mode
        model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        # Apply torch.jit compilation if possible
        try:
            # This is experimental and may not work for all models
            model = torch.jit.optimize_for_inference(model)
            logger.info("Applied torch.jit optimization")
        except Exception as jit_e:
            logger.debug(f"torch.jit optimization not available: {jit_e}")

        return model

    except Exception as e:
        logger.error(f"Model optimization failed: {e}")
        return model


def profile_model_memory(model, tokenizer, sample_text: str) -> Dict[str, Any]:
    """
    Profile model memory usage during inference.

    Args:
        model: Model to profile
        tokenizer: Tokenizer for the model
        sample_text: Sample text for profiling

    Returns:
        Dictionary with profiling results
    """
    try:
        profile_results = {}

        # Get initial memory
        initial_memory = get_memory_usage()
        profile_results['initial_memory'] = initial_memory

        # Perform inference
        with torch.no_grad():
            inputs = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)

            if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Memory after tokenization
            after_tokenization = get_memory_usage()
            profile_results['after_tokenization'] = after_tokenization

            # Perform inference
            outputs = model(**inputs)

            # Memory after inference
            after_inference = get_memory_usage()
            profile_results['after_inference'] = after_inference

        # Calculate deltas
        if 'system_used_gb' in initial_memory and 'system_used_gb' in after_inference:
            profile_results['memory_delta_gb'] = (
                after_inference['system_used_gb'] - initial_memory['system_used_gb']
            )

        return profile_results

    except Exception as e:
        logger.error(f"Memory profiling failed: {e}")
        return {}


class InferenceOptimizer:
    """Class for managing inference optimizations."""

    def __init__(self, model, tokenizer):
        """
        Initialize optimizer.

        Args:
            model: Model to optimize
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizations_applied = []

    def apply_memory_optimizations(self):
        """Apply memory optimization settings."""
        try:
            optimize_memory()
            self.optimizations_applied.append("memory_optimizations")
            logger.info("Memory optimizations applied")
        except Exception as e:
            logger.error(f"Failed to apply memory optimizations: {e}")

    def apply_model_optimizations(self):
        """Apply model-level optimizations."""
        try:
            self.model = optimize_model_for_inference(self.model)
            self.optimizations_applied.append("model_optimizations")
            logger.info("Model optimizations applied")
        except Exception as e:
            logger.error(f"Failed to apply model optimizations: {e}")

    def benchmark_performance(self, sample_text: str, num_runs: int = 50) -> Dict[str, Any]:
        """
        Benchmark model performance.

        Args:
            sample_text: Text for benchmarking
            num_runs: Number of benchmark runs

        Returns:
            Benchmark results
        """
        try:
            avg_time, sample_output = benchmark_inference(
                self.model, self.tokenizer, sample_text, num_runs
            )

            return {
                'average_time_ms': avg_time * 1000,
                'samples_per_second': 1 / avg_time if avg_time > 0 else 0,
                'num_runs': num_runs,
                'sample_output_shape': str(sample_output.last_hidden_state.shape) if hasattr(sample_output, 'last_hidden_state') else 'unknown',
                'optimizations_applied': self.optimizations_applied
            }

        except Exception as e:
            logger.error(f"Performance benchmark failed: {e}")
            return {}

    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report."""
        memory_info = get_memory_usage()

        return {
            'optimizations_applied': self.optimizations_applied,
            'memory_usage': memory_info,
            'device': next(self.model.parameters()).device.type if hasattr(self.model, 'parameters') else 'unknown',
            'model_dtype': str(next(self.model.parameters()).dtype) if hasattr(self.model, 'parameters') else 'unknown'
        }