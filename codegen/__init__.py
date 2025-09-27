"""
Code generation package for HuggingFace models.

This package provides modularized code generation utilities for:
- Model loading snippets
- Inference examples
- Performance optimization
- Fine-tuning scripts
- Dataset utilities
"""

from .snippets import (
    CodeSnippet,
    SnippetType,
    SnippetTemplate,
    SnippetValidator,
    SnippetMetadata
)
from .generators import (
    CodeSnippetGenerator,
    LoadingSnippetTemplate,
    InferenceSnippetTemplate,
    OptimizationSnippetTemplate,
    FineTuningSnippetTemplate,
    extract_features
)
from .optimization import (
    optimize_memory,
    benchmark_inference,
    optimized_inference,
    InferenceOptimizer,
    get_memory_usage,
    cleanup_memory
)
from .datasets import (
    TextDataset,
    CustomClassificationDataset,
    CustomTokenClassificationDataset,
    ImageTextDataset,
    AudioDataset,
    DatasetBuilder,
    create_dataloader,
    extract_features,
    compute_metrics,
    get_sample_data_for_task,
    CustomDataset  # Legacy alias
)

__all__ = [
    # Snippets
    'CodeSnippet',
    'SnippetType',
    'SnippetTemplate',
    'SnippetValidator',
    'SnippetMetadata',

    # Generators
    'CodeSnippetGenerator',
    'LoadingSnippetTemplate',
    'InferenceSnippetTemplate',
    'OptimizationSnippetTemplate',
    'FineTuningSnippetTemplate',

    # Optimization
    'optimize_memory',
    'benchmark_inference',
    'optimized_inference',
    'InferenceOptimizer',
    'get_memory_usage',
    'cleanup_memory',

    # Datasets
    'TextDataset',
    'CustomClassificationDataset',
    'CustomTokenClassificationDataset',
    'ImageTextDataset',
    'AudioDataset',
    'DatasetBuilder',
    'create_dataloader',
    'extract_features',
    'compute_metrics',
    'get_sample_data_for_task',
    'CustomDataset'  # Legacy alias
]