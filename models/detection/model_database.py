"""
Model database component for model type detection.

This module provides database access functionality for model information,
extracted from the original monolithic model_type_detector.py.
"""

from typing import Dict, List, Optional, Any
import sys
import os

# Add project root to path to import model_database
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from model_database import model_database as global_model_db
except ImportError:
    global_model_db = None


class ModelDatabase:
    """Wrapper for accessing model database information."""

    def __init__(self):
        """Initialize model database wrapper."""
        self.db = global_model_db

        # Cache for task-to-model-class mapping
        self._task_to_model_class = None
        self._supported_model_types = None

    def get_task_to_model_class_mapping(self) -> Dict[str, str]:
        """Get mapping from task types to model classes."""
        if self._task_to_model_class is None:
            self._task_to_model_class = {
                "text-classification": "AutoModelForSequenceClassification",
                "token-classification": "AutoModelForTokenClassification",
                "question-answering": "AutoModelForQuestionAnswering",
                "text-generation": "AutoModelForCausalLM",
                "text2text-generation": "AutoModelForSeq2SeqLM",
                "feature-extraction": "AutoModel",
                "fill-mask": "AutoModelForMaskedLM",
                "translation": "AutoModelForSeq2SeqLM",
                "summarization": "AutoModelForSeq2SeqLM",
                "image-classification": "AutoModelForImageClassification",
                "automatic-speech-recognition": "AutoModelForSpeechSeq2Seq",
            }
        return self._task_to_model_class

    def get_all_model_types(self) -> List[str]:
        """Get all supported model types."""
        if self._supported_model_types is None:
            if self.db is not None:
                try:
                    self._supported_model_types = self.db.get_all_model_types()
                except AttributeError:
                    # Fallback if method doesn't exist
                    self._supported_model_types = self._get_default_model_types()
            else:
                self._supported_model_types = self._get_default_model_types()

        return self._supported_model_types

    def get_model_info(self, model_type: str):
        """Get model information from database."""
        if self.db is not None:
            try:
                return self.db.get_model_info(model_type)
            except AttributeError:
                # Fallback if method doesn't exist
                return self._get_default_model_info(model_type)
        else:
            return self._get_default_model_info(model_type)

    def _get_default_model_types(self) -> List[str]:
        """Get default supported model types as fallback."""
        return [
            "bert", "distilbert", "roberta", "xlm-roberta", "albert",
            "deberta", "deberta-v2", "electra", "camembert", "flaubert",
            "xlnet", "longformer", "bigbird", "gpt2", "gpt-neo", "gpt-j",
            "llama", "bloom", "opt", "t5", "bart", "pegasus", "marian",
            "m2m100", "mbart", "blenderbot", "wav2vec2", "whisper",
            "vit", "deit", "swin", "clip"
        ]

    def _get_default_model_info(self, model_type: str):
        """Get default model information as fallback."""
        # Mock model info structure
        class MockModelInfo:
            def __init__(self, primary_tasks=None):
                self.primary_tasks = primary_tasks or []

        class MockTaskType:
            def __init__(self, value):
                self.value = value

        # Default task mappings for common model types
        default_tasks = {
            "bert": [MockTaskType("text-classification")],
            "distilbert": [MockTaskType("text-classification")],
            "roberta": [MockTaskType("text-classification")],
            "deberta": [MockTaskType("text-classification")],
            "electra": [MockTaskType("text-classification")],
            "gpt2": [MockTaskType("text-generation")],
            "gpt-neo": [MockTaskType("text-generation")],
            "llama": [MockTaskType("text-generation")],
            "bloom": [MockTaskType("text-generation")],
            "t5": [MockTaskType("text2text-generation")],
            "bart": [MockTaskType("text2text-generation")],
            "pegasus": [MockTaskType("summarization")],
        }

        if model_type in default_tasks:
            return MockModelInfo(default_tasks[model_type])
        else:
            # Default to text classification for unknown models
            return MockModelInfo([MockTaskType("text-classification")])

    def is_available(self) -> bool:
        """Check if the model database is available."""
        return self.db is not None

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the model database."""
        return {
            "available": self.is_available(),
            "total_model_types": len(self.get_all_model_types()),
            "total_task_mappings": len(self.get_task_to_model_class_mapping())
        }