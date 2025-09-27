"""
AutoConfig analyzer component for model type detection.

This module provides AutoConfig-based analysis functionality,
extracted from the original monolithic model_type_detector.py.
"""

from typing import Dict, List, Optional, Tuple, Any
from .model_database import ModelDatabase


class AutoConfigAnalyzer:
    """Handles AutoConfig-based model type analysis."""

    def __init__(self):
        """Initialize AutoConfig analyzer."""
        self.model_database = ModelDatabase()

    def analyze_with_autoconfig(self, model_path: str, model_name: str) -> Tuple[Optional[str], Optional[str], float]:
        """AutoConfig를 사용한 정밀한 모델 타입 감지."""
        try:
            # transformers의 AutoConfig 사용
            from transformers import AutoConfig

            print(f"[AUTOCONFIG_ANALYZER] AutoConfig loading started: {model_path}")

            # AutoConfig로 설정 로드
            config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            # 모델 타입 추출
            detected_model_type = getattr(config, 'model_type', None)
            if not detected_model_type:
                print(f"[AUTOCONFIG_ANALYZER] No model_type field found")
                return None, None, 0.0

            print(f"[AUTOCONFIG_ANALYZER] Detected model type: {detected_model_type}")

            # 데이터베이스에서 모델 정보 확인
            model_info = self.model_database.get_model_info(detected_model_type)
            if not model_info:
                print(f"[AUTOCONFIG_ANALYZER] No model info in database: {detected_model_type}")
                return None, None, 0.0

            # 아키텍처 분석을 통한 세부 태스크 결정
            architectures = getattr(config, 'architectures', [])
            if architectures:
                arch = architectures[0]
                task_type, confidence = self.determine_task_from_architecture(arch, config, model_name)

                if task_type:
                    model_class = self.model_database.get_task_to_model_class_mapping().get(task_type, "AutoModel")
                    print(f"[AUTOCONFIG_ANALYZER] Architecture-based task determination: {arch} -> {task_type}")
                    return task_type, model_class, confidence

            # 기본 태스크 사용
            if model_info.primary_tasks:
                primary_task = model_info.primary_tasks[0].value
                model_class = self.model_database.get_task_to_model_class_mapping().get(primary_task, "AutoModel")
                print(f"[AUTOCONFIG_ANALYZER] Using default task: {primary_task}")
                return primary_task, model_class, 0.85

            return None, None, 0.0

        except Exception as e:
            print(f"[AUTOCONFIG_ANALYZER] AutoConfig analysis error: {e}")
            return None, None, 0.0

    def determine_task_from_architecture(self, architecture: str, config: Any, model_name: str) -> Tuple[Optional[str], float]:
        """아키텍처와 설정을 기반으로 정확한 태스크 결정."""
        arch_lower = architecture.lower()

        # 1. 아키텍처 이름에서 직접 태스크 유형 판별
        if "forsequenceclassification" in arch_lower:
            return "text-classification", 0.95
        elif "fortokenclassification" in arch_lower:
            return "token-classification", 0.95
        elif "forquestionanswering" in arch_lower:
            return "question-answering", 0.95
        elif "forcausallm" in arch_lower:
            return "text-generation", 0.95
        elif "forseq2seqlm" in arch_lower or "forconditionalgeneration" in arch_lower:
            return "text2text-generation", 0.95
        elif "formaskedlm" in arch_lower:
            return "fill-mask", 0.95
        elif "forimageclassification" in arch_lower:
            return "image-classification", 0.95
        elif "forobjectdetection" in arch_lower:
            return "object-detection", 0.95
        elif "forspeechseq2seq" in arch_lower:
            return "automatic-speech-recognition", 0.95

        # 2. 기본 모델 타입 + 설정 분석
        if arch_lower.endswith("model"):
            # 레이블 정보 분석
            if hasattr(config, 'id2label') and hasattr(config, 'label2id'):
                id2label = config.id2label
                num_labels = len(id2label)

                # 레이블 패턴 분석
                label_values = list(id2label.values()) if isinstance(id2label, dict) else []
                label_str = " ".join(str(label).lower() for label in label_values)

                # NER 패턴 확인
                ner_patterns = ["b-", "i-", "o"]
                if any(pattern in label_str for pattern in ner_patterns):
                    return "token-classification", 0.9

                # 레이블 수에 따른 분류
                if num_labels <= 10:
                    return "text-classification", 0.85
                else:
                    return "token-classification", 0.8

            # multitask 모델 특별 처리
            if "multitask" in model_name.lower():
                return "text-classification", 0.8

            # 모델 이름 기반 추론
            if "ner" in model_name.lower() or "token" in model_name.lower():
                return "token-classification", 0.75
            elif "classification" in model_name.lower():
                return "text-classification", 0.75

            # 기본값: feature-extraction
            return "feature-extraction", 0.7

        return None, 0.0

    def extract_model_metadata(self, model_path: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from AutoConfig."""
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )

            metadata = {
                "model_type": getattr(config, 'model_type', None),
                "architectures": getattr(config, 'architectures', []),
                "vocab_size": getattr(config, 'vocab_size', None),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None),
                "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                "max_position_embeddings": getattr(config, 'max_position_embeddings', None),
                "num_labels": getattr(config, 'num_labels', None),
                "id2label": getattr(config, 'id2label', {}),
                "label2id": getattr(config, 'label2id', {}),
            }

            # Filter out None values
            return {k: v for k, v in metadata.items() if v is not None}

        except Exception as e:
            print(f"[AUTOCONFIG_ANALYZER] Metadata extraction error: {e}")
            return {}

    def is_autoconfig_available(self) -> bool:
        """Check if AutoConfig is available."""
        try:
            from transformers import AutoConfig
            return True
        except ImportError:
            return False

    def get_supported_models(self) -> List[str]:
        """Get list of models supported by AutoConfig."""
        if not self.is_autoconfig_available():
            return []

        # This would typically come from transformers library
        # For now, return common model types
        return [
            "bert", "distilbert", "roberta", "xlm-roberta", "albert",
            "deberta", "deberta-v2", "electra", "gpt2", "gpt-neo",
            "llama", "bloom", "t5", "bart", "pegasus"
        ]