"""
Configuration analyzer component for model type detection.

This module provides config.json-based analysis functionality,
extracted from the original monolithic model_type_detector.py.
"""

import json
import os
from typing import Dict, List, Optional, Tuple, Any
from .pattern_matchers import PatternMatcher
from .model_database import ModelDatabase


class ConfigAnalyzer:
    """Handles config.json-based model type analysis."""

    def __init__(self):
        """Initialize config analyzer."""
        self.pattern_matcher = PatternMatcher()
        self.model_database = ModelDatabase()

    def analyze_config(self, model_path: str) -> Tuple[Optional[str], Optional[str], float]:
        """config.json 파일을 분석하여 모델 타입 감지."""
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            print(f"[CONFIG_ANALYZER] config.json not found: {config_path}")
            return None, None, 0.0

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            print(f"[CONFIG_ANALYZER] config.json loaded successfully")

            # architectures 필드 분석 (가장 신뢰할 수 있는 정보)
            if "architectures" in config:
                architectures = config["architectures"]
                if architectures:
                    arch = architectures[0]  # 첫 번째 아키텍처 사용

                    # 직접 매핑된 아키텍처 확인
                    task_type = self.pattern_matcher.get_task_from_architecture(arch)
                    if task_type:
                        model_class = self.model_database.get_task_to_model_class_mapping().get(task_type, "AutoModel")
                        print(f"[CONFIG_ANALYZER] Architecture analysis: {arch} -> {task_type}")
                        return task_type, model_class, 0.95

                    # 아키텍처 이름에서 태스크 유형 동적 추출
                    task_type, confidence = self.extract_task_from_architecture(arch, config)
                    if task_type:
                        model_class = self.model_database.get_task_to_model_class_mapping().get(task_type, "AutoModel")
                        print(f"[CONFIG_ANALYZER] Dynamic architecture analysis: {arch} -> {task_type}")
                        return task_type, model_class, confidence

            # model_type 기반 추론
            model_type = config.get("model_type", "")
            if model_type:
                task_type, model_class, confidence = self.analyze_model_type_config(config, model_type)
                if task_type:
                    return task_type, model_class, confidence

            # task_specific_params 확인
            if "task_specific_params" in config:
                tasks = list(config["task_specific_params"].keys())
                if tasks:
                    task = tasks[0]
                    model_class = self.model_database.get_task_to_model_class_mapping().get(task, "AutoModel")
                    print(f"[CONFIG_ANALYZER] task_specific_params detected: {task}")
                    return task, model_class, 0.8

            print(f"[CONFIG_ANALYZER] config.json analysis complete, no clear task detection")
            return None, None, 0.0

        except Exception as e:
            print(f"[CONFIG_ANALYZER] config.json analysis error: {e}")
            return None, None, 0.0

    def analyze_model_type_config(self, config: Dict[str, Any], model_type: str) -> Tuple[Optional[str], Optional[str], float]:
        """Analyze config based on model_type field."""
        # Special cases for BERT-like models
        if model_type in ["bert", "distilbert", "roberta", "xlm-roberta", "deberta", "deberta-v2", "electra"]:
            # Enhanced analysis for BERT-like models including DeBERTa and ELECTRA
            num_labels = config.get("num_labels", 0)
            has_id2label = "id2label" in config
            has_label2id = "label2id" in config
            problem_type = config.get("problem_type", "")

            # Check for classification indicators
            if num_labels > 1 or has_id2label or has_label2id or problem_type:
                # Determine if it's sequence or token classification
                if (num_labels > 10 or
                    (has_id2label and len(config.get("id2label", {})) > 10) or
                    problem_type == "token_classification"):
                    # Likely token classification (NER usually has more labels)
                    print(f"[CONFIG_ANALYZER] Token classification detected: {model_type} (num_labels={num_labels})")
                    return "token-classification", "AutoModelForTokenClassification", 0.9
                else:
                    # Sequence classification (sentiment, text classification)
                    print(f"[CONFIG_ANALYZER] Sequence classification detected: {model_type} (num_labels={num_labels})")
                    return "text-classification", "AutoModelForSequenceClassification", 0.9

        elif model_type in ["gpt2", "gpt_neo", "llama", "bloom"]:
            print(f"[CONFIG_ANALYZER] Generation model type detected: {model_type}")
            return "text-generation", "AutoModelForCausalLM", 0.9

        elif model_type in ["t5", "bart", "pegasus"]:
            print(f"[CONFIG_ANALYZER] Seq2Seq model type detected: {model_type}")
            return "text2text-generation", "AutoModelForSeq2SeqLM", 0.9

        return None, None, 0.0

    def extract_task_from_architecture(self, architecture: str, config: Dict) -> Tuple[Optional[str], float]:
        """아키텍처 이름에서 태스크 유형을 동적으로 추출."""
        # Use pattern matcher for architecture analysis
        task_type, confidence = self.pattern_matcher.extract_task_from_architecture_pattern(architecture)
        if task_type:
            return task_type, confidence

        # Additional logic for base models
        if self.pattern_matcher.is_base_model(architecture):
            # 기본 모델인 경우 config 정보로 태스크 추정
            task_type, confidence = self.infer_task_from_config_details(architecture, config)
            if task_type:
                return task_type, confidence

            # 기본값: feature-extraction
            return "feature-extraction", 0.7

        return None, 0.0

    def infer_task_from_config_details(self, architecture: str, config: Dict) -> Tuple[Optional[str], float]:
        """config.json의 세부 정보로 태스크 추정."""

        # 현재 감지 중인 모델 이름 확인 (전역 컨텍스트에서)
        current_model_name = getattr(self, '_current_model_name', '')
        if current_model_name and 'multitask' in current_model_name.lower():
            print(f"[CONFIG_ANALYZER] multitask model detected (model name): {current_model_name}")
            return "text-classification", 0.9

        # Use pattern matcher for label analysis
        task_type, confidence = self.pattern_matcher.analyze_config_labels(config)
        if task_type:
            return task_type, confidence

        # 특수 필드 확인
        special_fields = {
            "task_specific_params": lambda x: list(x.keys())[0] if x else None,
            "finetuning_task": lambda x: x,
            "downstream_task": lambda x: x
        }

        for field, extractor in special_fields.items():
            if field in config:
                task = extractor(config[field])
                if task and task in self.model_database.get_task_to_model_class_mapping():
                    return task, 0.8

        # 모델 이름 분석 (최후 수단)
        model_name = config.get("_name_or_path", "")
        if model_name:
            # multitask 키워드 확인
            if "multitask" in model_name.lower():
                print(f"[CONFIG_ANALYZER] multitask model detected: {model_name}")
                return "text-classification", 0.8

            # 다른 태스크 키워드 확인
            task_keywords = {
                "sentiment": "text-classification",
                "classification": "text-classification",
                "ner": "token-classification",
                "entity": "token-classification",
                "qa": "question-answering",
                "question": "question-answering",
                "generation": "text-generation",
                "translation": "translation",
                "summarization": "summarization"
            }

            for keyword, task in task_keywords.items():
                if keyword in model_name.lower():
                    print(f"[CONFIG_ANALYZER] Model name keyword detected: {keyword} -> {task}")
                    return task, 0.75

        return None, 0.0

    def load_config_safely(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Safely load config.json file."""
        config_path = os.path.join(model_path, "config.json")

        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[CONFIG_ANALYZER] Error loading config: {e}")
            return None

    def extract_config_metadata(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract useful metadata from config."""
        metadata = {
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures", []),
            "num_labels": config.get("num_labels"),
            "vocab_size": config.get("vocab_size"),
            "hidden_size": config.get("hidden_size"),
            "max_position_embeddings": config.get("max_position_embeddings"),
            "has_labels": "id2label" in config and "label2id" in config,
            "label_count": len(config.get("id2label", {})),
            "problem_type": config.get("problem_type"),
            "task_specific_params": list(config.get("task_specific_params", {}).keys()),
        }

        # Filter out None values
        return {k: v for k, v in metadata.items() if v is not None}

    def validate_config_structure(self, config: Dict[str, Any]) -> Dict[str, bool]:
        """Validate config structure and return validation results."""
        validations = {
            "has_model_type": "model_type" in config,
            "has_architectures": "architectures" in config and len(config.get("architectures", [])) > 0,
            "has_vocab_size": "vocab_size" in config,
            "has_labels": "id2label" in config and "label2id" in config,
            "labels_consistent": self._validate_labels_consistency(config),
            "has_task_params": "task_specific_params" in config,
        }

        return validations

    def _validate_labels_consistency(self, config: Dict[str, Any]) -> bool:
        """Validate that id2label and label2id are consistent."""
        if "id2label" not in config or "label2id" not in config:
            return True  # Can't validate if not present

        id2label = config["id2label"]
        label2id = config["label2id"]

        # Check if they're inverse mappings
        try:
            for id_str, label in id2label.items():
                if label not in label2id or str(label2id[label]) != str(id_str):
                    return False
            return True
        except (KeyError, ValueError):
            return False