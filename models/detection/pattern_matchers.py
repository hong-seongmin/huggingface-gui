"""
Pattern matching component for model type detection.

This module provides pattern-based analysis for model names and architectures,
extracted from the original monolithic model_type_detector.py.
"""

import re
from typing import Dict, List, Optional, Tuple, Any


class PatternMatcher:
    """Handles pattern-based model type detection."""

    def __init__(self):
        """Initialize pattern matcher with comprehensive patterns."""

        # Model name patterns for different tasks - ENHANCED VERSION
        self.name_patterns = {
            "text-classification": [
                r"sentiment", r"classification", r"classifier", r"toxic", r"emotion",
                r"stance", r"intent", r"category", r"label", r"multilingual-sentiment",
                r"deberta.*classification", r"electra.*classification",
                r"hate", r"abuse", r"bias", r"polarity"
            ],
            "token-classification": [
                r"ner", r"named.*entity", r"token.*class", r"pos.*tag", r"chunk",
                r"electra.*ner", r"deberta.*ner", r"klue.*test", r"korean.*ner"
            ],
            "text-generation": [
                r"gpt", r"generator", r"chat", r"conversational", r"instruct",
                r"dialogue", r"story", r"llama", r"mistral", r"phi", r"qwen"
            ],
            "translation": [
                r"translate", r"translation", r"mt-", r"opus-mt", r"nllb",
                r"marian", r"m2m100", r"multilingual.*translation"
            ],
            "summarization": [
                r"summar", r"abstract", r"headline", r"pegasus"
            ],
            "question-answering": [
                r"qa", r"question", r"squad", r"answer", r"reading.*comprehension"
            ],
            "feature-extraction": [
                r"embedding", r"encode", r"sentence", r"similarity", r"retrieval",
                r"bge", r"e5", r"gte", r"instructor", r"all.*MiniLM"
            ],
            "text2text-generation": [
                r"t5", r"bart", r"multitask", r"multi.*task", r"unified"
            ]
        }

        # Architecture to task mapping - COMPREHENSIVE VERSION
        self.architecture_to_task = {
            # === SEQUENCE CLASSIFICATION (텍스트 분류) ===
            # BERT family
            "BertForSequenceClassification": "text-classification",
            "DistilBertForSequenceClassification": "text-classification",
            "RobertaForSequenceClassification": "text-classification",
            "XLMRobertaForSequenceClassification": "text-classification",
            "AlbertForSequenceClassification": "text-classification",

            # DeBERTa family (MISSING - KEY FIX)
            "DebertaForSequenceClassification": "text-classification",
            "DebertaV2ForSequenceClassification": "text-classification",
            "DebertaV3ForSequenceClassification": "text-classification",

            # ELECTRA family (MISSING - KEY FIX)
            "ElectraForSequenceClassification": "text-classification",

            # Other transformer variants
            "CamembertForSequenceClassification": "text-classification",
            "FlaubertForSequenceClassification": "text-classification",
            "XLNetForSequenceClassification": "text-classification",
            "ReformerForSequenceClassification": "text-classification",
            "LongformerForSequenceClassification": "text-classification",
            "BigBirdForSequenceClassification": "text-classification",
            "ConvBertForSequenceClassification": "text-classification",
            "MobileBertForSequenceClassification": "text-classification",
            "SqueezeBertForSequenceClassification": "text-classification",

            # === TOKEN CLASSIFICATION (NER, POS 태깅) ===
            # BERT family
            "BertForTokenClassification": "token-classification",
            "DistilBertForTokenClassification": "token-classification",
            "RobertaForTokenClassification": "token-classification",
            "XLMRobertaForTokenClassification": "token-classification",
            "AlbertForTokenClassification": "token-classification",

            # DeBERTa family (MISSING - KEY FIX)
            "DebertaForTokenClassification": "token-classification",
            "DebertaV2ForTokenClassification": "token-classification",

            # ELECTRA family (MISSING - KEY FIX)
            "ElectraForTokenClassification": "token-classification",

            # Other variants
            "CamembertForTokenClassification": "token-classification",
            "FlaubertForTokenClassification": "token-classification",
            "XLNetForTokenClassification": "token-classification",
            "LongformerForTokenClassification": "token-classification",
            "BigBirdForTokenClassification": "token-classification",

            # === QUESTION ANSWERING ===
            "BertForQuestionAnswering": "question-answering",
            "DistilBertForQuestionAnswering": "question-answering",
            "RobertaForQuestionAnswering": "question-answering",
            "XLMRobertaForQuestionAnswering": "question-answering",
            "AlbertForQuestionAnswering": "question-answering",
            "DebertaForQuestionAnswering": "question-answering",
            "DebertaV2ForQuestionAnswering": "question-answering",
            "ElectraForQuestionAnswering": "question-answering",
            "LongformerForQuestionAnswering": "question-answering",
            "BigBirdForQuestionAnswering": "question-answering",

            # === TEXT GENERATION ===
            "GPT2LMHeadModel": "text-generation",
            "GPTJForCausalLM": "text-generation",
            "GPTNeoForCausalLM": "text-generation",
            "GPTNeoXForCausalLM": "text-generation",
            "LlamaForCausalLM": "text-generation",
            "BloomForCausalLM": "text-generation",
            "OPTForCausalLM": "text-generation",
            "CodeGenForCausalLM": "text-generation",
            "PersimmonForCausalLM": "text-generation",
            "QwenForCausalLM": "text-generation",
            "MistralForCausalLM": "text-generation",
            "Phi3ForCausalLM": "text-generation",

            # === CONDITIONAL GENERATION (Seq2Seq) ===
            "T5ForConditionalGeneration": "text2text-generation",
            "BartForConditionalGeneration": "text2text-generation",
            "PegasusForConditionalGeneration": "summarization",
            "MarianMTModel": "translation",
            "M2M100ForConditionalGeneration": "translation",
            "MBartForConditionalGeneration": "text2text-generation",
            "ProphetNetForConditionalGeneration": "text2text-generation",
            "BlenderbotForConditionalGeneration": "text2text-generation",
            "BlenderbotSmallForConditionalGeneration": "text2text-generation",

            # === MASKED LANGUAGE MODELING ===
            "BertForMaskedLM": "fill-mask",
            "DistilBertForMaskedLM": "fill-mask",
            "RobertaForMaskedLM": "fill-mask",
            "XLMRobertaForMaskedLM": "fill-mask",
            "AlbertForMaskedLM": "fill-mask",
            "DebertaForMaskedLM": "fill-mask",
            "DebertaV2ForMaskedLM": "fill-mask",
            "ElectraForMaskedLM": "fill-mask",

            # === FEATURE EXTRACTION (순수 임베딩 모델만) ===
            # 주의: ForXXX가 없는 기본 모델들만 여기에 포함
            "BertModel": "feature-extraction",
            "DistilBertModel": "feature-extraction",
            "RobertaModel": "feature-extraction",
            "XLMRobertaModel": "feature-extraction",
            "AlbertModel": "feature-extraction",
            "DebertaModel": "feature-extraction",
            "DebertaV2Model": "feature-extraction",
            "ElectraModel": "feature-extraction",
            "CamembertModel": "feature-extraction",

            # === SPECIAL ARCHITECTURES ===
            # Image models
            "ViTForImageClassification": "image-classification",
            "DeiTForImageClassification": "image-classification",
            "SwinForImageClassification": "image-classification",

            # Audio models
            "Wav2Vec2ForCTC": "automatic-speech-recognition",
            "WhisperForConditionalGeneration": "automatic-speech-recognition",

            # Multi-modal
            "CLIPModel": "feature-extraction",
            "CLIPVisionModel": "feature-extraction",
            "CLIPTextModel": "feature-extraction",
        }

    def analyze_model_name(self, model_name: str) -> Tuple[Optional[str], float]:
        """Analyze model name patterns to infer task type."""
        model_name_lower = model_name.lower()

        scores = {}
        for task, patterns in self.name_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, model_name_lower):
                    score += 1

            if score > 0:
                # Pattern matching count-based confidence calculation
                confidence = min(score / len(patterns), 1.0) * 0.8
                scores[task] = confidence

        if scores:
            best_task = max(scores, key=scores.get)
            best_confidence = scores[best_task]
            print(f"[PATTERN_MATCHER] Model name analysis: {best_task} (confidence: {best_confidence:.2f})")
            return best_task, best_confidence

        return None, 0.0

    def get_task_from_architecture(self, architecture: str) -> Optional[str]:
        """Get task directly from architecture mapping."""
        return self.architecture_to_task.get(architecture)

    def extract_task_from_architecture_pattern(self, architecture: str) -> Tuple[Optional[str], float]:
        """Extract task from architecture name using pattern matching."""
        arch_lower = architecture.lower()

        # Architecture name to task pattern mapping
        task_patterns = {
            "sequenceclassification": "text-classification",
            "tokenclassification": "token-classification",
            "questionanswering": "question-answering",
            "causallm": "text-generation",
            "seq2seqlm": "text2text-generation",
            "conditionalgeneration": "text2text-generation",
            "maskedlm": "fill-mask",
            "imageclassification": "image-classification",
            "speechseq2seq": "automatic-speech-recognition"
        }

        for pattern, task in task_patterns.items():
            if pattern in arch_lower:
                print(f"[PATTERN_MATCHER] Architecture pattern match: {architecture} -> {task}")
                return task, 0.9

        return None, 0.0

    def is_base_model(self, architecture: str) -> bool:
        """Check if architecture represents a base model (for feature extraction)."""
        arch_lower = architecture.lower()
        return arch_lower.endswith("model") and not any(
            task_indicator in arch_lower
            for task_indicator in ["for", "head", "classification", "generation"]
        )

    def get_fallback_detection(self, model_name: str) -> Tuple[str, str]:
        """Enhanced fallback detection with smart multi-task handling."""
        model_name_lower = model_name.lower()

        # Multi-task models - PRIORITY CHECK
        if any(keyword in model_name_lower for keyword in ["multitask", "multi-task", "multi_task"]):
            # Analyze model name for dominant task hints
            if any(keyword in model_name_lower for keyword in ["sentiment", "classification", "emotion"]):
                print(f"[PATTERN_MATCHER] Multi-task model detected, defaulting to text-classification: {model_name}")
                return "text-classification", "AutoModelForSequenceClassification"
            elif any(keyword in model_name_lower for keyword in ["ner", "token", "entity"]):
                print(f"[PATTERN_MATCHER] Multi-task model detected, defaulting to token-classification: {model_name}")
                return "token-classification", "AutoModelForTokenClassification"
            else:
                # Generic multi-task - default to text classification as most common
                print(f"[PATTERN_MATCHER] Generic multi-task model detected, defaulting to text-classification: {model_name}")
                return "text-classification", "AutoModelForSequenceClassification"

        # DeBERTa models without clear architecture
        if "deberta" in model_name_lower:
            if any(keyword in model_name_lower for keyword in ["ner", "token", "entity", "klue"]):
                return "token-classification", "AutoModelForTokenClassification"
            else:
                return "text-classification", "AutoModelForSequenceClassification"

        # ELECTRA models without clear architecture
        if "electra" in model_name_lower:
            if any(keyword in model_name_lower for keyword in ["ner", "token", "entity", "klue"]):
                return "token-classification", "AutoModelForTokenClassification"
            else:
                return "text-classification", "AutoModelForSequenceClassification"

        # Specific task patterns (enhanced)
        if any(keyword in model_name_lower for keyword in ["sentiment", "classification", "classifier", "emotion", "toxic", "hate"]):
            return "text-classification", "AutoModelForSequenceClassification"
        elif any(keyword in model_name_lower for keyword in ["ner", "named-entity", "token-class", "pos-tag"]):
            return "token-classification", "AutoModelForTokenClassification"
        elif any(keyword in model_name_lower for keyword in ["bge", "embedding", "sentence", "similarity", "retrieval", "e5", "gte"]):
            return "feature-extraction", "AutoModel"
        elif any(keyword in model_name_lower for keyword in ["gpt", "generate", "chat", "llama", "mistral"]):
            return "text-generation", "AutoModelForCausalLM"
        elif any(keyword in model_name_lower for keyword in ["t5", "bart", "pegasus", "translate"]):
            return "text2text-generation", "AutoModelForSeq2SeqLM"
        elif any(keyword in model_name_lower for keyword in ["qa", "question", "squad"]):
            return "question-answering", "AutoModelForQuestionAnswering"
        else:
            # Last resort - but be smarter about it
            print(f"[PATTERN_MATCHER] WARNING: No pattern matched for {model_name}, defaulting to feature-extraction")
            return "feature-extraction", "AutoModel"

    def analyze_config_labels(self, config: Dict[str, Any]) -> Tuple[Optional[str], float]:
        """Analyze config labels to determine task type."""
        # Label information analysis (most reliable indicator)
        if "id2label" in config and "label2id" in config:
            id2label = config["id2label"]
            num_labels = len(id2label)

            # Label content analysis
            label_values = list(id2label.values()) if isinstance(id2label, dict) else []
            label_str = " ".join(str(label).lower() for label in label_values)

            # NER/Token Classification pattern check
            ner_patterns = ["b-", "i-", "o", "beginning-", "inside-", "outside-"]
            if any(pattern in label_str for pattern in ner_patterns):
                print(f"[PATTERN_MATCHER] NER label pattern detected: {label_values[:5]}...")
                return "token-classification", 0.95

            # Multi-label classification check
            if num_labels > 1:
                if num_labels <= 10:
                    print(f"[PATTERN_MATCHER] Sequence classification detected: {num_labels} labels")
                    return "text-classification", 0.9
                else:
                    print(f"[PATTERN_MATCHER] Token classification detected: {num_labels} labels")
                    return "token-classification", 0.85

        # num_labels check
        num_labels = config.get("num_labels", 0)
        if num_labels > 1:
            if num_labels <= 10:
                return "text-classification", 0.8
            else:
                return "token-classification", 0.8

        # problem_type check
        problem_type = config.get("problem_type", "")
        if problem_type:
            if "classification" in problem_type:
                return "text-classification", 0.85
            elif "token" in problem_type:
                return "token-classification", 0.85

        return None, 0.0

    def get_comprehensive_patterns(self) -> Dict[str, Any]:
        """Get comprehensive pattern information for debugging."""
        return {
            "name_patterns": self.name_patterns,
            "architecture_count": len(self.architecture_to_task),
            "supported_tasks": list(self.name_patterns.keys())
        }