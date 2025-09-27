"""
정밀한 모델 타입 자동 감지 시스템
Hugging Face 메타데이터 기반 정교한 분석을 통한 모델 태스크 유형 판별
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
from pathlib import Path
from model_database import model_database, ModelCategory, TaskType

class ModelTypeDetector:
    """
    Hugging Face 모델의 태스크 유형을 정밀하게 자동 감지하는 클래스
    config.json, 모델 구조, 토크나이저 설정 등을 종합 분석
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ModelTypeDetector")
        
        # 데이터베이스에서 태스크별 모델 클래스 매핑 가져오기
        self.task_to_model_class = model_database.get_task_to_model_class_mapping()
        
        # 지원되는 모델 타입 목록
        self.supported_model_types = model_database.get_all_model_types()
        
        # 아키텍처별 기본 태스크 매핑 (COMPREHENSIVE VERSION)
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
        
        # 모델명 패턴 분석 (보조 수단) - ENHANCED VERSION
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
        
        # 성공 사례 캐시
        self.detection_cache = {}
        
    def detect_model_type(self, model_name: str, model_path: str) -> Tuple[str, str, Dict[str, Any]]:
        """
        모델 타입을 정밀하게 감지
        
        Returns:
            (task_type, model_class, analysis_info)
        """
        analysis_info = {
            "detection_method": [],
            "confidence": 0.0,
            "fallback_used": False,
            "config_analysis": {},
            "name_analysis": {},
            "autoconfig_analysis": {},
            "errors": []
        }
        
        print(f"[DETECTOR] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 모델 타입 감지 시작: {model_name}")
        
        # 1단계: 캐시 확인
        cache_key = f"{model_name}:{model_path}"
        if cache_key in self.detection_cache:
            cached_result = self.detection_cache[cache_key]
            analysis_info["detection_method"].append("cache")
            analysis_info["confidence"] = 1.0
            print(f"[DETECTOR] 캐시에서 발견: {cached_result[0]}")
            return cached_result[0], cached_result[1], analysis_info
        
        # 2단계: AutoConfig 기반 분석 (새로운 메소드)
        task_type, model_class, autoconfig_confidence = self._analyze_with_autoconfig(model_path, model_name)
        if task_type and autoconfig_confidence > 0.8:
            analysis_info["detection_method"].append("autoconfig_analysis")
            analysis_info["confidence"] = autoconfig_confidence
            analysis_info["autoconfig_analysis"] = {"task": task_type, "class": model_class}
            
            # 캐시에 저장
            self.detection_cache[cache_key] = (task_type, model_class)
            print(f"[DETECTOR] AutoConfig 분석 성공: {task_type} -> {model_class}")
            return task_type, model_class, analysis_info
            
        # 3단계: config.json 기반 분석
        task_type, model_class, config_confidence = self._analyze_config(model_path)
        if task_type and config_confidence > 0.8:
            analysis_info["detection_method"].append("config_analysis")
            analysis_info["confidence"] = config_confidence
            analysis_info["config_analysis"] = {"task": task_type, "class": model_class}
            
            # 캐시에 저장
            self.detection_cache[cache_key] = (task_type, model_class)
            print(f"[DETECTOR] config.json 분석 성공: {task_type} -> {model_class}")
            return task_type, model_class, analysis_info
            
        # 4단계: 모델명 패턴 분석
        name_task, name_confidence = self._analyze_model_name(model_name)
        if name_task and name_confidence > 0.7:
            model_class = self.task_to_model_class.get(name_task, "AutoModel")
            analysis_info["detection_method"].append("name_pattern")
            analysis_info["confidence"] = name_confidence
            analysis_info["name_analysis"] = {"task": name_task, "class": model_class}
            
            print(f"[DETECTOR] 모델명 패턴 분석 성공: {name_task} -> {model_class}")
            return name_task, model_class, analysis_info
            
        # 5단계: 백업 전략 (기존 키워드 매칭)
        fallback_task, fallback_class = self._fallback_detection(model_name)
        analysis_info["detection_method"].append("fallback")
        analysis_info["fallback_used"] = True
        analysis_info["confidence"] = 0.5
        
        print(f"[DETECTOR] 백업 전략 사용: {fallback_task} -> {fallback_class}")
        return fallback_task, fallback_class, analysis_info
    
    def _analyze_with_autoconfig(self, model_path: str, model_name: str) -> Tuple[Optional[str], Optional[str], float]:
        """AutoConfig를 사용한 정밀한 모델 타입 감지"""
        try:
            # transformers의 AutoConfig 사용
            from transformers import AutoConfig
            
            print(f"[DETECTOR] AutoConfig 로딩 시작: {model_path}")
            
            # AutoConfig로 설정 로드
            config = AutoConfig.from_pretrained(
                model_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # 모델 타입 추출
            detected_model_type = getattr(config, 'model_type', None)
            if not detected_model_type:
                print(f"[DETECTOR] model_type 필드 없음")
                return None, None, 0.0
                
            print(f"[DETECTOR] 감지된 모델 타입: {detected_model_type}")
            
            # 데이터베이스에서 모델 정보 확인
            model_info = model_database.get_model_info(detected_model_type)
            if not model_info:
                print(f"[DETECTOR] 데이터베이스에서 모델 정보 없음: {detected_model_type}")
                return None, None, 0.0
            
            # 아키텍처 분석을 통한 세부 태스크 결정
            architectures = getattr(config, 'architectures', [])
            if architectures:
                arch = architectures[0]
                task_type, confidence = self._determine_task_from_architecture(arch, config, model_name)
                
                if task_type:
                    model_class = self.task_to_model_class.get(task_type, "AutoModel")
                    print(f"[DETECTOR] 아키텍처 기반 태스크 결정: {arch} -> {task_type}")
                    return task_type, model_class, confidence
            
            # 기본 태스크 사용
            if model_info.primary_tasks:
                primary_task = model_info.primary_tasks[0].value
                model_class = self.task_to_model_class.get(primary_task, "AutoModel")
                print(f"[DETECTOR] 기본 태스크 사용: {primary_task}")
                return primary_task, model_class, 0.85
                
            return None, None, 0.0
            
        except Exception as e:
            print(f"[DETECTOR] AutoConfig 분석 오류: {e}")
            return None, None, 0.0
    
    def _determine_task_from_architecture(self, architecture: str, config: Any, model_name: str) -> Tuple[Optional[str], float]:
        """아키텍처와 설정을 기반으로 정확한 태스크 결정"""
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
        
    def _analyze_config(self, model_path: str) -> Tuple[Optional[str], Optional[str], float]:
        """config.json 파일을 분석하여 모델 타입 감지"""
        config_path = os.path.join(model_path, "config.json")
        
        if not os.path.exists(config_path):
            print(f"[DETECTOR] config.json을 찾을 수 없음: {config_path}")
            return None, None, 0.0
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            print(f"[DETECTOR] config.json 로딩 성공")
            
            # architectures 필드 분석 (가장 신뢰할 수 있는 정보)
            if "architectures" in config:
                architectures = config["architectures"]
                if architectures:
                    arch = architectures[0]  # 첫 번째 아키텍처 사용
                    
                    # 직접 매핑된 아키텍처 확인
                    if arch in self.architecture_to_task:
                        task_type = self.architecture_to_task[arch]
                        model_class = self.task_to_model_class.get(task_type, "AutoModel")
                        print(f"[DETECTOR] architectures 분석: {arch} -> {task_type}")
                        return task_type, model_class, 0.95
                    
                    # 아키텍처 이름에서 태스크 유형 동적 추출
                    task_type, confidence = self._extract_task_from_architecture(arch, config)
                    if task_type:
                        model_class = self.task_to_model_class.get(task_type, "AutoModel")
                        print(f"[DETECTOR] 동적 아키텍처 분석: {arch} -> {task_type}")
                        return task_type, model_class, confidence
                        
            # model_type 기반 추론
            model_type = config.get("model_type", "")
            if model_type:
                # 특수 케이스들
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
                            print(f"[DETECTOR] Token classification detected: {model_type} (num_labels={num_labels})")
                            return "token-classification", "AutoModelForTokenClassification", 0.9
                        else:
                            # Sequence classification (sentiment, text classification)
                            print(f"[DETECTOR] Sequence classification detected: {model_type} (num_labels={num_labels})")
                            return "text-classification", "AutoModelForSequenceClassification", 0.9
                        
                elif model_type in ["gpt2", "gpt_neo", "llama", "bloom"]:
                    print(f"[DETECTOR] 생성 모델 타입 감지: {model_type}")
                    return "text-generation", "AutoModelForCausalLM", 0.9
                    
                elif model_type in ["t5", "bart", "pegasus"]:
                    print(f"[DETECTOR] Seq2Seq 모델 타입 감지: {model_type}")
                    return "text2text-generation", "AutoModelForSeq2SeqLM", 0.9
                    
            # task_specific_params 확인
            if "task_specific_params" in config:
                tasks = list(config["task_specific_params"].keys())
                if tasks:
                    task = tasks[0]
                    model_class = self.task_to_model_class.get(task, "AutoModel")
                    print(f"[DETECTOR] task_specific_params 감지: {task}")
                    return task, model_class, 0.8
                    
            print(f"[DETECTOR] config.json 분석 완료, 명확한 태스크 감지 실패")
            return None, None, 0.0
            
        except Exception as e:
            print(f"[DETECTOR] config.json 분석 오류: {e}")
            return None, None, 0.0
            
    def _analyze_model_name(self, model_name: str) -> Tuple[Optional[str], float]:
        """모델명 패턴 분석을 통한 태스크 추론"""
        model_name_lower = model_name.lower()
        
        scores = {}
        for task, patterns in self.name_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, model_name_lower):
                    score += 1
            
            if score > 0:
                # 패턴 매칭 개수에 따른 신뢰도 계산
                confidence = min(score / len(patterns), 1.0) * 0.8
                scores[task] = confidence
                
        if scores:
            best_task = max(scores, key=scores.get)
            best_confidence = scores[best_task]
            print(f"[DETECTOR] 모델명 패턴 분석: {best_task} (신뢰도: {best_confidence:.2f})")
            return best_task, best_confidence
            
        return None, 0.0
        
    def _extract_task_from_architecture(self, architecture: str, config: Dict) -> Tuple[Optional[str], float]:
        """아키텍처 이름에서 태스크 유형을 동적으로 추출"""
        arch_lower = architecture.lower()
        
        # 1. 아키텍처 이름에서 태스크 유형 패턴 매칭
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
                print(f"[DETECTOR] 아키텍처 패턴 매칭: {architecture} -> {task}")
                return task, 0.9
        
        # 2. 기본 모델 타입 + config 분석으로 세부 태스크 추정
        if arch_lower.endswith("model"):
            # 기본 모델인 경우 config 정보로 태스크 추정
            task_type, confidence = self._infer_task_from_config_details(architecture, config)
            if task_type:
                return task_type, confidence
        
        # 3. 알려진 모델 패턴 확인
        base_model_types = ["bert", "roberta", "distilbert", "electra", "deberta", "xlm", "gpt", "t5", "bart"]
        for model_type in base_model_types:
            if model_type in arch_lower:
                print(f"[DETECTOR] 기본 모델 타입 감지: {model_type}")
                # 기본 모델이므로 feature-extraction으로 설정
                return "feature-extraction", 0.7
        
        return None, 0.0
    
    def _infer_task_from_config_details(self, architecture: str, config: Dict) -> Tuple[Optional[str], float]:
        """config.json의 세부 정보로 태스크 추정"""
        
        # 0. 현재 감지 중인 모델 이름 확인 (전역 컨텍스트에서)
        current_model_name = getattr(self, '_current_model_name', '')
        if current_model_name and 'multitask' in current_model_name.lower():
            print(f"[DETECTOR] multitask 모델 감지 (모델명): {current_model_name}")
            return "text-classification", 0.9
        
        # 1. 레이블 정보 확인 (가장 강력한 지표)
        if "id2label" in config and "label2id" in config:
            id2label = config["id2label"]
            num_labels = len(id2label)
            
            # 레이블 내용 분석
            label_values = list(id2label.values()) if isinstance(id2label, dict) else []
            label_str = " ".join(str(label).lower() for label in label_values)
            
            # NER/Token Classification 패턴 확인
            ner_patterns = ["b-", "i-", "o", "beginning-", "inside-", "outside-"]
            if any(pattern in label_str for pattern in ner_patterns):
                print(f"[DETECTOR] NER 레이블 패턴 감지: {label_values[:5]}...")
                return "token-classification", 0.95
            
            # 다중 레이블 분류 확인
            if num_labels > 1:
                if num_labels <= 10:
                    print(f"[DETECTOR] 시퀀스 분류 감지: {num_labels}개 레이블")
                    return "text-classification", 0.9
                else:
                    print(f"[DETECTOR] 토큰 분류 감지: {num_labels}개 레이블")
                    return "token-classification", 0.85
        
        # 2. num_labels 확인
        num_labels = config.get("num_labels", 0)
        if num_labels > 1:
            if num_labels <= 10:
                return "text-classification", 0.8
            else:
                return "token-classification", 0.8
        
        # 3. problem_type 확인
        problem_type = config.get("problem_type", "")
        if problem_type:
            if "classification" in problem_type:
                return "text-classification", 0.85
            elif "token" in problem_type:
                return "token-classification", 0.85
        
        # 4. 특수 필드 확인
        special_fields = {
            "task_specific_params": lambda x: list(x.keys())[0] if x else None,
            "finetuning_task": lambda x: x,
            "downstream_task": lambda x: x
        }
        
        for field, extractor in special_fields.items():
            if field in config:
                task = extractor(config[field])
                if task and task in self.task_to_model_class:
                    return task, 0.8
        
        # 5. 모델 이름 분석 (최후 수단)
        model_name = config.get("_name_or_path", "")
        if model_name:
            # multitask 키워드 확인
            if "multitask" in model_name.lower():
                print(f"[DETECTOR] multitask 모델 감지: {model_name}")
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
                    print(f"[DETECTOR] 모델명 키워드 감지: {keyword} -> {task}")
                    return task, 0.75
        
        return None, 0.0
    
    def _fallback_detection(self, model_name: str) -> Tuple[str, str]:
        """Enhanced fallback detection with smart multi-task handling"""
        model_name_lower = model_name.lower()
        
        # Multi-task models - PRIORITY CHECK
        if any(keyword in model_name_lower for keyword in ["multitask", "multi-task", "multi_task"]):
            # Analyze model name for dominant task hints
            if any(keyword in model_name_lower for keyword in ["sentiment", "classification", "emotion"]):
                print(f"[DETECTOR] Multi-task model detected, defaulting to text-classification: {model_name}")
                return "text-classification", "AutoModelForSequenceClassification"
            elif any(keyword in model_name_lower for keyword in ["ner", "token", "entity"]):
                print(f"[DETECTOR] Multi-task model detected, defaulting to token-classification: {model_name}")
                return "token-classification", "AutoModelForTokenClassification"
            else:
                # Generic multi-task - default to text classification as most common
                print(f"[DETECTOR] Generic multi-task model detected, defaulting to text-classification: {model_name}")
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
            print(f"[DETECTOR] WARNING: No pattern matched for {model_name}, defaulting to feature-extraction")
            return "feature-extraction", "AutoModel"
            
    def get_transformers_class_name(self, model_class: str) -> str:
        """AutoModel 클래스명을 실제 transformers 클래스명으로 변환"""
        class_mapping = {
            "AutoModel": "AutoModel",
            "AutoModelForSequenceClassification": "AutoModelForSequenceClassification", 
            "AutoModelForCausalLM": "AutoModelForCausalLM",
            "AutoModelForSeq2SeqLM": "AutoModelForSeq2SeqLM",
            "AutoModelForQuestionAnswering": "AutoModelForQuestionAnswering",
            "AutoModelForTokenClassification": "AutoModelForTokenClassification",
            "AutoModelForMaskedLM": "AutoModelForMaskedLM",
        }
        return class_mapping.get(model_class, "AutoModel")
        
    def get_model_specific_class(self, model_class: str, model_type: str) -> str:
        """특정 모델 타입에 대한 구체적인 클래스 반환"""
        if model_class == "AutoModelForSequenceClassification":
            if model_type == "distilbert":
                return "DistilBertForSequenceClassification"
            elif model_type == "bert":
                return "BertForSequenceClassification" 
            elif model_type == "roberta":
                return "RobertaForSequenceClassification"
        elif model_class == "AutoModel":
            if model_type == "distilbert":
                return "DistilBertModel"
            elif model_type == "bert":
                return "BertModel"
            elif model_type == "roberta":
                return "RobertaModel"
                
        return model_class
        
    def clear_cache(self):
        """캐시 정리"""
        self.detection_cache.clear()
        print(f"[DETECTOR] 캐시 정리 완료")
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """캐시 통계 반환"""
        return {
            "cache_size": len(self.detection_cache),
            "cached_models": list(self.detection_cache.keys())
        }