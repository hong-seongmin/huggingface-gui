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

class ModelTypeDetector:
    """
    Hugging Face 모델의 태스크 유형을 정밀하게 자동 감지하는 클래스
    config.json, 모델 구조, 토크나이저 설정 등을 종합 분석
    """
    
    def __init__(self):
        self.logger = logging.getLogger("ModelTypeDetector")
        
        # 태스크별 모델 클래스 매핑
        self.task_to_model_class = {
            "text-classification": "AutoModelForSequenceClassification",
            "sentiment-analysis": "AutoModelForSequenceClassification", 
            "token-classification": "AutoModelForTokenClassification",
            "question-answering": "AutoModelForQuestionAnswering",
            "text-generation": "AutoModelForCausalLM",
            "text2text-generation": "AutoModelForSeq2SeqLM",
            "translation": "AutoModelForSeq2SeqLM",
            "summarization": "AutoModelForSeq2SeqLM",
            "feature-extraction": "AutoModel",
            "sentence-similarity": "AutoModel",
            "fill-mask": "AutoModelForMaskedLM",
            "image-classification": "AutoModelForImageClassification",
            "automatic-speech-recognition": "AutoModelForSpeechSeq2Seq",
        }
        
        # 아키텍처별 기본 태스크 매핑
        self.architecture_to_task = {
            # 분류 모델들
            "BertForSequenceClassification": "text-classification",
            "DistilBertForSequenceClassification": "text-classification",
            "RobertaForSequenceClassification": "text-classification",
            "XLMRobertaForSequenceClassification": "text-classification",
            "AlbertForSequenceClassification": "text-classification",
            
            # 생성 모델들
            "GPT2LMHeadModel": "text-generation",
            "GPTNeoForCausalLM": "text-generation",
            "LlamaForCausalLM": "text-generation",
            "BloomForCausalLM": "text-generation",
            
            # Seq2Seq 모델들
            "T5ForConditionalGeneration": "text2text-generation",
            "BartForConditionalGeneration": "text2text-generation",
            "PegasusForConditionalGeneration": "summarization",
            "MarianMTModel": "translation",
            
            # Q&A 모델들
            "BertForQuestionAnswering": "question-answering",
            "DistilBertForQuestionAnswering": "question-answering",
            
            # 기본 모델들 (임베딩용)
            "BertModel": "feature-extraction",
            "DistilBertModel": "feature-extraction", 
            "RobertaModel": "feature-extraction",
            "XLMRobertaModel": "feature-extraction",
        }
        
        # 모델명 패턴 분석 (보조 수단)
        self.name_patterns = {
            "text-classification": [
                r"sentiment", r"classification", r"classifier", r"toxic", r"emotion",
                r"stance", r"intent", r"category", r"label"
            ],
            "text-generation": [
                r"gpt", r"generator", r"chat", r"conversational", r"instruct",
                r"dialogue", r"story"
            ],
            "translation": [
                r"translate", r"translation", r"mt-", r"opus-mt", r"nllb"
            ],
            "summarization": [
                r"summar", r"abstract", r"headline"
            ],
            "question-answering": [
                r"qa", r"question", r"squad", r"answer"
            ],
            "feature-extraction": [
                r"embedding", r"encode", r"sentence", r"similarity", r"retrieval"
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
            
        # 2단계: config.json 기반 분석
        task_type, model_class, config_confidence = self._analyze_config(model_path)
        if task_type and config_confidence > 0.8:
            analysis_info["detection_method"].append("config_analysis")
            analysis_info["confidence"] = config_confidence
            analysis_info["config_analysis"] = {"task": task_type, "class": model_class}
            
            # 캐시에 저장
            self.detection_cache[cache_key] = (task_type, model_class)
            print(f"[DETECTOR] config.json 분석 성공: {task_type} -> {model_class}")
            return task_type, model_class, analysis_info
            
        # 3단계: 모델명 패턴 분석
        name_task, name_confidence = self._analyze_model_name(model_name)
        if name_task and name_confidence > 0.7:
            model_class = self.task_to_model_class.get(name_task, "AutoModel")
            analysis_info["detection_method"].append("name_pattern")
            analysis_info["confidence"] = name_confidence
            analysis_info["name_analysis"] = {"task": name_task, "class": model_class}
            
            print(f"[DETECTOR] 모델명 패턴 분석 성공: {name_task} -> {model_class}")
            return name_task, model_class, analysis_info
            
        # 4단계: 백업 전략 (기존 키워드 매칭)
        fallback_task, fallback_class = self._fallback_detection(model_name)
        analysis_info["detection_method"].append("fallback")
        analysis_info["fallback_used"] = True
        analysis_info["confidence"] = 0.5
        
        print(f"[DETECTOR] 백업 전략 사용: {fallback_task} -> {fallback_class}")
        return fallback_task, fallback_class, analysis_info
        
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
                    if arch in self.architecture_to_task:
                        task_type = self.architecture_to_task[arch]
                        model_class = self.task_to_model_class.get(task_type, "AutoModel")
                        print(f"[DETECTOR] architectures 분석: {arch} -> {task_type}")
                        return task_type, model_class, 0.95
                        
            # model_type 기반 추론
            model_type = config.get("model_type", "")
            if model_type:
                # 특수 케이스들
                if model_type in ["bert", "distilbert", "roberta", "xlm-roberta"]:
                    # num_labels로 분류 모델인지 판단
                    if config.get("num_labels", 0) > 1:
                        print(f"[DETECTOR] num_labels 감지: 분류 모델")
                        return "text-classification", "AutoModelForSequenceClassification", 0.9
                    elif "id2label" in config:
                        print(f"[DETECTOR] id2label 감지: 분류 모델")
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
        
    def _fallback_detection(self, model_name: str) -> Tuple[str, str]:
        """기존 키워드 매칭 백업 전략"""
        model_name_lower = model_name.lower()
        
        # 기존 로직 유지
        if any(keyword in model_name_lower for keyword in ["sentiment", "classification", "classifier"]):
            return "text-classification", "AutoModelForSequenceClassification"
        elif any(keyword in model_name_lower for keyword in ["bge", "embedding"]):
            return "feature-extraction", "AutoModel"
        elif any(keyword in model_name_lower for keyword in ["gpt", "generate", "chat"]):
            return "text-generation", "AutoModelForCausalLM"
        else:
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