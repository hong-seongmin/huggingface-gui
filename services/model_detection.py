"""
Model type detection service for API guide generation.
"""
from typing import List, Dict, Any


def detect_model_type_for_api_guide(model_name: str) -> str:
    """API 가이드에서 사용할 모델 타입을 감지합니다. 모든 HuggingFace 아키텍처 지원."""
    model_name_lower = model_name.lower()

    # === TEXT GENERATION MODELS ===
    # GPT 계열
    if any(keyword in model_name_lower for keyword in ['gpt', 'gpt2', 'gpt-3', 'gpt-4', 'gpt-neo', 'gpt-j']):
        return "text-generation"

    # Large Language Models
    elif any(keyword in model_name_lower for keyword in ['llama', 'llama2', 'llama3', 'alpaca', 'vicuna']):
        return "text-generation"

    # Mistral 계열
    elif any(keyword in model_name_lower for keyword in ['mistral', 'mixtral', 'mamba']):
        return "text-generation"

    # Phi 계열
    elif any(keyword in model_name_lower for keyword in ['phi', 'phi-2', 'phi-3']):
        return "text-generation"

    # Qwen 계열
    elif any(keyword in model_name_lower for keyword in ['qwen', 'qwen2', 'qwen-vl']):
        return "text-generation"

    # Gemma 계열
    elif any(keyword in model_name_lower for keyword in ['gemma', 'gemma-2', 'gemma-7b']):
        return "text-generation"

    # Code 생성 모델
    elif any(keyword in model_name_lower for keyword in ['codegen', 'code-llama', 'starcoder', 'codebert']):
        return "text-generation"

    # Chat/Instruct 모델
    elif any(keyword in model_name_lower for keyword in ['chat', 'instruct', 'dialogue', 'conversational']):
        return "text-generation"

    # === SEQUENCE-TO-SEQUENCE MODELS ===
    # T5 계열
    elif any(keyword in model_name_lower for keyword in ['t5', 'flan-t5', 'ul2']):
        return "text2text-generation"

    # BART 계열
    elif any(keyword in model_name_lower for keyword in ['bart', 'mbart', 'blenderbot']):
        return "text2text-generation"

    # === SUMMARIZATION MODELS ===
    elif any(keyword in model_name_lower for keyword in ['pegasus', 'led', 'longformer-encoder-decoder']):
        return "summarization"

    # === TRANSLATION MODELS ===
    elif any(keyword in model_name_lower for keyword in ['opus-mt', 'nllb', 'marian', 'm2m100', 'madlad400']):
        return "translation"

    # === CLASSIFICATION MODELS ===
    # DeBERTa 계열
    elif 'deberta' in model_name_lower:
        if any(keyword in model_name_lower for keyword in ['classification', 'sentiment', 'multitask']):
            return "text-classification"
        elif any(keyword in model_name_lower for keyword in ['ner', 'token']):
            return "token-classification"
        elif any(keyword in model_name_lower for keyword in ['qa', 'question']):
            return "question-answering"
        else:
            return "text-classification"

    # BERT 계열 (기본)
    elif any(keyword in model_name_lower for keyword in ['bert', 'kobert', 'roberta', 'electra', 'albert']):
        if any(keyword in model_name_lower for keyword in ['ner', 'token']):
            return "token-classification"
        elif any(keyword in model_name_lower for keyword in ['qa', 'question']):
            return "question-answering"
        elif any(keyword in model_name_lower for keyword in ['classification', 'sentiment']):
            return "text-classification"
        else:
            return "text-classification"

    # === EMBEDDING MODELS ===
    elif any(keyword in model_name_lower for keyword in ['sentence-transformer', 'bge', 'e5', 'gte', 'multilingual-e5']):
        return "feature-extraction"

    # === AUDIO MODELS ===
    elif any(keyword in model_name_lower for keyword in ['whisper', 'wav2vec', 'speech2text', 'asr']):
        return "automatic-speech-recognition"

    # === IMAGE MODELS ===
    elif any(keyword in model_name_lower for keyword in ['vit', 'swin', 'deit', 'resnet', 'efficientnet']):
        if any(keyword in model_name_lower for keyword in ['clip', 'blip', 'multimodal']):
            return "zero-shot-image-classification"
        else:
            return "image-classification"

    # CLIP 계열
    elif 'clip' in model_name_lower:
        return "zero-shot-image-classification"

    # === MULTIMODAL MODELS ===
    elif any(keyword in model_name_lower for keyword in ['blip', 'flamingo', 'kosmos', 'llava']):
        if any(keyword in model_name_lower for keyword in ['caption', 'describe']):
            return "image-to-text"
        elif any(keyword in model_name_lower for keyword in ['vqa', 'visual-question']):
            return "visual-question-answering"
        else:
            return "image-to-text"

    # === QUESTION ANSWERING MODELS ===
    elif any(keyword in model_name_lower for keyword in ['squadv2', 'korquad', 'qa', 'question-answering']):
        return "question-answering"

    # === FILL MASK MODELS ===
    elif any(keyword in model_name_lower for keyword in ['fill-mask', 'masked-lm', 'mlm']):
        return "fill-mask"

    # === ZERO-SHOT CLASSIFICATION ===
    elif any(keyword in model_name_lower for keyword in ['zero-shot', 'nli', 'entailment']):
        return "zero-shot-classification"

    # === TABLE QUESTION ANSWERING ===
    elif any(keyword in model_name_lower for keyword in ['tapas', 'table-qa', 'tabular']):
        return "table-question-answering"

    # === 기본값 (가장 일반적인 용도) ===
    else:
        # 텍스트 관련 모델이면 분류로 추정
        if any(keyword in model_name_lower for keyword in ['text', 'nlp', 'korean', 'multilingual']):
            return "text-classification"
        # 이미지 관련이면 이미지 분류로 추정
        elif any(keyword in model_name_lower for keyword in ['image', 'vision', 'visual']):
            return "image-classification"
        # 그 외는 가장 일반적인 텍스트 분류로 설정
        else:
            return "text-classification"


def get_supported_tasks_for_model(model_name: str) -> List[str]:
    """모델이 지원하는 태스크 목록을 반환합니다."""
    model_type = detect_model_type_for_api_guide(model_name)

    task_mappings = {
        "text-generation": ["텍스트 생성", "대화", "코드 생성", "창작"],
        "text2text-generation": ["텍스트 변환", "요약", "번역", "질의응답"],
        "text-classification": ["감정 분석", "텍스트 분류", "스팸 탐지", "주제 분류"],
        "token-classification": ["개체명 인식", "품사 태깅", "구문 분석"],
        "question-answering": ["질의응답", "독해", "정보 추출"],
        "summarization": ["요약", "문서 압축", "핵심 내용 추출"],
        "translation": ["번역", "언어 변환", "다국어 지원"],
        "feature-extraction": ["임베딩 생성", "벡터 변환", "유사도 계산"],
        "fill-mask": ["단어 예측", "문장 완성", "마스킹 복원"],
        "zero-shot-classification": ["제로샷 분류", "라벨 없는 분류"],
        "image-classification": ["이미지 분류", "객체 인식", "시각적 분석"],
        "zero-shot-image-classification": ["제로샷 이미지 분류", "CLIP 기반 분석"],
        "image-to-text": ["이미지 캡셔닝", "시각적 설명", "이미지 묘사"],
        "visual-question-answering": ["시각적 질의응답", "이미지 기반 QA"],
        "automatic-speech-recognition": ["음성 인식", "STT", "오디오 전사"],
        "table-question-answering": ["테이블 질의응답", "표 기반 분석"]
    }

    return task_mappings.get(model_type, ["일반 NLP 태스크"])


def get_model_architecture_info(model_name: str) -> Dict[str, Any]:
    """모델 아키텍처 정보를 반환합니다."""
    model_name_lower = model_name.lower()

    # Architecture mapping
    if 'gpt' in model_name_lower:
        return {
            "architecture": "GPT",
            "type": "Decoder-only Transformer",
            "strengths": ["텍스트 생성", "대화", "창작"],
            "typical_use_cases": ["챗봇", "코드 생성", "창작 지원"]
        }
    elif 'bert' in model_name_lower:
        return {
            "architecture": "BERT",
            "type": "Encoder-only Transformer",
            "strengths": ["텍스트 이해", "분류", "임베딩"],
            "typical_use_cases": ["감정 분석", "문서 분류", "검색"]
        }
    elif 't5' in model_name_lower:
        return {
            "architecture": "T5",
            "type": "Encoder-Decoder Transformer",
            "strengths": ["텍스트 변환", "요약", "번역"],
            "typical_use_cases": ["요약", "번역", "질의응답"]
        }
    elif 'bart' in model_name_lower:
        return {
            "architecture": "BART",
            "type": "Encoder-Decoder Transformer",
            "strengths": ["텍스트 생성", "요약", "번역"],
            "typical_use_cases": ["요약", "대화", "번역"]
        }
    else:
        return {
            "architecture": "Unknown",
            "type": "Transformer-based",
            "strengths": ["다양한 NLP 태스크"],
            "typical_use_cases": ["범용 NLP"]
        }