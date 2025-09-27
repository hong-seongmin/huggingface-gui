"""
AI task processing services for various machine learning tasks.
"""
import base64
import io
import os
from typing import Dict, List, Any, Union, Optional
from PIL import Image
import streamlit as st
from core.logging_config import get_logger

logger = get_logger(__name__)


# ============================================================================
# SENTIMENT ANALYSIS SERVICES
# ============================================================================

def analyze_sentiment_basic(text: str) -> Dict:
    """기본 감정 분석을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(text, task_type="sentiment-analysis")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Sentiment analysis failed: {e}")
        return {"error": str(e)}


def analyze_sentiment_detailed(text: str, top_k: int = 3) -> Dict:
    """상세한 감정 분석을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(
            text,
            task_type="sentiment-analysis",
            return_all_scores=True,
            top_k=top_k
        )
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Detailed sentiment analysis failed: {e}")
        return {"error": str(e)}


def analyze_sentiment_batch(texts: List[str], **kwargs) -> List[Dict]:
    """배치 감정 분석을 수행합니다."""
    results = []
    for text in texts:
        try:
            result = analyze_sentiment_detailed(text, **kwargs)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed for text: {text[:50]}...")
            results.append({"error": str(e), "text": text[:50]})
    return results


def analyze_sentiment_advanced(text: str, confidence_threshold: float = 0.8) -> Dict:
    """고급 감정 분석을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(
            text,
            task_type="sentiment-analysis",
            return_all_scores=True,
            confidence_threshold=confidence_threshold
        )

        # 신뢰도 필터링
        if isinstance(result, list):
            filtered_result = [
                item for item in result
                if item.get('score', 0) >= confidence_threshold
            ]
            result = filtered_result if filtered_result else result

        return {"status": "success", "result": result, "threshold": confidence_threshold}
    except Exception as e:
        logger.error(f"Advanced sentiment analysis failed: {e}")
        return {"error": str(e)}


def safe_analyze_sentiment(text: str, max_retries: int = 3, timeout: int = 30) -> Dict:
    """안전한 감정 분석 (재시도 및 타임아웃 지원)."""
    for attempt in range(max_retries):
        try:
            result = analyze_sentiment_basic(text)
            if "error" not in result:
                return result

            logger.warning(f"Sentiment analysis attempt {attempt + 1} failed: {result.get('error')}")

        except Exception as e:
            logger.error(f"Sentiment analysis attempt {attempt + 1} error: {e}")

        if attempt < max_retries - 1:
            import time
            time.sleep(2 ** attempt)  # Exponential backoff

    return {"error": f"감정 분석이 {max_retries}번의 시도 후에도 실패했습니다."}


# ============================================================================
# NER SERVICES
# ============================================================================

def extract_entities(text: str) -> Dict:
    """개체명 인식을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(text, task_type="ner")

        # 엔티티 그룹화
        if isinstance(result, list):
            grouped = group_entities_by_type(result)
            return {"status": "success", "entities": result, "grouped": grouped}

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return {"error": str(e)}


def group_entities_by_type(entities: List[Dict]) -> Dict[str, List]:
    """엔티티를 타입별로 그룹화합니다."""
    grouped = {}

    for entity in entities:
        entity_type = entity.get('entity_group') or entity.get('entity')
        if entity_type:
            if entity_type not in grouped:
                grouped[entity_type] = []
            grouped[entity_type].append(entity)

    return grouped


# ============================================================================
# EMBEDDING SERVICES
# ============================================================================

def get_embedding(text: str) -> Dict:
    """텍스트 임베딩을 생성합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(text, task_type="feature-extraction")

        # 임베딩 벡터 정보 추가
        if isinstance(result, list) and len(result) > 0:
            embedding = result[0] if isinstance(result[0], list) else result
            return {
                "status": "success",
                "embedding": embedding,
                "dimensions": len(embedding),
                "vector_norm": sum(x*x for x in embedding) ** 0.5
            }

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return {"error": str(e)}


def find_most_similar(query_text: str, document_texts: List[str], top_k: int = 3) -> Dict:
    """가장 유사한 문서를 찾습니다."""
    try:
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity

        # 쿼리 임베딩
        query_result = get_embedding(query_text)
        if "error" in query_result:
            return query_result

        query_embedding = np.array(query_result["embedding"]).reshape(1, -1)

        # 문서 임베딩들
        doc_embeddings = []
        for doc_text in document_texts:
            doc_result = get_embedding(doc_text)
            if "error" not in doc_result:
                doc_embeddings.append(doc_result["embedding"])

        if not doc_embeddings:
            return {"error": "문서 임베딩 생성에 실패했습니다."}

        # 유사도 계산
        doc_embeddings = np.array(doc_embeddings)
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

        # 상위 k개 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "document": document_texts[idx],
                "similarity": float(similarities[idx]),
                "rank": len(results) + 1
            })

        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Similarity search failed: {e}")
        return {"error": str(e)}


# ============================================================================
# TEXT GENERATION SERVICES
# ============================================================================

def text2text_generate(input_text: str, task_prefix: str = "") -> Dict:
    """텍스트-투-텍스트 생성을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        # T5 모델의 경우 태스크 접두사 추가
        if task_prefix:
            full_input = f"{task_prefix}: {input_text}"
        else:
            full_input = input_text

        result = st.session_state['model_manager'].predict(full_input, task_type="text2text-generation")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Text2text generation failed: {e}")
        return {"error": str(e)}


def advanced_text2text_generate(input_text: str, task_prefix: str = "") -> Dict:
    """고급 텍스트-투-텍스트 생성을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        # T5 고급 파라미터 설정
        full_input = f"{task_prefix}: {input_text}" if task_prefix else input_text

        result = st.session_state['model_manager'].predict(
            full_input,
            task_type="text2text-generation",
            max_length=200,
            min_length=10,
            num_beams=4,
            temperature=0.7,
            do_sample=True
        )

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Advanced text2text generation failed: {e}")
        return {"error": str(e)}


# ============================================================================
# SUMMARIZATION SERVICES
# ============================================================================

def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> Dict:
    """텍스트 요약을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(
            text,
            task_type="summarization",
            max_length=max_length,
            min_length=min_length
        )

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Text summarization failed: {e}")
        return {"error": str(e)}


def advanced_summarize(text: str, summary_type: str = "balanced") -> Dict:
    """고급 텍스트 요약을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        # 요약 타입별 파라미터 설정
        params = {
            "extractive": {"max_length": 100, "min_length": 20, "num_beams": 2},
            "abstractive": {"max_length": 200, "min_length": 50, "num_beams": 4},
            "balanced": {"max_length": 150, "min_length": 30, "num_beams": 3}
        }

        settings = params.get(summary_type, params["balanced"])

        result = st.session_state['model_manager'].predict(
            text,
            task_type="summarization",
            **settings,
            temperature=0.7,
            do_sample=True
        )

        return {"status": "success", "result": result, "type": summary_type}
    except Exception as e:
        logger.error(f"Advanced summarization failed: {e}")
        return {"error": str(e)}


# ============================================================================
# TRANSLATION SERVICES
# ============================================================================

def translate_text(text: str, source_lang: str = "en", target_lang: str = "ko") -> Dict:
    """텍스트 번역을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        result = st.session_state['model_manager'].predict(
            text,
            task_type="translation",
            src_lang=source_lang,
            tgt_lang=target_lang
        )

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return {"error": str(e)}


def multi_language_translate(text: str, target_languages: List[str] = ["ko", "ja", "zh"]) -> Dict[str, Dict]:
    """다중 언어 번역을 수행합니다."""
    results = {}

    for lang in target_languages:
        try:
            result = translate_text(text, target_lang=lang)
            results[lang] = result
        except Exception as e:
            logger.error(f"Translation to {lang} failed: {e}")
            results[lang] = {"error": str(e)}

    return results


def advanced_translate(text: str, source_lang: str = "en", target_lang: str = "ko",
                      quality: str = "balanced") -> Dict:
    """고급 번역을 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        # 품질별 파라미터 설정
        quality_params = {
            "fast": {"num_beams": 1, "max_length": 200},
            "balanced": {"num_beams": 4, "max_length": 300},
            "high": {"num_beams": 8, "max_length": 400, "temperature": 0.5}
        }

        params = quality_params.get(quality, quality_params["balanced"])

        result = st.session_state['model_manager'].predict(
            text,
            task_type="translation",
            src_lang=source_lang,
            tgt_lang=target_lang,
            **params
        )

        return {"status": "success", "result": result, "quality": quality}
    except Exception as e:
        logger.error(f"Advanced translation failed: {e}")
        return {"error": str(e)}


# ============================================================================
# IMAGE PROCESSING SERVICES
# ============================================================================

def encode_image_to_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩합니다."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding failed: {e}")
        raise


def prepare_image_data(image_input: Union[str, bytes, Image.Image]) -> str:
    """다양한 형태의 이미지 입력을 처리합니다."""
    try:
        if isinstance(image_input, str):
            # 파일 경로인 경우
            return encode_image_to_base64(image_input)
        elif isinstance(image_input, bytes):
            # 바이트 데이터인 경우
            return base64.b64encode(image_input).decode('utf-8')
        elif isinstance(image_input, Image.Image):
            # PIL Image 객체인 경우
            buffer = io.BytesIO()
            image_input.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode('utf-8')
        else:
            raise ValueError("지원하지 않는 이미지 형식입니다.")
    except Exception as e:
        logger.error(f"Image data preparation failed: {e}")
        raise


def classify_image_basic(image_path: str, top_k: int = 5) -> Dict:
    """기본 이미지 분류를 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        # 이미지 데이터 준비
        image_data = prepare_image_data(image_path)

        result = st.session_state['model_manager'].predict(
            image_data,
            task_type="image-classification",
            top_k=top_k
        )

        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"Image classification failed: {e}")
        return {"error": str(e)}


def classify_image_advanced(image_path: str, top_k: int = 10,
                           confidence_threshold: float = 0.1) -> Dict:
    """고급 이미지 분류를 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        image_data = prepare_image_data(image_path)

        result = st.session_state['model_manager'].predict(
            image_data,
            task_type="image-classification",
            top_k=top_k,
            confidence_threshold=confidence_threshold
        )

        # 신뢰도 필터링
        if isinstance(result, list):
            filtered_result = [
                item for item in result
                if item.get('score', 0) >= confidence_threshold
            ]
            result = filtered_result if filtered_result else result[:5]

        return {"status": "success", "result": result, "threshold": confidence_threshold}
    except Exception as e:
        logger.error(f"Advanced image classification failed: {e}")
        return {"error": str(e)}


# ============================================================================
# AUDIO PROCESSING SERVICES
# ============================================================================

def encode_audio_to_base64(audio_path: str) -> str:
    """오디오를 base64로 인코딩합니다."""
    try:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode('utf-8')
    except Exception as e:
        logger.error(f"Audio encoding failed: {e}")
        raise


def get_audio_info(audio_path: str) -> Dict:
    """오디오 파일 정보를 가져옵니다."""
    try:
        import wave
        import os

        file_size = os.path.getsize(audio_path)

        # WAV 파일인 경우 추가 정보 추출
        if audio_path.lower().endswith('.wav'):
            with wave.open(audio_path, 'rb') as wav_file:
                return {
                    "file_size": file_size,
                    "sample_rate": wav_file.getframerate(),
                    "channels": wav_file.getnchannels(),
                    "duration": wav_file.getnframes() / wav_file.getframerate(),
                    "sample_width": wav_file.getsampwidth()
                }
        else:
            return {"file_size": file_size}

    except Exception as e:
        logger.error(f"Audio info extraction failed: {e}")
        return {"error": str(e)}


def transcribe_audio_basic(audio_path: str) -> Dict:
    """기본 오디오 전사를 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        audio_data = encode_audio_to_base64(audio_path)
        audio_info = get_audio_info(audio_path)

        result = st.session_state['model_manager'].predict(
            audio_data,
            task_type="automatic-speech-recognition"
        )

        return {"status": "success", "result": result, "audio_info": audio_info}
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return {"error": str(e)}


def transcribe_audio_advanced(audio_path: str, language: str = "auto",
                             return_timestamps: bool = True) -> Dict:
    """고급 오디오 전사를 수행합니다."""
    try:
        if 'model_manager' not in st.session_state:
            return {"error": "모델 매니저가 초기화되지 않았습니다."}

        audio_data = encode_audio_to_base64(audio_path)
        audio_info = get_audio_info(audio_path)

        params = {
            "return_timestamps": return_timestamps,
            "chunk_length_s": 30,
            "stride_length_s": 5
        }

        if language != "auto":
            params["language"] = language

        result = st.session_state['model_manager'].predict(
            audio_data,
            task_type="automatic-speech-recognition",
            **params
        )

        return {
            "status": "success",
            "result": result,
            "audio_info": audio_info,
            "language": language
        }
    except Exception as e:
        logger.error(f"Advanced audio transcription failed: {e}")
        return {"error": str(e)}