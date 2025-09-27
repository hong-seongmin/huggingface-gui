"""
AI task UI components for Streamlit.
"""
import streamlit as st
import requests
from typing import Dict, List, Any, Optional, Union
from PIL import Image
import base64
import io
from core.logging_config import get_logger
from utils.validators import validate_text_input, validate_image_file, validate_audio_file
from utils.formatters import format_prediction_result, format_json

logger = get_logger(__name__)


def render_text_classification_task():
    """Render text classification task UI."""
    st.subheader("📝 텍스트 분류")

    # Input text
    text_input = st.text_area(
        "분류할 텍스트를 입력하세요",
        placeholder="예: 오늘 날씨가 정말 좋네요!",
        height=100
    )

    # Settings
    col1, col2 = st.columns(2)

    with col1:
        return_all_scores = st.checkbox("모든 점수 반환", value=False)

    with col2:
        top_k = st.slider("상위 K개 결과", 1, 10, 3)

    # Process button
    if st.button("🔍 텍스트 분류 실행", key="text_classification_btn"):
        if not text_input.strip():
            st.error("텍스트를 입력해주세요.")
            return

        is_valid, error_msg = validate_text_input(text_input)
        if not is_valid:
            st.error(error_msg)
            return

        with st.spinner("분류 중..."):
            try:
                result = perform_text_classification(
                    text_input,
                    return_all_scores=return_all_scores,
                    top_k=top_k
                )

                if result:
                    st.success("✅ 분류 완료!")
                    render_classification_results(result)
                else:
                    st.error("분류에 실패했습니다.")

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


def render_text_generation_task():
    """Render text generation task UI."""
    st.subheader("✨ 텍스트 생성")

    # Input prompt
    prompt_input = st.text_area(
        "생성을 위한 프롬프트를 입력하세요",
        placeholder="예: 인공지능의 미래는",
        height=100
    )

    # Settings
    col1, col2, col3 = st.columns(3)

    with col1:
        max_length = st.slider("최대 길이", 10, 500, 100)

    with col2:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, 0.1)

    with col3:
        num_return_sequences = st.slider("생성 개수", 1, 5, 1)

    # Process button
    if st.button("✨ 텍스트 생성 실행", key="text_generation_btn"):
        if not prompt_input.strip():
            st.error("프롬프트를 입력해주세요.")
            return

        with st.spinner("생성 중..."):
            try:
                result = perform_text_generation(
                    prompt_input,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=num_return_sequences
                )

                if result:
                    st.success("✅ 생성 완료!")
                    render_generation_results(result)
                else:
                    st.error("생성에 실패했습니다.")

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


def render_question_answering_task():
    """Render question answering task UI."""
    st.subheader("❓ 질문 답변")

    # Context input
    context_input = st.text_area(
        "컨텍스트 (답변의 근거가 되는 텍스트)",
        placeholder="예: 인공지능은 컴퓨터 과학의 한 분야로...",
        height=150
    )

    # Question input
    question_input = st.text_input(
        "질문",
        placeholder="예: 인공지능이란 무엇인가요?"
    )

    # Process button
    if st.button("❓ 질문 답변 실행", key="question_answering_btn"):
        if not context_input.strip() or not question_input.strip():
            st.error("컨텍스트와 질문을 모두 입력해주세요.")
            return

        with st.spinner("답변 생성 중..."):
            try:
                result = perform_question_answering(
                    question=question_input,
                    context=context_input
                )

                if result:
                    st.success("✅ 답변 완료!")
                    render_qa_results(result)
                else:
                    st.error("답변 생성에 실패했습니다.")

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


def render_summarization_task():
    """Render text summarization task UI."""
    st.subheader("📋 텍스트 요약")

    # Text input
    text_input = st.text_area(
        "요약할 텍스트를 입력하세요",
        placeholder="긴 텍스트를 입력하세요...",
        height=200
    )

    # Settings
    col1, col2 = st.columns(2)

    with col1:
        max_length = st.slider("최대 요약 길이", 20, 300, 150)

    with col2:
        min_length = st.slider("최소 요약 길이", 5, 100, 30)

    # Summary type
    summary_type = st.selectbox(
        "요약 스타일",
        ["balanced", "extractive", "abstractive"],
        help="balanced: 균형잡힌 요약, extractive: 핵심 문장 추출, abstractive: 새로운 문장 생성"
    )

    # Process button
    if st.button("📋 텍스트 요약 실행", key="summarization_btn"):
        if not text_input.strip():
            st.error("요약할 텍스트를 입력해주세요.")
            return

        if len(text_input.split()) < 20:
            st.warning("요약하기에는 텍스트가 너무 짧습니다.")

        with st.spinner("요약 중..."):
            try:
                result = perform_summarization(
                    text_input,
                    max_length=max_length,
                    min_length=min_length,
                    summary_type=summary_type
                )

                if result:
                    st.success("✅ 요약 완료!")
                    render_summarization_results(result)
                else:
                    st.error("요약에 실패했습니다.")

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


def render_translation_task():
    """Render translation task UI."""
    st.subheader("🌐 번역")

    # Text input
    text_input = st.text_area(
        "번역할 텍스트를 입력하세요",
        placeholder="예: Hello, how are you?",
        height=100
    )

    # Language settings
    col1, col2 = st.columns(2)

    with col1:
        source_lang = st.selectbox(
            "원본 언어",
            ["auto", "en", "ko", "ja", "zh", "es", "fr", "de"],
            help="auto: 자동 감지"
        )

    with col2:
        target_languages = st.multiselect(
            "번역할 언어",
            ["ko", "en", "ja", "zh", "es", "fr", "de"],
            default=["ko"] if source_lang != "ko" else ["en"]
        )

    # Process button
    if st.button("🌐 번역 실행", key="translation_btn"):
        if not text_input.strip():
            st.error("번역할 텍스트를 입력해주세요.")
            return

        if not target_languages:
            st.error("번역할 언어를 선택해주세요.")
            return

        with st.spinner("번역 중..."):
            try:
                result = perform_translation(
                    text_input,
                    source_lang=source_lang,
                    target_languages=target_languages
                )

                if result:
                    st.success("✅ 번역 완료!")
                    render_translation_results(result)
                else:
                    st.error("번역에 실패했습니다.")

            except Exception as e:
                st.error(f"오류 발생: {str(e)}")


def render_image_classification_task():
    """Render image classification task UI."""
    st.subheader("🖼️ 이미지 분류")

    # Image upload
    uploaded_file = st.file_uploader(
        "이미지를 업로드하세요",
        type=["jpg", "jpeg", "png", "gif", "bmp", "webp"]
    )

    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="업로드된 이미지", use_column_width=True)

        # Settings
        col1, col2 = st.columns(2)

        with col1:
            top_k = st.slider("상위 K개 결과", 1, 10, 5)

        with col2:
            confidence_threshold = st.slider("신뢰도 임계값", 0.0, 1.0, 0.1, 0.05)

        # Process button
        if st.button("🖼️ 이미지 분류 실행", key="image_classification_btn"):
            with st.spinner("분류 중..."):
                try:
                    result = perform_image_classification(
                        image,
                        top_k=top_k,
                        confidence_threshold=confidence_threshold
                    )

                    if result:
                        st.success("✅ 분류 완료!")
                        render_classification_results(result)
                    else:
                        st.error("분류에 실패했습니다.")

                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

    else:
        st.info("이미지를 업로드해주세요.")


def render_audio_transcription_task():
    """Render audio transcription task UI."""
    st.subheader("🎵 음성 인식")

    # Audio upload
    uploaded_file = st.file_uploader(
        "오디오 파일을 업로드하세요",
        type=["wav", "mp3", "flac", "m4a", "ogg"]
    )

    if uploaded_file is not None:
        # Display audio player
        st.audio(uploaded_file, format='audio/wav')

        # Settings
        col1, col2 = st.columns(2)

        with col1:
            language = st.selectbox(
                "언어",
                ["auto", "ko", "en", "ja", "zh"],
                help="auto: 자동 감지"
            )

        with col2:
            return_timestamps = st.checkbox("타임스탬프 반환", value=False)

        # Process button
        if st.button("🎵 음성 인식 실행", key="audio_transcription_btn"):
            with st.spinner("인식 중..."):
                try:
                    result = perform_audio_transcription(
                        uploaded_file,
                        language=language,
                        return_timestamps=return_timestamps
                    )

                    if result:
                        st.success("✅ 인식 완료!")
                        render_transcription_results(result)
                    else:
                        st.error("인식에 실패했습니다.")

                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

    else:
        st.info("오디오 파일을 업로드해주세요.")


def render_classification_results(result: Dict[str, Any]):
    """Render classification results."""
    if 'predictions' in result:
        st.subheader("📊 분류 결과")

        predictions = result['predictions']
        if isinstance(predictions, list):
            for i, pred in enumerate(predictions[:5], 1):
                label = pred.get('label', 'Unknown')
                score = pred.get('score', 0)

                # Progress bar for score
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.write(f"**{i}. {label}**")
                with col2:
                    st.progress(score)
                    st.write(f"{score:.1%}")
        else:
            st.write(predictions)


def render_generation_results(result: Dict[str, Any]):
    """Render text generation results."""
    if 'generated_text' in result:
        st.subheader("✨ 생성된 텍스트")

        if isinstance(result['generated_text'], list):
            for i, text in enumerate(result['generated_text'], 1):
                st.markdown(f"**생성 {i}:**")
                st.write(text)
                st.divider()
        else:
            st.write(result['generated_text'])


def render_qa_results(result: Dict[str, Any]):
    """Render question answering results."""
    if 'answer' in result:
        st.subheader("💡 답변")
        st.success(result['answer'])

        if 'score' in result:
            st.metric("신뢰도", f"{result['score']:.1%}")


def render_summarization_results(result: Dict[str, Any]):
    """Render summarization results."""
    if 'summary_text' in result:
        st.subheader("📋 요약")
        st.info(result['summary_text'])


def render_translation_results(result: Dict[str, Any]):
    """Render translation results."""
    if 'translations' in result:
        st.subheader("🌐 번역 결과")

        for lang, translation in result['translations'].items():
            st.markdown(f"**{lang}:**")
            st.write(translation)


def render_transcription_results(result: Dict[str, Any]):
    """Render transcription results."""
    if 'text' in result:
        st.subheader("📝 인식 결과")
        st.success(result['text'])

        if 'timestamps' in result:
            with st.expander("타임스탬프 보기"):
                st.json(result['timestamps'])


def perform_text_classification(
    text: str,
    return_all_scores: bool = False,
    top_k: int = 3
) -> Optional[Dict]:
    """Perform text classification (placeholder implementation)."""
    # This would connect to the actual model API
    # For now, return mock data
    return {
        'predictions': [
            {'label': 'POSITIVE', 'score': 0.95},
            {'label': 'NEGATIVE', 'score': 0.05}
        ]
    }


def perform_text_generation(
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    num_return_sequences: int = 1
) -> Optional[Dict]:
    """Perform text generation (placeholder implementation)."""
    return {
        'generated_text': f"{prompt} [생성된 텍스트가 여기에 표시됩니다]"
    }


def perform_question_answering(
    question: str,
    context: str
) -> Optional[Dict]:
    """Perform question answering (placeholder implementation)."""
    return {
        'answer': "답변이 여기에 표시됩니다.",
        'score': 0.85
    }


def perform_summarization(
    text: str,
    max_length: int = 150,
    min_length: int = 30,
    summary_type: str = "balanced"
) -> Optional[Dict]:
    """Perform text summarization (placeholder implementation)."""
    return {
        'summary_text': "요약 결과가 여기에 표시됩니다."
    }


def perform_translation(
    text: str,
    source_lang: str = "auto",
    target_languages: List[str] = ["ko"]
) -> Optional[Dict]:
    """Perform translation (placeholder implementation)."""
    return {
        'translations': {
            lang: f"번역 결과 ({lang})"
            for lang in target_languages
        }
    }


def perform_image_classification(
    image: Image.Image,
    top_k: int = 5,
    confidence_threshold: float = 0.1
) -> Optional[Dict]:
    """Perform image classification (placeholder implementation)."""
    return {
        'predictions': [
            {'label': 'dog', 'score': 0.85},
            {'label': 'cat', 'score': 0.12},
            {'label': 'bird', 'score': 0.03}
        ]
    }


def perform_audio_transcription(
    audio_file,
    language: str = "auto",
    return_timestamps: bool = False
) -> Optional[Dict]:
    """Perform audio transcription (placeholder implementation)."""
    result = {
        'text': "음성 인식 결과가 여기에 표시됩니다."
    }

    if return_timestamps:
        result['timestamps'] = [
            {'start': 0.0, 'end': 2.5, 'text': '음성'},
            {'start': 2.5, 'end': 5.0, 'text': '인식'},
            {'start': 5.0, 'end': 8.0, 'text': '결과입니다'}
        ]

    return result


def render_ai_tasks():
    """Render AI tasks interface."""
    st.header("🎯 AI 태스크")

    # Task selection
    task_options = {
        "텍스트 분류": "text_classification",
        "텍스트 생성": "text_generation",
        "질문 답변": "question_answering",
        "텍스트 요약": "summarization",
        "번역": "translation",
        "이미지 분류": "image_classification",
        "음성 인식": "audio_transcription"
    }

    selected_task = st.selectbox(
        "수행할 태스크를 선택하세요",
        options=list(task_options.keys()),
        index=0
    )

    st.divider()

    # Render selected task
    task_type = task_options[selected_task]

    if task_type == "text_classification":
        render_text_classification_task()
    elif task_type == "text_generation":
        render_text_generation_task()
    elif task_type == "question_answering":
        render_question_answering_task()
    elif task_type == "summarization":
        render_summarization_task()
    elif task_type == "translation":
        render_translation_task()
    elif task_type == "image_classification":
        render_image_classification_task()
    elif task_type == "audio_transcription":
        render_audio_transcription_task()

    # Task tips
    with st.expander("💡 태스크 사용 팁"):
        st.markdown("""
        - **모델 로드**: 태스크 실행 전에 해당 태스크를 지원하는 모델을 로드해주세요
        - **입력 길이**: 모델마다 최대 입력 길이 제한이 있습니다
        - **배치 처리**: 여러 텍스트를 동시에 처리하면 더 효율적입니다
        - **결과 해석**: 신뢰도 점수를 참고하여 결과를 해석하세요
        - **성능**: GPU를 사용하면 처리 속도가 크게 향상됩니다
        """)


def init_tasks_state():
    """Initialize AI tasks-related session state."""
    if 'selected_task' not in st.session_state:
        st.session_state['selected_task'] = 'text_classification'