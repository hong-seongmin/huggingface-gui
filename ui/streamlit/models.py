"""
Model management UI components for Streamlit.
"""
import streamlit as st
from typing import Dict, Any, Optional, List
from core.state_manager import state_manager
from core.cache_manager import cache_manager
from core.logging_config import get_logger
from utils.validators import validate_model_id
from utils.formatters import format_json, format_model_info

logger = get_logger(__name__)


def render_model_status_banner():
    """Render model status banner."""
    status_items = []

    if st.session_state.get('model_path_input', ''):
        status_items.append(f"입력 경로: {st.session_state['model_path_input']}")

    if st.session_state.get('current_model_analysis'):
        status_items.append("분석 결과 존재")

    if st.session_state.get('selected_cached_model', '직접 입력') != '직접 입력':
        status_items.append(f"캐시 선택: {st.session_state['selected_cached_model']}")

    if status_items:
        st.success(f"🟢 **저장된 상태**: {', '.join(status_items)}")
    else:
        st.info("🔵 **모델 관리**: 새로운 세션 - 아래에서 모델을 선택하거나 입력하세요")


def render_cached_model_selector():
    """Render cached model selector."""
    if not st.session_state.get('cache_info'):
        st.info("캐시 정보가 없습니다. 먼저 캐시를 스캔해주세요.")
        return

    with st.expander("🗂️ 캐시된 모델에서 선택", expanded=False):
        cached_models = []
        for repo in st.session_state['cache_info'].repos:
            cached_models.append(repo.repo_id)

        if cached_models:
            # 저장된 선택값 복원
            saved_selection = st.session_state.get('selected_cached_model', '직접 입력')
            try:
                default_index = (["직접 입력"] + cached_models).index(saved_selection)
            except ValueError:
                default_index = 0

            selected_cached_model = st.selectbox(
                "캐시된 모델 선택",
                options=["직접 입력"] + cached_models,
                index=default_index,
                key="cached_model_select"
            )

            if selected_cached_model != "직접 입력":
                # 캐시된 모델 선택 시 자동으로 경로 설정
                st.session_state['model_path_input'] = selected_cached_model
                st.session_state['selected_cached_model'] = selected_cached_model
                state_manager.save_state(st.session_state)
                st.success(f"✅ 선택된 모델: `{selected_cached_model}`")
            else:
                st.session_state['selected_cached_model'] = '직접 입력'
                state_manager.save_state(st.session_state)
        else:
            st.info("캐시된 모델이 없습니다.")


def render_model_input():
    """Render model path input section."""
    st.markdown("#### 🔗 모델 경로 입력")

    model_path = st.text_input(
        "모델 경로",
        key="model_path_input",
        placeholder="예: tabularisai/multilingual-sentiment-analysis",
        help="로컬 경로 또는 HuggingFace 모델 ID를 입력하세요"
    )

    # Validate input
    if model_path:
        is_valid, error_msg = validate_model_id(model_path)
        if not is_valid:
            st.warning(f"⚠️ {error_msg}")

    return model_path


def render_model_actions(model_path: str):
    """Render model action buttons."""
    st.markdown("#### ⚡ 액션")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        analyze_clicked = st.button("🔍 모델 분석", use_container_width=True, type="secondary")

    with col2:
        load_clicked = st.button("📤 모델 로드", use_container_width=True, type="primary")

    with col3:
        refresh_clicked = st.button("🔄 상태 새로고침", use_container_width=True)

    with col4:
        clear_clicked = st.button("🧹 입력 지우기", use_container_width=True)

    # Handle button actions
    if analyze_clicked:
        handle_analyze_model(model_path)

    if load_clicked:
        handle_load_model(model_path)

    if refresh_clicked:
        st.rerun()

    if clear_clicked:
        handle_clear_input()


def handle_analyze_model(model_path: str):
    """Handle model analysis action."""
    if not model_path:
        st.error("❌ 모델 경로를 입력하세요.")
        return

    try:
        with st.spinner("🔍 모델 분석 중..."):
            if 'model_manager' not in st.session_state:
                st.error("❌ 모델 매니저가 초기화되지 않았습니다.")
                return

            analysis = st.session_state['model_manager'].analyze_model(model_path)

            if 'error' in analysis:
                st.error(f"❌ 모델 분석 실패: {analysis['error']}")
            else:
                st.session_state['current_model_analysis'] = analysis
                state_manager.save_state(st.session_state)
                st.success("✅ 모델 분석 완료!")

                # Show analysis results
                render_analysis_results(analysis)

    except Exception as e:
        error_msg = f"모델 분석 중 오류 발생: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)


def handle_load_model(model_path: str):
    """Handle model loading action."""
    if not model_path:
        st.error("❌ 모델 경로를 입력하세요.")
        return

    try:
        if 'model_manager' not in st.session_state:
            st.error("❌ 모델 매니저가 초기화되지 않았습니다.")
            return

        # Generate model name
        if st.session_state['model_manager']._is_huggingface_model_id(model_path):
            model_name = model_path.split('/')[-1]
        else:
            model_name = model_path.replace('/', '_').replace('\\', '_')

        with st.spinner(f"📤 모델 로딩 중: {model_name}"):
            success = st.session_state['model_manager'].load_model(model_path, model_name)

            if success:
                st.success(f"✅ 모델 로드 성공: {model_name}")
                state_manager.save_state(st.session_state)
            else:
                st.error(f"❌ 모델 로드 실패: {model_name}")

    except Exception as e:
        error_msg = f"모델 로딩 중 오류 발생: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)


def handle_clear_input():
    """Handle clear input action."""
    st.session_state['model_path_input'] = ''
    st.session_state['selected_cached_model'] = '직접 입력'
    st.session_state['current_model_analysis'] = None
    state_manager.save_state(st.session_state)
    st.success("🧹 입력이 지워졌습니다.")
    st.rerun()


def render_analysis_results(analysis: Dict[str, Any]):
    """Render model analysis results."""
    if not analysis or 'error' in analysis:
        return

    st.subheader("📊 모델 분석 결과")

    # Basic model info
    if 'model_info' in analysis:
        with st.expander("📋 모델 기본 정보", expanded=True):
            model_info = analysis['model_info']
            st.markdown(format_model_info(model_info))

    # Architecture details
    if 'architecture' in analysis:
        with st.expander("🏗️ 아키텍처 정보"):
            st.json(analysis['architecture'])

    # Config information
    if 'config' in analysis:
        with st.expander("⚙️ 모델 설정"):
            st.json(analysis['config'])

    # Task information
    if 'supported_tasks' in analysis:
        with st.expander("🎯 지원 태스크"):
            tasks = analysis['supported_tasks']
            if isinstance(tasks, list):
                for task in tasks:
                    st.write(f"• {task}")
            else:
                st.write(tasks)

    # Performance metrics
    if 'performance' in analysis:
        with st.expander("📈 성능 정보"):
            perf = analysis['performance']
            col1, col2, col3 = st.columns(3)

            with col1:
                if 'memory_usage' in perf:
                    st.metric("메모리 사용량", f"{perf['memory_usage']:.1f} MB")

            with col2:
                if 'loading_time' in perf:
                    st.metric("로딩 시간", f"{perf['loading_time']:.2f}초")

            with col3:
                if 'model_size' in perf:
                    st.metric("모델 크기", f"{perf['model_size']:.1f} MB")


def render_loaded_models_status():
    """Render loaded models status."""
    if 'model_manager' not in st.session_state:
        return

    manager = st.session_state['model_manager']

    # Loaded models
    if manager.loaded_models:
        st.subheader("✅ 로드된 모델")
        for model_name, model_info in manager.loaded_models.items():
            with st.expander(f"📦 {model_name}", expanded=False):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.write(f"**경로**: {model_info.path}")
                    st.write(f"**로드 시간**: {model_info.load_time:.2f}초")
                    if hasattr(model_info, 'memory_usage'):
                        st.write(f"**메모리 사용량**: {model_info.memory_usage:.1f} MB")

                with col2:
                    if st.button(f"🗑️ 언로드", key=f"unload_{model_name}"):
                        if manager.unload_model(model_name):
                            st.success(f"모델 {model_name}이 언로드되었습니다.")
                            st.rerun()
                        else:
                            st.error(f"모델 {model_name} 언로드에 실패했습니다.")

    # Loading models
    if manager.loading_models:
        st.subheader("⏳ 로딩 중인 모델")
        for model_name in manager.loading_models:
            st.info(f"🔄 {model_name} 로딩 중...")

    # Failed models
    if manager.failed_models:
        st.subheader("❌ 로드 실패한 모델")
        for model_name, error in manager.failed_models.items():
            st.error(f"💥 {model_name}: {error}")


def render_model_load_section():
    """Render model loading section."""
    with st.container():
        st.subheader("📥 새 모델 로드")

        # Cached model selector
        render_cached_model_selector()

        # Model input
        model_path = render_model_input()

        # Action buttons
        render_model_actions(model_path)

        # Show current analysis if exists
        if st.session_state.get('current_model_analysis'):
            render_analysis_results(st.session_state['current_model_analysis'])


def render_model_management():
    """Render complete model management UI."""
    try:
        st.header("🤖 모델 관리")

        # Initialize state
        init_model_management_state()

        # Status banner
        render_model_status_banner()

        st.markdown("---")

        # Model loading section
        render_model_load_section()

        st.markdown("---")

        # Loaded models status
        render_loaded_models_status()

        # Model management tips
        with st.expander("💡 모델 관리 팁"):
            st.markdown("""
            - **모델 분석**: 로드하기 전에 모델을 분석하여 호환성을 확인하세요
            - **메모리 관리**: 큰 모델들은 많은 메모리를 사용합니다. 필요하지 않은 모델은 언로드하세요
            - **캐시 활용**: 한번 다운로드한 모델은 캐시에서 빠르게 로드됩니다
            """)
    except Exception as e:
        logger.error(f"Model management UI error: {e}")
        st.error(f"모델 관리 페이지에서 오류가 발생했습니다: {e}")
        st.info("페이지를 새로고침하거나 관리자에게 문의하세요.")


def init_model_management_state():
    """Initialize model management-related session state."""
    if 'model_path_input' not in st.session_state:
        st.session_state['model_path_input'] = ''

    if 'selected_cached_model' not in st.session_state:
        st.session_state['selected_cached_model'] = '직접 입력'

    if 'current_model_analysis' not in st.session_state:
        st.session_state['current_model_analysis'] = None


def get_model_suggestions(query: str) -> List[str]:
    """
    Get model suggestions based on query.

    Args:
        query: Search query

    Returns:
        List of suggested model IDs
    """
    # This would typically query HuggingFace Hub or a local index
    # For now, return some popular models
    popular_models = [
        "microsoft/DialoGPT-medium",
        "tabularisai/multilingual-sentiment-analysis",
        "bert-base-uncased",
        "gpt2",
        "distilbert-base-uncased",
        "facebook/blenderbot-400M-distill",
        "microsoft/DialoGPT-small",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
    ]

    if not query:
        return popular_models[:5]

    # Simple filtering
    filtered = [model for model in popular_models if query.lower() in model.lower()]
    return filtered[:5]