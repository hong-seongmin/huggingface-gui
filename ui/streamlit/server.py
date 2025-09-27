"""
FastAPI server management UI components for Streamlit.
"""
import streamlit as st
import time
import requests
from typing import Dict, List, Any, Optional
from core.state_manager import state_manager
from core.config import config
from core.logging_config import get_logger
from utils.formatters import format_json

logger = get_logger(__name__)


def render_server_status_banner():
    """Render server status banner."""
    if st.session_state.get('fastapi_server_running', False):
        st.success("🟢 **서버 상태**: 실행 중")
    else:
        loaded_models_count = get_loaded_models_count()
        if loaded_models_count == 0:
            st.error("❌ **서버 상태**: 로드된 모델이 없어 시작할 수 없음")
        else:
            st.warning("🟡 **서버 상태**: 중지됨 - 아래 버튼으로 시작하세요")


def render_server_metrics():
    """Render server metrics."""
    loaded_models_count = get_loaded_models_count()
    server_info = get_server_info()

    col1, col2, col3 = st.columns(3)

    with col1:
        active_servers = server_info.get('active_servers', [])
        is_running = len(active_servers) > 0

        if not is_running and 'fastapi_server' in st.session_state:
            is_running = (server_info.get('default_server_running', False) or
                         st.session_state['fastapi_server'].is_running())

        st.metric("서버 상태", "🟢 실행 중" if is_running else "🔴 중지됨")

    with col2:
        st.metric("로드된 모델 수", loaded_models_count)

    with col3:
        st.metric("활성 포트", len(st.session_state.get('model_ports', {})))


def render_model_port_settings():
    """Render model port configuration."""
    loaded_models = get_loaded_models()
    loaded_models_count = len(loaded_models)

    if loaded_models_count == 0:
        st.info("로드된 모델이 없습니다. 먼저 모델을 로드해주세요.")
        return

    st.subheader("🔧 모델별 포트 설정")

    # Initialize port settings
    if 'model_ports' not in st.session_state:
        st.session_state['model_ports'] = {}

    col_left, col_right = st.columns([2, 1])

    with col_left:
        render_port_configuration_table(loaded_models)

    with col_right:
        st.info("💡 **포트 설정 팁**\n- 각 모델마다 다른 포트 사용\n- 기본값: 8000, 8001, 8002...\n- 범위: 3000-65535")


def render_port_configuration_table(loaded_models: List[str]):
    """Render port configuration table."""
    server_info = get_server_info()
    active_servers = server_info.get('active_servers', [])

    for i, model_name in enumerate(loaded_models):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.write(f"🤖 **{model_name}**")

        with col2:
            # Default port (starting from 8000)
            default_port = config.API_PORT + i
            current_port = st.session_state['model_ports'].get(model_name, default_port)

            new_port = st.number_input(
                f"포트",
                min_value=3000,
                max_value=65535,
                value=current_port,
                key=f"port_{model_name}",
                help=f"{model_name} 모델의 API 포트"
            )

            if new_port != current_port:
                st.session_state['model_ports'][model_name] = new_port
                save_port_changes()

        with col3:
            # Check model server status
            model_port = st.session_state['model_ports'].get(model_name)
            model_running = False

            if model_port:
                for server in active_servers:
                    if server.get('port') == model_port:
                        model_running = True
                        break

            status = "🟢 실행중" if model_running else "⚪ 대기중"
            st.write(status)


def render_server_controls():
    """Render server control buttons."""
    st.subheader("🎛️ 서버 제어")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🚀 서버 시작", use_container_width=True, type="primary"):
            start_server()

    with col2:
        if st.button("⏹️ 서버 중지", use_container_width=True, type="secondary"):
            stop_server()

    with col3:
        if st.button("🔄 상태 새로고침", use_container_width=True):
            st.rerun()


def render_api_endpoints():
    """Render API endpoints documentation."""
    if not st.session_state.get('fastapi_server_running', False):
        st.info("서버를 시작하면 API 엔드포인트 정보가 표시됩니다.")
        return

    st.subheader("📚 API 엔드포인트")

    # Base URL
    base_url = f"http://{config.API_HOST}:{config.API_PORT}"

    # Endpoint documentation
    endpoints = [
        {
            "method": "GET",
            "path": "/",
            "description": "서버 정보 조회"
        },
        {
            "method": "GET",
            "path": "/models",
            "description": "로드된 모델 목록 조회"
        },
        {
            "method": "POST",
            "path": "/models/load",
            "description": "새 모델 로드"
        },
        {
            "method": "POST",
            "path": "/models/{model_name}/predict",
            "description": "모델 예측 실행"
        },
        {
            "method": "POST",
            "path": "/models/{model_name}/unload",
            "description": "모델 언로드"
        },
        {
            "method": "GET",
            "path": "/system/status",
            "description": "시스템 상태 조회"
        }
    ]

    # Display endpoints
    for endpoint in endpoints:
        with st.expander(f"{endpoint['method']} {endpoint['path']}"):
            st.write(f"**설명**: {endpoint['description']}")
            st.write(f"**전체 URL**: {base_url}{endpoint['path']}")

            # Example for POST endpoints
            if endpoint['method'] == 'POST':
                if 'predict' in endpoint['path']:
                    st.code("""
{
    "text": "분석할 텍스트",
    "return_all_scores": true,
    "top_k": 3
}
                    """, language='json')
                elif 'load' in endpoint['path']:
                    st.code("""
{
    "model_path": "microsoft/DialoGPT-medium",
    "model_name": "dialog_model"
}
                    """, language='json')

    # Swagger UI link
    swagger_url = f"{base_url}/docs"
    st.markdown(f"📖 **전체 API 문서**: [Swagger UI]({swagger_url})")


def render_server_logs():
    """Render server logs."""
    if not st.session_state.get('fastapi_server_running', False):
        return

    st.subheader("📋 서버 로그")

    # Get recent logs
    try:
        if 'fastapi_server' in st.session_state:
            logs = st.session_state['fastapi_server'].get_recent_logs()
            if logs:
                st.text_area("최근 로그", value=logs, height=200)
            else:
                st.info("로그가 없습니다.")
    except Exception as e:
        st.error(f"로그 조회 실패: {str(e)}")


def start_server():
    """Start FastAPI server."""
    try:
        loaded_models_count = get_loaded_models_count()

        if loaded_models_count == 0:
            st.error("❌ 로드된 모델이 없습니다. 먼저 모델을 로드해주세요.")
            return

        if 'fastapi_server' not in st.session_state:
            st.error("❌ FastAPI 서버가 초기화되지 않았습니다.")
            return

        with st.spinner("🚀 서버 시작 중..."):
            success = st.session_state['fastapi_server'].start_server()

            if success:
                st.session_state['fastapi_server_running'] = True
                state_manager.save_state(st.session_state)
                st.success("✅ FastAPI 서버가 시작되었습니다!")

                # Show server info
                server_info = get_server_info()
                if server_info:
                    st.info(f"서버 주소: http://{config.API_HOST}:{config.API_PORT}")

                st.rerun()
            else:
                st.error("❌ 서버 시작에 실패했습니다.")

    except Exception as e:
        error_msg = f"서버 시작 중 오류 발생: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)


def stop_server():
    """Stop FastAPI server."""
    try:
        if 'fastapi_server' not in st.session_state:
            st.warning("서버가 실행되고 있지 않습니다.")
            return

        with st.spinner("⏹️ 서버 중지 중..."):
            success = st.session_state['fastapi_server'].stop_server()

            if success:
                st.session_state['fastapi_server_running'] = False
                state_manager.save_state(st.session_state)
                st.success("✅ FastAPI 서버가 중지되었습니다.")
                st.rerun()
            else:
                st.warning("서버 중지에 실패했거나 이미 중지된 상태입니다.")

    except Exception as e:
        error_msg = f"서버 중지 중 오류 발생: {str(e)}"
        st.error(error_msg)
        logger.error(error_msg)


def test_api_endpoint():
    """Test API endpoint with sample request."""
    st.subheader("🧪 API 테스트")

    if not st.session_state.get('fastapi_server_running', False):
        st.warning("서버를 먼저 시작해주세요.")
        return

    # Select model for testing
    loaded_models = get_loaded_models()
    if not loaded_models:
        st.error("테스트할 모델이 없습니다.")
        return

    selected_model = st.selectbox("테스트할 모델 선택", loaded_models)

    # Test input
    test_text = st.text_input(
        "테스트 텍스트",
        value="This is a test message.",
        placeholder="API 테스트용 텍스트를 입력하세요"
    )

    if st.button("🧪 API 테스트 실행"):
        if test_text.strip():
            with st.spinner("API 테스트 중..."):
                result = perform_api_test(selected_model, test_text)
                if result:
                    st.success("✅ API 테스트 성공!")
                    st.json(result)
                else:
                    st.error("❌ API 테스트 실패")
        else:
            st.error("테스트 텍스트를 입력해주세요.")


def perform_api_test(model_name: str, text: str) -> Optional[Dict]:
    """Perform API test request."""
    try:
        model_port = st.session_state['model_ports'].get(model_name, config.API_PORT)
        url = f"http://{config.API_HOST}:{model_port}/models/{model_name}/predict"

        data = {
            "text": text,
            "return_all_scores": True,
            "top_k": 3
        }

        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        logger.error(f"API 테스트 실패: {str(e)}")
        st.error(f"API 요청 실패: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"API 테스트 중 오류: {str(e)}")
        st.error(f"오류 발생: {str(e)}")
        return None


def get_loaded_models() -> List[str]:
    """Get list of loaded models."""
    if 'model_manager' in st.session_state:
        return st.session_state['model_manager'].get_loaded_models()
    return []


def get_loaded_models_count() -> int:
    """Get count of loaded models."""
    return len(get_loaded_models())


def get_server_info() -> Dict[str, Any]:
    """Get server information."""
    if 'fastapi_server' in st.session_state:
        return st.session_state['fastapi_server'].get_server_info()
    return {}


def save_port_changes():
    """Save port changes with rate limiting."""
    if not hasattr(save_port_changes, 'last_save'):
        save_port_changes.last_save = 0

    current_time = time.time()
    if current_time - save_port_changes.last_save > 5:  # Save only every 5 seconds
        state_manager.save_state(st.session_state)
        save_port_changes.last_save = current_time


def render_fastapi_server():
    """Render complete FastAPI server management UI."""
    st.subheader("🚀 FastAPI 서버")

    # Initialize state
    init_server_state()

    # Status banner
    render_server_status_banner()

    # Server metrics
    render_server_metrics()

    st.divider()

    # Port settings
    render_model_port_settings()

    st.divider()

    # Server controls
    render_server_controls()

    st.divider()

    # API endpoints documentation
    render_api_endpoints()

    # API testing
    if st.session_state.get('fastapi_server_running', False):
        st.divider()
        test_api_endpoint()

    # Server tips
    with st.expander("💡 서버 사용 팁"):
        st.markdown("""
        - **모델 로드**: 서버 시작 전에 반드시 모델을 로드해주세요
        - **포트 설정**: 각 모델마다 다른 포트를 사용하여 동시 서비스 가능
        - **API 문서**: `/docs` 엔드포인트에서 Swagger UI를 통한 상세 문서 확인
        - **성능**: GPU를 사용하면 API 응답 속도가 크게 향상됩니다
        - **보안**: 프로덕션 환경에서는 인증 및 HTTPS 설정 권장
        """)


def init_server_state():
    """Initialize server-related session state."""
    if 'fastapi_server_running' not in st.session_state:
        st.session_state['fastapi_server_running'] = False

    if 'model_ports' not in st.session_state:
        st.session_state['model_ports'] = {}


def check_server_health() -> bool:
    """Check if server is healthy."""
    try:
        response = requests.get(f"http://{config.API_HOST}:{config.API_PORT}/", timeout=5)
        return response.status_code == 200
    except:
        return False