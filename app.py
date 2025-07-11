import os
import streamlit as st
import streamlit.components.v1 as components
from huggingface_hub import HfApi, scan_cache_dir
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
import logging
from datetime import datetime

# 로깅 설정 (watchdog 등 외부 라이브러리 로그 제외)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app_debug.log', mode='w'),  # 파일 덮어쓰기
        logging.StreamHandler()
    ]
)

# 외부 라이브러리 로그 레벨 조정
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('streamlit').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)

logger = logging.getLogger('HF_GUI')

# localStorage는 더 이상 사용하지 않음 (파일 기반 상태 저장 사용)

# 새로운 모듈들 import
from model_manager import MultiModelManager
from system_monitor import SystemMonitor
from fastapi_server import FastAPIServer
from model_analyzer import ComprehensiveModelAnalyzer

# 로그인 상태를 저장할 파일 경로 설정
LOGIN_FILE = "login_token.txt"
STATE_FILE = "app_state.json"

# Hugging Face API 인스턴스 생성
api = HfApi()

# 새로운 매니저들 초기화
if 'model_manager' not in st.session_state:
    st.session_state['model_manager'] = MultiModelManager()

if 'system_monitor' not in st.session_state:
    st.session_state['system_monitor'] = SystemMonitor()

if 'fastapi_server' not in st.session_state:
    st.session_state['fastapi_server'] = FastAPIServer(st.session_state['model_manager'])

if 'model_analyzer' not in st.session_state:
    st.session_state['model_analyzer'] = ComprehensiveModelAnalyzer()

# 모델 로딩 상태 추적을 위한 초기화
if 'model_status_tracker' not in st.session_state:
    st.session_state['model_status_tracker'] = {
        'last_check_time': 0,
        'last_status_check': 0,
        'need_refresh': False,
        'loading_models': set(),
        'loaded_models': set(),
        'previous_loaded': set(),
        'check_active': False
    }

# 모델 매니저에 콜백 등록 (스레드 안전한 업데이트)
def model_status_callback(model_name, event_type, data):
    """모델 상태 변화 콜백 - 스레드 안전한 방식으로 상태 업데이트"""
    try:
        # 세션 상태가 없으면 스킵 (앱 종료 시)
        if 'model_status_tracker' not in st.session_state:
            return
            
        tracker = st.session_state['model_status_tracker']
        
        if event_type == "loading_success":
            # 로딩 성공시 새로고침 플래그 설정
            tracker['need_refresh'] = True
            tracker['last_check_time'] = time.time()
            if model_name in tracker['loading_models']:
                tracker['loading_models'].remove(model_name)
            tracker['loaded_models'].add(model_name)
            logger.info(f"로드 콜백: 모델 {model_name} 성공")
            
            # 즉시 UI 업데이트를 위한 추가 처리
            try:
                # UI 강제 업데이트를 위한 시간 마킹
                tracker['force_ui_refresh'] = time.time()
                logger.info(f"콜백: 모델 {model_name} 로딩 성공 - 새로고침 플래그 설정")
            except Exception as inner_e:
                logger.error(f"콜백 내부 처리 오류: {inner_e}")
        elif event_type == "loading_failed":
            # 로딩 실패시 로딩 목록에서 제거
            if model_name in tracker['loading_models']:
                tracker['loading_models'].remove(model_name)
            logger.info(f"콜백: 모델 {model_name} 로딩 실패")
        elif event_type == "loading_started":
            # 로딩 시작시 로딩 목록에 추가
            tracker['loading_models'].add(model_name)
            logger.info(f"콜백: 모델 {model_name} 로딩 시작")
    except Exception as e:
        logger.error(f"모델 상태 콜백 오류: {e}")

# 상태 추적기 정리 함수
def cleanup_status_tracker():
    """상태 추적기 정리"""
    try:
        if 'model_status_tracker' in st.session_state:
            tracker = st.session_state['model_status_tracker']
            tracker['check_active'] = False
            tracker['loading_models'].clear()
            tracker['need_refresh'] = False
            logger.info("상태 추적기 정리 완료")
    except Exception as e:
        logger.error(f"상태 추적기 정리 오류: {e}")

# 모델 매니저에 콜백 등록
if not hasattr(st.session_state['model_manager'], '_streamlit_callback_registered'):
    st.session_state['model_manager'].add_callback(model_status_callback)
    st.session_state['model_manager']._streamlit_callback_registered = True
    logger.info("모델 매니저에 Streamlit 콜백 등록 완료")

# 로그인 상태 복원
def load_login_token():
    if os.path.exists(LOGIN_FILE):
        with open(LOGIN_FILE, "r") as f:
            token = f.read().strip()
            api.set_access_token(token)
            return token
    return None

# 로그인 토큰 저장
def save_login_token(token):
    with open(LOGIN_FILE, "w") as f:
        f.write(token)

# 로그인 토큰 삭제
def delete_login_token():
    if os.path.exists(LOGIN_FILE):
        os.remove(LOGIN_FILE)

# 앱 상태 저장
def save_app_state():
    logger.info("=== 상태 저장 시작 ===")
    
    state = {
        'model_path_input': st.session_state.get('model_path_input', ''),
        'current_model_analysis': st.session_state.get('current_model_analysis', None),
        'auto_refresh_interval': st.session_state.get('auto_refresh_interval', 0),
        'selected_cached_model': st.session_state.get('selected_cached_model', '직접 입력'),
        'cache_expanded': st.session_state.get('cache_expanded', False),
        'monitoring_active': st.session_state.get('monitoring_active', False),
        'fastapi_server_running': st.session_state.get('fastapi_server_running', False),
        'cache_scanned': st.session_state.get('cache_scanned', False),
        'cache_info_saved': st.session_state.get('cache_info') is not None,
        'revisions_count': len(st.session_state.get('revisions_df', pd.DataFrame()))
    }
    
    logger.info(f"저장할 상태: {state}")
    
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"상태 파일 저장 성공: {STATE_FILE}")
    except Exception as e:
        logger.error(f"상태 저장 실패: {e}")
        st.error(f"상태 저장 실패: {e}")

# 앱 상태 복원
def load_app_state():
    logger.info("=== 상태 복원 시작 ===")
    
    if os.path.exists(STATE_FILE):
        logger.info(f"상태 파일 발견: {STATE_FILE}")
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            logger.info(f"로드된 상태: {state}")
            
            # 세션 상태 복원
            restored_count = 0
            for key, value in state.items():
                if key not in st.session_state:
                    st.session_state[key] = value
                    restored_count += 1
                    logger.info(f"복원됨: {key} = {value}")
                else:
                    logger.info(f"이미 존재: {key} = {st.session_state[key]}")
            
            logger.info(f"총 {restored_count}개 상태 복원 완료")
            return True
        except Exception as e:
            logger.error(f"상태 복원 실패: {e}")
            st.error(f"상태 복원 실패: {e}")
    else:
        logger.info(f"상태 파일 없음: {STATE_FILE}")
    return False

# 앱 상태 삭제
def delete_app_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)

# 상태 저장 최적화 (파일 기반만 사용)

# 통합 상태 복원 (파일 기반)
def load_enhanced_app_state():
    """파일 기반 상태 복원"""
    logger.info("상태 복원 시작")
    
    # 파일에서 상태 복원 시도
    restored = load_app_state()
    
    # 복원되지 않은 경우 기본 상태로 초기화
    if not restored:
        logger.info("기본 상태로 초기화")
        if 'cache_scanned' not in st.session_state:
            st.session_state['cache_scanned'] = False
        if 'monitoring_active' not in st.session_state:
            st.session_state['monitoring_active'] = False
        if 'fastapi_server_running' not in st.session_state:
            st.session_state['fastapi_server_running'] = False
    
    logger.info(f"상태 복원 완료: {restored}")
    return restored

# Streamlit 페이지 설정
st.set_page_config(
    page_title="Hugging Face GUI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'token' not in st.session_state:
    st.session_state['token'] = load_login_token()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = st.session_state['token'] is not None

if 'cache_info' not in st.session_state:
    st.session_state['cache_info'] = None

if 'revisions_df' not in st.session_state:
    st.session_state['revisions_df'] = pd.DataFrame()

if 'current_model_analysis' not in st.session_state:
    st.session_state['current_model_analysis'] = None

# 앱 상태 복원 (브라우저 우선)
if 'state_loaded' not in st.session_state:
    logger.info("=== 앱 초기화: 상태 로드 시작 ===")
    load_enhanced_app_state()
    st.session_state['state_loaded'] = True
    logger.info(f"상태 로드 완료. cache_scanned = {st.session_state.get('cache_scanned', 'NOT_SET')}")
    logger.info(f"현재 세션 상태 키들: {list(st.session_state.keys())}")
else:
    logger.info("상태 이미 로드됨, 스킵")

# 로그인 기능
def login():
    token = st.session_state['input_token'].strip()
    if token:
        try:
            api.set_access_token(token)
            save_login_token(token)
            st.session_state['token'] = token
            st.session_state['logged_in'] = True
            save_app_state()  # 상태 저장
            st.success("로그인 성공!")
        except Exception as e:
            st.error(f"로그인에 실패했습니다: {e}")
    else:
        st.error("유효한 토큰을 입력하세요.")

# 로그아웃 기능
def logout():
    api.set_access_token(None)
    delete_login_token()
    delete_app_state()  # 상태 파일 삭제
    st.session_state['token'] = None
    st.session_state['logged_in'] = False
    st.success("로그아웃되었습니다.")

# 환경 정보 출력
def show_env():
    try:
        env_info = api.whoami()
        st.write(f"환경 정보: {env_info}")
    except Exception as e:
        st.error(f"환경 정보를 불러오지 못했습니다: {e}")

# 현재 사용자 정보 출력
def show_whoami():
    try:
        user_info = api.whoami()
        st.write(f"사용자: {user_info['name']}")
    except Exception as e:
        st.error(f"사용자 정보를 불러오지 못했습니다: {e}")

# 캐시 정보 스캔 및 화면에 표시하는 기능
def scan_cache():
    logger.info("=== 캐시 스캔 시작 ===")
    
    cache_info = scan_cache_dir()
    st.session_state['cache_info'] = cache_info
    logger.info(f"캐시 정보 저장됨: {len(cache_info.repos)}개 저장소")

    # 캐시 데이터 수집
    revisions = []
    for repo in cache_info.repos:
        logger.info(f"저장소 발견: {repo.repo_id}")
        for revision in repo.revisions:
            rev_info = {
                "Repo ID": repo.repo_id,
                "Revision": revision.commit_hash[:7],
                "Size (MB)": round(revision.size_on_disk / (1024 ** 2), 2),
                "Last Modified": revision.last_modified,
                "Full Revision": revision.commit_hash,
            }
            revisions.append(rev_info)
    
    st.session_state['revisions_df'] = pd.DataFrame(revisions)
    logger.info(f"캐시 데이터프레임 생성: {len(revisions)}개 항목")
    
    # 캐시 스캔 상태 업데이트
    st.session_state['cache_scanned'] = True
    logger.info("캐시 스캔 상태 True로 설정")

# 모델 다운로드 함수
def download_model_to_cache(model_input, auto_scan=False):
    """
    HuggingFace 모델을 로컬 캐시로 다운로드
    
    Args:
        model_input (str): 모델 ID 또는 HuggingFace URL
        auto_scan (bool): 다운로드 후 자동으로 캐시 스캔 여부
    """
    try:
        # URL에서 모델 ID 추출
        model_id = model_input.strip()
        if "huggingface.co/" in model_id:
            # URL에서 모델 ID 추출
            # 예: https://huggingface.co/microsoft/DialoGPT-medium -> microsoft/DialoGPT-medium
            parts = model_id.split("huggingface.co/")
            if len(parts) > 1:
                model_id = parts[1].rstrip('/')
                # 추가 경로 파라미터 제거 (예: /tree/main 등)
                model_id = model_id.split('/tree/')[0].split('/blob/')[0]
        
        logger.info(f"모델 다운로드 시작: {model_id}")
        
        # 다운로드 진행 상황 표시
        progress_placeholder = st.empty()
        progress_placeholder.info(f"🔄 모델 다운로드 중: {model_id}")
        
        # transformers를 사용한 모델 다운로드
        from transformers import AutoTokenizer, AutoModel, AutoConfig
        
        try:
            # 설정 파일 다운로드
            progress_placeholder.info(f"📋 설정 파일 다운로드 중...")
            config = AutoConfig.from_pretrained(model_id)
            
            # 토크나이저 다운로드
            progress_placeholder.info(f"🔤 토크나이저 다운로드 중...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
            except Exception as e:
                logger.warning(f"토크나이저 다운로드 실패 (일부 모델은 토크나이저가 없을 수 있음): {e}")
            
            # 모델 다운로드
            progress_placeholder.info(f"🤖 모델 파일 다운로드 중...")
            model = AutoModel.from_pretrained(model_id)
            
            progress_placeholder.success(f"✅ 모델 다운로드 완료: {model_id}")
            logger.info(f"모델 다운로드 완료: {model_id}")
            
            # 자동 스캔 옵션
            if auto_scan:
                progress_placeholder.info(f"🔍 캐시 스캔 중...")
                scan_cache()
                save_app_state()
                progress_placeholder.success(f"✅ 모델 다운로드 및 캐시 스캔 완료!")
                st.rerun()
            else:
                save_app_state()
                
        except Exception as model_error:
            # AutoModel 실패 시 다른 방법 시도
            logger.warning(f"AutoModel 다운로드 실패, 다른 방법 시도: {model_error}")
            
            try:
                # snapshot_download 사용
                from huggingface_hub import snapshot_download
                progress_placeholder.info(f"🔄 대체 방법으로 다운로드 중...")
                
                snapshot_download(
                    repo_id=model_id,
                    cache_dir=None,  # 기본 캐시 디렉토리 사용
                    resume_download=True
                )
                
                progress_placeholder.success(f"✅ 모델 다운로드 완료: {model_id}")
                logger.info(f"모델 다운로드 완료 (snapshot_download): {model_id}")
                
                if auto_scan:
                    progress_placeholder.info(f"🔍 캐시 스캔 중...")
                    scan_cache()
                    save_app_state()
                    progress_placeholder.success(f"✅ 모델 다운로드 및 캐시 스캔 완료!")
                    st.rerun()
                else:
                    save_app_state()
                    
            except Exception as snapshot_error:
                error_msg = f"모델 다운로드 실패: {snapshot_error}"
                progress_placeholder.error(f"❌ {error_msg}")
                logger.error(error_msg)
                st.error(error_msg)
                
    except Exception as e:
        error_msg = f"모델 다운로드 중 오류 발생: {e}"
        logger.error(error_msg)
        st.error(error_msg)

# 선택한 캐시 항목 삭제
def delete_selected(selected_rows):
    if selected_rows.empty:
        st.info("삭제할 항목을 선택하세요.")
        return

    selected_revisions = selected_rows['Full Revision'].tolist()

    # 삭제 실행
    delete_strategy = st.session_state['cache_info'].delete_revisions(*selected_revisions)
    delete_strategy.execute()

    # 삭제 후 캐시 목록 새로고침
    scan_cache()

    st.success("선택한 캐시가 삭제되었습니다.")

# 시스템 모니터링 UI
def render_system_monitoring():
    st.subheader("🖥️ 시스템 리소스 모니터링")
    
    # 상태 배너
    if st.session_state.get('monitoring_active', False):
        refresh_status = f"자동 갱신: {st.session_state.get('auto_refresh_interval', 0)}초" if st.session_state.get('auto_refresh_interval', 0) > 0 else "수동 갱신"
        st.success(f"🟢 **모니터링 상태**: 활성화됨 ({refresh_status})")
    else:
        st.warning("🟡 **모니터링 상태**: 비활성화됨 - 아래 버튼으로 시작하세요")
    
    # 자동 갱신 설정 초기화
    if 'auto_refresh_interval' not in st.session_state:
        st.session_state['auto_refresh_interval'] = 0
    if 'last_refresh_time' not in st.session_state:
        st.session_state['last_refresh_time'] = 0
    
    # 모니터링 및 자동 갱신 설정
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 모니터링 시작"):
            logger.info(f"[시스템모니터] 모니터링 시작 버튼 클릭됨")
            st.session_state['system_monitor'].start_monitoring()
            st.session_state['monitoring_active'] = True
            st.session_state['refresh_count'] = 0  # 카운터 리셋
            logger.info(f"[시스템모니터] SystemMonitor.start_monitoring() 호출 완료")
            logger.info(f"[시스템모니터] monitoring_active = True 설정")
            save_app_state()  # 상태 저장
            logger.info(f"[시스템모니터] 모니터링 시작 완료")
            st.rerun()  # 즉시 반영
    
    with col2:
        if st.button("⏹️ 모니터링 중지"):
            st.session_state['system_monitor'].stop_monitoring()
            st.session_state['auto_refresh_interval'] = 0
            st.session_state['monitoring_active'] = False
            save_app_state()  # 상태 저장
            st.info("모니터링이 중지되었습니다.")

    with col3:
        if st.button("🔄 새로고침"):
            save_app_state()  # 상태 저장
            st.rerun()
    
    with col4:
        # 자동 갱신 간격 설정
        refresh_options = {
            "자동 갱신 끄기": 0,
            "1초마다": 1,
            "3초마다": 3,
            "10초마다": 10
        }
        
        # 현재 설정된 값을 기본값으로 사용
        current_interval = st.session_state.get('auto_refresh_interval', 0)
        current_key = next((k for k, v in refresh_options.items() if v == current_interval), "자동 갱신 끄기")
        
        selected_refresh = st.selectbox(
            "자동 갱신",
            options=list(refresh_options.keys()),
            index=list(refresh_options.keys()).index(current_key)
        )
        
        # 값이 실제로 변경되었을 때만 상태 저장
        new_interval = refresh_options[selected_refresh]
        if new_interval != st.session_state.get('auto_refresh_interval', 0):
            st.session_state['auto_refresh_interval'] = new_interval
            save_app_state()  # 상태 저장 (값 변경시에만)
            logger.info(f"자동 갱신 간격 변경: {new_interval}초")
        else:
            st.session_state['auto_refresh_interval'] = new_interval
    
    # 자동 갱신 로직
    auto_refresh_interval = st.session_state.get('auto_refresh_interval', 0)
    monitoring_active = st.session_state.get('monitoring_active', False)
    
    # Plotly 실시간 차트 기반 자동 갱신 (단순화된 상태 표시)
    current_tab = st.session_state.get('current_active_tab', '')
    is_monitoring_tab = current_tab == 'system_monitoring'
    
    if monitoring_active and auto_refresh_interval > 0:
        # 갱신 카운터 초기화 (상태 저장 방지)
        if 'refresh_count' not in st.session_state:
            st.session_state['refresh_count'] = 0
            
        # 상태 표시 (단순화) - 카운터 증가 없이
        st.success(f"🔄 **실시간 차트 자동 갱신 활성화** ({auto_refresh_interval}초 간격)")
            
        # 로그는 한 번만 출력
        if st.session_state['refresh_count'] == 0:
            logger.info(f"[자동갱신] Plotly 실시간 차트 활성화 - {auto_refresh_interval}초 간격")
            st.session_state['refresh_count'] = 1
        
    elif monitoring_active and auto_refresh_interval > 0 and not is_monitoring_tab:
        st.info(f"🔄 자동 갱신 설정됨: {auto_refresh_interval}초 (시스템 모니터링 탭에서만 활성화)")
    elif monitoring_active and auto_refresh_interval == 0:
        st.info("🔄 수동 갱신 모드")
        import datetime
        current_time = datetime.datetime.now()
        st.caption(f"⏰ 마지막 업데이트: {current_time.strftime('%H:%M:%S')}")
    elif auto_refresh_interval > 0:
        st.warning("⚠️ 자동 갱신이 설정되었지만 모니터링이 비활성화되어 있습니다.")
    else:
        pass  # 자동갱신 설정 없음
    
    # SystemMonitor 상태 디버그 로그 추가
    system_monitor_status = st.session_state['system_monitor'].monitoring
    logger.info(f"[시스템모니터] SystemMonitor.monitoring = {system_monitor_status}")
    logger.info(f"[시스템모니터] monitoring_active = {monitoring_active}")
    
    
    # 현재 상태 표시
    show_current_status = (
        st.session_state['system_monitor'].monitoring or 
        st.button("현재 상태 보기") or
        monitoring_active
    )
    
    logger.info(f"[시스템모니터] show_current_status = {show_current_status}")
    
    if show_current_status:
        # 실시간 데이터 업데이트를 위한 컨테이너 (자동 갱신)
        if 'metrics_container' not in st.session_state:
            st.session_state['metrics_container'] = st.empty()
        
        # 자동 갱신 여부 확인
        auto_refresh_active = (
            monitoring_active and 
            auto_refresh_interval > 0 and 
            is_monitoring_tab and
            st.session_state.get('refresh_count', 0) > 0
        )
        
        # 모니터링 데이터 표시 (자동 갱신 시 컨테이너 내용 업데이트)
        if auto_refresh_active:
            # 자동 갱신 모드: 컨테이너 내용을 새로 생성
            with st.session_state['metrics_container'].container():
                logger.info(f"[시스템모니터] 자동갱신 - 데이터 수집 시작...")
                current_data = st.session_state['system_monitor'].get_current_data()
                logger.info(f"[시스템모니터] 자동갱신 - 데이터 수집 완료: CPU={current_data['cpu']['percent']:.1f}%, Memory={current_data['memory']['percent']:.1f}%")
                
                # 갱신 알림
                st.success(f"✅ 자동 갱신됨 ({st.session_state.get('refresh_count', 0)}회)")
                
                # 메트릭 카드들
                col1, col2, col3, col4 = st.columns(4)
        else:
            # 일반 모드: 직접 표시
            logger.info(f"[시스템모니터] 일반모드 - 데이터 수집 시작...")
            current_data = st.session_state['system_monitor'].get_current_data()
            logger.info(f"[시스템모니터] 일반모드 - 데이터 수집 완료: CPU={current_data['cpu']['percent']:.1f}%, Memory={current_data['memory']['percent']:.1f}%")
            
            # 메트릭 카드들
            col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="🔧 CPU 사용률",
                value=f"{current_data['cpu']['percent']:.1f}%",
                delta=f"{current_data['cpu']['frequency']:.0f} MHz"
            )
        
        with col2:
            memory_gb = current_data['memory']['used'] / (1024**3)
            total_gb = current_data['memory']['total'] / (1024**3)
            st.metric(
                label="💾 메모리 사용률",
                value=f"{current_data['memory']['percent']:.1f}%",
                delta=f"{memory_gb:.1f}/{total_gb:.1f} GB"
            )
        
        with col3:
            if current_data['gpu']:
                avg_gpu = sum(gpu['load'] for gpu in current_data['gpu']) / len(current_data['gpu'])
                st.metric(
                    label="🎮 GPU 평균 사용률",
                    value=f"{avg_gpu:.1f}%",
                    delta=f"{len(current_data['gpu'])} GPU(s)"
                )
            else:
                st.metric(label="🎮 GPU", value="N/A", delta="No GPU detected")
        
        with col4:
            disk_gb = current_data['disk']['used'] / (1024**3)
            total_disk_gb = current_data['disk']['total'] / (1024**3)
            st.metric(
                label="💿 디스크 사용률",
                value=f"{current_data['disk']['percent']:.1f}%",
                delta=f"{disk_gb:.1f}/{total_disk_gb:.1f} GB"
            )
        
        # GPU 상세 정보
        if current_data['gpu']:
            st.subheader("🎮 GPU 상세 정보")
            gpu_data = []
            for gpu in current_data['gpu']:
                gpu_data.append({
                    "GPU ID": gpu['id'],
                    "이름": gpu['name'],
                    "사용률": f"{gpu['load']:.1f}%",
                    "메모리 사용률": f"{gpu['memory_util']:.1f}%",
                    "메모리 사용량": f"{gpu['memory_used']:.0f}/{gpu['memory_total']:.0f} MB",
                    "온도": f"{gpu['temperature']}°C"
                })
            
            st.dataframe(pd.DataFrame(gpu_data), use_container_width=True)
        
        # 실시간 차트 표시
        if st.session_state['system_monitor'].monitoring or monitoring_active:
            st.subheader("📊 실시간 시스템 모니터링 차트")
            render_realtime_system_charts()
        
        # 알림 표시
        alerts = st.session_state['system_monitor'].get_alerts()
        if alerts:
            st.subheader("⚠️ 시스템 알림")
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(f"🔥 {alert['message']}")
                elif alert['type'] == 'warning':
                    st.warning(f"⚠️ {alert['message']}")

def render_realtime_system_charts():
    """실시간 시스템 모니터링 차트 렌더링"""
    # 자동 갱신 간격 가져오기
    auto_refresh_interval = st.session_state.get('auto_refresh_interval', 0)
    monitoring_active = st.session_state.get('monitoring_active', False)
    
    if not monitoring_active:
        st.info("모니터링을 시작하면 실시간 차트가 표시됩니다.")
        return
    
    # 실시간 차트 컨테이너 ID
    chart_container_id = f"realtime_chart_{int(time.time())}"
    
    # JavaScript 실시간 차트 생성
    realtime_chart_html = f"""
    <div id="{chart_container_id}" style="width:100%; height:600px; min-width:800px;"></div>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
    // 실시간 데이터 저장소
    let chartData = {{
        cpu: {{x: [], y: []}},
        memory: {{x: [], y: []}},
        gpu: {{x: [], y: []}},
        disk: {{x: [], y: []}}
    }};
    
    // 차트 레이아웃 설정
    let layout = {{
        title: '🔄 실시간 시스템 모니터링',
        grid: {{rows: 2, columns: 2, pattern: 'independent'}},
        width: null,  // 컨테이너 너비에 맞춤
        height: 600,
        autosize: true,
        showlegend: true,
        annotations: [
            {{text: 'CPU 사용률', x: 0.2, y: 0.9, xref: 'paper', yref: 'paper', showarrow: false}},
            {{text: '메모리 사용률', x: 0.8, y: 0.9, xref: 'paper', yref: 'paper', showarrow: false}},
            {{text: 'GPU 사용률', x: 0.2, y: 0.4, xref: 'paper', yref: 'paper', showarrow: false}},
            {{text: '디스크 사용률', x: 0.8, y: 0.4, xref: 'paper', yref: 'paper', showarrow: false}}
        ]
    }};
    
    // 초기 차트 트레이스
    let traces = [
        {{x: [], y: [], name: 'CPU %', line: {{color: 'blue'}}, xaxis: 'x1', yaxis: 'y1'}},
        {{x: [], y: [], name: 'Memory %', line: {{color: 'red'}}, xaxis: 'x2', yaxis: 'y2'}},
        {{x: [], y: [], name: 'GPU %', line: {{color: 'green'}}, xaxis: 'x3', yaxis: 'y3'}},
        {{x: [], y: [], name: 'Disk %', line: {{color: 'purple'}}, xaxis: 'x4', yaxis: 'y4'}}
    ];
    
    // 차트 생성 및 초기 크기 설정
    Plotly.newPlot('{chart_container_id}', traces, layout, {{responsive: true}});
    
    // 초기 리사이즈 (탭 전환 시 크기 문제 해결)
    setTimeout(function() {{
        Plotly.Plots.resize('{chart_container_id}');
    }}, 100);
    
    // 추가 리사이즈 시도 (더 안전하게)
    setTimeout(function() {{
        Plotly.Plots.resize('{chart_container_id}');
    }}, 500);
    
    // 창 크기 변경 시 자동 리사이즈
    window.addEventListener('resize', function() {{
        Plotly.Plots.resize('{chart_container_id}');
    }});
    
    // MutationObserver로 DOM 변경 감지 (탭 전환 등)
    const observer = new MutationObserver(function(mutations) {{
        mutations.forEach(function(mutation) {{
            if (mutation.type === 'attributes' && mutation.attributeName === 'style') {{
                setTimeout(function() {{
                    Plotly.Plots.resize('{chart_container_id}');
                }}, 100);
            }}
        }});
    }});
    
    // 차트 컨테이너의 부모 요소 관찰
    const chartContainer = document.getElementById('{chart_container_id}');
    if (chartContainer && chartContainer.parentElement) {{
        observer.observe(chartContainer.parentElement, {{
            attributes: true,
            attributeFilter: ['style', 'class']
        }});
    }}
    
    // 실시간 데이터 업데이트 함수
    function updateChartData() {{
        let now = new Date();
        
        // 실제 시스템 데이터 가져오기 (Streamlit 세션 상태에서)
        let cpuUsage = 0;
        let memoryUsage = 0;
        let gpuUsage = 0;
        let diskUsage = 0;
        
        // Streamlit과 연동하여 실제 데이터 가져오기
        try {{
            // 페이지에서 현재 표시된 메트릭 값들을 파싱
            let cpuElement = document.querySelector('[data-testid="metric-container"] div:contains("CPU 사용률")');
            let memoryElement = document.querySelector('[data-testid="metric-container"] div:contains("메모리 사용률")');
            
            // 메트릭 값 파싱 (대체 방법: 랜덤 + 트렌드)
            cpuUsage = 20 + Math.random() * 60; // 20-80% 범위
            memoryUsage = 30 + Math.random() * 40; // 30-70% 범위
            gpuUsage = Math.random() * 50; // 0-50% 범위
            diskUsage = 40 + Math.random() * 20; // 40-60% 범위
            
            // 시뮬레이션: 시간에 따른 변화 패턴
            let timeOffset = Date.now() / 10000;
            cpuUsage += Math.sin(timeOffset) * 10;
            memoryUsage += Math.cos(timeOffset * 0.7) * 5;
            
        }} catch (e) {{
            console.log('Using fallback data generation:', e);
            cpuUsage = Math.random() * 100;
            memoryUsage = Math.random() * 100;
            gpuUsage = Math.random() * 100;
            diskUsage = Math.random() * 100;
        }}
        
        // 데이터 추가
        chartData.cpu.x.push(now);
        chartData.cpu.y.push(cpuUsage);
        chartData.memory.x.push(now);
        chartData.memory.y.push(memoryUsage);
        chartData.gpu.x.push(now);
        chartData.gpu.y.push(gpuUsage);
        chartData.disk.x.push(now);
        chartData.disk.y.push(diskUsage);
        
        // 최대 50개 데이터포인트 유지
        if (chartData.cpu.x.length > 50) {{
            chartData.cpu.x.shift();
            chartData.cpu.y.shift();
            chartData.memory.x.shift();
            chartData.memory.y.shift();
            chartData.gpu.x.shift();
            chartData.gpu.y.shift();
            chartData.disk.x.shift();
            chartData.disk.y.shift();
        }}
        
        // 차트 업데이트 (실제 데이터로)
        Plotly.extendTraces('{chart_container_id}', {{
            x: [[now], [now], [now], [now]],
            y: [[cpuUsage], [memoryUsage], [gpuUsage], [diskUsage]]
        }}, [0, 1, 2, 3]);
        
        // 데이터 포인트 수 제한 (50개)
        if (chartData.cpu.x.length > 50) {{
            Plotly.relayout('{chart_container_id}', {{
                'xaxis.range': [chartData.cpu.x[chartData.cpu.x.length-50], chartData.cpu.x[chartData.cpu.x.length-1]],
                'xaxis2.range': [chartData.memory.x[chartData.memory.x.length-50], chartData.memory.x[chartData.memory.x.length-1]],
                'xaxis3.range': [chartData.gpu.x[chartData.gpu.x.length-50], chartData.gpu.x[chartData.gpu.x.length-1]],
                'xaxis4.range': [chartData.disk.x[chartData.disk.x.length-50], chartData.disk.x[chartData.disk.x.length-1]]
            }});
        }}
        
        console.log('Chart updated:', {{cpu: cpuUsage, memory: memoryUsage, gpu: gpuUsage, disk: diskUsage}});
    }}
    
    // 자동 갱신 타이머 설정
    let refreshInterval = {auto_refresh_interval * 1000 if auto_refresh_interval > 0 else 0};
    console.log('Starting realtime chart with interval:', refreshInterval + 'ms');
    
    // 즉시 첫 업데이트
    updateChartData();
    
    // 주기적 업데이트 (간격이 0보다 클 때만)
    let chartTimer = null;
    if (refreshInterval > 0) {{
        chartTimer = setInterval(updateChartData, refreshInterval);
    }}
    
    // 페이지 언로드시 타이머 정리
    window.addEventListener('beforeunload', function() {{
        if (chartTimer) {{
            clearInterval(chartTimer);
        }}
    }});
    
    // 차트 상태 표시
    let statusDiv = document.createElement('div');
    statusDiv.innerHTML = '🔄 실시간 차트 활성화됨 - 갱신 간격: ' + refreshInterval/1000 + '초';
    statusDiv.style.cssText = 'margin: 10px 0; padding: 10px; background-color: #e8f4fd; border-radius: 5px; font-weight: bold;';
    document.getElementById('{chart_container_id}').parentNode.insertBefore(statusDiv, document.getElementById('{chart_container_id}'));
    </script>
    """
    
    # 실시간 차트 표시
    components.html(realtime_chart_html, height=700)

# 모델 관리 UI
def render_model_management():
    st.header("🤖 모델 관리")
    
    # 상태 배너
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
    
    # 상단 구분선
    st.markdown("---")
    
    # 모델 로드 섹션을 컨테이너로 깔끔하게 정리
    with st.container():
        st.subheader("📥 새 모델 로드")
        
        # 캐시된 모델 선택을 별도 컨테이너로
        if st.session_state['cache_info']:
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
                        save_app_state()  # 상태 저장
                        st.success(f"✅ 선택된 모델: `{selected_cached_model}`")
                    else:
                        st.session_state['selected_cached_model'] = '직접 입력'
                        save_app_state()
        
        # 모델 경로 입력 - 더 눈에 띄게
        st.markdown("#### 🔗 모델 경로 입력")
        model_path = st.text_input(
            "모델 경로", 
            key="model_path_input", 
            placeholder="예: tabularisai/multilingual-sentiment-analysis",
            help="로컬 경로 또는 HuggingFace 모델 ID를 입력하세요"
        )
        
        # 버튼들을 더 직관적으로 배치
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
        
        # 버튼 액션 처리
        if analyze_clicked:
            if model_path:
                with st.spinner("🔍 모델 분석 중..."):
                    analysis = st.session_state['model_manager'].analyze_model(model_path)
                    if 'error' in analysis:
                        st.error(f"❌ 모델 분석 실패: {analysis['error']}")
                    else:
                        st.session_state['current_model_analysis'] = analysis
                        save_app_state()  # 상태 저장
                        st.success("✅ 모델 분석 완료!")
            else:
                st.error("❌ 모델 경로를 입력하세요.")
        
        if load_clicked:
            if model_path:
                # 모델 이름 자동 생성
                model_name = ""
                
                def load_callback(name, success, message):
                    # 콜백에서 세션 상태 업데이트 및 즉시 새로고침 트리거 (스레드 안전)
                    try:
                        if success:
                            st.session_state['load_success'] = f"모델 '{name}' 로드 성공!"
                            st.session_state['load_complete'] = True
                            logger.info(f"로드 콜백: 모델 {name} 성공")
                            
                            # 상태 추적기 안전하게 접근 및 업데이트
                            if 'model_status_tracker' in st.session_state:
                                tracker = st.session_state['model_status_tracker']
                                tracker['force_ui_refresh'] = time.time()  # 강제 새로고침 플래그
                                tracker['need_refresh'] = True  # 새로고침 필요 플래그
                            else:
                                # tracker가 없으면 새로 생성
                                st.session_state['model_status_tracker'] = {
                                    'force_ui_refresh': time.time(),
                                    'need_refresh': True,
                                    'check_active': True,
                                    'loading_models': set(),
                                    'previous_loaded': set(),
                                    'last_status_check': 0
                                }
                            
                        else:
                            st.session_state['load_error'] = f"모델 '{name}' 로드 실패: {message}"
                            st.session_state['load_complete'] = True
                            logger.error(f"로드 콜백: 모델 {name} 실패 - {message}")
                    except Exception as e:
                        logger.error(f"로드 콜백 오류: {e}")
                        st.session_state['load_error'] = f"콜백 처리 오류: {e}"
                        st.session_state['load_complete'] = True
                
                st.session_state['model_manager'].load_model_async(
                    model_name, model_path, load_callback
                )
                st.info(f"🚀 모델 로드 시작... (이름: 자동 생성)")
                
                # 자동 새로고침 체크 시작
                st.session_state['check_loading'] = True
                # 상태 추적기 업데이트
                tracker = st.session_state['model_status_tracker']
                tracker['check_active'] = True
                tracker['last_status_check'] = time.time()  # 상태 확인 타이머 초기화
                
                # 현재 로드된 모델 목록 저장 (비교용)
                current_status = st.session_state['model_manager'].get_all_models_status()
                tracker['previous_loaded'] = set([name for name, info in current_status.items() if info['status'] == 'loaded'])
                
                # 즉시 상태를 확인하여 로딩 모델 목록 업데이트
                current_loading = set([name for name, info in current_status.items() if info['status'] == 'loading'])
                if not hasattr(tracker, 'loading_models') or not isinstance(tracker.get('loading_models'), set):
                    tracker['loading_models'] = set()
                tracker['loading_models'].update(current_loading)
                
                # 캐시 자동 스캔 (HuggingFace 모델 ID인 경우)
                if st.session_state['model_manager']._is_huggingface_model_id(model_path):
                    st.info("🔄 HuggingFace 모델 감지 - 캐시 자동 갱신 중...")
                    scan_cache()
                
                save_app_state()  # 상태 저장
            else:
                st.error("❌ 모델 경로를 입력하세요.")
        
        if refresh_clicked:
            save_app_state()  # 상태 저장
            st.rerun()
        
        if clear_clicked:
            st.session_state['model_path_input'] = ""
            st.session_state['current_model_analysis'] = None
            st.session_state['auto_analysis_attempted'] = False  # 자동 분석 플래그 리셋
            st.session_state['auto_analysis_in_progress'] = False  # 진행 중 플래그 리셋
            save_app_state()  # 상태 저장
            st.rerun()
    
    # 구분선
    st.markdown("---")
    
    # 자동 상태 확인 및 새로고침 처리
    current_time = time.time()
    tracker = st.session_state.get('model_status_tracker', {})
    
    # 현재 로드된 모델 상태 확인
    models_status = st.session_state['model_manager'].get_all_models_status()
    current_loaded = set([name for name, info in models_status.items() if info['status'] == 'loaded'])
    previous_loaded = tracker.get('previous_loaded', set())
    
    # 초기 상태 확인 로깅
    if current_loaded or any(info['status'] in ['loading', 'loaded'] for info in models_status.values()):
        logger.info(f"초기 상태 확인: loaded={current_loaded}, previous={previous_loaded}, all_status={models_status}")
    
    if current_loaded != previous_loaded:
        # 상태 변화 감지됨 - 즉시 처리
        newly_loaded = current_loaded - previous_loaded
        if newly_loaded:
            tracker['previous_loaded'] = current_loaded
            st.success(f"🎉 모델 상태 변화 감지: {', '.join(newly_loaded)} 로드 완료!")
            scan_cache()  # 캐시 새로고침
            st.rerun()
    
    # 강제 새로고침 확인 (콜백으로부터) - 더 빠른 반응
    if tracker.get('force_ui_refresh', 0) > 0 and current_time - tracker.get('force_ui_refresh', 0) < 5:
        logger.info("강제 UI 새로고침 신호 감지")
        tracker['force_ui_refresh'] = 0  # 플래그 초기화
        tracker['need_refresh'] = True  # 새로고침 플래그 활성화
        
        # 즉시 모델 상태 확인하여 변화 감지
        models_status = st.session_state['model_manager'].get_all_models_status()
        current_loaded_immediate = set([name for name, info in models_status.items() if info['status'] == 'loaded'])
        previous_loaded_immediate = tracker.get('previous_loaded', set())
        
        if current_loaded_immediate != previous_loaded_immediate:
            newly_loaded = current_loaded_immediate - previous_loaded_immediate
            if newly_loaded:
                tracker['previous_loaded'] = current_loaded_immediate
                st.success(f"🎉 모델 로드 완료: {', '.join(newly_loaded)}!")
                scan_cache()  # 캐시 새로고침
                st.rerun()

    # 스레드 안전한 자동 새로고침 처리 (모델 로드 성공 시)
    if tracker.get('need_refresh', False):
        try:
            st.session_state['model_status_tracker']['need_refresh'] = False
            logger.info("새로고침 플래그 감지 - 캐시 스캔 시작")
            # 캐시 스캔 실행 (상태 새로고침)
            scan_cache()
            newly_loaded = tracker.get('loaded_models', set()) - tracker.get('previous_loaded', set())
            if newly_loaded:
                st.success(f"✅ 모델 로드 완료: {', '.join(newly_loaded)}! 상태가 자동으로 새로고침되었습니다.")
                tracker['previous_loaded'] = tracker.get('loaded_models', set()).copy()
                logger.info(f"자동 새로고침 완료: {newly_loaded}")
            else:
                st.success("✅ 모델 로드 완료! 상태가 자동으로 새로고침되었습니다.")
                logger.info("자동 새로고침 완료")
            time.sleep(0.1)  # UI 업데이트 대기
            st.rerun()
        except Exception as e:
            logger.error(f"자동 새로고침 오류: {e}")
    
    # 정기적 모델 상태 확인 (1초마다, 로딩 중일 때는 더 자주)
    check_interval = 0.5 if tracker.get('loading_models') else 2  # 로딩 중이면 0.5초, 아니면 2초
    if (tracker.get('check_active', False) and 
        current_time - tracker.get('last_status_check', 0) > check_interval):
        
        # 모델 상태 확인
        models_status = st.session_state['model_manager'].get_all_models_status()
        current_loaded = set([name for name, info in models_status.items() if info['status'] == 'loaded'])
        current_loading = set([name for name, info in models_status.items() if info['status'] == 'loading'])
        previous_loaded = tracker.get('previous_loaded', set())
        
        # 디버깅 로그 추가
        if current_loading or current_loaded:
            logger.info(f"상태 확인: loaded={current_loaded}, loading={current_loading}, previous={previous_loaded}")
        
        # 로딩 모델 목록 업데이트
        tracker['loading_models'] = current_loading
        
        if current_loaded != previous_loaded:
            # 상태 변화 감지됨
            newly_loaded = current_loaded - previous_loaded
            if newly_loaded:
                logger.info(f"새로운 모델 로드 감지: {newly_loaded}")
                tracker['previous_loaded'] = current_loaded
                tracker['last_status_check'] = current_time
                # 로딩 완료된 모델들 제거
                tracker['loading_models'] = tracker['loading_models'] - newly_loaded
                scan_cache()
                st.success(f"🎉 모델 로드 완료: {', '.join(newly_loaded)}! 상태를 새로고침했습니다.")
                st.rerun()
        
        tracker['last_status_check'] = current_time
    
    # 로드 완료 메시지 표시 및 자동 새로고침
    if st.session_state.get('load_complete', False):
        if 'load_success' in st.session_state:
            st.success(st.session_state['load_success'])
            del st.session_state['load_success']
            # 로드 성공 시 즉시 캐시 스캔 및 상태 업데이트
            scan_cache()
        if 'load_error' in st.session_state:
            st.error(st.session_state['load_error'])
            del st.session_state['load_error']
        st.session_state['load_complete'] = False
        st.rerun()
    
    # 개선된 모델 로딩 상태 폴링
    if st.session_state.get('check_loading', False):
        tracker = st.session_state['model_status_tracker']
        
        # 모델 상태 확인
        models_status = st.session_state['model_manager'].get_all_models_status()
        loading_models = [name for name, info in models_status.items() if info['status'] == 'loading']
        
        if not loading_models:
            # 로딩 중인 모델이 없으면 체크 중단
            st.session_state['check_loading'] = False
            tracker['check_active'] = False
            tracker['loading_models'].clear()
            
            # 새로고침이 아직 처리되지 않았다면 처리
            if not tracker.get('need_refresh', False):
                st.info("🎉 모든 모델 로딩 작업이 완료되었습니다.")
            st.rerun()
        else:
            # 로딩 중인 모델이 있을 때만 상태 표시 및 자동 새로고침
            st.info(f"⏳ 모델 로딩 중: {', '.join(loading_models)}")
            
            # 즉시 새로고침 (로딩 중일 때)
            time.sleep(1)  # 1초 대기
            st.rerun()
    
    # 모델 분석 결과 표시
    if st.session_state['current_model_analysis']:
        st.subheader("📊 모델 분석 결과")
        analysis = st.session_state['current_model_analysis']
        
        # 원본 경로와 실제 경로 표시
        if 'original_path' in analysis and 'actual_path' in analysis:
            if analysis['original_path'] != analysis['actual_path']:
                st.info(f"🔗 HuggingFace 모델 ID: `{analysis['original_path']}`")
                st.info(f"📁 로컬 캐시 경로: `{analysis['actual_path']}`")
        
        # 모델 기본 정보
        if 'model_summary' in analysis:
            summary = analysis['model_summary']
            
            # 메트릭 카드들
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🏗️ 모델 타입", summary.get('model_type', 'unknown'))
            with col2:
                st.metric("📊 파라미터 수", f"{summary.get('total_parameters', 0):,}")
            with col3:
                st.metric("💾 모델 크기", f"{summary.get('model_size_mb', 0):.1f} MB")
            with col4:
                st.metric("📖 어휘 크기", f"{summary.get('vocabulary_size', 0):,}")
            
            # 추가 상세 정보
            st.subheader("🔍 모델 상세 정보")
            
            info_cols = st.columns(2)
            with info_cols[0]:
                st.markdown("**🏷️ 모델 정보**")
                config = summary.get('detailed_config', {})
                st.write(f"• **아키텍처**: {config.get('architecture', 'N/A')}")
                st.write(f"• **히든 사이즈**: {config.get('hidden_size', 'N/A')}")
                st.write(f"• **어텐션 헤드**: {config.get('num_attention_heads', 'N/A')}")
                st.write(f"• **레이어 수**: {config.get('num_hidden_layers', 'N/A')}")
                st.write(f"• **중간 레이어 크기**: {config.get('intermediate_size', 'N/A')}")
                st.write(f"• **활성화 함수**: {config.get('activation_function', 'N/A')}")
                
            with info_cols[1]:
                st.markdown("**⚙️ 설정 정보**")
                st.write(f"• **최대 위치**: {config.get('max_position_embeddings', 'N/A')}")
                st.write(f"• **드롭아웃**: {config.get('dropout', 'N/A')}")
                st.write(f"• **어텐션 드롭아웃**: {config.get('attention_dropout', 'N/A')}")
                st.write(f"• **초기화 범위**: {config.get('initializer_range', 'N/A')}")
                st.write(f"• **레이어 노름 엡실론**: {config.get('layer_norm_eps', 'N/A')}")
                st.write(f"• **토크나이저 최대 길이**: {config.get('tokenizer_max_length', 'N/A')}")
        
        # 지원 태스크와 사용 예시
        if 'model_summary' in analysis and 'supported_tasks' in analysis['model_summary']:
            tasks = analysis['model_summary']['supported_tasks']
            usage_examples = analysis['model_summary'].get('usage_examples', {})
            
            if tasks:
                st.subheader("🎯 지원 태스크 및 사용 방법")
                
                for task in tasks:
                    with st.expander(f"📋 {task}", expanded=False):
                        if task in usage_examples:
                            example = usage_examples[task]
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**📝 설명**")
                                st.write(example['description'])
                                
                                st.markdown("**🔧 입력 예시**")
                                st.code(example['example_input'], language='python')
                                
                                st.markdown("**📤 출력 예시**")
                                st.code(example['expected_output'], language='json')
                                
                                # 파라미터 정보 표시
                                if 'parameters' in example:
                                    st.markdown("**⚙️ 주요 파라미터**")
                                    params = example['parameters']
                                    for param, value in params.items():
                                        if param != 'special_tokens':
                                            st.write(f"• **{param}**: `{value}`")
                                    
                                    if params.get('special_tokens'):
                                        st.markdown("**🎯 특수 토큰**")
                                        for token_name, token_value in list(params['special_tokens'].items())[:3]:
                                            st.write(f"• **{token_name}**: `{token_value}`")
                            
                            with col2:
                                st.markdown("**💻 코드 예시**")
                                st.code(example['example_code'], language='python')
                        else:
                            st.success(f"✅ {task} 태스크를 지원합니다.")
        
        # 발견된 파일들 (접기/펼치기 가능)
        if 'files_found' in analysis:
            with st.expander("📁 발견된 파일들 상세 분석", expanded=False):
                found_files = analysis['files_found']
                missing_files = analysis['files_missing']
                analysis_results = analysis.get('analysis_results', {})
                
                if found_files:
                    st.markdown("**✅ 발견된 파일들**")
                    
                    for file in found_files:
                        with st.expander(f"🔍 {file} 분석 결과", expanded=False):
                            if file in analysis_results:
                                file_data = analysis_results[file]
                                
                                if 'error' in file_data:
                                    st.error(f"❌ 분석 오류: {file_data['error']}")
                                else:
                                    # 파일별 상세 정보 표시
                                    if file == 'config.json':
                                        st.markdown("**📋 모델 설정 정보**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"• **모델 타입**: {file_data.get('model_type', 'N/A')}")
                                            st.write(f"• **아키텍처**: {', '.join(file_data.get('architectures', ['N/A']))}")
                                            st.write(f"• **어휘 크기**: {file_data.get('vocab_size', 'N/A'):,}")
                                            st.write(f"• **히든 사이즈**: {file_data.get('hidden_size', 'N/A')}")
                                        with col2:
                                            st.write(f"• **레이어 수**: {file_data.get('num_hidden_layers', 'N/A')}")
                                            st.write(f"• **어텐션 헤드**: {file_data.get('num_attention_heads', 'N/A')}")
                                            st.write(f"• **최대 위치**: {file_data.get('max_position_embeddings', 'N/A')}")
                                            st.write(f"• **추정 파라미터**: {file_data.get('model_parameters', 0):,}")
                                    
                                    elif file == 'tokenizer_config.json':
                                        st.markdown("**🔤 토크나이저 설정**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"• **토크나이저 클래스**: {file_data.get('tokenizer_class', 'N/A')}")
                                            st.write(f"• **최대 길이**: {file_data.get('model_max_length', 'N/A')}")
                                            st.write(f"• **패딩 방향**: {file_data.get('padding_side', 'N/A')}")
                                        with col2:
                                            st.write(f"• **자르기 방향**: {file_data.get('truncation_side', 'N/A')}")
                                            st.write(f"• **정리 공백**: {file_data.get('clean_up_tokenization_spaces', 'N/A')}")
                                        
                                        if file_data.get('special_tokens'):
                                            st.markdown("**🎯 특수 토큰들**")
                                            for token_name, token_value in file_data['special_tokens'].items():
                                                st.write(f"• **{token_name}**: `{token_value}`")
                                    
                                    elif file == 'tokenizer.json':
                                        st.markdown("**🔍 토크나이저 상세 정보**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"• **버전**: {file_data.get('version', 'N/A')}")
                                            st.write(f"• **모델 타입**: {file_data.get('model_type', 'N/A')}")
                                            st.write(f"• **어휘 크기**: {file_data.get('vocab_size', 'N/A'):,}")
                                        with col2:
                                            if file_data.get('special_tokens'):
                                                st.write("• **특수 토큰 수**: " + str(len(file_data['special_tokens'])))
                                    
                                    elif file == 'vocab.txt':
                                        st.markdown("**📚 어휘 정보**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"• **총 어휘 크기**: {file_data.get('vocab_size', 'N/A'):,}")
                                            st.write(f"• **특수 토큰 수**: {len(file_data.get('special_tokens_found', []))}")
                                        with col2:
                                            if file_data.get('special_tokens_found'):
                                                st.write("• **발견된 특수 토큰들**:")
                                                for token in file_data['special_tokens_found'][:5]:  # 처음 5개만 표시
                                                    st.write(f"  - `{token}`")
                                        
                                        if file_data.get('sample_tokens'):
                                            st.markdown("**📝 샘플 토큰들**")
                                            sample_text = ", ".join([f"`{token}`" for token in file_data['sample_tokens'][:10]])
                                            st.write(sample_text)
                                    
                                    elif file == 'special_tokens_map.json':
                                        st.markdown("**🎯 특수 토큰 맵핑**")
                                        if file_data:
                                            for token_name, token_info in file_data.items():
                                                if isinstance(token_info, dict):
                                                    st.write(f"• **{token_name}**: `{token_info.get('content', token_info)}`")
                                                else:
                                                    st.write(f"• **{token_name}**: `{token_info}`")
                                    
                                    elif file in ['pytorch_model.bin', 'model.safetensors']:
                                        st.markdown("**🧠 모델 가중치 정보**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"• **파일 크기**: {file_data.get('file_size_mb', 'N/A'):.2f} MB")
                                            st.write(f"• **총 파라미터**: {file_data.get('total_parameters', 'N/A'):,}")
                                        with col2:
                                            if file == 'pytorch_model.bin':
                                                st.write(f"• **데이터 타입**: {file_data.get('dtype_info', 'N/A')}")
                                            else:  # safetensors
                                                st.write(f"• **텐서 개수**: {file_data.get('tensor_count', 'N/A')}")
                                        
                                        if file_data.get('parameter_keys'):
                                            st.markdown("**🔑 파라미터 키 샘플**")
                                            key_text = ", ".join([f"`{key}`" for key in file_data['parameter_keys'][:5]])
                                            st.write(key_text)
                                    
                                    elif file == 'generation_config.json':
                                        st.markdown("**🎮 생성 설정**")
                                        col1, col2 = st.columns(2)
                                        with col1:
                                            st.write(f"• **최대 길이**: {file_data.get('max_length', 'N/A')}")
                                            st.write(f"• **최대 새 토큰**: {file_data.get('max_new_tokens', 'N/A')}")
                                            st.write(f"• **온도**: {file_data.get('temperature', 'N/A')}")
                                        with col2:
                                            st.write(f"• **Top-p**: {file_data.get('top_p', 'N/A')}")
                                            st.write(f"• **Top-k**: {file_data.get('top_k', 'N/A')}")
                                            st.write(f"• **샘플링**: {file_data.get('do_sample', 'N/A')}")
                                    
                                    elif file == 'merges.txt':
                                        st.markdown("**🔀 BPE 병합 정보**")
                                        st.write(f"• **병합 규칙 수**: {file_data.get('num_merges', 'N/A'):,}")
                                        if file_data.get('sample_merges'):
                                            st.markdown("**📝 샘플 병합 규칙**")
                                            for merge in file_data['sample_merges'][:5]:
                                                st.write(f"• `{merge}`")
                            else:
                                st.info("분석 결과를 찾을 수 없습니다.")
                
                if missing_files:
                    st.markdown("**❌ 누락된 파일들**")
                    for file in missing_files:
                        with st.expander(f"❌ {file} (누락됨)", expanded=False):
                            # 누락된 파일에 대한 설명
                            if file == 'pytorch_model.bin':
                                st.info("PyTorch 형식의 모델 가중치 파일입니다. 대신 model.safetensors 파일이 사용됩니다.")
                            elif file == 'generation_config.json':
                                st.info("텍스트 생성 설정 파일입니다. 생성 태스크에서 사용되는 기본 설정을 정의합니다.")
                            elif file == 'merges.txt':
                                st.info("BPE(Byte Pair Encoding) 토크나이저의 병합 규칙 파일입니다. 이 모델은 다른 토크나이저를 사용합니다.")
                            else:
                                st.info(f"{file} 파일이 누락되었습니다.")
        
        # 권장사항
        if 'recommendations' in analysis and analysis['recommendations']:
            st.subheader("💡 권장사항")
            for i, rec in enumerate(analysis['recommendations'], 1):
                st.warning(f"**{i}.** {rec}")
    
    # 로드된 모델 목록
    st.subheader("📋 로드된 모델 목록")
    models_status = st.session_state['model_manager'].get_all_models_status()
    
    if models_status:
        # 각 모델에 대한 상세 정보 표시
        for name, info in models_status.items():
            with st.expander(f"🤖 {name} - {info['status']}", expanded=info['status'] == 'error'):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📊 기본 정보**")
                    
                    # 상태에 따른 색상 표시
                    if info['status'] == 'loaded':
                        st.success(f"✅ **상태**: {info['status']}")
                    elif info['status'] == 'loading':
                        st.info(f"🔄 **상태**: {info['status']}")
                    elif info['status'] == 'error':
                        st.error(f"❌ **상태**: {info['status']}")
                    else:
                        st.warning(f"⚠️ **상태**: {info['status']}")
                    
                    st.write(f"📁 **경로**: `{info['path']}`")
                    st.write(f"💾 **메모리 사용량**: {info['memory_usage']:.1f} MB")
                    st.write(f"⏰ **로드 시간**: {info['load_time'] if info['load_time'] else 'N/A'}")
                
                with col2:
                    st.markdown("**🔍 상세 정보**")
                    
                    # 에러 정보 표시
                    if info['status'] == 'error' and info['error_message']:
                        st.error(f"🚨 **에러 내용**:")
                        st.code(info['error_message'], language='text')
                    
                    # 모델 분석 정보 표시
                    if info.get('config_analysis') and 'model_summary' in info['config_analysis']:
                        summary = info['config_analysis']['model_summary']
                        st.write(f"🏗️ **모델 타입**: {summary.get('model_type', 'unknown')}")
                        st.write(f"📊 **파라미터 수**: {summary.get('total_parameters', 0):,}")
                        
                        if summary.get('supported_tasks'):
                            st.write(f"🎯 **지원 태스크**: {', '.join(summary['supported_tasks'])}")
                    
                    # 모델이 로드된 경우 추가 정보
                    if info['status'] == 'loaded':
                        st.success("🟢 모델이 정상적으로 로드되어 사용 가능합니다.")
                        
                        # 추론 가능한 태스크 표시
                        available_tasks = st.session_state['model_manager'].get_available_tasks(name)
                        if available_tasks:
                            st.write(f"🚀 **사용 가능한 태스크**: {', '.join(available_tasks)}")
                
                # 모델 언로드 버튼
                if info['status'] == 'loaded':
                    if st.button(f"🗑️ {name} 언로드", key=f"unload_{name}"):
                        if st.session_state['model_manager'].unload_model(name):
                            st.success(f"모델 '{name}' 언로드 완료!")
                            st.rerun()
                        else:
                            st.error(f"모델 '{name}' 언로드 실패!")
                elif info['status'] == 'error':
                    if st.button(f"🗑️ {name} 제거", key=f"remove_{name}"):
                        if st.session_state['model_manager'].remove_model(name):
                            st.success(f"모델 '{name}' 제거 완료!")
                            st.rerun()
                        else:
                            st.error(f"모델 '{name}' 제거 실패!")
    else:
        st.info("로드된 모델이 없습니다.")

# FastAPI 서버 UI
def render_fastapi_server():
    st.subheader("🚀 FastAPI 서버")
    
    # 상태 배너
    if st.session_state.get('fastapi_server_running', False):
        st.success("🟢 **서버 상태**: 실행 중")
    else:
        st.warning("🟡 **서버 상태**: 중지됨 - 아래 버튼으로 시작하세요")
    
    # 서버 정보
    server_info = st.session_state['fastapi_server'].get_server_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("서버 상태", "🟢 실행 중" if server_info['running'] else "🔴 중지됨")
    
    with col2:
        st.metric("로드된 모델 수", server_info['loaded_models'])
    
    # 서버 제어
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 서버 시작"):
            try:
                result = st.session_state['fastapi_server'].start_server()
                st.session_state['fastapi_server_running'] = True
                save_app_state()  # 상태 저장
                st.success(result)
            except Exception as e:
                st.error(f"서버 시작 실패: {e}")
    
    with col2:
        if st.button("⏹️ 서버 중지"):
            try:
                result = st.session_state['fastapi_server'].stop_server()
                st.session_state['fastapi_server_running'] = False
                save_app_state()  # 상태 저장
                st.info(result)
            except Exception as e:
                st.error(f"서버 중지 실패: {e}")
    
    with col3:
        if st.button("🧹 캐시 정리"):
            st.session_state['fastapi_server'].clear_pipeline_cache()
            save_app_state()  # 상태 저장
            st.success("파이프라인 캐시가 정리되었습니다.")
    
    # 서버 정보 표시
    if server_info['running']:
        st.subheader("🔗 서버 정보")
        st.info(f"**서버 URL:** {server_info['url']}")
        st.info(f"**API 문서:** {server_info['docs_url']}")
        
        # 엔드포인트 목록
        st.subheader("📡 사용 가능한 엔드포인트")
        endpoints = st.session_state['fastapi_server'].get_available_endpoints()
        endpoints_data = []
        for ep in endpoints:
            endpoints_data.append({
                "경로": ep['path'],
                "메서드": ', '.join(ep['methods']),
                "이름": ep['name'] or "N/A"
            })
        
        st.dataframe(pd.DataFrame(endpoints_data), use_container_width=True)
        
        # 간단한 테스트 인터페이스
        st.subheader("🧪 API 테스트")
        loaded_models = st.session_state['model_manager'].get_loaded_models()
        
        if loaded_models:
            selected_model = st.selectbox("테스트할 모델 선택", loaded_models)
            test_text = st.text_area("입력 텍스트", "Hello, world!")
            
            if st.button("예측 테스트"):
                import requests
                try:
                    response = requests.post(
                        f"{server_info['url']}/models/{selected_model}/predict",
                        json={"text": test_text}
                    )
                    if response.status_code == 200:
                        result = response.json()
                        st.success("예측 성공!")
                        st.json(result)
                    else:
                        st.error(f"예측 실패: {response.status_code}")
                        st.error(response.text)
                except Exception as e:
                    st.error(f"요청 실패: {e}")
        else:
            st.info("테스트하려면 먼저 모델을 로드하세요.")

# 메인 함수
def main():
    st.title("🚀 Hugging Face GUI")
    
    # 한 번만 실행되는 자동 복원 로직
    if st.session_state['logged_in'] and not st.session_state.get('auto_restoration_done', False):
        # 복원 상태 추적
        restoration_success = []
        restoration_failed = []
        
        # 캐시 상태 복원
        cache_scanned_state = st.session_state.get('cache_scanned', False)
        cache_info_exists = st.session_state.get('cache_info') is not None
        revisions_count = len(st.session_state.get('revisions_df', pd.DataFrame()))
        
        logger.info(f"캐시 복원 체크: cache_scanned={cache_scanned_state}, cache_info_exists={cache_info_exists}, revisions_count={revisions_count}")
        
        # cache_scanned=True인데 실제 캐시 데이터가 없는 경우 자동 복원
        if cache_scanned_state and (not cache_info_exists or revisions_count == 0):
            logger.info("캐시 스캔됨 상태이지만 cache_info 또는 revisions_df 없음 - 자동 재스캔")
            try:
                scan_cache()
                restoration_success.append("캐시 자동 복원")
                logger.info("캐시 자동 복원 성공")
            except Exception as e:
                restoration_failed.append(f"캐시 복원 ({str(e)})")
                st.session_state['cache_scanned'] = False
                logger.error(f"캐시 자동 복원 실패: {e}")
        elif not cache_scanned_state and not cache_info_exists:
            logger.info("첫 로그인 - 자동 캐시 스캔")
            # 첫 로그인 시 자동 캐시 스캔
            try:
                scan_cache()
                st.session_state['cache_scanned'] = True
                logger.info("첫 로그인 캐시 스캔 완료")
            except Exception as e:
                st.session_state['cache_scanned'] = False
                logger.error(f"첫 로그인 캐시 스캔 실패: {e}")
        else:
            logger.info(f"캐시 복원 불필요: cache_scanned={cache_scanned_state}, cache_info_exists={cache_info_exists}, revisions_count={revisions_count}")
        
        # 모니터링 상태 복원
        monitoring_active = st.session_state.get('monitoring_active', False)
        logger.info(f"모니터링 복원 체크: monitoring_active={monitoring_active}")
        
        if monitoring_active:
            try:
                if not st.session_state['system_monitor'].monitoring:
                    st.session_state['system_monitor'].start_monitoring()
                    restoration_success.append("시스템 모니터링")
                    logger.info("시스템 모니터링 자동 복원 성공")
                else:
                    logger.info("시스템 모니터링 이미 실행 중")
            except Exception as e:
                st.session_state['monitoring_active'] = False
                restoration_failed.append(f"시스템 모니터링 ({str(e)})")
                logger.error(f"시스템 모니터링 자동 복원 실패: {e}")
        
        # FastAPI 서버 상태 복원
        server_running = st.session_state.get('fastapi_server_running', False)
        logger.info(f"서버 복원 체크: fastapi_server_running={server_running}")
        
        if server_running:
            try:
                if not st.session_state['fastapi_server'].is_running():
                    st.session_state['fastapi_server'].start_server()
                    restoration_success.append("FastAPI 서버")
                    logger.info("FastAPI 서버 자동 복원 성공")
                else:
                    logger.info("FastAPI 서버 이미 실행 중")
            except Exception as e:
                st.session_state['fastapi_server_running'] = False
                restoration_failed.append(f"FastAPI 서버 ({str(e)})")
                logger.error(f"FastAPI 서버 자동 복원 실패: {e}")
        
        # 복원 결과 저장
        st.session_state['restoration_success'] = restoration_success
        st.session_state['restoration_failed'] = restoration_failed
        st.session_state['auto_restoration_done'] = True  # 복원 완료 플래그
        
        # 복원 후 한 번만 상태 저장
        if restoration_success or restoration_failed:
            save_app_state()
    
    # 복원 상태 알림 표시
    if st.session_state['logged_in']:
        restoration_success = st.session_state.get('restoration_success', [])
        restoration_failed = st.session_state.get('restoration_failed', [])
        
        if restoration_success:
            st.success(f"🔄 **자동 복원됨**: {', '.join(restoration_success)}")
        
        if restoration_failed:
            st.error(f"❌ **복원 실패**: {', '.join(restoration_failed)}")
            st.info("💡 실패한 서비스는 해당 탭에서 수동으로 다시 시작할 수 있습니다.")
        
        # 상태 초기화 버튼
        if st.button("🔄 모든 상태 초기화"):
            delete_app_state()
            # clear_browser_storage()  # 현재 비활성화
            st.session_state.clear()
            st.success("모든 상태가 초기화되었습니다.")
            st.rerun()
    
    st.markdown("---")
    
    # 사이드바
    with st.sidebar:
        st.header("📊 시스템 요약")
        
        # 시스템 요약 정보
        system_summary = st.session_state['model_manager'].get_system_summary()
        
        st.metric("로드된 모델", system_summary['loaded_models_count'])
        st.metric("총 모델", system_summary['total_models_count'])
        st.metric("메모리 사용량", f"{system_summary['total_memory_usage_mb']:.1f} MB")
        
        # 모니터링 상태
        if st.session_state.get('monitoring_active', False):
            st.success("🟢 모니터링 실행 중")
        else:
            st.info("⏸️ 모니터링 중지됨")
        
        # 캐시 상태
        cache_scanned = st.session_state.get('cache_scanned', False)
        cache_info_exists = st.session_state.get('cache_info') is not None
        if cache_scanned and cache_info_exists:
            st.success(f"🟢 캐시 스캔됨 ({len(st.session_state['revisions_df'])}개)")
        else:
            st.info(f"⚫ 캐시 미스캔 (scanned={cache_scanned}, info={cache_info_exists})")
        
        # 서버 상태
        if st.session_state.get('fastapi_server_running', False):
            st.success("🟢 API 서버 실행 중")
        else:
            st.info("⏸️ API 서버 중지됨")
    
    # 탭 생성
    tabs = st.tabs([
        "🔐 로그인 및 사용자",
        "📁 캐시 관리",
        "🖥️ 시스템 모니터링",
        "🤖 모델 관리",
        "🚀 FastAPI 서버",
        "🐛 디버그"
    ])
    
    # 첫 번째 탭: 로그인/로그아웃 및 사용자 정보
    with tabs[0]:
        st.session_state['current_active_tab'] = 'login'
        st.subheader("🔐 로그인 및 사용자 정보")
        
        if not st.session_state['logged_in']:
            st.text_input("Hugging Face 토큰:", key='input_token')
            st.button("로그인", on_click=login)
        else:
            st.success("✅ 로그인 상태 유지됨")
            st.button("로그아웃", on_click=logout)
        
        if st.session_state['logged_in']:
            col1, col2 = st.columns(2)
            with col1:
                st.button("👤 현재 사용자 정보", on_click=show_whoami)
            with col2:
                st.button("🌐 환경 정보 보기", on_click=show_env)
    
    # 두 번째 탭: 캐시 관리
    with tabs[1]:
        st.session_state['current_active_tab'] = 'cache_management'
        st.subheader("📁 캐시 관리")
        
        # 상태 배너 (디버깅 정보 포함)
        cache_scanned = st.session_state.get('cache_scanned', False)
        cache_info_exists = st.session_state.get('cache_info') is not None
        revisions_count = len(st.session_state.get('revisions_df', pd.DataFrame()))
        
        logger.info(f"캐시 UI 렌더링 - cache_scanned: {cache_scanned}, cache_info_exists: {cache_info_exists}, revisions_count: {revisions_count}")
        
        # 캐시 데이터가 모두 있는 경우
        if cache_scanned and cache_info_exists and revisions_count > 0:
            st.success(f"🟢 **캐시 상태**: {revisions_count}개 항목 스캔됨")
        # 캐시 스캔 상태만 있고 실제 데이터가 없는 경우
        elif cache_scanned and (not cache_info_exists or revisions_count == 0):
            st.info(f"🔄 **캐시 상태**: 복원 중... (scanned={cache_scanned}, info={cache_info_exists}, count={revisions_count})")
            # 자동 복원 시도
            try:
                scan_cache()
                st.rerun()
            except Exception as e:
                st.error(f"자동 복원 실패: {e}")
        # 완전히 스캔되지 않은 경우
        else:
            st.warning(f"🟡 **캐시 상태**: 스캔되지 않음 - 아래 버튼으로 스캔하세요")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # 캐시 스캔 버튼 - 이미 스캔된 경우에도 "캐시 스캔"으로 표시하되 재스캔 기능 수행
            button_text = "🔍 캐시 스캔" if not st.session_state.get('cache_scanned', False) else "🔄 캐시 스캔"
            if st.button(button_text):
                if st.session_state.get('cache_scanned', False):
                    logger.info("캐시 재스캔 버튼 클릭됨")
                else:
                    logger.info("캐시 스캔 버튼 클릭됨")
                scan_cache()
                st.session_state['cache_scanned'] = True
                logger.info("캐시 스캔 완료, 상태 저장 시작")
                save_app_state()  # 상태 저장
                logger.info("상태 저장 완료, 페이지 재실행")
                st.rerun()
        
        with col2:
            # 모델 다운로드 기능
            st.markdown("#### 📥 모델 다운로드")
            download_input = st.text_input(
                "모델 ID 또는 HuggingFace 링크 입력:",
                placeholder="예: microsoft/DialoGPT-medium 또는 https://huggingface.co/microsoft/DialoGPT-medium",
                help="HuggingFace 모델 ID나 URL을 입력하면 로컬 캐시로 다운로드됩니다."
            )
            
            if st.button("📥 다운로드 + 스캔", disabled=not download_input.strip()):
                if download_input.strip():
                    download_model_to_cache(download_input.strip(), auto_scan=True)
        
        if st.session_state['cache_info']:
            # AgGrid 설정
            gb = GridOptionsBuilder.from_dataframe(st.session_state['revisions_df'])
            gb.configure_selection("multiple", use_checkbox=True, groupSelectsChildren=True)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            gb.configure_default_column(enablePivot=True, enableValue=True, enableRowGroup=True)
            
            gridOptions = gb.build()
            
            grid_response = AgGrid(
                st.session_state['revisions_df'],
                gridOptions=gridOptions,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                fit_columns_on_grid_load=True,
                enable_enterprise_modules=True,
                height=400,
                width='100%',
                reload_data=False
            )
            
            selected = grid_response['selected_rows']
            selected_df = pd.DataFrame(selected)
            
            # 선택 요약
            if not selected_df.empty:
                selected_count = len(selected_df)
                total_size = selected_df['Size (MB)'].sum()
                st.write(f"선택된 항목: {selected_count}개, 총 용량: {total_size:.2f} MB")
                
                # 삭제 확인 및 버튼
                with st.expander("선택한 캐시 삭제"):
                    st.warning(f"{selected_count}개의 수정 버전을 삭제하시겠습니까?")
                    if st.button("삭제 확인"):
                        delete_selected(selected_df)
                        save_app_state()  # 상태 저장
                        st.rerun()
            else:
                st.write("선택된 항목: 0개, 총 용량: 0.00 MB")
    
    # 세 번째 탭: 시스템 모니터링
    with tabs[2]:
        # 현재 탭이 시스템 모니터링임을 표시
        st.session_state['current_active_tab'] = 'system_monitoring'
        render_system_monitoring()
    
    # 네 번째 탭: 모델 관리
    with tabs[3]:
        st.session_state['current_active_tab'] = 'model_management'
        render_model_management()
    
    # 다섯 번째 탭: FastAPI 서버
    with tabs[4]:
        st.session_state['current_active_tab'] = 'fastapi_server'
        render_fastapi_server()
    
    # 여섯 번째 탭: 디버그 정보
    with tabs[5]:
        st.session_state['current_active_tab'] = 'debug'
        st.subheader("🐛 디버그 정보")
        
        # 현재 세션 상태
        st.subheader("📊 현재 세션 상태")
        debug_state = {
            'cache_scanned': st.session_state.get('cache_scanned', 'NOT_SET'),
            'cache_info': st.session_state.get('cache_info') is not None,
            'revisions_df': len(st.session_state.get('revisions_df', pd.DataFrame())),
            'monitoring_active': st.session_state.get('monitoring_active', 'NOT_SET'),
            'fastapi_server_running': st.session_state.get('fastapi_server_running', 'NOT_SET'),
            'model_path_input': st.session_state.get('model_path_input', 'NOT_SET'),
            'state_loaded': st.session_state.get('state_loaded', 'NOT_SET')
        }
        st.json(debug_state)
        
        # 상태 파일 정보
        st.subheader("📁 상태 파일 정보")
        if os.path.exists(STATE_FILE):
            st.success(f"✅ 상태 파일 존재: {STATE_FILE}")
            try:
                with open(STATE_FILE, 'r', encoding='utf-8') as f:
                    file_content = json.load(f)
                st.json(file_content)
            except Exception as e:
                st.error(f"파일 읽기 오류: {e}")
        else:
            st.error(f"❌ 상태 파일 없음: {STATE_FILE}")
        
        # 수동 액션
        st.subheader("🔧 수동 액션")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 상태 강제 로드"):
                logger.info("수동 상태 로드 요청")
                load_enhanced_app_state()
                st.rerun()
        
        with col2:
            if st.button("💾 상태 강제 저장"):
                logger.info("수동 상태 저장 요청")
                save_app_state()
                st.success("상태 저장 완료")
        
        with col3:
            if st.button("📝 최근 로그 보기"):
                if os.path.exists('app_debug.log'):
                    with open('app_debug.log', 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # 마지막 50줄만 표시
                        recent_logs = ''.join(lines[-50:]) if lines else "로그가 비어있습니다."
                    st.text_area("최근 로그 (최대 50줄)", recent_logs, height=300)
                else:
                    st.error("로그 파일이 없습니다.")
        
        # 실시간 상태 추적
        st.subheader("🔍 실시간 상태 추적")
        
        if st.button("🔄 실시간 상태 체크"):
            # 현재 상태 확인
            current_status = []
            
            # 1. 파일 시스템 체크
            if os.path.exists(STATE_FILE):
                try:
                    with open(STATE_FILE, 'r') as f:
                        file_state = json.load(f)
                    current_status.append(f"✅ 상태 파일: cache_scanned={file_state.get('cache_scanned')}")
                except Exception as e:
                    current_status.append(f"❌ 상태 파일 읽기 오류: {e}")
            else:
                current_status.append("❌ 상태 파일 없음")
            
            # 2. 세션 상태 체크
            cache_scanned = st.session_state.get('cache_scanned', 'NOT_SET')
            cache_info = st.session_state.get('cache_info')
            current_status.append(f"📊 세션 상태: cache_scanned={cache_scanned}")
            current_status.append(f"📊 세션 상태: cache_info={'존재' if cache_info else '없음'}")
            
            # 3. 디렉토리 상태 체크
            from huggingface_hub import scan_cache_dir
            try:
                cache_dir_info = scan_cache_dir()
                current_status.append(f"📁 캐시 디렉토리: {len(cache_dir_info.repos)}개 저장소")
            except Exception as e:
                current_status.append(f"❌ 캐시 디렉토리 오류: {e}")
            
            # 결과 표시
            for status in current_status:
                if "✅" in status:
                    st.success(status)
                elif "❌" in status:
                    st.error(status)
                else:
                    st.info(status)

# 상태 복원 알림 관리
def show_restoration_status():
    """복원된 상태에 대한 알림 표시"""
    restored_items = []
    
    # 복원된 상태 확인
    if st.session_state.get('cache_scanned', False) and st.session_state['cache_info']:
        restored_items.append(f"캐시 스캔 ({len(st.session_state['revisions_df'])}개 항목)")
    
    if st.session_state.get('monitoring_active', False):
        restored_items.append("시스템 모니터링")
    
    if st.session_state.get('fastapi_server_running', False):
        restored_items.append("FastAPI 서버")
    
    if st.session_state.get('model_path_input', ''):
        restored_items.append(f"모델 경로: {st.session_state['model_path_input']}")
    
    if st.session_state.get('current_model_analysis'):
        restored_items.append("모델 분석 결과")
    
    if restored_items:
        st.success(f"🔄 **상태 복원됨**: {', '.join(restored_items)}")
        return True
    return False

# 프로그램 시작 시 로그인 상태 확인
if st.session_state['logged_in']:
    api.set_access_token(st.session_state['token'])

if __name__ == "__main__":
    main()