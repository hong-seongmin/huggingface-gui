import os
import streamlit as st
from huggingface_hub import HfApi, scan_cache_dir
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import json
from datetime import datetime

# 새로운 모듈들 import
from model_manager import MultiModelManager
from system_monitor import SystemMonitor
from fastapi_server import FastAPIServer
from model_analyzer import ComprehensiveModelAnalyzer

# 로그인 상태를 저장할 파일 경로 설정
LOGIN_FILE = "login_token.txt"

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

# 로그인 기능
def login():
    token = st.session_state['input_token'].strip()
    if token:
        try:
            api.set_access_token(token)
            save_login_token(token)
            st.session_state['token'] = token
            st.session_state['logged_in'] = True
            st.success("로그인 성공!")
        except Exception as e:
            st.error(f"로그인에 실패했습니다: {e}")
    else:
        st.error("유효한 토큰을 입력하세요.")

# 로그아웃 기능
def logout():
    api.set_access_token(None)
    delete_login_token()
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
    cache_info = scan_cache_dir()
    st.session_state['cache_info'] = cache_info

    # 캐시 데이터 수집
    revisions = []
    for repo in cache_info.repos:
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
    
    # 자동 갱신 설정 초기화
    if 'auto_refresh_interval' not in st.session_state:
        st.session_state['auto_refresh_interval'] = 0
    if 'last_refresh_time' not in st.session_state:
        st.session_state['last_refresh_time'] = 0
    
    # 모니터링 및 자동 갱신 설정
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🚀 모니터링 시작"):
            st.session_state['system_monitor'].start_monitoring()
            st.success("모니터링이 시작되었습니다.")
    
    with col2:
        if st.button("⏹️ 모니터링 중지"):
            st.session_state['system_monitor'].stop_monitoring()
            st.session_state['auto_refresh_interval'] = 0
            st.info("모니터링이 중지되었습니다.")
    
    with col3:
        if st.button("🔄 새로고침"):
            st.rerun()
    
    with col4:
        # 자동 갱신 간격 설정
        refresh_options = {
            "자동 갱신 끄기": 0,
            "1초마다": 1,
            "3초마다": 3,
            "10초마다": 10
        }
        selected_refresh = st.selectbox(
            "자동 갱신",
            options=list(refresh_options.keys()),
            index=0
        )
        st.session_state['auto_refresh_interval'] = refresh_options[selected_refresh]
    
    # 자동 갱신 로직
    if st.session_state['auto_refresh_interval'] > 0:
        import time
        current_time = time.time()
        
        # 자동 갱신 상태 표시
        st.info(f"🔄 {st.session_state['auto_refresh_interval']}초마다 자동 갱신 중...")
        
        # 지정된 간격이 지났으면 갱신
        if current_time - st.session_state['last_refresh_time'] >= st.session_state['auto_refresh_interval']:
            st.session_state['last_refresh_time'] = current_time
            st.rerun()
        
        # 다음 갱신까지 남은 시간 계산 및 표시
        remaining_time = st.session_state['auto_refresh_interval'] - (current_time - st.session_state['last_refresh_time'])
        if remaining_time > 0:
            st.write(f"⏰ 다음 갱신까지: {remaining_time:.1f}초")
            # 페이지 자동 갱신을 위한 JavaScript 추가
            st.markdown(f"""
                <script>
                setTimeout(function(){{
                    window.location.reload();
                }}, {remaining_time * 1000});
                </script>
            """, unsafe_allow_html=True)
    
    # 현재 상태 표시
    if st.session_state['system_monitor'].monitoring or st.button("현재 상태 보기"):
        current_data = st.session_state['system_monitor'].get_current_data()
        
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
        
        # 차트 표시
        if st.session_state['system_monitor'].monitoring:
            render_system_charts()
        
        # 알림 표시
        alerts = st.session_state['system_monitor'].get_alerts()
        if alerts:
            st.subheader("⚠️ 시스템 알림")
            for alert in alerts:
                if alert['type'] == 'error':
                    st.error(f"🔥 {alert['message']}")
                elif alert['type'] == 'warning':
                    st.warning(f"⚠️ {alert['message']}")

def render_system_charts():
    """시스템 모니터링 차트 렌더링"""
    history = st.session_state['system_monitor'].get_history()
    
    if not history['cpu']:
        st.info("데이터 수집 중...")
        return
    
    # 차트 생성
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('CPU 사용률', '메모리 사용률', 'GPU 사용률', '디스크 사용률'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CPU 차트
    cpu_df = pd.DataFrame(history['cpu'])
    fig.add_trace(
        go.Scatter(
            x=cpu_df['timestamp'], 
            y=cpu_df['percent'], 
            name='CPU %',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    # 메모리 차트
    memory_df = pd.DataFrame(history['memory'])
    fig.add_trace(
        go.Scatter(
            x=memory_df['timestamp'], 
            y=memory_df['percent'], 
            name='Memory %',
            line=dict(color='red')
        ),
        row=1, col=2
    )
    
    # GPU 차트
    if history['gpu'] and history['gpu'][0]:
        gpu_data = []
        for gpu_snapshot in history['gpu']:
            for gpu in gpu_snapshot:
                gpu_data.append({
                    'timestamp': gpu['timestamp'],
                    'gpu_id': gpu['gpu_id'],
                    'load': gpu['load']
                })
        
        if gpu_data:
            gpu_df = pd.DataFrame(gpu_data)
            colors = ['green', 'orange', 'purple', 'brown']
            for i, gpu_id in enumerate(gpu_df['gpu_id'].unique()):
                gpu_subset = gpu_df[gpu_df['gpu_id'] == gpu_id]
                fig.add_trace(
                    go.Scatter(
                        x=gpu_subset['timestamp'], 
                        y=gpu_subset['load'], 
                        name=f'GPU {gpu_id}',
                        line=dict(color=colors[i % len(colors)])
                    ),
                    row=2, col=1
                )
    
    # 디스크 차트
    disk_df = pd.DataFrame(history['disk'])
    fig.add_trace(
        go.Scatter(
            x=disk_df['timestamp'], 
            y=disk_df['percent'], 
            name='Disk %',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# 모델 관리 UI
def render_model_management():
    st.subheader("🤖 모델 관리")
    
    # 모델 로드 섹션
    st.subheader("📥 모델 로드")
    
    # 캐시된 모델에서 선택 옵션 추가
    if st.session_state['cache_info']:
        st.subheader("🗂️ 캐시된 모델에서 선택")
        cached_models = []
        for repo in st.session_state['cache_info'].repos:
            cached_models.append(repo.repo_id)
        
        if cached_models:
            selected_cached_model = st.selectbox(
                "캐시된 모델 선택 (선택사항)", 
                options=["직접 입력"] + cached_models,
                index=0
            )
            
            if selected_cached_model != "직접 입력":
                # 캐시된 모델 선택 시 자동으로 경로 설정
                st.session_state['model_path_input'] = selected_cached_model
                st.success(f"선택된 모델: {selected_cached_model}")
    
    model_path = st.text_input("모델 경로 (로컬 경로 또는 HuggingFace 모델 ID)", key="model_path_input", placeholder="예: tabularisai/multilingual-sentiment-analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔍 모델 분석"):
            if model_path:
                with st.spinner("모델 분석 중..."):
                    analysis = st.session_state['model_manager'].analyze_model(model_path)
                    if 'error' in analysis:
                        st.error(f"모델 분석 실패: {analysis['error']}")
                    else:
                        st.session_state['current_model_analysis'] = analysis
                        st.success("모델 분석 완료!")
            else:
                st.error("모델 경로를 입력하세요.")
    
    with col2:
        if st.button("📤 모델 로드"):
            if model_path:
                # 모델 이름 자동 생성
                model_name = ""
                
                def load_callback(name, success, message):
                    # 콜백에서 세션 상태 업데이트
                    if success:
                        st.session_state['load_success'] = f"모델 '{name}' 로드 성공!"
                        st.session_state['load_complete'] = True
                    else:
                        st.session_state['load_error'] = f"모델 '{name}' 로드 실패: {message}"
                        st.session_state['load_complete'] = True
                
                st.session_state['model_manager'].load_model_async(
                    model_name, model_path, load_callback
                )
                st.info(f"모델 로드 시작... (이름: 자동 생성)")
                
                # 자동 새로고침 체크 시작
                st.session_state['check_loading'] = True
                
                # 캐시 자동 스캔 (HuggingFace 모델 ID인 경우)
                if st.session_state['model_manager']._is_huggingface_model_id(model_path):
                    st.info("🔄 HuggingFace 모델 감지 - 캐시 자동 갱신 중...")
                    scan_cache()
            else:
                st.error("모델 경로를 입력하세요.")
    
    with col3:
        if st.button("🔄 상태 새로고침"):
            st.rerun()
    
    # 로드 완료 메시지 표시 및 자동 새로고침
    if st.session_state.get('load_complete', False):
        if 'load_success' in st.session_state:
            st.success(st.session_state['load_success'])
            del st.session_state['load_success']
        if 'load_error' in st.session_state:
            st.error(st.session_state['load_error'])
            del st.session_state['load_error']
        st.session_state['load_complete'] = False
        st.rerun()
    
    # 페이지 로드 시 자동 새로고침 체크 (로딩 상태 폴링)
    if st.session_state.get('check_loading', False):
        # 짧은 간격으로 모델 상태 확인
        import time
        time.sleep(1)
        
        # 모델 상태 확인
        models_status = st.session_state['model_manager'].get_all_models_status()
        loading_models = [name for name, info in models_status.items() if info['status'] == 'loading']
        
        if not loading_models:
            # 로딩 중인 모델이 없으면 체크 중단
            st.session_state['check_loading'] = False
        
        # 자동 새로고침
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
                st.success(result)
            except Exception as e:
                st.error(f"서버 시작 실패: {e}")
    
    with col2:
        if st.button("⏹️ 서버 중지"):
            try:
                result = st.session_state['fastapi_server'].stop_server()
                st.info(result)
            except Exception as e:
                st.error(f"서버 중지 실패: {e}")
    
    with col3:
        if st.button("🧹 캐시 정리"):
            st.session_state['fastapi_server'].clear_pipeline_cache()
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
        if st.session_state['system_monitor'].monitoring:
            st.success("🟢 모니터링 실행 중")
        else:
            st.info("⏸️ 모니터링 중지됨")
        
        # 서버 상태
        if st.session_state['fastapi_server'].is_running():
            st.success("🟢 API 서버 실행 중")
        else:
            st.info("⏸️ API 서버 중지됨")
    
    # 탭 생성
    tabs = st.tabs([
        "🔐 로그인 및 사용자",
        "📁 캐시 관리",
        "🖥️ 시스템 모니터링",
        "🤖 모델 관리",
        "🚀 FastAPI 서버"
    ])
    
    # 첫 번째 탭: 로그인/로그아웃 및 사용자 정보
    with tabs[0]:
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
        st.subheader("📁 캐시 관리")
        if st.button("🔍 캐시 스캔"):
            scan_cache()
        
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
                        st.rerun()
            else:
                st.write("선택된 항목: 0개, 총 용량: 0.00 MB")
    
    # 세 번째 탭: 시스템 모니터링
    with tabs[2]:
        render_system_monitoring()
    
    # 네 번째 탭: 모델 관리
    with tabs[3]:
        render_model_management()
    
    # 다섯 번째 탭: FastAPI 서버
    with tabs[4]:
        render_fastapi_server()

# 프로그램 시작 시 로그인 상태 확인 및 캐시 스캔
if st.session_state['logged_in']:
    api.set_access_token(st.session_state['token'])
    if st.session_state['cache_info'] is None:
        scan_cache()

if __name__ == "__main__":
    main()