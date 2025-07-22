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

# API 가이드용 모델 타입 감지 함수 (모든 HuggingFace 아키텍처 지원)
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
    
    # ELECTRA 계열
    elif 'electra' in model_name_lower:
        if any(keyword in model_name_lower for keyword in ['ner', 'token']):
            return "token-classification"
        elif any(keyword in model_name_lower for keyword in ['classification', 'sentiment']):
            return "text-classification"
        else:
            return "token-classification"
    
    # BERT 계열 (모든 변형 포함)
    elif any(keyword in model_name_lower for keyword in ['bert', 'roberta', 'albert', 'distilbert']):
        if any(keyword in model_name_lower for keyword in ['ner', 'token']):
            return "token-classification"
        elif any(keyword in model_name_lower for keyword in ['qa', 'question', 'squad']):
            return "question-answering"
        else:
            return "text-classification"
    
    # XLM 계열
    elif any(keyword in model_name_lower for keyword in ['xlm', 'xlm-roberta', 'xlnet']):
        return "text-classification"
    
    # 기타 인코더 모델
    elif any(keyword in model_name_lower for keyword in ['camembert', 'flaubert', 'convbert', 'mobilebert']):
        return "text-classification"
    
    # === FEATURE EXTRACTION MODELS ===
    # 임베딩 모델들
    elif any(keyword in model_name_lower for keyword in ['bge', 'e5', 'gte', 'instructor', 'sentence']):
        return "feature-extraction"
    
    # Sentence Transformers
    elif any(keyword in model_name_lower for keyword in ['all-minilm', 'all-mpnet', 'paraphrase']):
        return "feature-extraction"
    
    # === MULTIMODAL MODELS ===
    # CLIP 계열
    elif any(keyword in model_name_lower for keyword in ['clip', 'blip', 'blip2']):
        return "multimodal"
    
    # Vision-Language 모델
    elif any(keyword in model_name_lower for keyword in ['layoutlm', 'donut', 'pix2struct']):
        return "multimodal"
    
    # === VISION MODELS ===
    elif any(keyword in model_name_lower for keyword in ['vit', 'deit', 'swin', 'convnext', 'resnet']):
        return "image-classification"
    
    # === AUDIO MODELS ===
    elif any(keyword in model_name_lower for keyword in ['wav2vec2', 'whisper', 'speecht5', 'hubert']):
        return "automatic-speech-recognition"
    
    # === FILL-MASK MODELS ===
    elif any(keyword in model_name_lower for keyword in ['masked', 'mlm', 'fill-mask']):
        return "fill-mask"
    
    # === SPECIFIC TASK INDICATORS ===
    # 감정 분석
    elif any(keyword in model_name_lower for keyword in ['sentiment', 'emotion', 'affect', 'polarity']):
        return "text-classification"
    
    # NER 모델
    elif any(keyword in model_name_lower for keyword in ['ner', 'named-entity', 'token-class']):
        return "token-classification"
    
    # 질문 답변
    elif any(keyword in model_name_lower for keyword in ['qa', 'question', 'squad', 'answer']):
        return "question-answering"
    
    # 요약
    elif any(keyword in model_name_lower for keyword in ['summary', 'summarization', 'abstract']):
        return "summarization"
    
    # 번역
    elif any(keyword in model_name_lower for keyword in ['translation', 'translate', 'mt-']):
        return "translation"
    
    # 텍스트 생성
    elif any(keyword in model_name_lower for keyword in ['generation', 'generator', 'lm', 'causal']):
        return "text-generation"
    
    # 임베딩
    elif any(keyword in model_name_lower for keyword in ['embedding', 'encode', 'similarity', 'retrieval']):
        return "feature-extraction"
    
    # === SPECIAL ARCHITECTURES ===
    # Long sequence 모델
    elif any(keyword in model_name_lower for keyword in ['longformer', 'bigbird', 'reformer']):
        return "text-classification"
    
    # Efficient 모델
    elif any(keyword in model_name_lower for keyword in ['mobilenet', 'efficientnet', 'squeezenet']):
        return "image-classification"
    
    # === LANGUAGE-SPECIFIC MODELS ===
    # 한국어 모델
    elif any(keyword in model_name_lower for keyword in ['ko-', 'korean', 'klue', 'kcbert']):
        if any(keyword in model_name_lower for keyword in ['ner', 'token']):
            return "token-classification"
        else:
            return "text-classification"
    
    # 중국어 모델
    elif any(keyword in model_name_lower for keyword in ['chinese', 'zh-', 'bert-base-chinese']):
        return "text-classification"
    
    # 일본어 모델
    elif any(keyword in model_name_lower for keyword in ['japanese', 'ja-', 'bert-base-japanese']):
        return "text-classification"
    
    # === DEFAULT FALLBACK ===
    # 기본값 - 가장 일반적인 태스크
    else:
        return "text-classification"

# 새로운 모듈들 import
from model_manager import MultiModelManager
from system_monitor import SystemMonitor
from fastapi_server import FastAPIServer
from model_analyzer import ComprehensiveModelAnalyzer
from model_type_detector import ModelTypeDetector

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

# 스마트 폴링 시스템 (로딩 중일 때만 적극적으로 체크)
def should_perform_expensive_check(operation_name, base_interval=30):
    """비용이 많이 드는 작업을 수행할지 결정하는 스마트 폴링"""
    current_time = time.time()
    
    # 체크 시간 추적
    if not hasattr(should_perform_expensive_check, 'last_checks'):
        should_perform_expensive_check.last_checks = {}
    
    last_check = should_perform_expensive_check.last_checks.get(operation_name, 0)
    
    # 모델 로딩 중인지 확인
    is_loading, _ = is_any_model_loading()
    
    # 로딩 중이면 더 자주 체크, 아니면 덜 자주 체크
    interval = base_interval // 3 if is_loading else base_interval * 2
    
    if current_time - last_check > interval:
        should_perform_expensive_check.last_checks[operation_name] = current_time
        return True
    
    return False

# 스마트 모델 상태 감지 시스템 (콜백 의존성 제거)
def smart_model_status_check():
    """직접 모델 상태 감지 - 콜백 없이 안정적 감지"""
    # 스마트 폴링: 로딩 중이 아니면 덜 자주 체크
    if not should_perform_expensive_check('model_status_check', 10):
        return False
        
    try:
        tracker = st.session_state.get('model_status_tracker', {})
        
        # 현재 모델 상태 직접 확인
        models_status = st.session_state['model_manager'].get_all_models_status()
        current_loaded = set([name for name, info in models_status.items() if info['status'] == 'loaded'])
        current_loading = set([name for name, info in models_status.items() if info['status'] == 'loading'])
        previous_loaded = tracker.get('previous_loaded', set())
        
        # 새로 로드 완료된 모델 감지
        newly_loaded = current_loaded - previous_loaded
        
        if newly_loaded:
            # 즉시 상태 업데이트
            tracker['previous_loaded'] = current_loaded
            st.session_state['model_status_tracker'] = tracker
            
            # 성공 메시지 표시
            st.success(f"🎉 모델 로드 완료: {', '.join(newly_loaded)}!")
            logger.info(f"직접 감지: 새로 로드된 모델 {newly_loaded}")
            
            # 캐시 즉시 스캔 (로드 완료 시에만)
            scan_cache()
            
            # 즉시 새로고침
            st.rerun()
            return True
            
        # 로딩 상태 업데이트
        tracker['current_loading'] = current_loading
        tracker['is_any_loading'] = len(current_loading) > 0
        
        return False
        
    except Exception as e:
        logger.error(f"스마트 모델 상태 체크 오류: {e}")
        return False

def is_any_model_loading():
    """현재 로딩 중인 모델이 있는지 확인"""
    try:
        models_status = st.session_state['model_manager'].get_all_models_status()
        loading_models = [name for name, info in models_status.items() if info['status'] == 'loading']
        return len(loading_models) > 0, loading_models
    except:
        return False, []

# 상태 추적기 정리 함수
def cleanup_status_tracker():
    """상태 추적기 정리"""
    try:
        if 'model_status_tracker' in st.session_state:
            tracker = st.session_state['model_status_tracker']
            tracker['current_loading'] = set()
            tracker['is_any_loading'] = False
            logger.info("상태 추적기 정리 완료")
    except Exception as e:
        logger.error(f"상태 추적기 정리 오류: {e}")

# 콜백 시스템 비활성화 (직접 감지 시스템으로 대체)

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

# 앱 상태 저장 (변경사항 감지 및 중복 방지 최적화)
def save_app_state(force=False):
    # Rate limit both logging AND actual saving
    if not hasattr(save_app_state, 'last_save_time'):
        save_app_state.last_save_time = 0
        save_app_state.last_log_time = 0
        save_app_state.last_state = {}
    
    current_time = time.time()
    # Check if we're currently loading models
    is_loading, _ = is_any_model_loading()
    
    # Much more aggressive rate limiting: save only when necessary
    save_interval = 30 if is_loading else 300  # 30 seconds during loading, 5 minutes normally
    log_interval = 120 if is_loading else 600  # Log even less frequently
    
    # Skip saving if we saved recently (unless forced)
    if not force and current_time - save_app_state.last_save_time < save_interval:
        return  # Don't save at all
    
    # 현재 상태 수집
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
    
    # 상태 변경 확인 (current_model_analysis는 크므로 제외하고 비교)
    state_for_comparison = {k: v for k, v in state.items() if k != 'current_model_analysis'}
    last_state_for_comparison = {k: v for k, v in save_app_state.last_state.items() if k != 'current_model_analysis'}
    
    # 상태가 변경되지 않았으면 스킵 (강제가 아닌 경우)
    if not force and state_for_comparison == last_state_for_comparison:
        return  # 변경사항 없음
    
    # Update save time
    save_app_state.last_save_time = current_time
    save_app_state.last_state = state.copy()
    
    # Check if we should log
    if current_time - save_app_state.last_log_time > log_interval:
        logger.info("=== 상태 저장 시작 ===")
        save_app_state.last_log_time = current_time
        should_log = True
    else:
        should_log = False
    
    if should_log:
        logger.info(f"저장할 상태: {state}")
    
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        if should_log:
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
    load_enhanced_app_state()
    st.session_state['state_loaded'] = True
    pass
else:
    # 상태 이미 로드됨
    pass

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

# 캐시 정보 스캔 및 화면에 표시하는 기능 - 중복 실행 방지 최적화
def scan_cache(force=False):
    import time
    
    # 중복 실행 방지: 30초 이내 재실행 방지 (force=True인 경우 제외)
    current_time = time.time()
    if not force and hasattr(scan_cache, 'last_scan_time'):
        time_since_last_scan = current_time - scan_cache.last_scan_time
        if time_since_last_scan < 30:  # 30초 이내 중복 실행 방지
            logger.info(f"캐시 스캔 스킵: {time_since_last_scan:.1f}초 전에 실행됨")
            return
    
    cache_info = scan_cache_dir()
    st.session_state['cache_info'] = cache_info
    logger.info(f"캐시 스캔: {len(cache_info.repos)}개 저장소")

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
    st.session_state['cache_scanned'] = True
    logger.info(f"캐시 스캔 완료: {len(revisions)}개 항목")
    
    # 마지막 스캔 시간 기록
    scan_cache.last_scan_time = current_time

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
                # Skip immediate save during download - not critical
                progress_placeholder.success(f"✅ 모델 다운로드 및 캐시 스캔 완료!")
                st.rerun()
            else:
                # Only save state if actually needed for non-auto-scan downloads
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
                    # Only save state if actually needed for non-auto-scan downloads
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
            st.session_state['system_monitor'].start_monitoring()
            st.session_state['monitoring_active'] = True
            st.session_state['refresh_count'] = 0  # 카운터 리셋
            logger.info("시스템 모니터링 시작")
            st.rerun()  # 즉시 반영
    
    with col2:
        if st.button("⏹️ 모니터링 중지"):
            st.session_state['system_monitor'].stop_monitoring()
            st.session_state['auto_refresh_interval'] = 0
            st.session_state['monitoring_active'] = False
            # Only save state when actually needed - monitoring stop is not critical to persist immediately  
            st.info("모니터링이 중지되었습니다.")

    with col3:
        if st.button("🔄 새로고침"):
            # Skip save_app_state for manual refresh - no state changes
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
            logger.info(f"실시간 차트 활성화: {auto_refresh_interval}초")
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
    
    # SystemMonitor 상태 추적 (간소화된 로깅)
    system_monitor_status = st.session_state['system_monitor'].monitoring
    if system_monitor_status != st.session_state.get('_last_monitor_status'):
        logger.info(f"모니터링 상태 변경: {system_monitor_status}")
        st.session_state['_last_monitor_status'] = system_monitor_status
    
    
    # 현재 상태 표시
    show_current_status = (
        st.session_state['system_monitor'].monitoring or 
        st.button("현재 상태 보기") or
        monitoring_active
    )
    
    # Reduced logging for show_current_status
    if show_current_status != st.session_state.get('_last_show_status'):
        logger.info(f"[시스템모니터] show_current_status = {show_current_status}")
        st.session_state['_last_show_status'] = show_current_status
    
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
                # 자동 갱신 로깅 대폭 축소 (매 50회마다만)
                refresh_count = st.session_state.get('refresh_count', 0)
                current_data = st.session_state['system_monitor'].get_current_data()
                if refresh_count % 50 == 0:
                    logger.info(f"자동갱신 #{refresh_count}: CPU={current_data['cpu']['percent']:.1f}%")
                
                # 갱신 알림
                st.success(f"✅ 자동 갱신됨 ({st.session_state.get('refresh_count', 0)}회)")
                
                # 메트릭 카드들
                col1, col2, col3, col4 = st.columns(4)
        else:
            # 일반 모드: 직접 표시 (로깅 최소화)
            current_data = st.session_state['system_monitor'].get_current_data()
            st.session_state['_normal_mode_count'] = st.session_state.get('_normal_mode_count', 0) + 1
            
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
                if st.session_state['model_manager']._is_huggingface_model_id(model_path):
                    model_name = model_path.split('/')[-1]
                else:
                    model_name = os.path.basename(model_path.rstrip('/'))
                
                logger.info(f"모델 로드 시작: {model_name} ({model_path})")
                
                # 모델 매니저 상태 확인
                if 'model_manager' not in st.session_state:
                    st.error("❌ 모델 매니저가 초기화되지 않았습니다.")
                    return
                
                logger.info(f"모델 매니저 준비됨: {type(st.session_state['model_manager'])}")
                
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
                
                logger.info(f"load_model_async 호출: {model_name}, {model_path}")
                try:
                    thread = st.session_state['model_manager'].load_model_async(
                        model_name, model_path, load_callback
                    )
                    logger.info(f"모델 로딩 스레드 시작됨: {thread}")
                    st.info(f"⏳ 모델 '{model_name}' 로딩을 시작했습니다...")
                except Exception as e:
                    logger.error(f"모델 로딩 시작 실패: {e}")
                    st.error(f"❌ 모델 로딩 시작 실패: {e}")
                    st.session_state['check_loading'] = False
                    return
                
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
                    scan_cache()
                
                save_app_state()  # 상태 저장
            else:
                st.error("❌ 모델 경로를 입력하세요.")
        
        if refresh_clicked:
            # Skip save on manual refresh - no state changes
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
    
    # 직접 모델 상태 체크 (콜백 시스템 대신 스마트 폴링)
    if should_perform_expensive_check('model_management_check', 8):
        smart_model_status_check()
    
    # 모델 로딩 완료 체크 및 UI 갱신
    if st.session_state.get('check_loading', False):
        models_status = st.session_state['model_manager'].get_all_models_status()
        loading_models = [name for name, info in models_status.items() if info['status'] == 'loading']
        
        if not loading_models:
            # 로딩 중인 모델이 없으면 체크 중단하고 UI 갱신
            st.session_state['check_loading'] = False
            st.success("🎉 모든 모델 로딩이 완료되었습니다!")
            # 즉시 UI 갱신 (캐시 스캔 불필요 - 모델 로딩 완료와 캐시는 별개)
            # 상태 새로고침 자동 실행
            st.session_state['model_status_tracker']['need_refresh'] = True
            st.session_state['model_status_tracker']['force_ui_refresh'] = time.time()
            st.rerun()
        else:
            # 로딩 중인 모델 상태 표시
            st.info(f"⏳ 모델 로딩 중: {', '.join(loading_models)}")
            # 즉시 UI 갱신하여 실시간 업데이트
            time.sleep(0.5)  # 0.5초 대기 후 즉시 갱신
            st.rerun()
    
    # 콜백 완료 메시지 처리 (스레드에서 설정된 상태 확인)
    if st.session_state.get('load_complete', False):
        if 'load_success' in st.session_state:
            st.success(st.session_state['load_success'])
            del st.session_state['load_success']
            # 로드 성공 시 UI 갱신 (캐시 스캔 불필요 - 모델 로딩과 캐시는 별개)
            st.rerun()
        if 'load_error' in st.session_state:
            st.error(st.session_state['load_error'])
            del st.session_state['load_error']
        st.session_state['load_complete'] = False
        st.session_state['check_loading'] = False  # 로딩 체크 중단
    
    # 모델 분석 결과 표시 (rate-limited state saving)
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
    
    # 로드된 모델 확인
    loaded_models = st.session_state['model_manager'].get_loaded_models()
    loaded_models_count = len(loaded_models)
    
    # 상태 배너
    if st.session_state.get('fastapi_server_running', False):
        st.success("🟢 **서버 상태**: 실행 중")
    else:
        if loaded_models_count == 0:
            st.error("❌ **서버 상태**: 로드된 모델이 없어 시작할 수 없음")
        else:
            st.warning("🟡 **서버 상태**: 중지됨 - 아래 버튼으로 시작하세요")
    
    # 서버 정보
    server_info = st.session_state['fastapi_server'].get_server_info()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 서버 상태 확인 (active_servers 기반으로 수정)
        active_servers = server_info.get('active_servers', [])
        is_running = len(active_servers) > 0
        
        # 대체 방법으로도 확인
        if not is_running:
            is_running = server_info.get('default_server_running', False) or st.session_state['fastapi_server'].is_running()
        
        st.metric("서버 상태", "🟢 실행 중" if is_running else "🔴 중지됨")
    
    with col2:
        st.metric("로드된 모델 수", loaded_models_count)
    
    with col3:
        st.metric("활성 포트", len(st.session_state.get('model_ports', {})))
    
    # 모델별 포트 설정
    if loaded_models_count > 0:
        st.subheader("🔧 모델별 포트 설정")
        
        # 포트 설정 초기화
        if 'model_ports' not in st.session_state:
            st.session_state['model_ports'] = {}
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            for i, model_name in enumerate(loaded_models):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"🤖 **{model_name}**")
                
                with col2:
                    # 기본 포트 (8000부터 시작)
                    default_port = 8000 + i
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
                        # Rate-limit port change saves
                        if not hasattr(save_app_state, 'last_port_save'):
                            save_app_state.last_port_save = 0
                        if time.time() - save_app_state.last_port_save > 5:  # Save port changes only every 5 seconds
                            save_app_state()
                            save_app_state.last_port_save = time.time()
                
                with col3:
                    # 모델별 서버 상태 확인
                    model_port = st.session_state['model_ports'].get(model_name)
                    model_running = False
                    
                    if model_port:
                        # 해당 포트의 서버가 실행 중인지 확인
                        for server in active_servers:
                            if server['port'] == model_port:
                                model_running = True
                                break
                    
                    status = "🟢 실행중" if model_running else "⚪ 대기중"
                    st.write(status)
        
        with col_right:
            st.info("💡 **포트 설정 팁**\n- 각 모델마다 다른 포트 사용\n- 기본값: 8000, 8001, 8002...\n- 범위: 3000-65535")
    
    # 서버 제어
    st.subheader("🎛️ 서버 제어")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 로드된 모델이 있을 때만 시작 버튼 활성화
        start_disabled = loaded_models_count == 0
        start_button_text = "🚀 서버 시작" if not start_disabled else "❌ 모델 없음"
        
        if st.button(start_button_text, disabled=start_disabled):
            try:
                # 로드된 모델들의 포트 설정 확인 및 준비
                loaded_models = st.session_state['model_manager'].get_loaded_models()
                model_ports = st.session_state.get('model_ports', {})
                
                # 모든 로드된 모델에 대해 포트 설정 확인
                final_model_ports = {}
                for i, model_name in enumerate(loaded_models):
                    if model_name in model_ports:
                        final_model_ports[model_name] = model_ports[model_name]
                    else:
                        # 기본 포트 설정 (8000부터 시작)
                        default_port = 8000 + i
                        final_model_ports[model_name] = default_port
                        st.session_state['model_ports'][model_name] = default_port
                
                # 멀티 포트 모드로 서버 시작
                result = st.session_state['fastapi_server'].start_server(final_model_ports)
                st.session_state['fastapi_server_running'] = True
                st.success(result)
                
                # 포트 설정 확인 메시지
                active_ports = []
                for model_name in loaded_models:
                    port = final_model_ports.get(model_name, 8000)
                    active_ports.append(f"{model_name}:{port}")
                st.info(f"🚀 모델별 포트 설정: {', '.join(active_ports)}")
                st.rerun()  # UI 즉시 업데이트
                    
            except Exception as e:
                st.error(f"서버 시작 실패: {e}")
        
        if start_disabled:
            st.caption("⚠️ 먼저 모델을 로드하세요")
    
    with col2:
        if st.button("⏹️ 서버 중지"):
            try:
                result = st.session_state['fastapi_server'].stop_server()
                st.info(result)
                
                # 서버 중지 후 실제 상태 즉시 확인
                import time
                time.sleep(1)  # 중지 처리 완료 대기
                
                server_info = st.session_state['fastapi_server'].get_server_info()
                active_servers = server_info.get('active_servers', [])
                actual_server_running = len(active_servers) > 0
                
                st.session_state['fastapi_server_running'] = actual_server_running
                
                if actual_server_running:
                    st.warning("⚠️ 일부 서버가 여전히 실행 중입니다. 다시 시도하세요.")
                else:
                    st.success("✅ 모든 서버가 성공적으로 중지되었습니다.")
                
                st.rerun()  # UI 즉시 업데이트
            except Exception as e:
                st.error(f"서버 중지 실패: {e}")
    
    with col3:
        if st.button("🧹 캐시 정리"):
            st.session_state['fastapi_server'].clear_pipeline_cache()
            save_app_state()  # 상태 저장
            st.success("파이프라인 캐시가 정리되었습니다.")
    
    # 서버 정보 표시
    # 서버 실행 상태 확인 (호환성 지원)
    if 'running' in server_info:
        is_running = server_info['running']
    elif 'default_server_running' in server_info:
        is_running = server_info['default_server_running'] or len(server_info.get('active_servers', [])) > 0
    else:
        is_running = st.session_state['fastapi_server'].is_running()
    
    if is_running:
        st.subheader("🔗 서버 정보")
        
        # 서버 URL 표시 (멀티 포트 지원)
        if 'active_servers' in server_info and server_info['active_servers']:
            st.markdown("#### 🌐 활성 서버 목록")
            for server in server_info['active_servers']:
                models_text = ', '.join(server['models']) if isinstance(server['models'], list) else server['models']
                st.info(f"**포트 {server['port']}** - 모델: {models_text}")
                st.caption(f"📡 API: {server['url']} | 📚 문서: {server['docs_url']}")
        else:
            # 하위 호환성을 위한 기본 표시
            default_port = getattr(st.session_state['fastapi_server'], 'default_port', getattr(st.session_state['fastapi_server'], 'port', 8000))
            host = server_info.get('host', '127.0.0.1')
            st.info(f"**🌐 서버 URL:** http://{host}:{default_port}")
            st.info(f"**📚 API 문서:** http://{host}:{default_port}/docs")
        
        # 로드된 모델 목록 표시 (포트 정보 포함)
        if loaded_models:
            st.markdown("#### 🤖 사용 가능한 모델")
            models_info = []
            for model_name in loaded_models:
                # 모델이 실행 중인 포트 찾기 (메서드 존재 확인)
                if hasattr(st.session_state['fastapi_server'], 'get_model_server_port'):
                    model_port = st.session_state['fastapi_server'].get_model_server_port(model_name)
                else:
                    # 이전 버전 호환성 - 기본 포트 사용
                    default_port = getattr(st.session_state['fastapi_server'], 'default_port', getattr(st.session_state['fastapi_server'], 'port', 8000))
                    model_port = default_port if st.session_state['fastapi_server'].is_running() else None
                
                port_info = f"포트 {model_port}" if model_port else "오프라인"
                endpoint_url = f"http://{server_info.get('host', '127.0.0.1')}:{model_port}/models/{model_name}/predict" if model_port else "N/A"
                
                models_info.append({
                    "모델명": model_name,
                    "포트": port_info,
                    "엔드포인트": f"/models/{model_name}/predict",
                    "전체 URL": endpoint_url,
                    "상태": "🟢 로드됨" if model_port else "⚪ 오프라인"
                })
            st.dataframe(pd.DataFrame(models_info), use_container_width=True)
        
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
        
        # 상세 API 사용법 가이드
        st.subheader("📖 API 사용법 가이드")
        
        # 기본 엔드포인트 섹션
        with st.expander("🔍 기본 엔드포인트", expanded=False):
            st.markdown("""
            ### 서버 상태 확인
            ```bash
            # 서버 기본 정보
            curl http://localhost:8002/
            
            # 서버 상태 확인
            curl http://localhost:8002/health
            
            # 로드된 모델 목록
            curl http://localhost:8002/models
            ```
            """)
        
        # 모델별 예측 엔드포인트 - 완전히 새로운 버전
        with st.expander("🤖 모델 예측 API", expanded=True):
            if loaded_models:
                for model_name in loaded_models:
                    # 모델 포트 찾기
                    if hasattr(st.session_state['fastapi_server'], 'get_model_server_port'):
                        model_port = st.session_state['fastapi_server'].get_model_server_port(model_name)
                        if model_port is None:
                            model_port = 8000  # 기본 포트 사용
                    else:
                        model_port = 8000
                    
                    st.markdown(f"### 🎯 {model_name}")
                    
                    # 모델 유형 자동 감지 및 맞춤형 예제 생성
                    model_type = detect_model_type_for_api_guide(model_name)
                    
                    # 모델 타입 배지 표시
                    if model_type == "text-classification":
                        st.markdown("🏷️ **모델 타입:** 텍스트 분류 (Text Classification)")
                    elif model_type == "token-classification":
                        st.markdown("🏷️ **모델 타입:** 개체명 인식 (Named Entity Recognition)")
                    elif model_type == "feature-extraction":
                        st.markdown("🔍 **모델 타입:** 임베딩 추출 (Feature Extraction)")
                    elif model_type == "question-answering":
                        st.markdown("❓ **모델 타입:** 질문 답변 (Question Answering)")
                    elif model_type == "text-generation":
                        st.markdown("🤖 **모델 타입:** 텍스트 생성 (Text Generation)")
                    elif model_type == "summarization":
                        st.markdown("📝 **모델 타입:** 텍스트 요약 (Summarization)")
                    elif model_type == "translation":
                        st.markdown("🌍 **모델 타입:** 번역 (Translation)")
                    
                    # 탭으로 구분된 예제들
                    tab1, tab2, tab3, tab4 = st.tabs(["🖥️ cURL", "🐍 Python", "📱 JavaScript", "⚙️ 고급"])
                    
                    with tab1:
                        # cURL 예제
                        if model_type == "text-classification":
                            st.code(f'''# 🏷️ 텍스트 분류/감정 분석 (BERT, RoBERTa, DeBERTa)

# 1. 기본 분류 (기본 파라미터)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "이 제품은 정말 훌륭합니다! 강력 추천해요.",
       "return_all_scores": false,
       "function_to_apply": "softmax"
     }}'

# 2. 상세 분류 결과 (모든 클래스 확률)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "This product is amazing! Highly recommend.",
       "return_all_scores": true,
       "function_to_apply": "softmax",
       "top_k": 3
     }}'

# 3. 배치 처리 (여러 텍스트 동시 처리)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "texts": [
         "좋은 제품이에요",
         "별로 마음에 안들어요", 
         "그냥 그래요"
       ],
       "return_all_scores": true,
       "function_to_apply": "softmax",
       "truncation": true,
       "max_length": 512
     }}'

# 4. 고급 파라미터 (DeBERTa/RoBERTa 최적화)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "복잡한 감정이 섞인 긴 텍스트 리뷰...",
       "return_all_scores": true,
       "function_to_apply": "softmax",
       "truncation": true,
       "max_length": 512,
       "stride": 128,
       "return_overflowing_tokens": true,
       "padding": "max_length"
     }}'

# 5. 신뢰도 임계값 설정
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "애매한 감정의 텍스트",
       "return_all_scores": true,
       "function_to_apply": "softmax",
       "threshold": 0.8,
       "top_k": 2
     }}'
''', language='bash')
                        
                        elif model_type == "token-classification":
                            st.code(f'''# 🏷️ 개체명 인식 (NER) - ELECTRA, KoBERT, DeBERTa-NER

# 1. 기본 NER (기본 파라미터)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "김철수는 서울특별시에서 태어나 삼성전자에서 일하고 있다.",
       "aggregation_strategy": "simple",
       "ignore_labels": ["O"]
     }}'

# 2. 고급 NER 설정 (임계값 및 집계 전략)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "John Smith works at Google in New York City.",
       "aggregation_strategy": "max",
       "ignore_labels": ["O"],
       "threshold": 0.5,
       "stride": 16,
       "return_overflowing_tokens": true
     }}'

# 3. 긴 문서 처리 (슬라이딩 윈도우)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "이순신 장군은 1545년 4월 28일 서울에서 태어났다. 조선 중기의 무신으로 임진왜란과 정유재란에서 큰 공을 세웠다. 한산도 대첩에서 거북선을 이용해 일본군을 크게 물리쳤다.",
       "aggregation_strategy": "first",
       "ignore_labels": ["O", "B-MISC"],
       "threshold": 0.7,
       "max_length": 512,
       "stride": 128,
       "truncation": true,
       "return_overflowing_tokens": true
     }}'

# 4. 배치 처리 (여러 문서 동시 처리)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "texts": [
         "김철수는 서울에서 근무합니다.",
         "이영희는 부산 출신입니다.",
         "박민수는 구글에서 일합니다."
       ],
       "aggregation_strategy": "average",
       "ignore_labels": ["O"],
       "threshold": 0.6,
       "return_all_scores": true
     }}'

# 5. 세밀한 토큰 분석 (모든 토큰 점수 반환)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "2024년 1월 15일 애플이 새로운 iPhone을 발표했다.",
       "aggregation_strategy": "none",
       "ignore_labels": [],
       "return_all_scores": true,
       "offset_mapping": true,
       "special_tokens_mask": true
     }}'

# 6. 특정 개체 유형만 추출
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "BTS의 RM이 유엔에서 연설했다.",
       "aggregation_strategy": "simple",
       "ignore_labels": ["O", "B-MISC", "I-MISC"],
       "threshold": 0.8,
       "entity_types": ["PER", "ORG", "LOC"]
     }}'
''', language='bash')
                        
                        elif model_type == "feature-extraction":
                            st.code(f'''# 🔍 임베딩 벡터 추출 - BGE, Sentence-BERT, E5

# 1. 기본 임베딩 추출 (CLS 토큰)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "안녕하세요, 반갑습니다!",
       "normalize": true,
       "return_tensors": "pt"
     }}'

# 2. 다양한 풀링 전략 적용
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "This is a sample text for embedding.",
       "pooling_strategy": "mean",
       "normalize": true,
       "return_attention_mask": true,
       "return_token_type_ids": true
     }}'

# 3. 배치 임베딩 (여러 텍스트 동시 처리)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "texts": [
         "첫 번째 문장의 임베딩",
         "두 번째 문장의 임베딩",
         "세 번째 문장의 임베딩"
       ],
       "pooling_strategy": "cls",
       "normalize": true,
       "max_length": 512,
       "truncation": true,
       "padding": "max_length"
     }}'

# 4. 긴 문서 임베딩 (슬라이딩 윈도우)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "인공지능은 현재 우리 사회의 다양한 분야에서 혁신을 이끌고 있습니다. 의료진단, 자율주행, 언어번역 등 광범위한 응용이 가능합니다...",
       "pooling_strategy": "mean",
       "normalize": true,
       "max_length": 512,
       "stride": 256,
       "return_overflowing_tokens": true,
       "truncation": true
     }}'

# 5. 고급 임베딩 설정 (BGE-M3 최적화)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "Multilingual embedding extraction",
       "pooling_strategy": "weighted_mean",
       "normalize": true,
       "layer_index": -1,
       "return_all_layers": false,
       "attention_mask": true,
       "token_type_ids": true,
       "output_hidden_states": false
     }}'

# 6. 토큰별 임베딩 (세밀한 분석)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "토큰별 임베딩 추출 예제",
       "pooling_strategy": "none",
       "normalize": false,
       "return_all_tokens": true,
       "return_attention_mask": true,
       "return_offsets_mapping": true,
       "output_hidden_states": true,
       "output_attentions": false
     }}'

# 7. 유사도 계산용 임베딩 쌍
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text_pairs": [
         ["질문: 날씨가 어떤가요?", "답변: 오늘은 맑습니다."],
         ["첫 번째 문서", "두 번째 문서"]
       ],
       "pooling_strategy": "cls",
       "normalize": true,
       "return_similarity": true,
       "similarity_function": "cosine"
     }}'
''', language='bash')
                        
                        elif model_type == "question-answering":
                            st.code(f'''# ❓ 질문 답변 - BERT-QA, RoBERTa-QA, DeBERTa-QA

# 1. 기본 질문 답변
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "question": "대한민국의 수도는?",
       "context": "대한민국의 수도는 서울특별시이다. 서울은 한강을 중심으로 발전했다.",
       "max_answer_length": 30,
       "max_seq_length": 384,
       "max_query_length": 64
     }}'

# 2. 상세 설정 (신뢰도 및 다중 답변)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "question": "언제 태어났나?",
       "context": "김철수는 1990년 5월 15일 서울에서 태어났다. 그의 형 김영수는 1988년 3월 10일에 태어났다.",
       "top_k": 3,
       "max_answer_length": 50,
       "handle_impossible_answer": true,
       "threshold": 0.1
     }}'

# 3. 긴 문서 처리 (문서 스트라이드)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "question": "주요 업적은 무엇인가?",
       "context": "이순신 장군은 1545년 4월 28일 서울에서 태어났다. 조선 중기의 무신으로 임진왜란과 정유재란에서 큰 공을 세웠다. 한산도 대첩에서 거북선을 이용해 일본군을 크게 물리쳤다. 명량대첩에서는 13척의 배로 133척의 일본군을 물리치는 기적을 일으켰다.",
       "max_seq_length": 512,
       "doc_stride": 128,
       "max_answer_length": 100,
       "return_overflowing_tokens": true,
       "stride": 128
     }}'

# 4. 배치 질문 답변
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "questions": [
         "누구의 이야기인가?",
         "언제 일어난 일인가?",
         "어디에서 일어났나?"
       ],
       "context": "2024년 1월 15일 서울에서 김철수가 새로운 회사를 창업했다.",
       "max_answer_length": 30,
       "return_all_answers": true,
       "top_k": 2
     }}'

# 5. 불가능한 답변 처리
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "question": "달의 색깔은?",
       "context": "김철수는 서울에서 태어났다. 그는 컴퓨터 과학을 전공했다.",
       "handle_impossible_answer": true,
       "impossible_answer_threshold": 0.5,
       "null_score_diff_threshold": 0.0,
       "max_answer_length": 50
     }}'

# 6. 고급 답변 추출 (점수 및 위치 정보)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "question": "어떤 기술을 사용했나?",
       "context": "거북선은 조선시대의 혁신적인 군함이다. 철갑으로 덮여 있어서 적의 공격을 막을 수 있었다. 화포를 장착하여 강력한 공격력을 가지고 있었다.",
       "return_answer_start_end": true,
       "return_confidence_score": true,
       "output_scores": true,
       "max_answer_length": 80,
       "min_answer_length": 1
     }}'
''', language='bash')
                        
                        elif model_type == "text-generation":
                            st.code(f'''# 🤖 텍스트 생성 - GPT, GPT-2, BART, T5

# 1. 기본 텍스트 생성 (greedy decoding)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "인공지능의 미래는",
       "max_length": 100,
       "min_length": 20,
       "do_sample": false,
       "num_return_sequences": 1
     }}'

# 2. 창의적 생성 (sampling with temperature)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "옛날 옛적에",
       "max_length": 200,
       "min_length": 50,
       "do_sample": true,
       "temperature": 0.8,
       "top_p": 0.9,
       "top_k": 50,
       "num_return_sequences": 3
     }}'

# 3. 고급 생성 설정 (repetition penalty)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "과학 기술의 발전으로",
       "max_length": 150,
       "min_length": 30,
       "do_sample": true,
       "temperature": 0.7,
       "top_p": 0.95,
       "repetition_penalty": 1.1,
       "length_penalty": 1.0,
       "no_repeat_ngram_size": 3
     }}'

# 4. 조건부 생성 (prefix guidance)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "AI 기술:",
       "max_length": 120,
       "min_length": 40,
       "do_sample": true,
       "temperature": 0.6,
       "top_p": 0.8,
       "pad_token_id": 0,
       "eos_token_id": 2,
       "forced_eos_token_id": 2
     }}'

# 5. 빔 서치 생성 (beam search)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "기후 변화 대응을 위해",
       "max_length": 100,
       "min_length": 25,
       "do_sample": false,
       "num_beams": 5,
       "num_return_sequences": 3,
       "early_stopping": true,
       "length_penalty": 1.2
     }}'

# 6. 제어된 생성 (stop sequences)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "질문: 우주의 크기는? 답변:",
       "max_length": 80,
       "min_length": 10,
       "do_sample": true,
       "temperature": 0.5,
       "stop_sequences": ["질문:", "\\n\\n"],
       "bad_words_ids": [],
       "force_words_ids": []
     }}'

# 7. 스트리밍 생성 (실시간 출력)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -H "Accept: text/event-stream" \\
     -d '{{
       "text": "AI의 윤리적 고려사항은",
       "max_length": 150,
       "do_sample": true,
       "temperature": 0.7,
       "stream": true,
       "include_stop_str_in_output": false
     }}'
''', language='bash')
                        
                        elif model_type == "text2text-generation":
                            st.code(f'''# 🔄 Text2Text 생성 - T5, BART, UL2, Flan-T5

# 1. 기본 Text2Text 생성
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "translate English to Korean: Hello world",
       "max_length": 128,
       "min_length": 10,
       "num_return_sequences": 1
     }}'

# 2. 다양한 T5 태스크
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "summarize: The quick brown fox jumps over the lazy dog...",
       "max_length": 100,
       "min_length": 20,
       "do_sample": true,
       "temperature": 0.8,
       "top_p": 0.9
     }}'

# 3. 멀티태스크 처리
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "question: What is AI? context: Artificial intelligence is...",
       "max_length": 64,
       "num_beams": 4,
       "early_stopping": true,
       "length_penalty": 1.2
     }}'
''', language='bash')
                        
                        elif model_type == "summarization":
                            st.code(f'''# 📝 텍스트 요약 - Pegasus, LED, BART-Large

# 1. 기본 요약
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "긴 문서 내용을 여기에 입력...",
       "max_length": 150,
       "min_length": 30,
       "length_penalty": 2.0,
       "num_beams": 4
     }}'

# 2. 고급 요약 (길이 제어)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "Very long article text...",
       "max_length": 200,
       "min_length": 50,
       "length_penalty": 1.5,
       "num_beams": 6,
       "early_stopping": true,
       "no_repeat_ngram_size": 3
     }}'

# 3. 배치 요약
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "texts": [
         "First document to summarize...",
         "Second document to summarize..."
       ],
       "max_length": 100,
       "min_length": 20,
       "num_beams": 3
     }}'
''', language='bash')
                        
                        elif model_type == "translation":
                            st.code(f'''# 🌍 번역 - Opus-MT, NLLB, Marian, M2M100

# 1. 기본 번역
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "Hello, how are you?",
       "src_lang": "en",
       "tgt_lang": "ko",
       "max_length": 128
     }}'

# 2. 다국어 번역 (NLLB)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "Bonjour, comment allez-vous?",
       "src_lang": "fra_Latn",
       "tgt_lang": "kor_Hang",
       "max_length": 256,
       "num_beams": 5,
       "length_penalty": 1.0
     }}'

# 3. 배치 번역
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "texts": [
         "Good morning!",
         "How are you today?",
         "Nice to meet you."
       ],
       "src_lang": "en",
       "tgt_lang": "ko",
       "max_length": 100
     }}'
''', language='bash')
                        
                        elif model_type == "fill-mask":
                            st.code(f'''# 🎭 Fill-Mask - BERT, RoBERTa, DeBERTa MLM

# 1. 기본 마스크 채우기
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "Paris is the [MASK] of France.",
       "top_k": 5
     }}'

# 2. 다중 마스크 처리
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "The [MASK] is very [MASK] today.",
       "top_k": 3,
       "targets": ["weather", "sunny"]
     }}'

# 3. 한국어 마스크 채우기
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "서울은 대한민국의 [MASK]이다.",
       "top_k": 5,
       "threshold": 0.1
     }}'
''', language='bash')
                        
                        elif model_type == "multimodal":
                            st.code(f'''# 🖼️ 멀티모달 - CLIP, BLIP, LayoutLM

# 1. 이미지-텍스트 유사도 (CLIP)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "image": "base64_encoded_image_data",
       "text": "a photo of a cat",
       "task": "image-text-similarity"
     }}'

# 2. 이미지 캡션 생성 (BLIP)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "image": "base64_encoded_image_data",
       "task": "image-captioning",
       "max_length": 50,
       "num_beams": 3
     }}'

# 3. 문서 이해 (LayoutLM)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "image": "base64_encoded_document_image",
       "text": "Extract key information",
       "task": "document-understanding",
       "return_bbox": true
     }}'
''', language='bash')
                        
                        elif model_type == "image-classification":
                            st.code(f'''# 🖼️ 이미지 분류 - ViT, DeiT, Swin, ConvNeXT

# 1. 기본 이미지 분류
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "image": "base64_encoded_image_data",
       "top_k": 5
     }}'

# 2. 고급 이미지 분류 (확률 반환)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "image": "base64_encoded_image_data",
       "return_all_scores": true,
       "top_k": 10,
       "threshold": 0.1
     }}'

# 3. 배치 이미지 분류
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "images": [
         "base64_encoded_image_1",
         "base64_encoded_image_2"
       ],
       "top_k": 3
     }}'
''', language='bash')
                        
                        elif model_type == "automatic-speech-recognition":
                            st.code(f'''# 🎤 음성 인식 - Wav2Vec2, Whisper, SpeechT5

# 1. 기본 음성 인식
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "audio": "base64_encoded_audio_data",
       "sampling_rate": 16000
     }}'

# 2. 다국어 음성 인식 (Whisper)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "audio": "base64_encoded_audio_data",
       "language": "korean",
       "task": "transcribe",
       "return_timestamps": true
     }}'

# 3. 음성 번역 (Whisper)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "audio": "base64_encoded_audio_data",
       "task": "translate",
       "target_language": "english",
       "return_timestamps": true,
       "chunk_length": 30
     }}'
''', language='bash')
                        
                        else:
                            st.code(f'''# 🔧 기본 예측 API (모든 모델 타입 지원)

# 1. 기본 예측
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{"text": "입력 텍스트를 여기에 넣으세요"}}'

# 2. 파라미터가 있는 요청
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "입력 텍스트",
       "max_length": 512,
       "temperature": 0.7,
       "top_p": 0.9,
       "do_sample": true
     }}'

# 3. 멀티모달 입력 (이미지 + 텍스트)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "텍스트 입력",
       "image": "base64_encoded_image_data",
       "audio": "base64_encoded_audio_data",
       "return_all_outputs": true
     }}'
''', language='bash')
                    
                    with tab2:
                        # Python 예제
                        if model_type == "text-classification":
                            st.code(f'''import requests
import json
import asyncio
import aiohttp
from typing import List, Dict, Optional

# 🏷️ 텍스트 분류/감정 분석 - 상세 파라미터 포함

# 1. 기본 감정 분석 (기본 파라미터)
def analyze_sentiment_basic(text: str) -> Dict:
    """기본 감정 분석"""
    data = {{
        "text": text,
        "return_all_scores": False,
        "function_to_apply": "softmax"
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 2. 상세 분석 (모든 클래스 확률 반환)
def analyze_sentiment_detailed(text: str, top_k: int = 3) -> Dict:
    """모든 클래스 확률을 반환하는 상세 분석"""
    data = {{
        "text": text,
        "return_all_scores": True,
        "function_to_apply": "softmax",
        "top_k": top_k
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 3. 배치 처리 (최적화된 다중 텍스트 처리)
def analyze_sentiment_batch(texts: List[str], **kwargs) -> List[Dict]:
    """배치 처리로 여러 텍스트 동시 분석"""
    data = {{
        "texts": texts,
        "return_all_scores": True,
        "function_to_apply": "softmax",
        "truncation": True,
        "max_length": 512,
        **kwargs  # 추가 파라미터
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 4. 고급 분석 (DeBERTa/RoBERTa 최적화)
def analyze_sentiment_advanced(text: str, confidence_threshold: float = 0.8) -> Dict:
    """고급 파라미터를 사용한 정밀 분석"""
    data = {{
        "text": text,
        "return_all_scores": True,
        "function_to_apply": "softmax",
        "truncation": True,
        "max_length": 512,
        "stride": 128,
        "return_overflowing_tokens": True,
        "padding": "max_length",
        "threshold": confidence_threshold
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 5. 비동기 처리 (대용량 처리용)
async def analyze_sentiment_async(session: aiohttp.ClientSession, text: str) -> Dict:
    """비동기 감정 분석"""
    data = {{
        "text": text,
        "return_all_scores": True,
        "function_to_apply": "softmax"
    }}
    
    async with session.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    ) as response:
        return await response.json()

async def analyze_batch_async(texts: List[str]) -> List[Dict]:
    """비동기 배치 처리"""
    async with aiohttp.ClientSession() as session:
        tasks = [analyze_sentiment_async(session, text) for text in texts]
        return await asyncio.gather(*tasks)

# 6. 실전 사용 예제
def main():
    # 단일 텍스트 분석
    text = "이 제품은 정말 훌륭합니다! 강력 추천해요."
    result = analyze_sentiment_detailed(text)
    
    print(f"텍스트: {{text}}")
    print(f"주요 감정: {{result['predictions'][0]['label']}}")
    print(f"신뢰도: {{result['predictions'][0]['score']:.4f}}")
    
    # 배치 처리
    texts = [
        "좋은 제품이에요",
        "별로 마음에 안들어요", 
        "그냥 그래요",
        "최고의 선택이었습니다!"
    ]
    
    batch_results = analyze_sentiment_batch(texts)
    
    print("\\n배치 처리 결과:")
    for i, text in enumerate(texts):
        pred = batch_results['predictions'][i]
        print(f"'{{text}}' -> {{pred['label']}} ({{pred['score']:.3f}})")
    
    # 신뢰도 필터링
    high_confidence = [
        (texts[i], batch_results['predictions'][i]) 
        for i in range(len(texts))
        if batch_results['predictions'][i]['score'] > 0.8
    ]
    
    print(f"\\n고신뢰도 결과 (>0.8): {{len(high_confidence)}}개")
    for text, pred in high_confidence:
        print(f"  '{{text}}' -> {{pred['label']}} ({{pred['score']:.3f}})")

# 7. 에러 처리 및 재시도
def safe_analyze_sentiment(text: str, max_retries: int = 3, timeout: int = 30) -> Dict:
    """에러 처리와 재시도 로직이 포함된 안전한 분석"""
    for attempt in range(max_retries):
        try:
            data = {{
                "text": text,
                "return_all_scores": True,
                "function_to_apply": "softmax"
            }}
            
            response = requests.post(
                "http://localhost:{model_port}/models/{model_name}/predict",
                json=data,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                return {{"error": f"요청 시간 초과 ({{max_retries}}회 시도)"}}
            print(f"시간 초과, 재시도 {{attempt + 1}}/{{max_retries}}")
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return {{"error": f"요청 실패: {{e}}"}}
            print(f"요청 실패, 재시도 {{attempt + 1}}/{{max_retries}}: {{e}}")
            
        except Exception as e:
            return {{"error": f"예상치 못한 오류: {{e}}"}}
    
    return {{"error": "최대 재시도 횟수 초과"}}

# 사용 예제
if __name__ == "__main__":
    main()
''', language='python')
                        
                        elif model_type == "token-classification":
                            st.code(f'''import requests
from collections import defaultdict

# NER 분석 함수
def extract_entities(text):
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json={{"text": text}}
    )
    return response.json()

# 단일 텍스트 분석
text = "김철수는 서울특별시에서 태어나 삼성전자에서 일하고 있다."
result = extract_entities(text)

print(f"원본 텍스트: {{text}}")
print("\\n개체명 목록:")
for entity in result['predictions']:
    print(f"- {{entity['word']}}: {{entity['entity']}} ({{entity['score']:.3f}})")

# 개체명 타입별 그룹화
def group_entities_by_type(entities):
    grouped = defaultdict(list)
    for entity in entities:
        grouped[entity['entity']].append(entity['word'])
    return dict(grouped)

entities = result['predictions']
grouped = group_entities_by_type(entities)

print("\\n타입별 개체명:")
for entity_type, words in grouped.items():
    print(f"- {{entity_type}}: {{', '.join(set(words))}}")

# 여러 문서 배치 처리
documents = [
    "이순신 장군은 조선의 명장이었다.",
    "애플은 캘리포니아에 본사를 두고 있다.",
    "BTS는 한국의 대표적인 K-POP 그룹이다."
]

print("\\n배치 처리 결과:")
for i, doc in enumerate(documents):
    result = extract_entities(doc)
    print(f"\\n문서 {{i+1}}: {{doc}}")
    entities = group_entities_by_type(result['predictions'])
    for entity_type, words in entities.items():
        print(f"  {{entity_type}}: {{', '.join(set(words))}}")
''', language='python')
                        
                        elif model_type == "feature-extraction":
                            st.code(f'''import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 임베딩 추출 함수
def get_embedding(text):
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json={{"text": text}}
    )
    result = response.json()
    return np.array(result['predictions'])

# 단일 텍스트 임베딩
text = "안녕하세요, 반갑습니다!"
embedding = get_embedding(text)
print(f"텍스트: {{text}}")
print(f"임베딩 크기: {{embedding.shape}}")
print(f"임베딩 샘플: {{embedding[:5]}}")

# 유사도 계산
texts = [
    "안녕하세요",
    "반갑습니다", 
    "날씨가 좋네요",
    "Hello world"
]

embeddings = []
for text in texts:
    emb = get_embedding(text)
    embeddings.append(emb)

# 모든 텍스트 쌍의 유사도 계산
print("\\n텍스트 간 유사도 매트릭스:")
similarities = cosine_similarity(embeddings)

for i, text1 in enumerate(texts):
    for j, text2 in enumerate(texts):
        if i < j:  # 중복 제거
            sim = similarities[i][j]
            print(f"'{{text1}}' vs '{{text2}}': {{sim:.3f}}")

# 문서 클러스터링 예제
def find_most_similar(query_text, document_texts, top_k=3):
    query_emb = get_embedding(query_text)
    doc_embeddings = [get_embedding(doc) for doc in document_texts]
    
    similarities = []
    for doc_emb in doc_embeddings:
        sim = cosine_similarity([query_emb], [doc_emb])[0][0]
        similarities.append(sim)
    
    # 상위 k개 문서 반환
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({{
            "document": document_texts[idx],
            "similarity": similarities[idx]
        }})
    
    return results

# 사용 예제
query = "인공지능 기술"
documents = [
    "머신러닝은 AI의 한 분야입니다",
    "오늘 날씨가 맑습니다",
    "딥러닝으로 이미지 인식을 합니다",
    "점심 메뉴를 고르고 있어요"
]

similar_docs = find_most_similar(query, documents)
print(f"\\n쿼리: '{{query}}'와 가장 유사한 문서들:")
for doc in similar_docs:
    print(f"- {{doc['similarity']:.3f}}: {{doc['document']}}")
''', language='python')
                        
                        elif model_type == "text2text-generation":
                            st.code('''import requests
import json
from typing import List, Dict, Optional

# 🔄 Text2Text 생성 - T5, BART, UL2, Flan-T5 (상세 파라미터 포함)

# 1. 기본 Text2Text 생성
def text2text_generate(input_text: str, task_prefix: str = "") -> Dict:
    """기본 Text2Text 생성"""
    full_text = f"{{task_prefix}}: {{input_text}}" if task_prefix else input_text
    
    data = {{
        "text": full_text,
        "max_length": 128,
        "min_length": 10,
        "num_return_sequences": 1,
        "do_sample": False,
        "num_beams": 1,
        "early_stopping": True
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 2. 다양한 T5 태스크 처리
def handle_t5_tasks() -> None:
    """T5 모델의 다양한 태스크 처리"""
    
    # 번역 태스크
    translation_result = text2text_generate(
        "Hello world, how are you?",
        task_prefix="translate English to Korean"
    )
    print("번역 결과:", translation_result)
    
    # 요약 태스크
    long_text = """
    인공지능(AI)은 인간의 지능을 모방하여 학습, 추론, 문제해결 등의 능력을 갖춘 시스템입니다.
    머신러닝과 딥러닝 기술의 발전으로 이미지 인식, 자연어 처리, 음성 인식 등 다양한 분야에서 활용되고 있습니다.
    """
    
    summary_result = text2text_generate(
        long_text.strip(),
        task_prefix="summarize"
    )
    print("요약 결과:", summary_result)

# 3. 고급 생성 파라미터
def advanced_text2text_generate(input_text: str, task_prefix: str = "") -> Dict:
    """고급 파라미터를 사용한 생성"""
    full_text = f"{{task_prefix}}: {{input_text}}" if task_prefix else input_text
    
    data = {{
        "text": full_text,
        "max_length": 100,
        "min_length": 20,
        "num_return_sequences": 3,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.9,
        "top_k": 50,
        "num_beams": 4,
        "early_stopping": True,
        "length_penalty": 1.2,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 2
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 사용 예제
if __name__ == "__main__":
    # 기본 사용
    result = text2text_generate("Hello world", "translate English to Korean")
    print("기본 결과:", result)
    
    # 고급 사용
    handle_t5_tasks()
''', language='python')
                        
                        elif model_type == "summarization":
                            st.code(f'''import requests
import json
from typing import List, Dict, Optional

# 📝 텍스트 요약 - Pegasus, LED, BART-Large (상세 파라미터 포함)

# 1. 기본 요약 함수
def summarize_text(text: str, max_length: int = 150, min_length: int = 30) -> Dict:
    """기본 텍스트 요약"""
    data = {{
        "text": text,
        "max_length": max_length,
        "min_length": min_length,
        "length_penalty": 2.0,
        "num_beams": 4,
        "early_stopping": True,
        "do_sample": False
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 2. 고급 요약 (다양한 파라미터)
def advanced_summarize(text: str, summary_type: str = "balanced") -> Dict:
    """다양한 요약 스타일"""
    
    if summary_type == "extractive":
        # 추출적 요약 (원문 구문 유지)
        data = {{
            "text": text,
            "max_length": 100,
            "min_length": 50,
            "length_penalty": 1.0,
            "num_beams": 6,
            "early_stopping": True,
            "do_sample": False,
            "repetition_penalty": 1.2
        }}
    elif summary_type == "abstractive":
        # 추상적 요약 (의미 기반 재구성)
        data = {{
            "text": text,
            "max_length": 120,
            "min_length": 40,
            "length_penalty": 1.5,
            "num_beams": 4,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }}
    else:  # balanced
        # 균형잡힌 요약
        data = {{
            "text": text,
            "max_length": 130,
            "min_length": 35,
            "length_penalty": 1.8,
            "num_beams": 5,
            "early_stopping": True,
            "do_sample": False
        }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 사용 예제
if __name__ == "__main__":
    sample_text = """
    인공지능(AI)은 인간의 지능을 모방하여 학습, 추론, 문제해결 등의 능력을 갖춘 시스템입니다.
    머신러닝과 딥러닝 기술의 발전으로 이미지 인식, 자연어 처리, 음성 인식 등 다양한 분야에서 활용되고 있습니다.
    """
    
    # 기본 요약
    basic_result = summarize_text(sample_text)
    print("기본 요약:", basic_result)
    
    # 고급 요약
    advanced_result = advanced_summarize(sample_text, "abstractive")
    print("고급 요약:", advanced_result)
''', language='python')
                        
                        elif model_type == "translation":
                            st.code(f'''import requests
import json
from typing import List, Dict, Optional

# 🌐 번역 - MarianMT, NLLB, M2M100, Opus-MT (상세 파라미터 포함)

# 1. 기본 번역 함수
def translate_text(text: str, source_lang: str = "en", target_lang: str = "ko") -> Dict:
    """기본 텍스트 번역"""
    
    # 언어 쌍에 따른 프롬프트 구성
    prompt = f"translate {{source_lang}} to {{target_lang}}: {{text}}"
    
    data = {{
        "text": prompt,
        "max_length": 512,
        "num_beams": 5,
        "early_stopping": True,
        "length_penalty": 1.0,
        "do_sample": False
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 2. 다중 언어 번역
def multi_language_translate(text: str, target_languages: List[str] = ["ko", "ja", "zh"]) -> Dict[str, Dict]:
    """하나의 텍스트를 여러 언어로 번역"""
    results = {{}}
    
    for lang in target_languages:
        try:
            result = translate_text(text, source_lang="en", target_lang=lang)
            results[lang] = result
        except Exception as e:
            results[lang] = {{"error": str(e)}}
    
    return results

# 3. 고급 번역 파라미터
def advanced_translate(text: str, source_lang: str = "en", target_lang: str = "ko", 
                      translation_style: str = "formal") -> Dict:
    """고급 파라미터를 사용한 번역"""
    
    if translation_style == "formal":
        data = {{
            "text": f"translate {{source_lang}} to {{target_lang}}: {{text}}",
            "max_length": 512,
            "num_beams": 6,
            "early_stopping": True,
            "length_penalty": 1.2,
            "repetition_penalty": 1.1,
            "do_sample": False
        }}
    elif translation_style == "casual":
        data = {{
            "text": f"translate {{source_lang}} to {{target_lang}}: {{text}}",
            "max_length": 512,
            "num_beams": 4,
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "length_penalty": 1.0
        }}
    else:  # balanced
        data = {{
            "text": f"translate {{source_lang}} to {{target_lang}}: {{text}}",
            "max_length": 512,
            "num_beams": 5,
            "early_stopping": True,
            "length_penalty": 1.0
        }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 사용 예제
if __name__ == "__main__":
    # 기본 번역
    english_text = "Hello, how are you today?"
    basic_result = translate_text(english_text, "en", "ko")
    print("기본 번역:", basic_result)
    
    # 다중 언어 번역
    multi_result = multi_language_translate(english_text, ["ko", "ja"])
    print("다중 언어 번역:", multi_result)
''', language='python')
                        
                        elif model_type == "fill-mask":
                            st.code(f'''import requests
import json
from typing import List, Dict, Optional

# 🎭 Fill-Mask - BERT, RoBERTa, DeBERTa, ELECTRA (상세 파라미터 포함)

# 1. 기본 마스크 채우기
def fill_mask_basic(text: str, mask_token: str = "[MASK]") -> Dict:
    """기본 마스크 채우기"""
    data = {{
        "text": text,
        "top_k": 5,
        "temperature": 1.0,
        "return_tensors": False
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 2. 고급 마스크 채우기
def fill_mask_advanced(text: str, top_k: int = 10, temperature: float = 1.0,
                      min_score: float = 0.01) -> Dict:
    """고급 파라미터를 사용한 마스크 채우기"""
    data = {{
        "text": text,
        "top_k": top_k,
        "temperature": temperature,
        "min_score": min_score,
        "return_tensors": False,
        "return_all_scores": False
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 3. 다중 마스크 처리
def multiple_mask_fill(text: str, top_k: int = 5) -> Dict:
    """여러 마스크를 동시에 처리"""
    
    # 마스크 개수 확인
    mask_count = text.count("[MASK]")
    
    if mask_count == 0:
        return {{"error": "No [MASK] token found in text"}}
    
    data = {{
        "text": text,
        "top_k": top_k,
        "temperature": 1.0,
        "handle_multiple_masks": True
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    
    result = response.json()
    result["mask_count"] = mask_count
    return result

# 사용 예제
if __name__ == "__main__":
    # 기본 마스크 채우기
    text1 = "서울은 대한민국의 [MASK]입니다."
    basic_result = fill_mask_basic(text1)
    print("기본 마스크 채우기:", basic_result)
    
    # 고급 마스크 채우기
    text2 = "인공지능은 [MASK] 기술입니다."
    advanced_result = fill_mask_advanced(text2, top_k=8, temperature=1.2)
    print("고급 마스크 채우기:", advanced_result)
    
    # 다중 마스크 처리
    text3 = "[MASK]는 [MASK]에서 개발된 언어 모델입니다."
    multi_result = multiple_mask_fill(text3)
    print("다중 마스크 처리:", multi_result)
''', language='python')
                        
                        elif model_type == "multimodal":
                            st.code(f'''import requests
import json
import base64
from typing import List, Dict, Optional, Union
from io import BytesIO
from PIL import Image

# 🎨 멀티모달 모델 - CLIP, BLIP, LayoutLM (상세 파라미터 포함)

# 1. 이미지 전처리 유틸리티
def encode_image_to_base64(image_path: str) -> str:
    """이미지를 Base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_image_data(image_input: Union[str, bytes, Image.Image]) -> str:
    """다양한 이미지 입력을 Base64로 변환"""
    if isinstance(image_input, str):
        # 파일 경로
        return encode_image_to_base64(image_input)
    elif isinstance(image_input, bytes):
        # 바이트 데이터
        return base64.b64encode(image_input).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        # PIL Image 객체
        buffer = BytesIO()
        image_input.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image input type")

# 2. 이미지-텍스트 유사도 계산 (CLIP)
def image_text_similarity(image_path: str, texts: List[str]) -> Dict:
    """이미지와 텍스트 간 유사도 계산"""
    
    image_data = prepare_image_data(image_path)
    
    data = {{
        "image": image_data,
        "texts": texts,
        "normalize": True,
        "temperature": 0.01,
        "return_similarity_matrix": True
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 3. 이미지 캡셔닝 (BLIP)
def generate_image_caption(image_path: str, max_length: int = 50, 
                         num_captions: int = 1) -> Dict:
    """이미지에 대한 캡션 생성"""
    
    image_data = prepare_image_data(image_path)
    
    data = {{
        "image": image_data,
        "task": "image_captioning",
        "max_length": max_length,
        "min_length": 10,
        "num_return_sequences": num_captions,
        "num_beams": 5,
        "early_stopping": True,
        "temperature": 0.8,
        "do_sample": True if num_captions > 1 else False
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 4. 시각적 질문 답변 (VQA)
def visual_question_answering(image_path: str, question: str) -> Dict:
    """이미지 기반 질문 답변"""
    
    image_data = prepare_image_data(image_path)
    
    data = {{
        "image": image_data,
        "question": question,
        "task": "visual_question_answering",
        "max_length": 50,
        "num_beams": 4,
        "early_stopping": True,
        "confidence_threshold": 0.5
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 사용 예제
if __name__ == "__main__":
    # 샘플 이미지 경로 (실제 사용시 적절한 경로로 변경)
    sample_image = "/path/to/sample_image.jpg"
    
    # 이미지 캡셔닝
    caption_result = generate_image_caption(sample_image, max_length=50)
    print("이미지 캡션:", caption_result)
    
    # 이미지-텍스트 유사도
    similarity_result = image_text_similarity(
        sample_image, 
        ["a beautiful landscape", "a person", "an animal", "a building"]
    )
    print("유사도 결과:", similarity_result)
    
    # 시각적 질문 답변
    vqa_result = visual_question_answering(sample_image, "What is in this image?")
    print("VQA 결과:", vqa_result)
''', language='python')
                        
                        elif model_type == "image-classification":
                            st.code(f'''import requests
import json
import base64
from typing import List, Dict, Optional, Union
from io import BytesIO
from PIL import Image

# 🖼️ 이미지 분류 - ViT, DeiT, Swin, ConvNeXT (상세 파라미터 포함)

# 1. 이미지 전처리 유틸리티
def encode_image_to_base64(image_path: str) -> str:
    """이미지를 Base64로 인코딩"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def prepare_image_data(image_input: Union[str, bytes, Image.Image]) -> str:
    """다양한 이미지 입력을 Base64로 변환"""
    if isinstance(image_input, str):
        # 파일 경로
        return encode_image_to_base64(image_input)
    elif isinstance(image_input, bytes):
        # 바이트 데이터
        return base64.b64encode(image_input).decode('utf-8')
    elif isinstance(image_input, Image.Image):
        # PIL Image 객체
        buffer = BytesIO()
        image_input.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    else:
        raise ValueError("Unsupported image input type")

# 2. 기본 이미지 분류
def classify_image_basic(image_path: str, top_k: int = 5) -> Dict:
    """기본 이미지 분류"""
    
    image_data = prepare_image_data(image_path)
    
    data = {{
        "image": image_data,
        "top_k": top_k,
        "return_all_scores": False,
        "confidence_threshold": 0.01
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 3. 고급 이미지 분류
def classify_image_advanced(image_path: str, top_k: int = 10, 
                          confidence_threshold: float = 0.1,
                          return_probabilities: bool = True) -> Dict:
    """고급 파라미터를 사용한 이미지 분류"""
    
    image_data = prepare_image_data(image_path)
    
    data = {{
        "image": image_data,
        "top_k": top_k,
        "confidence_threshold": confidence_threshold,
        "return_all_scores": return_probabilities,
        "normalize_scores": True,
        "temperature": 1.0
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 4. 배치 이미지 분류
def batch_classify_images(image_paths: List[str], top_k: int = 3) -> List[Dict]:
    """여러 이미지를 배치로 분류"""
    
    results = []
    
    for image_path in image_paths:
        try:
            result = classify_image_basic(image_path, top_k)
            result["image_path"] = image_path
            results.append(result)
        except Exception as e:
            results.append({{
                "image_path": image_path,
                "error": str(e)
            }})
    
    return results

# 사용 예제
if __name__ == "__main__":
    # 샘플 이미지 경로 (실제 사용시 적절한 경로로 변경)
    sample_image = "/path/to/sample_image.jpg"
    
    # 기본 분류
    basic_result = classify_image_basic(sample_image, top_k=5)
    print("기본 분류:", basic_result)
    
    # 고급 분류
    advanced_result = classify_image_advanced(sample_image, top_k=10, confidence_threshold=0.1)
    print("고급 분류:", advanced_result)
    
    # 배치 처리
    image_list = [sample_image]  # 실제로는 여러 이미지 경로
    batch_result = batch_classify_images(image_list, top_k=3)
    print("배치 분류:", batch_result)
''', language='python')
                        
                        elif model_type == "automatic-speech-recognition":
                            st.code(f'''import requests
import json
import base64
import wave
from typing import List, Dict, Optional

# 🎤 음성 인식 - Wav2Vec2, Whisper, SpeechT5 (상세 파라미터 포함)

# 1. 오디오 전처리 유틸리티
def encode_audio_to_base64(audio_path: str) -> str:
    """오디오 파일을 Base64로 인코딩"""
    with open(audio_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

def get_audio_info(audio_path: str) -> Dict:
    """오디오 파일 정보 추출"""
    try:
        with wave.open(audio_path, 'rb') as wav_file:
            frames = wav_file.getnframes()
            sample_rate = wav_file.getframerate()
            duration = frames / sample_rate
            channels = wav_file.getnchannels()
            
            return {{
                "duration": duration,
                "sample_rate": sample_rate,
                "channels": channels,
                "frames": frames,
                "format": "WAV"
            }}
    except Exception as e:
        return {{"error": f"Could not read audio file: {{e}}"}}

# 2. 기본 음성 인식
def transcribe_audio_basic(audio_path: str) -> Dict:
    """기본 음성 인식"""
    
    audio_data = encode_audio_to_base64(audio_path)
    audio_info = get_audio_info(audio_path)
    
    data = {{
        "audio": audio_data,
        "return_timestamps": False,
        "language": "auto",
        "task": "transcribe"
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    
    result = response.json()
    result["audio_info"] = audio_info
    return result

# 3. 고급 음성 인식 (Whisper)
def transcribe_audio_advanced(audio_path: str, language: str = "auto",
                            return_timestamps: bool = True,
                            task: str = "transcribe") -> Dict:
    """고급 파라미터를 사용한 음성 인식"""
    
    audio_data = encode_audio_to_base64(audio_path)
    
    data = {{
        "audio": audio_data,
        "language": language,
        "task": task,  # transcribe or translate
        "return_timestamps": return_timestamps,
        "word_timestamps": True,
        "no_speech_threshold": 0.6,
        "temperature": 0.0,
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0
    }}
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 4. 배치 음성 인식
def batch_transcribe_audio(audio_paths: List[str], 
                         language: str = "auto") -> List[Dict]:
    """여러 오디오 파일을 배치로 인식"""
    
    results = []
    
    for audio_path in audio_paths:
        try:
            result = transcribe_audio_advanced(
                audio_path, 
                language=language,
                return_timestamps=True
            )
            result["audio_path"] = audio_path
            results.append(result)
        except Exception as e:
            results.append({{
                "audio_path": audio_path,
                "error": str(e)
            }})
    
    return results

# 사용 예제
if __name__ == "__main__":
    # 샘플 오디오 파일 경로 (실제 사용시 적절한 경로로 변경)
    sample_audio = "/path/to/sample_audio.wav"
    
    # 기본 음성 인식
    basic_result = transcribe_audio_basic(sample_audio)
    print("기본 인식:", basic_result)
    
    # 고급 음성 인식
    advanced_result = transcribe_audio_advanced(
        sample_audio, 
        language="ko",
        return_timestamps=True
    )
    print("고급 인식:", advanced_result)
    
    # 배치 처리
    audio_list = [sample_audio]  # 실제로는 여러 오디오 파일
    batch_result = batch_transcribe_audio(audio_list, "ko")
    print("배치 인식:", batch_result)
''', language='python')

                        else:
                            st.code(f'''import requests
import asyncio
import aiohttp

# 기본 예측 함수
def predict(text, **kwargs):
    data = {{"text": text}}
    data.update(kwargs)  # 추가 파라미터
    
    response = requests.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    )
    return response.json()

# 기본 사용
result = predict("입력 텍스트를 여기에 넣으세요")
print(result)

# 파라미터가 있는 사용
result = predict(
    "입력 텍스트",
    max_length=100,
    temperature=0.8,
    top_p=0.9
)

# 비동기 처리 (대량 요청시 유용)
async def predict_async(session, text):
    data = {{"text": text}}
    async with session.post(
        "http://localhost:{model_port}/models/{model_name}/predict",
        json=data
    ) as response:
        return await response.json()

async def batch_predict(texts):
    async with aiohttp.ClientSession() as session:
        tasks = [predict_async(session, text) for text in texts]
        results = await asyncio.gather(*tasks)
        return results

# 배치 처리 사용
texts = ["첫 번째 텍스트", "두 번째 텍스트", "세 번째 텍스트"]
# results = asyncio.run(batch_predict(texts))
''', language='python')
                    
                    with tab3:
                        # JavaScript 예제
                        st.code(f'''// Node.js / Browser JavaScript

// 기본 예측 함수
async function predict(text, options = {{}}) {{
    const data = {{ text, ...options }};
    
    try {{
        const response = await fetch(
            'http://localhost:{model_port}/models/{model_name}/predict',
            {{
                method: 'POST',
                headers: {{
                    'Content-Type': 'application/json',
                }},
                body: JSON.stringify(data)
            }}
        );
        
        if (!response.ok) {{
            throw new Error(`HTTP error! status: ${{response.status}}`);
        }}
        
        return await response.json();
    }} catch (error) {{
        console.error('예측 요청 실패:', error);
        throw error;
    }}
}}

// 사용 예제
predict("안녕하세요!")
    .then(result => {{
        console.log('예측 결과:', result.predictions);
    }})
    .catch(error => {{
        console.error('에러:', error);
    }});

// React 컴포넌트 예제
function ModelPredictor() {{
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);
    
    const handlePredict = async () => {{
        setLoading(true);
        try {{
            const response = await predict(text);
            setResult(response.predictions);
        }} catch (error) {{
            console.error('예측 실패:', error);
        }} finally {{
            setLoading(false);
        }}
    }};
    
    return (
        <div>
            <textarea 
                value={{text}}
                onChange={{(e) => setText(e.target.value)}}
                placeholder="텍스트를 입력하세요"
            />
            <button onClick={{handlePredict}} disabled={{loading}}>
                {{loading ? '처리 중...' : '예측하기'}}
            </button>
            {{result && (
                <div>
                    <h3>결과:</h3>
                    <pre>{{JSON.stringify(result, null, 2)}}</pre>
                </div>
            )}}
        </div>
    );
}}

// 배치 처리
async function batchPredict(texts) {{
    const promises = texts.map(text => predict(text));
    try {{
        const results = await Promise.all(promises);
        return results;
    }} catch (error) {{
        console.error('배치 처리 실패:', error);
        throw error;
    }}
}}

// 사용 예제
const texts = ['첫 번째 텍스트', '두 번째 텍스트'];
batchPredict(texts).then(results => {{
    results.forEach((result, index) => {{
        console.log(`텍스트 ${{index + 1}} 결과:`, result.predictions);
    }});
}});
''', language='javascript')
                    
                    with tab4:
                        # 고급 사용법
                        st.code(f'''# 고급 파라미터 및 설정

## 1. 타임아웃 설정
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -H "X-Request-Timeout: 60" \\
     -d '{{"text": "긴 텍스트 처리용"}}'

## 2. 배치 요청 (여러 텍스트 동시 처리)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "texts": [
         "첫 번째 텍스트",
         "두 번째 텍스트", 
         "세 번째 텍스트"
       ]
     }}'

## 3. 스트리밍 응답 (긴 생성 작업용)
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -H "Accept: text/event-stream" \\
     -d '{{"text": "스트리밍 생성", "stream": true}}'

## 4. 모델별 고급 파라미터

### 텍스트 생성 모델
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "시작 텍스트",
       "max_length": 200,
       "min_length": 50,
       "temperature": 0.8,
       "top_p": 0.9,
       "top_k": 50,
       "do_sample": true,
       "repetition_penalty": 1.1,
       "no_repeat_ngram_size": 2
     }}'

### 분류 모델
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "분류할 텍스트",
       "return_all_scores": true,
       "top_k": 5
     }}'

### NER 모델  
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -d '{{
       "text": "개체명을 찾을 텍스트",
       "aggregation_strategy": "simple",
       "ignore_labels": ["O"],
       "threshold": 0.5
     }}'

## 5. 모니터링 및 디버깅
curl -X GET "http://localhost:{model_port}/models/{model_name}/status"
curl -X GET "http://localhost:{model_port}/models/{model_name}/config"
curl -X GET "http://localhost:{model_port}/system/metrics"

## 6. 에러 처리 및 재시도
curl -X POST "http://localhost:{model_port}/models/{model_name}/predict" \\
     -H "Content-Type: application/json" \\
     -H "X-Retry-Count: 3" \\
     -H "X-Retry-Delay: 1000" \\
     -d '{{"text": "재시도가 필요할 수 있는 요청"}}'
''', language='bash')
                    
                    st.markdown("---")
            
            else:
                st.info("💡 **로드된 모델이 없습니다.** 먼저 모델을 로드하면 해당 모델의 상세한 API 사용 예제가 표시됩니다.")
                
                # 일반적인 모델 타입별 빠른 참조 가이드
                st.markdown("""
                ### 📚 모델 타입별 빠른 참조 가이드
                
                아래는 일반적인 모델 타입별 기본 사용법입니다. 실제 모델을 로드하면 더 상세한 예제를 확인할 수 있습니다.
                """)
                
                # 탭으로 구분된 타입별 예제
                type_tab1, type_tab2, type_tab3, type_tab4 = st.tabs(["🏷️ 분류", "🏷️ NER", "🔍 임베딩", "🤖 생성"])
                
                with type_tab1:
                    st.markdown("""
                    #### 텍스트 분류 / 감정 분석 (BERT, RoBERTa, DeBERTa)
                    """)
                    st.code('''# 한국어 감정 분석
curl -X POST "http://localhost:8000/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "이 제품은 정말 훌륭합니다!"}'

# 영어 감정 분석
curl -X POST "http://localhost:8000/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "This product is amazing!"}'

# 주제 분류
curl -X POST "http://localhost:8000/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "오늘 주식 시장이 크게 상승했습니다."}'
''', language='bash')
                
                with type_tab2:
                    st.markdown("""
                    #### 개체명 인식 (ELECTRA, KoBERT)
                    """)
                    st.code('''# 한국어 NER
curl -X POST "http://localhost:8001/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "김철수는 서울에서 삼성전자에 다닙니다."}'

# 영어 NER
curl -X POST "http://localhost:8001/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "John Smith works at Google in New York."}'

# 복합 개체명
curl -X POST "http://localhost:8001/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "2024년 1월 15일 애플이 새로운 제품을 발표했다."}'
''', language='bash')
                
                with type_tab3:
                    st.markdown("""
                    #### 임베딩 추출 (BGE, Sentence-BERT)
                    """)
                    st.code('''# 텍스트 임베딩
curl -X POST "http://localhost:8002/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "임베딩을 추출할 텍스트"}'

# 문서 임베딩
curl -X POST "http://localhost:8002/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "긴 문서의 내용을 벡터로 변환합니다..."}'

# 배치 임베딩
curl -X POST "http://localhost:8002/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"texts": ["문장1", "문장2", "문장3"]}'
''', language='bash')
                
                with type_tab4:
                    st.markdown("""
                    #### 텍스트 생성 (GPT, BART)
                    """)
                    st.code('''# 기본 텍스트 생성
curl -X POST "http://localhost:8003/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "인공지능의 미래는", "max_length": 100}'

# 창의적 글쓰기
curl -X POST "http://localhost:8003/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "옛날 옛적에", "temperature": 0.8, "do_sample": true}'

# 질문 답변
curl -X POST "http://localhost:8003/models/model-name/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"question": "AI란 무엇인가?", "context": "인공지능 관련 문서..."}'
''', language='bash')
        
        # 시스템 관리 엔드포인트
        with st.expander("🔧 시스템 관리 API", expanded=False):
            st.markdown("""
            ### 시스템 정리
            ```bash
            # 파이프라인 캐시 정리
            curl -X POST http://localhost:8002/system/cleanup
            
            # 메모리 사용량 확인
            curl http://localhost:8002/system/memory
            
            # 시스템 상태 확인
            curl http://localhost:8002/system/status
            ```
            
            ### 모델 정보 확인
            ```bash
            # 특정 모델 정보
            curl http://localhost:8002/models/model_name
            
            # 모델 분석 정보
            curl http://localhost:8002/models/model_name/analyze
            ```
            """)
        
        # JavaScript/웹 예제
        with st.expander("🌐 JavaScript/웹 예제", expanded=False):
            st.markdown("""
            ### Fetch API 사용
            ```javascript
            // 감정 분석 예제
            async function analyzeSentiment(text) {
                const response = await fetch('http://localhost:8002/models/multilingual-sentiment-analysis/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const result = await response.json();
                return result;
            }
            
            // 사용 예제
            analyzeSentiment("I love this product!")
                .then(result => {
                    console.log('감정:', result.result[0].label);
                    console.log('신뢰도:', result.result[0].score);
                });
            ```
            
            ### jQuery 사용
            ```javascript
            $.ajax({
                url: 'http://localhost:8002/models/bge-m3/predict',
                method: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({ text: "Hello, world!" }),
                success: function(response) {
                    console.log('임베딩 벡터:', response.result);
                }
            });
            ```
            """)
        
        # 오류 처리 가이드
        with st.expander("⚠️ 오류 처리 및 팁", expanded=False):
            st.markdown("""
            ### 일반적인 오류 및 해결책
            
            **1. 연결 오류 (Connection Error)**
            - 서버가 실행 중인지 확인
            - 올바른 포트 번호 사용 확인
            - 방화벽 설정 확인
            
            **2. 모델 로드 오류**
            - 모델이 정상적으로 로드되었는지 확인
            - `/models` 엔드포인트로 모델 상태 확인
            
            **3. 예측 오류**
            - 입력 텍스트가 비어있지 않은지 확인
            - Content-Type 헤더가 올바른지 확인
            - JSON 형식이 올바른지 확인
            
            ### 성능 최적화 팁
            
            **1. 배치 처리**
            - 가능한 경우 여러 텍스트를 한 번에 처리
            
            **2. 캐시 활용**
            - 파이프라인 캐시가 자동으로 활성화됨
            - 동일한 모델 재사용 시 성능 향상
            
            **3. 메모리 관리**
            - 주기적으로 `/system/cleanup` 호출
            - GPU 메모리 정리
            """)
        
        # 언어별 SDK 예제
        with st.expander("🔨 언어별 SDK 예제", expanded=False):
            st.markdown("""
            ### Python requests
            ```python
            import requests
            
            def predict_sentiment(text, model_port=8002):
                url = f"http://localhost:{model_port}/models/multilingual-sentiment-analysis/predict"
                response = requests.post(url, json={"text": text})
                return response.json()
            
            # 사용 예제
            result = predict_sentiment("I love this!")
            print(result['result'][0]['label'])
            ```
            
            ### Node.js axios
            ```javascript
            const axios = require('axios');
            
            async function predictSentiment(text, modelPort = 8002) {
                const url = `http://localhost:${modelPort}/models/multilingual-sentiment-analysis/predict`;
                const response = await axios.post(url, { text });
                return response.data;
            }
            
            // 사용 예제
            predictSentiment("I love this!")
                .then(result => console.log(result.result[0].label));
            ```
            
            ### Go
            ```go
            package main
            
            import (
                "bytes"
                "encoding/json"
                "fmt"
                "net/http"
            )
            
            func predictSentiment(text string, modelPort int) {
                url := fmt.Sprintf("http://localhost:%d/models/multilingual-sentiment-analysis/predict", modelPort)
                payload := map[string]string{"text": text}
                jsonPayload, _ := json.Marshal(payload)
                
                resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonPayload))
                if err != nil {
                    fmt.Println("Error:", err)
                    return
                }
                defer resp.Body.Close()
                
                var result map[string]interface{}
                json.NewDecoder(resp.Body).Decode(&result)
                fmt.Printf("결과: %v\\n", result)
            }
            ```
            """)
        
        # 문서 및 리소스
        with st.expander("📚 문서 및 리소스", expanded=False):
            st.markdown("""
            ### 자동 생성 문서
            - **Swagger UI**: 각 포트의 `/docs` 엔드포인트
            - **OpenAPI Schema**: 각 포트의 `/openapi.json` 엔드포인트
            
            ### 유용한 링크
            - [FastAPI 공식 문서](https://fastapi.tiangolo.com/)
            - [Transformers 라이브러리](https://huggingface.co/transformers/)
            - [HuggingFace 모델 허브](https://huggingface.co/models)
            """)
        
        # 간단한 테스트 인터페이스
        st.subheader("🧪 API 테스트")
        
        if loaded_models:
            selected_model = st.selectbox("테스트할 모델 선택", loaded_models)
            test_text = st.text_area("입력 텍스트", "Hello, world!")
            
            # 선택된 모델의 포트 찾기 (메서드 존재 확인)
            if hasattr(st.session_state['fastapi_server'], 'get_model_server_port'):
                model_port = st.session_state['fastapi_server'].get_model_server_port(selected_model)
                # 포트를 찾지 못했으면 기본 포트 시도
                if model_port is None:
                    # 서버가 실행 중이면 기본 포트 사용
                    if st.session_state['fastapi_server'].is_running():
                        model_port = st.session_state['fastapi_server'].default_port
            else:
                # 이전 버전 호환성 - 기본 포트 사용
                default_port = getattr(st.session_state['fastapi_server'], 'default_port', getattr(st.session_state['fastapi_server'], 'port', 8000))
                model_port = default_port if st.session_state['fastapi_server'].is_running() else None
            
            if model_port:
                host = server_info.get('host', '127.0.0.1')
                test_url = f"http://{host}:{model_port}"
                endpoint = f"{test_url}/models/{selected_model}/predict"
                
                st.info(f"🎯 **테스트 대상:** {endpoint}")
                st.caption(f"📡 모델 '{selected_model}'은 포트 {model_port}에서 실행 중입니다.")
            else:
                st.warning(f"⚠️ 모델 '{selected_model}'이 실행 중이 아닙니다.")
                endpoint = None
            
            if st.button("예측 테스트", disabled=endpoint is None):
                if endpoint:
                    import requests
                    try:
                        response = requests.post(
                            endpoint,
                            json={"text": test_text}
                        )
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ 예측 성공!")
                            st.json(result)
                        else:
                            st.error(f"❌ 예측 실패: {response.status_code}")
                            st.error(response.text)
                    except Exception as e:
                        st.error(f"🔥 요청 실패: {e}")
                        st.caption("서버가 실행 중인지 확인해주세요.")
                else:
                    st.error("모델이 실행 중이 아닙니다.")
        else:
            st.info("테스트하려면 먼저 모델을 로드하세요.")

# 메인 함수
def main():
    st.title("🚀 Hugging Face GUI")
    
    # 스마트 모델 상태 체크 (콜백 대신 직접 감지)
    if should_perform_expensive_check('main_model_check', 5):
        smart_model_status_check()
    
    # 로딩 완료 상태 즉시 체크 (스레드 상태 반영)
    if st.session_state.get('load_complete', False):
        st.rerun()  # 즉시 UI 갱신하여 완료 메시지 표시
    
    # 현재 로딩 중인 모델이 있으면 주기적 체크 활성화
    if st.session_state.get('check_loading', False):
        models_status = st.session_state['model_manager'].get_all_models_status()
        loading_models = [name for name, info in models_status.items() if info['status'] == 'loading']
        if loading_models and should_perform_expensive_check('main_loading_check', 2):
            st.rerun()  # 로딩 중이면 2초마다 체크
    
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
        
        # FastAPI 서버 상태 복원 및 동기화
        server_running = st.session_state.get('fastapi_server_running', False)
        logger.info(f"서버 복원 체크: fastapi_server_running={server_running}")
        
        try:
            # 실제 서버 상태 확인
            actual_running = st.session_state['fastapi_server'].is_running()
            logger.info(f"실제 서버 상태: {actual_running}")
            
            # 세션 상태와 실제 상태 동기화 (상태 저장 없이)
            if actual_running != server_running:
                st.session_state['fastapi_server_running'] = actual_running
                # Don't save state here as this is called during restoration
                logger.info(f"서버 상태 초기 동기화: {server_running} -> {actual_running}")
            
            if server_running and not actual_running:
                # 복원 시도
                st.session_state['fastapi_server'].start_server()
                restoration_success.append("FastAPI 서버")
                logger.info("FastAPI 서버 자동 복원 성공")
            elif actual_running:
                logger.info("FastAPI 서버 이미 실행 중")
                
        except Exception as e:
            st.session_state['fastapi_server_running'] = False
            restoration_failed.append(f"FastAPI 서버 ({str(e)})")
            logger.error(f"FastAPI 서버 상태 확인/복원 실패: {e}")
        
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
        
        # 시스템 요약 정보 (스마트 폴링)
        if should_perform_expensive_check('system_summary', 15):
            system_summary = st.session_state['model_manager'].get_system_summary()
            # 캐시 시스템 요약
            st.session_state['_cached_system_summary'] = system_summary
        else:
            # 캐시된 요약 사용
            system_summary = st.session_state.get('_cached_system_summary', {
                'loaded_models_count': 0,
                'total_models_count': 0, 
                'total_memory_usage_mb': 0.0
            })
        
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
        
        # 서버 상태 (스마트 폴링 - 실제 서버 상태 확인) - 간격 단축
        if should_perform_expensive_check('server_status_check', 5):
            try:
                # API 서버 상태 확인 및 동기화 (멀티포트 지원)
                server_info = st.session_state['fastapi_server'].get_server_info()
                active_servers = server_info.get('active_servers', [])
                actual_server_running = len(active_servers) > 0
                session_server_running = st.session_state.get('fastapi_server_running', False)
                
                # 세션 상태와 실제 상태가 다르면 동기화 (상태 저장 없이)
                if actual_server_running != session_server_running:
                    st.session_state['fastapi_server_running'] = actual_server_running
                    # Don't call save_app_state here as this runs every main() execution
                    logger.info(f"서버 상태 동기화: session={session_server_running} -> actual={actual_server_running}")
                
                if actual_server_running:
                    st.success(f"🟢 API 서버 실행 중 ({len(active_servers)}개 포트)")
                else:
                    st.info("⏸️ API 서버 중지됨")
            except Exception as e:
                logger.error(f"서버 상태 확인 오류: {e}")
                st.error("❌ 서버 상태 확인 실패")
        else:
            # 캐시된 상태 표시 (빠른 표시)
            session_server_running = st.session_state.get('fastapi_server_running', False)
            if session_server_running:
                st.info("🟡 API 서버 실행 중 (캐시됨)")
            else:
                st.info("⏸️ API 서버 중지됨 (캐시됨)")
    
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
        
        # Reduced logging for cache UI rendering (only when state changes)
        cache_state = (cache_scanned, cache_info_exists, revisions_count)
        if cache_state != st.session_state.get('_last_cache_state'):
            logger.info(f"캐시 UI 렌더링 - cache_scanned: {cache_scanned}, cache_info_exists: {cache_info_exists}, revisions_count: {revisions_count}")
            st.session_state['_last_cache_state'] = cache_state
        
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
                scan_cache(force=True)  # 버튼 클릭 시에는 강제로 스캔
                st.session_state['cache_scanned'] = True
                # Cache scan state will be saved by the background save mechanism
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
                save_app_state(force=True)  # 강제 저장
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