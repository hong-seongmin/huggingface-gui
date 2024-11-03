import os
import streamlit as st
from huggingface_hub import HfApi, scan_cache_dir
from datetime import datetime

# 로그인 상태를 저장할 파일 경로 설정
LOGIN_FILE = "login_token.txt"

# Hugging Face API 인스턴스 생성
api = HfApi()

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

# 캐시 정보 스캔
@st.cache_data
def get_cache_info():
    return scan_cache_dir()


# 캐시 데이터를 정리
def process_cache_info(cache_info):
    revisions = []
    for repo in cache_info.repos:
        for revision in repo.revisions:
            # last_modified를 datetime 형식으로 변환
            last_modified = (
                datetime.fromtimestamp(revision.last_modified).strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(revision.last_modified, (int, float))
                else "Unknown"
            )
            rev_info = {
                "repo_id": repo.repo_id,
                "revision": revision.commit_hash[:7],
                "size_MB": revision.size_on_disk / (1024 ** 2),
                "last_modified": last_modified,
            }
            revisions.append(rev_info)
    return revisions

# 선택한 캐시 삭제
def delete_selected_revisions(selected_revisions):
    delete_strategy = cache_info.delete_revisions(*selected_revisions)
    delete_strategy.execute()

# 초기 설정
st.set_page_config(page_title="Hugging Face 캐시 및 사용자 관리", layout="wide")

# 사이드바에 로그인 상태 표시
st.sidebar.header("로그인 상태")
token = load_login_token()
if token:
    st.sidebar.success("로그인 상태 유지됨")
else:
    st.sidebar.warning("로그인 필요")

# 탭 생성
tab_user, tab_cache = st.tabs(["로그인 및 사용자 정보", "캐시 관리"])

### 첫 번째 탭: 로그인 및 사용자 정보 ###
with tab_user:
    st.header("로그인 및 사용자 정보")

    # 로그인 섹션
    with st.expander("로그인 / 로그아웃"):
        token_input = st.text_input("Hugging Face 토큰:", type="password")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("로그인"):
                if token_input:
                    try:
                        api.set_access_token(token_input)
                        save_login_token(token_input)
                        st.success("로그인 성공!")
                        token = token_input
                    except Exception as e:
                        st.error(f"로그인에 실패했습니다: {e}")
                else:
                    st.error("유효한 토큰을 입력하세요.")
        with col2:
            if st.button("로그아웃"):
                api.set_access_token(None)
                delete_login_token()
                st.success("로그아웃되었습니다.")
                token = None

    # 사용자 정보 섹션
    with st.expander("사용자 정보 및 환경 정보"):
        col1, col2 = st.columns(2)
        if st.button("현재 사용자 정보"):
            try:
                user_info = api.whoami()
                st.json(user_info)
            except Exception as e:
                st.error(f"사용자 정보를 불러오지 못했습니다: {e}")

        if st.button("환경 정보 보기"):
            try:
                env_info = api.whoami()
                st.write(f"환경 정보: {env_info}")
            except Exception as e:
                st.error(f"환경 정보를 불러오지 못했습니다: {e}")

### 두 번째 탭: 캐시 관리 ###
with tab_cache:
    st.header("캐시 관리")

    # 캐시 스캔 버튼
    if st.button("캐시 스캔"):
        cache_info = get_cache_info()
        revisions = process_cache_info(cache_info)
        st.session_state.revisions = revisions
    else:
        cache_info = get_cache_info()
        revisions = process_cache_info(cache_info)
        st.session_state.revisions = revisions

    # 캐시 데이터 로드
    revisions = st.session_state.get('revisions', [])

    if not revisions:
        st.info("캐시가 비어 있습니다.")
    else:
        # 캐시 데이터 표시
        st.subheader("캐시 항목")
        # 선택된 항목을 저장할 리스트
        selected_revisions = []

        # 표 형식으로 캐시 데이터 표시
        for i, rev in enumerate(revisions):
            col1, col2, col3, col4, col5 = st.columns([1, 3, 2, 2, 3])
            with col1:
                selected = st.checkbox("", key=f"select_{i}")
                if selected:
                    selected_revisions.append(rev['revision'])
            with col2:
                st.text(rev['repo_id'])
            with col3:
                st.text(rev['revision'])
            with col4:
                st.text(f"{rev['size_MB']:.2f} MB")
            with col5:
                st.text(rev['last_modified'])

        # 전체 선택 / 해제 버튼
        st.markdown("---")
        col_all1, col_all2 = st.columns(2)
        with col_all1:
            if st.button("전체 선택"):
                for i in range(len(revisions)):
                    st.session_state[f"select_{i}"] = True
        with col_all2:
            if st.button("전체 해제"):
                for i in range(len(revisions)):
                    st.session_state[f"select_{i}"] = False

        # 선택된 항목과 총 용량 표시
        selected_count = len(selected_revisions)
        total_size = sum([rev['size_MB'] for rev in revisions if rev['revision'] in selected_revisions])
        st.write(f"선택된 항목: {selected_count}개, 총 용량: {total_size:.2f} MB")

        # 선택된 캐시 삭제 버튼
        if st.button("선택한 캐시 삭제"):
            if selected_revisions:
                confirm = st.warning(f"{selected_count}개의 수정 버전을 삭제하시겠습니까?", icon="⚠️")
                if st.button("삭제 실행"):
                    try:
                        delete_selected_revisions(selected_revisions)
                        st.success("선택한 캐시가 삭제되었습니다.")
                        # 캐시 재스캔
                        cache_info = get_cache_info()
                        revisions = process_cache_info(cache_info)
                        st.session_state.revisions = revisions
                    except Exception as e:
                        st.error(f"캐시 삭제에 실패했습니다: {e}")
            else:
                st.info("삭제할 항목을 선택하세요.")
