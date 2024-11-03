import os
import streamlit as st
from huggingface_hub import HfApi, scan_cache_dir
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode

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

# Streamlit 페이지 설정
st.set_page_config(page_title="Hugging Face 캐시 및 사용자 관리", layout="wide")

# 세션 상태 초기화
if 'token' not in st.session_state:
    st.session_state['token'] = load_login_token()

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = st.session_state['token'] is not None

if 'cache_info' not in st.session_state:
    st.session_state['cache_info'] = None

if 'revisions_df' not in st.session_state:
    st.session_state['revisions_df'] = pd.DataFrame()

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

# 메인 함수
def main():
    st.title("Hugging Face 캐시 및 사용자 관리")

    # 탭 생성
    tabs = st.tabs(["로그인 및 사용자 정보", "캐시 관리"])

    # 첫 번째 탭: 로그인 / 로그아웃 및 사용자 정보
    with tabs[0]:
        st.subheader("로그인 및 사용자 정보")

        if not st.session_state['logged_in']:
            st.text_input("Hugging Face 토큰:", key='input_token')
            st.button("로그인", on_click=login)
        else:
            st.write("로그인 상태 유지됨")
            st.button("로그아웃", on_click=logout)

        if st.session_state['logged_in']:
            st.button("현재 사용자 정보", on_click=show_whoami)
            st.button("환경 정보 보기", on_click=show_env)

    # 두 번째 탭: 캐시 관리
    with tabs[1]:
        st.subheader("캐시 관리")
        if st.button("캐시 스캔"):
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
                        st.rerun()  # 페이지 갱신
            else:
                st.write("선택된 항목: 0개, 총 용량: 0.00 MB")

# 프로그램 시작 시 로그인 상태 확인 및 캐시 스캔
if st.session_state['logged_in']:
    api.set_access_token(st.session_state['token'])
    if st.session_state['cache_info'] is None:
        scan_cache()

main()
