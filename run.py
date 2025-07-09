import os, inspect
os.chdir(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
import customtkinter as ctk
from huggingface_hub import HfApi, scan_cache_dir
from tkinter import messagebox, ttk
import tkinter as tk
import threading
import time
from datetime import datetime
import json

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
model_manager = MultiModelManager()
system_monitor = SystemMonitor()
fastapi_server = FastAPIServer(model_manager)
model_analyzer = ComprehensiveModelAnalyzer()

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

# GUI 초기 설정
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# 메인 윈도우 생성
app = ctk.CTk()
app.title("Hugging Face GUI")
app.geometry("1200x800")

# 전역 변수들
cache_info = None
revisions = []
sort_orders = {}
checkboxes = []
current_model_analysis = None
system_data_labels = {}
model_listbox = None
loaded_models_listbox = None
auto_refresh_interval = 0
auto_refresh_timer = None

# 시스템 모니터링 데이터 업데이트 콜백
def update_system_display(data):
    """시스템 모니터링 데이터 업데이트"""
    try:
        # CPU 정보 업데이트
        if 'cpu_percent_label' in system_data_labels:
            system_data_labels['cpu_percent_label'].configure(
                text=f"CPU: {data['cpu']['percent']:.1f}%"
            )
        
        # 메모리 정보 업데이트
        if 'memory_percent_label' in system_data_labels:
            memory_gb = data['memory']['used'] / (1024**3)
            total_gb = data['memory']['total'] / (1024**3)
            system_data_labels['memory_percent_label'].configure(
                text=f"Memory: {data['memory']['percent']:.1f}% ({memory_gb:.1f}/{total_gb:.1f} GB)"
            )
        
        # GPU 정보 업데이트
        if 'gpu_info_label' in system_data_labels:
            if data['gpu']:
                avg_gpu = sum(gpu['load'] for gpu in data['gpu']) / len(data['gpu'])
                system_data_labels['gpu_info_label'].configure(
                    text=f"GPU: {avg_gpu:.1f}% (Average)"
                )
            else:
                system_data_labels['gpu_info_label'].configure(text="GPU: N/A")
        
        # 디스크 정보 업데이트
        if 'disk_info_label' in system_data_labels:
            disk_gb = data['disk']['used'] / (1024**3)
            total_disk_gb = data['disk']['total'] / (1024**3)
            system_data_labels['disk_info_label'].configure(
                text=f"Disk: {data['disk']['percent']:.1f}% ({disk_gb:.1f}/{total_disk_gb:.1f} GB)"
            )
        
        # 시간 정보 업데이트
        if 'time_label' in system_data_labels:
            system_data_labels['time_label'].configure(
                text=f"Last Update: {data['timestamp'].strftime('%H:%M:%S')}"
            )
        
    except Exception as e:
        print(f"Display update error: {e}")

# 시스템 모니터 콜백 등록
system_monitor.add_callback(lambda data: app.after(0, update_system_display, data))

# 모델 관리 콜백
def model_status_callback(model_name, event_type, data):
    """모델 상태 변경 콜백"""
    def update_ui():
        if event_type == "loading_started":
            messagebox.showinfo("알림", f"모델 '{model_name}' 로드 시작")
        elif event_type == "downloading":
            messagebox.showinfo("다운로드", f"HuggingFace 모델 '{data.get('model_id', model_name)}' 다운로드 중...")
            # HuggingFace 모델 다운로드 시 캐시 자동 갱신
            scan_cache()
        elif event_type == "loading_success":
            success_msg = f"모델 '{model_name}' 로드 완료!"
            if 'original_path' in data and 'actual_path' in data:
                if data['original_path'] != data['actual_path']:
                    success_msg += f"\n다운로드 완료: {data['original_path']}"
            messagebox.showinfo("성공", success_msg)
            update_model_lists()
            # 로드 완료 후 캐시 갱신
            scan_cache()
        elif event_type == "loading_error":
            messagebox.showerror("오류", f"모델 '{model_name}' 로드 실패: {data.get('error', 'Unknown error')}")
            update_model_lists()
        elif event_type == "unloaded":
            messagebox.showinfo("알림", f"모델 '{model_name}' 언로드 완료")
            update_model_lists()
    
    app.after(0, update_ui)

# 모델 매니저 콜백 등록
model_manager.add_callback(model_status_callback)

# 캐시 정보 스캔 및 화면에 표시하는 기능
def scan_cache():
    global cache_info, revisions, sort_orders
    cache_info = scan_cache_dir()

    # 캐시 데이터 수집
    revisions = []
    for repo in cache_info.repos:
        for revision in repo.revisions:
            rev_info = {
                "repo_id": repo.repo_id,
                "revision": revision.commit_hash,
                "size": revision.size_on_disk,
                "last_modified": revision.last_modified,
            }
            revisions.append(rev_info)
    
    # 각 열의 초기 정렬 순서 (True: 오름차순, False: 내림차순)
    sort_orders = {
        "repo_id": True,
        "revision": True,
        "size": False,
        "last_modified": True,
    }

    display_cache_items()
    
    # 캐시된 모델 목록 갱신
    refresh_cached_models()

# 선택된 항목의 개수와 총 용량을 합산하여 표시하는 함수
def update_selection_summary():
    selected_count = 0
    total_size = 0
    for var, rev in checkboxes:
        if var.get():
            selected_count += 1
            total_size += rev['size']

    label_selection_summary.configure(
        text=f"선택된 항목: {selected_count}개, 총 용량: {total_size / (1024 ** 2):.2f} MB"
    )

# 캐시 항목을 표 형태로 표시하는 기능
def display_cache_items():
    for widget in cache_table_frame.winfo_children():
        widget.destroy()

    # 표 헤더 생성
    headers = ["선택", "Repo ID", "Revision", "Size (MB)", "Last Modified"]
    columns = ["select", "repo_id", "revision", "size", "last_modified"]
    column_widths = [50, 200, 150, 100, 200]

    for col, (header, col_name, width) in enumerate(zip(headers, columns, column_widths)):
        label = ctk.CTkLabel(master=cache_table_frame, text=header, font=("Arial", 12, "bold"))
        label.grid(row=0, column=col, padx=5, pady=5)
        label.bind("<Button-1>", lambda e, cn=col_name: sort_by_column(cn))
        label.configure(width=width)
        label.configure(cursor="hand2")

    global checkboxes
    checkboxes = []

    # 데이터 표시
    for row, rev in enumerate(revisions, start=1):
        var = ctk.BooleanVar()
        cb = ctk.CTkCheckBox(master=cache_table_frame, text="", variable=var, command=update_selection_summary)
        cb.grid(row=row, column=0, padx=5, pady=5)
        checkboxes.append((var, rev))

        # Repo ID
        repo_id_label = ctk.CTkLabel(master=cache_table_frame, text=rev['repo_id'])
        repo_id_label.grid(row=row, column=1, padx=5, pady=5)
        repo_id_label.configure(width=column_widths[1])

        # Revision
        revision_label = ctk.CTkLabel(master=cache_table_frame, text=rev['revision'][:7])
        revision_label.grid(row=row, column=2, padx=5, pady=5)
        revision_label.configure(width=column_widths[2])

        # Size (MB)
        size_label = ctk.CTkLabel(master=cache_table_frame, text=f"{rev['size'] / (1024 ** 2):.2f}")
        size_label.grid(row=row, column=3, padx=5, pady=5)
        size_label.configure(width=column_widths[3])

        # Last Modified
        last_modified_label = ctk.CTkLabel(master=cache_table_frame, text=str(rev['last_modified']))
        last_modified_label.grid(row=row, column=4, padx=5, pady=5)
        last_modified_label.configure(width=column_widths[4])

    # 전체 선택/해제 버튼
    select_all_btn = ctk.CTkButton(master=cache_table_frame, text="전체 선택", command=select_all)
    select_all_btn.grid(row=len(revisions) + 1, column=0, padx=5, pady=5)

    deselect_all_btn = ctk.CTkButton(master=cache_table_frame, text="전체 해제", command=deselect_all)
    deselect_all_btn.grid(row=len(revisions) + 1, column=1, padx=5, pady=5)

    update_selection_summary()

# 열 정렬 함수
def sort_by_column(column_name):
    global revisions, sort_orders
    reverse = not sort_orders[column_name]
    sort_orders[column_name] = reverse

    if column_name == "size":
        revisions.sort(key=lambda x: x[column_name], reverse=reverse)
    elif column_name == "last_modified":
        revisions.sort(key=lambda x: x[column_name] or '', reverse=reverse)
    else:
        revisions.sort(key=lambda x: x[column_name].lower(), reverse=reverse)

    display_cache_items()

# 전체 선택 기능
def select_all():
    for var, _ in checkboxes:
        var.set(True)
    update_selection_summary()

# 전체 해제 기능
def deselect_all():
    for var, _ in checkboxes:
        var.set(False)
    update_selection_summary()

# 선택한 캐시 항목 삭제
def delete_selected():
    selected_revisions = [rev['revision'] for var, rev in checkboxes if var.get()]
    if not selected_revisions:
        messagebox.showinfo("알림", "삭제할 항목을 선택하세요.")
        return

    confirm = messagebox.askyesno("확인", f"{len(selected_revisions)}개의 수정 버전을 삭제하시겠습니까?")
    if confirm:
        delete_strategy = cache_info.delete_revisions(*selected_revisions)
        delete_strategy.execute()
        messagebox.showinfo("완료", "선택한 캐시가 삭제되었습니다.")
        scan_cache()
    else:
        messagebox.showinfo("취소", "삭제가 취소되었습니다.")

# 로그인 기능
def login():
    token = entry_token.get().strip()
    if token:
        try:
            api.set_access_token(token)
            save_login_token(token)
            label_status.configure(text="로그인 성공!")
        except Exception as e:
            messagebox.showerror("로그인 오류", f"로그인에 실패했습니다: {e}")
    else:
        messagebox.showerror("오류", "유효한 토큰을 입력하세요.")

# 로그아웃 기능
def logout():
    api.set_access_token(None)
    delete_login_token()
    label_status.configure(text="로그아웃되었습니다.")

# 환경 정보 출력
def show_env():
    try:
        env_info = api.whoami()
        label_env_info.configure(text=f"환경 정보: {env_info}")
    except Exception as e:
        label_env_info.configure(text=f"환경 정보를 불러오지 못했습니다: {e}")

# 현재 사용자 정보 출력
def show_whoami():
    try:
        user_info = api.whoami()
        label_user_info.configure(text=f"사용자: {user_info['name']}")
    except Exception as e:
        label_user_info.configure(text=f"사용자 정보를 불러오지 못했습니다: {e}")

# 모델 분석 기능
def analyze_model():
    model_path = entry_model_path.get().strip()
    if not model_path:
        messagebox.showerror("오류", "모델 경로를 입력하세요.")
        return
    
    def analyze_thread():
        try:
            analysis = model_manager.analyze_model(model_path)
            if 'error' in analysis:
                app.after(0, lambda: messagebox.showerror("분석 오류", f"모델 분석 실패: {analysis['error']}"))
            else:
                app.after(0, display_model_analysis, analysis)
        except Exception as e:
            app.after(0, lambda: messagebox.showerror("분석 오류", f"모델 분석 실패: {e}"))
    
    threading.Thread(target=analyze_thread, daemon=True).start()
    messagebox.showinfo("알림", "모델 분석을 시작합니다...")

# 모델 분석 결과 표시
def display_model_analysis(analysis):
    global current_model_analysis
    current_model_analysis = analysis
    
    # 분석 결과 윈도우 생성
    analysis_window = ctk.CTkToplevel(app)
    analysis_window.title("모델 분석 결과")
    analysis_window.geometry("800x600")
    
    # 스크롤 프레임 생성
    scrollable_frame = ctk.CTkScrollableFrame(analysis_window, width=750, height=550)
    scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    # 원본 경로와 실제 경로 표시
    if 'original_path' in analysis and 'actual_path' in analysis:
        if analysis['original_path'] != analysis['actual_path']:
            path_info_frame = ctk.CTkFrame(scrollable_frame)
            path_info_frame.pack(fill="x", pady=5)
            
            ctk.CTkLabel(path_info_frame, text=f"🔗 HuggingFace 모델 ID: {analysis['original_path']}", text_color="blue").pack(anchor="w", padx=10, pady=2)
            ctk.CTkLabel(path_info_frame, text=f"📁 로컬 캐시 경로: {analysis['actual_path']}", text_color="gray").pack(anchor="w", padx=10, pady=2)
    
    # 모델 요약 정보
    if 'model_summary' in analysis:
        summary = analysis['model_summary']
        
        ctk.CTkLabel(scrollable_frame, text="📊 모델 기본 정보", font=("Arial", 16, "bold")).pack(pady=10)
        
        basic_info_frame = ctk.CTkFrame(scrollable_frame)
        basic_info_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(basic_info_frame, text=f"🏗️ 모델 타입: {summary.get('model_type', 'unknown')}", font=("Arial", 12, "bold")).pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(basic_info_frame, text=f"📊 파라미터 수: {summary.get('total_parameters', 0):,}").pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(basic_info_frame, text=f"💾 모델 크기: {summary.get('model_size_mb', 0):.1f} MB").pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(basic_info_frame, text=f"📖 어휘 크기: {summary.get('vocabulary_size', 0):,}").pack(anchor="w", padx=10, pady=2)
        
        # 상세 정보 섹션
        ctk.CTkLabel(scrollable_frame, text="🔍 모델 상세 정보", font=("Arial", 14, "bold")).pack(pady=(20, 10))
        
        detail_frame = ctk.CTkFrame(scrollable_frame)
        detail_frame.pack(fill="x", pady=5)
        
        config = summary.get('detailed_config', {})
        ctk.CTkLabel(detail_frame, text=f"• 아키텍처: {config.get('architecture', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 히든 사이즈: {config.get('hidden_size', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 어텐션 헤드: {config.get('num_attention_heads', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 레이어 수: {config.get('num_hidden_layers', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 최대 위치: {config.get('max_position_embeddings', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 드롭아웃: {config.get('dropout', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 중간 레이어 크기: {config.get('intermediate_size', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        ctk.CTkLabel(detail_frame, text=f"• 활성화 함수: {config.get('activation_function', 'N/A')}").pack(anchor="w", padx=10, pady=1)
        
        # 지원 태스크 섹션
        if summary.get('supported_tasks'):
            ctk.CTkLabel(scrollable_frame, text="🎯 지원 태스크 및 사용 방법", font=("Arial", 14, "bold")).pack(pady=(20, 10))
            
            usage_examples = summary.get('usage_examples', {})
            
            for task in summary['supported_tasks']:
                task_frame = ctk.CTkFrame(scrollable_frame)
                task_frame.pack(fill="x", pady=5)
                
                # 태스크 이름
                ctk.CTkLabel(task_frame, text=f"📋 {task}", font=("Arial", 12, "bold"), text_color="green").pack(anchor="w", padx=10, pady=5)
                
                if task in usage_examples:
                    example = usage_examples[task]
                    
                    # 설명
                    ctk.CTkLabel(task_frame, text=f"설명: {example['description']}", text_color="gray").pack(anchor="w", padx=20, pady=2)
                    
                    # 코드 예시 (간단하게)
                    code_lines = example['example_code'].split('\n')[:3]  # 처음 3줄만 표시
                    code_preview = '\n'.join(code_lines) + "..."
                    ctk.CTkLabel(task_frame, text=f"코드 예시:\n{code_preview}", font=("Consolas", 10), text_color="blue").pack(anchor="w", padx=20, pady=2)
                    
                    # 입력/출력 예시
                    ctk.CTkLabel(task_frame, text=f"입력: {example['example_input']}", font=("Consolas", 9), text_color="orange").pack(anchor="w", padx=20, pady=1)
                    ctk.CTkLabel(task_frame, text=f"출력: {example['expected_output'][:50]}...", font=("Consolas", 9), text_color="purple").pack(anchor="w", padx=20, pady=1)
                else:
                    ctk.CTkLabel(task_frame, text=f"✅ {task} 태스크를 지원합니다.", text_color="green").pack(anchor="w", padx=20, pady=2)
    
    # 발견된 파일들 (접기/펼치기 버튼)
    if 'files_found' in analysis:
        files_button_frame = ctk.CTkFrame(scrollable_frame)
        files_button_frame.pack(fill="x", pady=(20, 5))
        
        files_visible = ctk.BooleanVar(value=False)
        
        def toggle_files():
            if files_visible.get():
                files_content_frame.pack_forget()
                files_toggle_btn.configure(text="📁 발견된 파일들 보기 ▼")
                files_visible.set(False)
            else:
                files_content_frame.pack(fill="x", pady=5)
                files_toggle_btn.configure(text="📁 발견된 파일들 숨기기 ▲")
                files_visible.set(True)
        
        files_toggle_btn = ctk.CTkButton(files_button_frame, text="📁 발견된 파일들 보기 ▼", command=toggle_files)
        files_toggle_btn.pack(padx=10, pady=5)
        
        files_content_frame = ctk.CTkFrame(scrollable_frame)
        analysis_results = analysis.get('analysis_results', {})
        
        # 발견된 파일들
        if analysis['files_found']:
            found_label = ctk.CTkLabel(files_content_frame, text="✅ 발견된 파일들", font=("Arial", 12, "bold"), text_color="green")
            found_label.pack(anchor="w", padx=10, pady=(10, 5))
            
            for file in analysis['files_found']:
                # 파일별 상세 정보 표시
                file_frame = ctk.CTkFrame(files_content_frame)
                file_frame.pack(fill="x", padx=10, pady=2)
                
                # 파일 이름
                ctk.CTkLabel(file_frame, text=f"🔍 {file}", font=("Arial", 11, "bold"), text_color="green").pack(anchor="w", padx=5, pady=2)
                
                # 파일 분석 결과
                if file in analysis_results:
                    file_data = analysis_results[file]
                    
                    if 'error' in file_data:
                        ctk.CTkLabel(file_frame, text=f"❌ 오류: {file_data['error']}", text_color="red").pack(anchor="w", padx=15, pady=1)
                    else:
                        # 파일별 상세 정보
                        if file == 'config.json':
                            ctk.CTkLabel(file_frame, text=f"  • 모델 타입: {file_data.get('model_type', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 아키텍처: {', '.join(file_data.get('architectures', ['N/A']))}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 어휘 크기: {file_data.get('vocab_size', 'N/A'):,}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 파라미터 추정: {file_data.get('model_parameters', 0):,}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file == 'tokenizer_config.json':
                            ctk.CTkLabel(file_frame, text=f"  • 토크나이저: {file_data.get('tokenizer_class', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 최대 길이: {file_data.get('model_max_length', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 패딩 방향: {file_data.get('padding_side', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file == 'tokenizer.json':
                            ctk.CTkLabel(file_frame, text=f"  • 버전: {file_data.get('version', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 어휘 크기: {file_data.get('vocab_size', 'N/A'):,}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file == 'vocab.txt':
                            ctk.CTkLabel(file_frame, text=f"  • 총 어휘: {file_data.get('vocab_size', 'N/A'):,}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 특수 토큰: {len(file_data.get('special_tokens_found', []))}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file == 'special_tokens_map.json':
                            token_count = len(file_data) if file_data else 0
                            ctk.CTkLabel(file_frame, text=f"  • 특수 토큰 수: {token_count}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file in ['pytorch_model.bin', 'model.safetensors']:
                            ctk.CTkLabel(file_frame, text=f"  • 파일 크기: {file_data.get('file_size_mb', 'N/A'):.2f} MB", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 총 파라미터: {file_data.get('total_parameters', 'N/A'):,}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            if file == 'safetensors':
                                ctk.CTkLabel(file_frame, text=f"  • 텐서 개수: {file_data.get('tensor_count', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file == 'generation_config.json':
                            ctk.CTkLabel(file_frame, text=f"  • 최대 길이: {file_data.get('max_length', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                            ctk.CTkLabel(file_frame, text=f"  • 온도: {file_data.get('temperature', 'N/A')}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                        
                        elif file == 'merges.txt':
                            ctk.CTkLabel(file_frame, text=f"  • 병합 규칙: {file_data.get('num_merges', 'N/A'):,}", text_color="gray").pack(anchor="w", padx=15, pady=1)
                else:
                    ctk.CTkLabel(file_frame, text="  • 분석 결과 없음", text_color="orange").pack(anchor="w", padx=15, pady=1)
        
        # 누락된 파일들
        if analysis['files_missing']:
            missing_label = ctk.CTkLabel(files_content_frame, text="❌ 누락된 파일들", font=("Arial", 12, "bold"), text_color="red")
            missing_label.pack(anchor="w", padx=10, pady=(10, 5))
            
            for file in analysis['files_missing']:
                file_frame = ctk.CTkFrame(files_content_frame)
                file_frame.pack(fill="x", padx=10, pady=2)
                
                ctk.CTkLabel(file_frame, text=f"❌ {file}", text_color="red").pack(anchor="w", padx=5, pady=2)
                
                # 누락된 파일 설명
                if file == 'pytorch_model.bin':
                    ctk.CTkLabel(file_frame, text="  • PyTorch 가중치 파일 (safetensors로 대체)", text_color="gray").pack(anchor="w", padx=15, pady=1)
                elif file == 'generation_config.json':
                    ctk.CTkLabel(file_frame, text="  • 텍스트 생성 설정 파일", text_color="gray").pack(anchor="w", padx=15, pady=1)
                elif file == 'merges.txt':
                    ctk.CTkLabel(file_frame, text="  • BPE 토크나이저 병합 규칙", text_color="gray").pack(anchor="w", padx=15, pady=1)
                else:
                    ctk.CTkLabel(file_frame, text="  • 선택적 파일", text_color="gray").pack(anchor="w", padx=15, pady=1)
    
    # 권장사항
    if 'recommendations' in analysis and analysis['recommendations']:
        ctk.CTkLabel(scrollable_frame, text="💡 권장사항", font=("Arial", 14, "bold")).pack(pady=(20, 10))
        
        rec_frame = ctk.CTkFrame(scrollable_frame)
        rec_frame.pack(fill="x", pady=5)
        
        for i, rec in enumerate(analysis['recommendations'], 1):
            ctk.CTkLabel(rec_frame, text=f"{i}. {rec}", text_color="orange").pack(anchor="w", padx=10, pady=2)

# 모델 로드 기능
def load_model():
    model_path = entry_model_path.get().strip()
    
    if not model_path:
        messagebox.showerror("오류", "모델 경로를 입력하세요.")
        return
    
    # 모델 이름 자동 생성
    model_name = ""
    
    # 비동기 로드 시작
    model_manager.load_model_async(model_name, model_path)
    messagebox.showinfo("알림", f"모델 로드를 시작합니다... (이름: 자동 생성)")

# 모델 언로드 기능
def unload_model():
    if not loaded_models_listbox.curselection():
        messagebox.showinfo("알림", "언로드할 모델을 선택하세요.")
        return
    
    selected_idx = loaded_models_listbox.curselection()[0]
    model_name = loaded_models_listbox.get(selected_idx)
    
    if model_manager.unload_model(model_name):
        messagebox.showinfo("성공", f"모델 '{model_name}' 언로드 완료!")
        update_model_lists()
    else:
        messagebox.showerror("오류", f"모델 '{model_name}' 언로드 실패!")

# 모델 상세 정보 보기
def show_model_details():
    if not model_listbox.curselection():
        messagebox.showinfo("알림", "모델을 선택하세요.")
        return
    
    selected_idx = model_listbox.curselection()[0]
    selected_text = model_listbox.get(selected_idx)
    
    # 모델 이름 추출 (아이콘과 상태 정보 제거)
    model_name = selected_text.split(" | ")[0].split(" ", 1)[1]  # 아이콘 제거
    
    model_info = model_manager.get_model_info(model_name)
    if not model_info:
        messagebox.showerror("오류", "모델 정보를 찾을 수 없습니다.")
        return
    
    # 상세 정보 창 생성
    details_window = ctk.CTkToplevel(app)
    details_window.title(f"모델 상세 정보 - {model_name}")
    details_window.geometry("600x500")
    
    # 스크롤 프레임
    details_scrollable = ctk.CTkScrollableFrame(details_window, width=550, height=450)
    details_scrollable.pack(fill="both", expand=True, padx=10, pady=10)
    
    # 기본 정보
    ctk.CTkLabel(details_scrollable, text=f"📊 {model_name} 상세 정보", font=("Arial", 16, "bold")).pack(pady=10)
    
    basic_frame = ctk.CTkFrame(details_scrollable)
    basic_frame.pack(fill="x", pady=5)
    
    ctk.CTkLabel(basic_frame, text=f"📁 경로: {model_info.path}").pack(anchor="w", padx=10, pady=2)
    ctk.CTkLabel(basic_frame, text=f"📊 상태: {model_info.status}").pack(anchor="w", padx=10, pady=2)
    ctk.CTkLabel(basic_frame, text=f"💾 메모리 사용량: {model_info.memory_usage:.1f} MB").pack(anchor="w", padx=10, pady=2)
    ctk.CTkLabel(basic_frame, text=f"⏰ 로드 시간: {model_info.load_time if model_info.load_time else 'N/A'}").pack(anchor="w", padx=10, pady=2)
    
    # 에러 정보
    if model_info.status == 'error' and model_info.error_message:
        ctk.CTkLabel(details_scrollable, text="🚨 에러 정보", font=("Arial", 14, "bold"), text_color="red").pack(pady=(20, 10))
        
        error_frame = ctk.CTkFrame(details_scrollable)
        error_frame.pack(fill="x", pady=5)
        
        error_text = ctk.CTkTextbox(error_frame, height=100)
        error_text.pack(fill="x", padx=10, pady=10)
        error_text.insert("0.0", model_info.error_message)
        error_text.configure(state="disabled")
    
    # 모델 분석 정보
    if model_info.config_analysis and 'model_summary' in model_info.config_analysis:
        summary = model_info.config_analysis['model_summary']
        
        ctk.CTkLabel(details_scrollable, text="🔍 모델 분석 정보", font=("Arial", 14, "bold")).pack(pady=(20, 10))
        
        analysis_frame = ctk.CTkFrame(details_scrollable)
        analysis_frame.pack(fill="x", pady=5)
        
        ctk.CTkLabel(analysis_frame, text=f"🏗️ 모델 타입: {summary.get('model_type', 'unknown')}").pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(analysis_frame, text=f"📊 파라미터 수: {summary.get('total_parameters', 0):,}").pack(anchor="w", padx=10, pady=2)
        ctk.CTkLabel(analysis_frame, text=f"💾 모델 크기: {summary.get('model_size_mb', 0):.1f} MB").pack(anchor="w", padx=10, pady=2)
        
        if summary.get('supported_tasks'):
            ctk.CTkLabel(analysis_frame, text=f"🎯 지원 태스크: {', '.join(summary['supported_tasks'])}").pack(anchor="w", padx=10, pady=2)
    
    # 닫기 버튼
    close_btn = ctk.CTkButton(details_scrollable, text="닫기", command=details_window.destroy)
    close_btn.pack(pady=20)

# 모델 목록 업데이트
def update_model_lists():
    # 로드된 모델 목록 업데이트
    loaded_models_listbox.delete(0, tk.END)
    for model_name in model_manager.get_loaded_models():
        loaded_models_listbox.insert(tk.END, model_name)
    
    # 전체 모델 목록 업데이트 (상태별 색상 구분)
    model_listbox.delete(0, tk.END)
    models_status = model_manager.get_all_models_status()
    
    for model_name, info in models_status.items():
        status = info['status']
        memory_usage = info['memory_usage']
        
        # 상태 아이콘 추가
        if status == 'loaded':
            icon = "✅"
        elif status == 'loading':
            icon = "🔄"
        elif status == 'error':
            icon = "❌"
        else:
            icon = "⚠️"
        
        # 에러 정보 표시
        if status == 'error' and info['error_message']:
            error_short = info['error_message'][:50] + "..." if len(info['error_message']) > 50 else info['error_message']
            display_text = f"{icon} {model_name} | {status} | {memory_usage:.1f}MB | ERROR: {error_short}"
        else:
            display_text = f"{icon} {model_name} | {status} | {memory_usage:.1f}MB"
        
        model_listbox.insert(tk.END, display_text)

# 자동 갱신 기능
def start_auto_refresh():
    global auto_refresh_timer
    if auto_refresh_interval > 0:
        # 시스템 정보 업데이트
        update_system_display(system_monitor.get_current_data())
        
        # 다음 갱신 예약
        auto_refresh_timer = app.after(auto_refresh_interval * 1000, start_auto_refresh)

def stop_auto_refresh():
    global auto_refresh_timer
    if auto_refresh_timer:
        app.after_cancel(auto_refresh_timer)
        auto_refresh_timer = None

def set_auto_refresh_interval(interval):
    global auto_refresh_interval
    auto_refresh_interval = interval
    
    # 기존 타이머 중지
    stop_auto_refresh()
    
    # 새 간격으로 시작
    if interval > 0:
        start_auto_refresh()
        update_auto_refresh_status()

def update_auto_refresh_status():
    if 'auto_refresh_status_label' in system_data_labels:
        if auto_refresh_interval > 0:
            system_data_labels['auto_refresh_status_label'].configure(
                text=f"🔄 {auto_refresh_interval}초마다 자동 갱신 중",
                text_color="green"
            )
        else:
            system_data_labels['auto_refresh_status_label'].configure(
                text="자동 갱신 비활성화",
                text_color="gray"
            )

# 수동 새로고침 기능
def manual_refresh():
    if system_monitor.monitoring:
        update_system_display(system_monitor.get_current_data())
    else:
        messagebox.showinfo("알림", "먼저 시스템 모니터링을 시작하세요.")

# 시스템 모니터링 시작
def start_monitoring():
    system_monitor.start_monitoring()
    messagebox.showinfo("알림", "시스템 모니터링을 시작합니다.")

# 시스템 모니터링 중지
def stop_monitoring():
    system_monitor.stop_monitoring()
    stop_auto_refresh()
    set_auto_refresh_interval(0)
    messagebox.showinfo("알림", "시스템 모니터링을 중지합니다.")

# FastAPI 서버 시작
def start_fastapi_server():
    try:
        result = fastapi_server.start_server()
        messagebox.showinfo("성공", result)
        update_server_status()
    except Exception as e:
        messagebox.showerror("오류", f"서버 시작 실패: {e}")

# FastAPI 서버 중지
def stop_fastapi_server():
    try:
        result = fastapi_server.stop_server()
        messagebox.showinfo("알림", result)
        update_server_status()
    except Exception as e:
        messagebox.showerror("오류", f"서버 중지 실패: {e}")

# 서버 상태 업데이트
def update_server_status():
    server_info = fastapi_server.get_server_info()
    if server_info['running']:
        server_status_label.configure(text=f"🟢 서버 실행 중: {server_info['url']}")
    else:
        server_status_label.configure(text="🔴 서버 중지됨")

# TabView 생성
tab_view = ctk.CTkTabview(app)
tab_view.pack(fill='both', expand=True, padx=20, pady=20)

# 첫 번째 탭: 로그인 / 로그아웃 및 사용자 정보
tab_user = tab_view.add("로그인 및 사용자 정보")

frame_login = ctk.CTkFrame(master=tab_user)
frame_login.pack(pady=10)

label_token = ctk.CTkLabel(master=frame_login, text="Hugging Face 토큰:")
label_token.pack(side="left", padx=10)

entry_token = ctk.CTkEntry(master=frame_login, width=300)
entry_token.pack(side="left")

button_login = ctk.CTkButton(master=frame_login, text="로그인", command=login)
button_login.pack(side="left", padx=10)

button_logout = ctk.CTkButton(master=frame_login, text="로그아웃", command=logout)
button_logout.pack(side="left", padx=10)

label_status = ctk.CTkLabel(master=tab_user, text="")
label_status.pack(pady=10)

# 사용자 정보 및 환경 정보 버튼
button_whoami = ctk.CTkButton(master=tab_user, text="현재 사용자 정보", command=show_whoami)
button_whoami.pack(pady=5)

button_env = ctk.CTkButton(master=tab_user, text="환경 정보 보기", command=show_env)
button_env.pack(pady=5)

# 사용자 정보 및 환경 정보 출력 라벨
label_user_info = ctk.CTkLabel(master=tab_user, text="")
label_user_info.pack(pady=5)

label_env_info = ctk.CTkLabel(master=tab_user, text="")
label_env_info.pack(pady=5)

# 두 번째 탭: 캐시 관리
tab_cache = tab_view.add("캐시 관리")

# 캐시 스캔 버튼
cache_scan_btn = ctk.CTkButton(master=tab_cache, text="캐시 스캔", command=scan_cache)
cache_scan_btn.pack(pady=10)

# 캐시 테이블 프레임
cache_table_frame = ctk.CTkScrollableFrame(tab_cache, width=1000, height=400)
cache_table_frame.pack(fill="both", expand=True, padx=10, pady=10)

# 선택된 항목과 용량 요약 표시 라벨
label_selection_summary = ctk.CTkLabel(master=tab_cache, text="선택된 항목: 0개, 총 용량: 0.00 MB")
label_selection_summary.pack(pady=10)

delete_btn = ctk.CTkButton(master=tab_cache, text="선택한 캐시 삭제", command=delete_selected)
delete_btn.pack(pady=10)

# 세 번째 탭: 시스템 모니터링
tab_monitoring = tab_view.add("시스템 모니터링")

# 모니터링 제어 버튼
monitor_control_frame = ctk.CTkFrame(tab_monitoring)
monitor_control_frame.pack(pady=10)

start_monitor_btn = ctk.CTkButton(monitor_control_frame, text="모니터링 시작", command=start_monitoring)
start_monitor_btn.pack(side="left", padx=5)

stop_monitor_btn = ctk.CTkButton(monitor_control_frame, text="모니터링 중지", command=stop_monitoring)
stop_monitor_btn.pack(side="left", padx=5)

# 자동 갱신 설정
auto_refresh_frame = ctk.CTkFrame(tab_monitoring)
auto_refresh_frame.pack(pady=10)

ctk.CTkLabel(auto_refresh_frame, text="자동 갱신 간격:", font=("Arial", 12, "bold")).pack(side="left", padx=5)

refresh_1s_btn = ctk.CTkButton(auto_refresh_frame, text="1초", command=lambda: set_auto_refresh_interval(1), width=60)
refresh_1s_btn.pack(side="left", padx=2)

refresh_3s_btn = ctk.CTkButton(auto_refresh_frame, text="3초", command=lambda: set_auto_refresh_interval(3), width=60)
refresh_3s_btn.pack(side="left", padx=2)

refresh_10s_btn = ctk.CTkButton(auto_refresh_frame, text="10초", command=lambda: set_auto_refresh_interval(10), width=60)
refresh_10s_btn.pack(side="left", padx=2)

refresh_off_btn = ctk.CTkButton(auto_refresh_frame, text="끄기", command=lambda: set_auto_refresh_interval(0), width=60)
refresh_off_btn.pack(side="left", padx=2)

# 수동 새로고침 버튼
manual_refresh_btn = ctk.CTkButton(auto_refresh_frame, text="🔄 새로고침", command=manual_refresh, width=80)
manual_refresh_btn.pack(side="left", padx=5)

# 자동 갱신 상태 표시
system_data_labels['auto_refresh_status_label'] = ctk.CTkLabel(auto_refresh_frame, text="자동 갱신 비활성화", text_color="gray")
system_data_labels['auto_refresh_status_label'].pack(side="left", padx=10)

# 시스템 정보 표시 프레임
system_info_frame = ctk.CTkFrame(tab_monitoring)
system_info_frame.pack(fill="x", pady=10, padx=10)

# 시스템 정보 라벨들
system_data_labels['cpu_percent_label'] = ctk.CTkLabel(system_info_frame, text="CPU: 0%")
system_data_labels['cpu_percent_label'].pack(pady=5)

system_data_labels['memory_percent_label'] = ctk.CTkLabel(system_info_frame, text="Memory: 0%")
system_data_labels['memory_percent_label'].pack(pady=5)

system_data_labels['gpu_info_label'] = ctk.CTkLabel(system_info_frame, text="GPU: N/A")
system_data_labels['gpu_info_label'].pack(pady=5)

system_data_labels['disk_info_label'] = ctk.CTkLabel(system_info_frame, text="Disk: 0%")
system_data_labels['disk_info_label'].pack(pady=5)

system_data_labels['time_label'] = ctk.CTkLabel(system_info_frame, text="Last Update: N/A")
system_data_labels['time_label'].pack(pady=5)

# 네 번째 탭: 모델 관리
tab_models = tab_view.add("모델 관리")

# 캐시된 모델 선택 프레임
cache_select_frame = ctk.CTkFrame(tab_models)
cache_select_frame.pack(pady=10, padx=10, fill="x")

ctk.CTkLabel(cache_select_frame, text="🗂️ 캐시된 모델에서 선택:", font=("Arial", 12, "bold")).pack(pady=5)

# 캐시된 모델 목록 가져오기
def get_cached_models():
    if cache_info:
        return [repo.repo_id for repo in cache_info.repos]
    return []

cached_models_var = ctk.StringVar(value="직접 입력")
cached_models_menu = ctk.CTkOptionMenu(
    cache_select_frame, 
    variable=cached_models_var,
    values=["직접 입력"] + get_cached_models(),
    command=lambda choice: update_model_path_from_cache(choice)
)
cached_models_menu.pack(pady=5)

def update_model_path_from_cache(choice):
    if choice != "직접 입력":
        entry_model_path.delete(0, tk.END)
        entry_model_path.insert(0, choice)
        messagebox.showinfo("캐시 모델 선택", f"선택된 모델: {choice}")

def refresh_cached_models():
    cached_models_menu.configure(values=["직접 입력"] + get_cached_models())

# 모델 입력 프레임
model_input_frame = ctk.CTkFrame(tab_models)
model_input_frame.pack(pady=10, padx=10, fill="x")

ctk.CTkLabel(model_input_frame, text="모델 경로 (로컬 경로 또는 HuggingFace 모델 ID):").grid(row=0, column=0, padx=5, pady=5, sticky="w")
entry_model_path = ctk.CTkEntry(model_input_frame, width=400, placeholder_text="예: tabularisai/multilingual-sentiment-analysis")
entry_model_path.grid(row=0, column=1, padx=5, pady=5)

# 모델 버튼들
model_buttons_frame = ctk.CTkFrame(tab_models)
model_buttons_frame.pack(pady=10)

analyze_model_btn = ctk.CTkButton(model_buttons_frame, text="모델 분석", command=analyze_model)
analyze_model_btn.pack(side="left", padx=5)

load_model_btn = ctk.CTkButton(model_buttons_frame, text="모델 로드", command=load_model)
load_model_btn.pack(side="left", padx=5)

unload_model_btn = ctk.CTkButton(model_buttons_frame, text="모델 언로드", command=unload_model)
unload_model_btn.pack(side="left", padx=5)

update_lists_btn = ctk.CTkButton(model_buttons_frame, text="목록 새로고침", command=update_model_lists)
update_lists_btn.pack(side="left", padx=5)

details_btn = ctk.CTkButton(model_buttons_frame, text="상세 정보", command=show_model_details)
details_btn.pack(side="left", padx=5)

# 모델 목록 표시
models_display_frame = ctk.CTkFrame(tab_models)
models_display_frame.pack(fill="both", expand=True, padx=10, pady=10)

# 로드된 모델 목록
loaded_models_frame = ctk.CTkFrame(models_display_frame)
loaded_models_frame.pack(side="left", fill="both", expand=True, padx=5)

ctk.CTkLabel(loaded_models_frame, text="로드된 모델", font=("Arial", 14, "bold")).pack(pady=5)
loaded_models_listbox = tk.Listbox(loaded_models_frame, height=10)
loaded_models_listbox.pack(fill="both", expand=True, padx=5, pady=5)

# 전체 모델 목록
all_models_frame = ctk.CTkFrame(models_display_frame)
all_models_frame.pack(side="right", fill="both", expand=True, padx=5)

ctk.CTkLabel(all_models_frame, text="전체 모델", font=("Arial", 14, "bold")).pack(pady=5)
model_listbox = tk.Listbox(all_models_frame, height=10)
model_listbox.pack(fill="both", expand=True, padx=5, pady=5)

# 다섯 번째 탭: FastAPI 서버
tab_server = tab_view.add("FastAPI 서버")

# 서버 제어 버튼
server_control_frame = ctk.CTkFrame(tab_server)
server_control_frame.pack(pady=10)

start_server_btn = ctk.CTkButton(server_control_frame, text="서버 시작", command=start_fastapi_server)
start_server_btn.pack(side="left", padx=5)

stop_server_btn = ctk.CTkButton(server_control_frame, text="서버 중지", command=stop_fastapi_server)
stop_server_btn.pack(side="left", padx=5)

# 서버 상태 표시
server_status_label = ctk.CTkLabel(tab_server, text="🔴 서버 중지됨")
server_status_label.pack(pady=10)

# 서버 정보 표시
server_info_frame = ctk.CTkFrame(tab_server)
server_info_frame.pack(fill="x", pady=10, padx=10)

ctk.CTkLabel(server_info_frame, text="서버 정보", font=("Arial", 14, "bold")).pack(pady=5)
ctk.CTkLabel(server_info_frame, text="기본 URL: http://127.0.0.1:8000").pack(pady=2)
ctk.CTkLabel(server_info_frame, text="API 문서: http://127.0.0.1:8000/docs").pack(pady=2)

# 프로그램 시작 시 로그인 상태 확인
token = load_login_token()
if token:
    label_status.configure(text="로그인 상태 유지됨")
else:
    label_status.configure(text="로그인 필요")

# 초기 설정
update_model_lists()
update_server_status()

# 캐시 스캔 실행
scan_cache()

# 캐시된 모델 목록 갱신
refresh_cached_models()

# 메인 루프 시작
if __name__ == "__main__":
    app.mainloop()