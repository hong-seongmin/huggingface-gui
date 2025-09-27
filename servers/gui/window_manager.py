"""
Window manager for desktop application GUI layout and components.

This module handles the creation and management of GUI windows, tabs, and layouts,
extracted from the UI creation code in servers/run.py.
"""

import customtkinter as ctk
from tkinter import ttk
import tkinter as tk


class WindowManager:
    """Manages GUI windows, tabs, and layout components."""

    def __init__(self):
        """Initialize window manager."""
        self.main_window = None
        self.tabview = None
        self.tabs = {}
        self.widgets = {}
        self.system_data_labels = {}

    def create_main_window(self):
        """Create and configure the main application window."""
        # GUI 초기 설정
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")

        # 메인 윈도우 생성
        self.main_window = ctk.CTk()
        self.main_window.title("Hugging Face GUI")
        self.main_window.geometry("1200x800")

        return self.main_window

    def create_tab_view(self, parent):
        """Create the main tab view."""
        self.tabview = ctk.CTkTabview(parent, width=1150, height=750)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=20)

        return self.tabview

    def create_login_tab(self, tabview):
        """Create the login/authentication tab."""
        tab_login = tabview.add("🔐 로그인")
        self.tabs['login'] = tab_login

        # 로그인 프레임
        login_frame = ctk.CTkFrame(tab_login)
        login_frame.pack(pady=20, padx=20, fill="x")

        ctk.CTkLabel(login_frame, text="Hugging Face Token", font=("Arial", 14, "bold")).pack(pady=10)

        # 토큰 입력
        self.widgets['entry_token'] = ctk.CTkEntry(login_frame, placeholder_text="hf_...", width=400)
        self.widgets['entry_token'].pack(pady=10)

        # 로그인 버튼
        self.widgets['btn_login'] = ctk.CTkButton(login_frame, text="로그인")
        self.widgets['btn_login'].pack(pady=5)

        # 로그아웃 버튼
        self.widgets['btn_logout'] = ctk.CTkButton(login_frame, text="로그아웃")
        self.widgets['btn_logout'].pack(pady=5)

        # 상태 라벨
        self.widgets['label_status'] = ctk.CTkLabel(login_frame, text="로그인 필요", text_color="red")
        self.widgets['label_status'].pack(pady=10)

        # 사용자 정보 버튼들
        button_frame = ctk.CTkFrame(tab_login)
        button_frame.pack(pady=20, padx=20, fill="x")

        self.widgets['btn_env'] = ctk.CTkButton(button_frame, text="환경 정보")
        self.widgets['btn_env'].pack(side="left", padx=10, pady=10)

        self.widgets['btn_whoami'] = ctk.CTkButton(button_frame, text="사용자 정보")
        self.widgets['btn_whoami'].pack(side="left", padx=10, pady=10)

        return tab_login

    def create_cache_tab(self, tabview):
        """Create the cache management tab."""
        tab_cache = tabview.add("💾 캐시 관리")
        self.tabs['cache'] = tab_cache

        # 캐시 관리 버튼들
        button_frame = ctk.CTkFrame(tab_cache)
        button_frame.pack(pady=10, padx=10, fill="x")

        self.widgets['btn_scan'] = ctk.CTkButton(button_frame, text="캐시 스캔")
        self.widgets['btn_scan'].pack(side="left", padx=5, pady=5)

        self.widgets['btn_select_all'] = ctk.CTkButton(button_frame, text="전체 선택")
        self.widgets['btn_select_all'].pack(side="left", padx=5, pady=5)

        self.widgets['btn_deselect_all'] = ctk.CTkButton(button_frame, text="전체 해제")
        self.widgets['btn_deselect_all'].pack(side="left", padx=5, pady=5)

        self.widgets['btn_delete'] = ctk.CTkButton(button_frame, text="선택 삭제", fg_color="red", hover_color="darkred")
        self.widgets['btn_delete'].pack(side="right", padx=5, pady=5)

        # 캐시 정보 표시 영역
        self.widgets['cache_frame'] = ctk.CTkScrollableFrame(tab_cache)
        self.widgets['cache_frame'].pack(fill="both", expand=True, padx=10, pady=10)

        # 선택 요약 정보
        self.widgets['summary_label'] = ctk.CTkLabel(tab_cache, text="선택된 항목: 0개, 총 용량: 0 MB", text_color="blue")
        self.widgets['summary_label'].pack(pady=5)

        return tab_cache

    def create_models_tab(self, tabview):
        """Create the models management tab."""
        tab_models = tabview.add("🤖 모델 관리")
        self.tabs['models'] = tab_models

        # 모델 분석 버튼
        analysis_frame = ctk.CTkFrame(tab_models)
        analysis_frame.pack(pady=10, padx=10, fill="x")

        self.widgets['btn_analyze'] = ctk.CTkButton(analysis_frame, text="모델 분석")
        self.widgets['btn_analyze'].pack(pady=10)

        # 모델 목록 (왼쪽)
        list_frame = ctk.CTkFrame(tab_models)
        list_frame.pack(side="left", fill="both", expand=True, padx=(10, 5), pady=10)

        ctk.CTkLabel(list_frame, text="사용 가능한 모델", font=("Arial", 14, "bold")).pack(pady=5)
        self.widgets['model_listbox'] = tk.Listbox(list_frame, height=15)
        self.widgets['model_listbox'].pack(fill="both", expand=True, padx=10, pady=5)

        # 로드된 모델 목록 (오른쪽)
        loaded_frame = ctk.CTkFrame(tab_models)
        loaded_frame.pack(side="right", fill="both", expand=True, padx=(5, 10), pady=10)

        ctk.CTkLabel(loaded_frame, text="로드된 모델", font=("Arial", 14, "bold")).pack(pady=5)
        self.widgets['loaded_models_listbox'] = tk.Listbox(loaded_frame, height=15)
        self.widgets['loaded_models_listbox'].pack(fill="both", expand=True, padx=10, pady=5)

        return tab_models

    def create_analysis_tab(self, tabview):
        """Create the model analysis results tab."""
        tab_analysis = tabview.add("🔍 분석 결과")
        self.tabs['analysis'] = tab_analysis

        # 분석 결과 표시 영역
        self.widgets['analysis_frame'] = ctk.CTkScrollableFrame(tab_analysis)
        self.widgets['analysis_frame'].pack(fill="both", expand=True, padx=10, pady=10)

        # 기본 메시지
        self.widgets['analysis_label'] = ctk.CTkLabel(self.widgets['analysis_frame'], text="모델을 선택하고 분석 버튼을 클릭하세요.", text_color="gray")
        self.widgets['analysis_label'].pack(pady=20)

        return tab_analysis

    def create_monitoring_tab(self, tabview):
        """Create the system monitoring tab."""
        tab_monitoring = tabview.add("📊 시스템 모니터링")
        self.tabs['monitoring'] = tab_monitoring

        # 시스템 정보 프레임
        system_frame = ctk.CTkFrame(tab_monitoring)
        system_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(system_frame, text="시스템 리소스", font=("Arial", 16, "bold")).pack(pady=10)

        # CPU 정보
        self.system_data_labels['cpu_percent_label'] = ctk.CTkLabel(system_frame, text="CPU: 로딩 중...", font=("Arial", 12))
        self.system_data_labels['cpu_percent_label'].pack(pady=2)

        # 메모리 정보
        self.system_data_labels['memory_percent_label'] = ctk.CTkLabel(system_frame, text="Memory: 로딩 중...", font=("Arial", 12))
        self.system_data_labels['memory_percent_label'].pack(pady=2)

        # GPU 정보
        self.system_data_labels['gpu_info_label'] = ctk.CTkLabel(system_frame, text="GPU: 로딩 중...", font=("Arial", 12))
        self.system_data_labels['gpu_info_label'].pack(pady=2)

        # 디스크 정보
        self.system_data_labels['disk_info_label'] = ctk.CTkLabel(system_frame, text="Disk: 로딩 중...", font=("Arial", 12))
        self.system_data_labels['disk_info_label'].pack(pady=2)

        # 시간 정보
        self.system_data_labels['time_label'] = ctk.CTkLabel(system_frame, text="Time: 로딩 중...", font=("Arial", 12))
        self.system_data_labels['time_label'].pack(pady=2)

        # 자동 새로고침 설정
        refresh_frame = ctk.CTkFrame(tab_monitoring)
        refresh_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(refresh_frame, text="자동 새로고침 설정", font=("Arial", 14, "bold")).pack(pady=5)

        self.widgets['refresh_interval'] = ctk.CTkEntry(refresh_frame, placeholder_text="간격(초)", width=100)
        self.widgets['refresh_interval'].pack(side="left", padx=5, pady=5)

        self.widgets['btn_start_refresh'] = ctk.CTkButton(refresh_frame, text="시작")
        self.widgets['btn_start_refresh'].pack(side="left", padx=5, pady=5)

        self.widgets['btn_stop_refresh'] = ctk.CTkButton(refresh_frame, text="중지")
        self.widgets['btn_stop_refresh'].pack(side="left", padx=5, pady=5)

        return tab_monitoring

    def create_server_tab(self, tabview):
        """Create the server management tab."""
        tab_server = tabview.add("🚀 서버")
        self.tabs['server'] = tab_server

        # 서버 제어 버튼들
        server_control_frame = ctk.CTkFrame(tab_server)
        server_control_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(server_control_frame, text="FastAPI 서버 제어", font=("Arial", 14, "bold")).pack(pady=5)

        button_frame = ctk.CTkFrame(server_control_frame)
        button_frame.pack(pady=10)

        self.widgets['btn_start_server'] = ctk.CTkButton(button_frame, text="서버 시작", fg_color="green", hover_color="darkgreen")
        self.widgets['btn_start_server'].pack(side="left", padx=10, pady=5)

        self.widgets['btn_stop_server'] = ctk.CTkButton(button_frame, text="서버 중지", fg_color="red", hover_color="darkred")
        self.widgets['btn_stop_server'].pack(side="left", padx=10, pady=5)

        # 서버 상태
        self.widgets['server_status_label'] = ctk.CTkLabel(server_control_frame, text="서버 상태: 중지됨", text_color="red")
        self.widgets['server_status_label'].pack(pady=5)

        # 서버 정보 표시
        server_info_frame = ctk.CTkFrame(tab_server)
        server_info_frame.pack(fill="x", pady=10, padx=10)

        ctk.CTkLabel(server_info_frame, text="서버 정보", font=("Arial", 14, "bold")).pack(pady=5)
        ctk.CTkLabel(server_info_frame, text="기본 URL: http://127.0.0.1:8000").pack(pady=2)
        ctk.CTkLabel(server_info_frame, text="API 문서: http://127.0.0.1:8000/docs").pack(pady=2)

        return tab_server

    def get_widget(self, name):
        """Get widget by name."""
        return self.widgets.get(name)

    def get_tab(self, name):
        """Get tab by name."""
        return self.tabs.get(name)

    def get_system_label(self, name):
        """Get system monitoring label by name."""
        return self.system_data_labels.get(name)

    def setup_all_tabs(self, parent):
        """Setup all tabs in the application."""
        tabview = self.create_tab_view(parent)

        self.create_login_tab(tabview)
        self.create_cache_tab(tabview)
        self.create_models_tab(tabview)
        self.create_analysis_tab(tabview)
        self.create_monitoring_tab(tabview)
        self.create_server_tab(tabview)

        return tabview