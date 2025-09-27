"""
Event handlers for desktop application GUI components.

This module contains all event handling logic extracted from servers/run.py,
including login/logout, cache operations, model analysis, and UI interactions.
"""

import os
import threading
import json
from datetime import datetime
from huggingface_hub import HfApi
from tkinter import messagebox


class EventHandlers:
    """Centralized event handler class for GUI interactions."""

    def __init__(self, app_instance):
        """Initialize event handlers with reference to app instance."""
        self.app = app_instance
        self.api = HfApi()

    # Login/Authentication handlers
    def handle_login(self):
        """Handle user login process."""
        try:
            from servers.run import api, label_status, save_login_token

            token = self.app.entry_token.get().strip()
            if not token:
                messagebox.showerror("오류", "토큰을 입력해주세요.")
                return

            api.set_access_token(token)

            try:
                user_info = api.whoami()
                save_login_token(token)
                label_status.configure(text=f"로그인 성공: {user_info['name']}")
                messagebox.showinfo("성공", f"로그인 성공!\n사용자: {user_info['name']}")
            except Exception as e:
                label_status.configure(text="로그인 실패")
                messagebox.showerror("로그인 실패", f"유효하지 않은 토큰입니다.\n{str(e)}")

        except Exception as e:
            messagebox.showerror("오류", f"로그인 처리 중 오류 발생: {str(e)}")

    def handle_logout(self):
        """Handle user logout process."""
        try:
            from servers.run import api, label_status, delete_login_token

            api.unset_access_token()
            delete_login_token()
            label_status.configure(text="로그아웃 완료")
            messagebox.showinfo("완료", "로그아웃이 완료되었습니다.")

        except Exception as e:
            messagebox.showerror("오류", f"로그아웃 처리 중 오류 발생: {str(e)}")

    # Cache management handlers
    def handle_scan_cache(self):
        """Handle cache scanning operation."""
        try:
            from servers.run import scan_cache
            scan_cache()

        except Exception as e:
            messagebox.showerror("오류", f"캐시 스캔 중 오류 발생: {str(e)}")

    def handle_select_all(self):
        """Handle select all checkboxes."""
        try:
            from servers.run import select_all
            select_all()

        except Exception as e:
            messagebox.showerror("오류", f"전체 선택 중 오류 발생: {str(e)}")

    def handle_deselect_all(self):
        """Handle deselect all checkboxes."""
        try:
            from servers.run import deselect_all
            deselect_all()

        except Exception as e:
            messagebox.showerror("오류", f"전체 해제 중 오류 발생: {str(e)}")

    def handle_delete_selected(self):
        """Handle deletion of selected cache items."""
        try:
            from servers.run import delete_selected
            delete_selected()

        except Exception as e:
            messagebox.showerror("오류", f"선택 항목 삭제 중 오류 발생: {str(e)}")

    # Model analysis handlers
    def handle_analyze_model(self):
        """Handle model analysis operation."""
        try:
            from servers.run import analyze_model
            analyze_model()

        except Exception as e:
            messagebox.showerror("오류", f"모델 분석 중 오류 발생: {str(e)}")

    # System info handlers
    def handle_show_env(self):
        """Handle showing environment information."""
        try:
            from servers.run import show_env
            show_env()

        except Exception as e:
            messagebox.showerror("오류", f"환경 정보 표시 중 오류 발생: {str(e)}")

    def handle_show_whoami(self):
        """Handle showing user information."""
        try:
            from servers.run import show_whoami
            show_whoami()

        except Exception as e:
            messagebox.showerror("오류", f"사용자 정보 표시 중 오류 발생: {str(e)}")

    # Server management handlers
    def handle_start_server(self):
        """Handle FastAPI server start."""
        try:
            from servers.run import fastapi_server

            def start_server_thread():
                try:
                    fastapi_server.start_server()
                    messagebox.showinfo("서버 시작", "FastAPI 서버가 시작되었습니다.")
                except Exception as e:
                    messagebox.showerror("서버 오류", f"서버 시작 실패: {str(e)}")

            server_thread = threading.Thread(target=start_server_thread, daemon=True)
            server_thread.start()

        except Exception as e:
            messagebox.showerror("오류", f"서버 시작 중 오류 발생: {str(e)}")

    def handle_stop_server(self):
        """Handle FastAPI server stop."""
        try:
            from servers.run import fastapi_server
            fastapi_server.stop_server()
            messagebox.showinfo("서버 중지", "FastAPI 서버가 중지되었습니다.")

        except Exception as e:
            messagebox.showerror("오류", f"서버 중지 중 오류 발생: {str(e)}")

    # Column sorting handlers
    def handle_sort_by_column(self, column_name):
        """Handle column sorting in cache table."""
        try:
            from servers.run import sort_by_column
            sort_by_column(column_name)

        except Exception as e:
            messagebox.showerror("오류", f"정렬 중 오류 발생: {str(e)}")

    # System monitoring handlers
    def handle_system_update(self, data):
        """Handle system monitoring data updates."""
        try:
            from servers.run import update_system_display
            update_system_display(data)

        except Exception as e:
            print(f"시스템 업데이트 중 오류 발생: {str(e)}")

    # Model loading handlers
    def handle_model_status_callback(self, model_name, event_type, data):
        """Handle model loading status callbacks."""
        try:
            from servers.run import model_status_callback
            model_status_callback(model_name, event_type, data)

        except Exception as e:
            print(f"모델 상태 콜백 중 오류 발생: {str(e)}")

    # Auto-refresh handlers
    def handle_toggle_auto_refresh(self):
        """Handle toggling auto-refresh functionality."""
        try:
            # This would need to be implemented based on the auto-refresh logic in run.py
            pass
        except Exception as e:
            messagebox.showerror("오류", f"자동 새로고침 토글 중 오류 발생: {str(e)}")

    # Utility handlers
    def handle_refresh_cached_models(self):
        """Handle refreshing the cached models list."""
        try:
            from servers.run import refresh_cached_models
            refresh_cached_models()

        except Exception as e:
            messagebox.showerror("오류", f"캐시된 모델 새로고침 중 오류 발생: {str(e)}")

    def handle_update_model_lists(self):
        """Handle updating model lists in the GUI."""
        try:
            from servers.run import update_model_lists
            update_model_lists()

        except Exception as e:
            messagebox.showerror("오류", f"모델 목록 업데이트 중 오류 발생: {str(e)}")

    def handle_update_server_status(self):
        """Handle updating server status display."""
        try:
            from servers.run import update_server_status
            update_server_status()

        except Exception as e:
            messagebox.showerror("오류", f"서버 상태 업데이트 중 오류 발생: {str(e)}")