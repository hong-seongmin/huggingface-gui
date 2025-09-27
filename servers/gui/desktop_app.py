"""
Main DesktopApp class for the Hugging Face GUI application.

This class coordinates all GUI components and serves as the main entry point
for the desktop application, replacing the procedural approach in run.py.
"""

import os
import inspect
import threading
import time
from datetime import datetime

from huggingface_hub import HfApi
from tkinter import messagebox

from .window_manager import WindowManager
from .event_handlers import EventHandlers


class DesktopApp:
    """Main desktop application class for Hugging Face GUI."""

    def __init__(self):
        """Initialize the desktop application."""
        # Change to servers directory to match original script behavior
        self._setup_working_directory()

        # Initialize core components
        self.window_manager = WindowManager()
        self.event_handlers = EventHandlers(self)

        # Initialize external components
        self._initialize_external_components()

        # GUI components
        self.main_window = None
        self.tabview = None

        # Global state variables
        self.cache_info = None
        self.revisions = []
        self.sort_orders = {}
        self.checkboxes = []
        self.current_model_analysis = None
        self.auto_refresh_interval = 0
        self.auto_refresh_timer = None

    def _setup_working_directory(self):
        """Setup working directory to match original script."""
        current_file = inspect.getfile(inspect.currentframe())
        servers_dir = os.path.dirname(os.path.dirname(os.path.abspath(current_file)))
        if os.path.exists(servers_dir):
            os.chdir(servers_dir)

    def _initialize_external_components(self):
        """Initialize external components like model managers and monitors."""
        try:
            # Import these after changing directory to avoid import issues
            from model_manager import MultiModelManager
            from system_monitor import SystemMonitor
            from fastapi_server import FastAPIServer
            from model_analyzer import ComprehensiveModelAnalyzer

            # Initialize components
            self.model_manager = MultiModelManager()
            self.system_monitor = SystemMonitor()
            self.fastapi_server = FastAPIServer(self.model_manager)
            self.model_analyzer = ComprehensiveModelAnalyzer()

            # Initialize HuggingFace API
            self.api = HfApi()

        except ImportError as e:
            print(f"Warning: Could not import external components: {e}")
            # Create mock components for testing
            self._create_mock_components()

    def _create_mock_components(self):
        """Create mock components when external components are not available."""
        class MockComponent:
            def __getattr__(self, name):
                return lambda *args, **kwargs: None

        self.model_manager = MockComponent()
        self.system_monitor = MockComponent()
        self.fastapi_server = MockComponent()
        self.model_analyzer = MockComponent()
        self.api = HfApi()

    def _setup_event_bindings(self):
        """Setup event bindings for GUI components."""
        # Login tab events
        login_btn = self.window_manager.get_widget('btn_login')
        if login_btn:
            login_btn.configure(command=self.event_handlers.handle_login)

        logout_btn = self.window_manager.get_widget('btn_logout')
        if logout_btn:
            logout_btn.configure(command=self.event_handlers.handle_logout)

        env_btn = self.window_manager.get_widget('btn_env')
        if env_btn:
            env_btn.configure(command=self.event_handlers.handle_show_env)

        whoami_btn = self.window_manager.get_widget('btn_whoami')
        if whoami_btn:
            whoami_btn.configure(command=self.event_handlers.handle_show_whoami)

        # Cache tab events
        scan_btn = self.window_manager.get_widget('btn_scan')
        if scan_btn:
            scan_btn.configure(command=self.event_handlers.handle_scan_cache)

        select_all_btn = self.window_manager.get_widget('btn_select_all')
        if select_all_btn:
            select_all_btn.configure(command=self.event_handlers.handle_select_all)

        deselect_all_btn = self.window_manager.get_widget('btn_deselect_all')
        if deselect_all_btn:
            deselect_all_btn.configure(command=self.event_handlers.handle_deselect_all)

        delete_btn = self.window_manager.get_widget('btn_delete')
        if delete_btn:
            delete_btn.configure(command=self.event_handlers.handle_delete_selected)

        # Models tab events
        analyze_btn = self.window_manager.get_widget('btn_analyze')
        if analyze_btn:
            analyze_btn.configure(command=self.event_handlers.handle_analyze_model)

        # Server tab events
        start_server_btn = self.window_manager.get_widget('btn_start_server')
        if start_server_btn:
            start_server_btn.configure(command=self.event_handlers.handle_start_server)

        stop_server_btn = self.window_manager.get_widget('btn_stop_server')
        if stop_server_btn:
            stop_server_btn.configure(command=self.event_handlers.handle_stop_server)

        # Monitoring tab events
        start_refresh_btn = self.window_manager.get_widget('btn_start_refresh')
        if start_refresh_btn:
            start_refresh_btn.configure(command=self.event_handlers.handle_toggle_auto_refresh)

        stop_refresh_btn = self.window_manager.get_widget('btn_stop_refresh')
        if stop_refresh_btn:
            stop_refresh_btn.configure(command=self.event_handlers.handle_toggle_auto_refresh)

    def _setup_callbacks(self):
        """Setup callbacks for external components."""
        try:
            # System monitor callback
            if hasattr(self.system_monitor, 'set_callback'):
                self.system_monitor.set_callback(self.event_handlers.handle_system_update)

            # Model manager callback
            if hasattr(self.model_manager, 'set_status_callback'):
                self.model_manager.set_status_callback(self.event_handlers.handle_model_status_callback)

        except Exception as e:
            print(f"Warning: Could not setup callbacks: {e}")

    def _load_initial_state(self):
        """Load initial application state."""
        try:
            # Load login token
            from servers.run import load_login_token
            token = load_login_token()

            status_label = self.window_manager.get_widget('label_status')
            if token and status_label:
                status_label.configure(text="로그인 상태 유지됨")
            elif status_label:
                status_label.configure(text="로그인 필요")

        except Exception as e:
            print(f"Warning: Could not load initial state: {e}")

    def _perform_initial_operations(self):
        """Perform initial operations like cache scanning."""
        try:
            # Initial setup operations
            self.event_handlers.handle_update_model_lists()
            self.event_handlers.handle_update_server_status()
            self.event_handlers.handle_scan_cache()
            self.event_handlers.handle_refresh_cached_models()

        except Exception as e:
            print(f"Warning: Could not perform initial operations: {e}")

    def create_gui(self):
        """Create the complete GUI interface."""
        # Create main window
        self.main_window = self.window_manager.create_main_window()

        # Setup all tabs
        self.tabview = self.window_manager.setup_all_tabs(self.main_window)

        # Setup event bindings
        self._setup_event_bindings()

        # Setup callbacks
        self._setup_callbacks()

        return self.main_window

    def initialize(self):
        """Initialize the application with all components."""
        # Create GUI
        self.create_gui()

        # Load initial state
        self._load_initial_state()

        # Perform initial operations
        self._perform_initial_operations()

    def run(self):
        """Run the desktop application."""
        try:
            # Initialize the application
            self.initialize()

            # Start the main event loop
            if self.main_window:
                self.main_window.mainloop()
            else:
                raise RuntimeError("Main window not created")

        except Exception as e:
            messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
            raise

    def get_widget(self, name):
        """Get a widget by name through window manager."""
        return self.window_manager.get_widget(name)

    def get_tab(self, name):
        """Get a tab by name through window manager."""
        return self.window_manager.get_tab(name)

    # Properties to maintain compatibility with original script
    @property
    def entry_token(self):
        """Get the token entry widget."""
        return self.window_manager.get_widget('entry_token')

    @property
    def label_status(self):
        """Get the status label widget."""
        return self.window_manager.get_widget('label_status')

    @property
    def model_listbox(self):
        """Get the model listbox widget."""
        return self.window_manager.get_widget('model_listbox')

    @property
    def loaded_models_listbox(self):
        """Get the loaded models listbox widget."""
        return self.window_manager.get_widget('loaded_models_listbox')

    def cleanup(self):
        """Cleanup resources when application exits."""
        try:
            # Stop auto-refresh timer
            if self.auto_refresh_timer:
                self.auto_refresh_timer.cancel()

            # Stop server if running
            if hasattr(self.fastapi_server, 'stop_server'):
                self.fastapi_server.stop_server()

            # Cleanup system monitor
            if hasattr(self.system_monitor, 'stop'):
                self.system_monitor.stop()

        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")


def main():
    """Main entry point for the desktop application."""
    app = DesktopApp()

    try:
        app.run()
    finally:
        app.cleanup()


if __name__ == "__main__":
    main()