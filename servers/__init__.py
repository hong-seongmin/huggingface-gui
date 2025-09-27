"""
Servers package for FastAPI and desktop application management.
"""

try:
    from .fastapi_server import FastAPIServerManager, FastAPIServer
    _fastapi_available = True
except ImportError:
    # Fallback classes if FastAPI is not available
    class FastAPIServerManager:
        def __init__(self):
            pass
        def start_server(self, *args, **kwargs):
            raise RuntimeError("FastAPI server not available")

    class FastAPIServer:
        def __init__(self):
            pass

    _fastapi_available = False

try:
    from .gui.desktop_app import DesktopApp
    from .run import main as desktop_main
    _desktop_available = True
except ImportError:
    # Fallback classes if desktop dependencies are not available
    class DesktopApp:
        def __init__(self):
            pass
        def run(self):
            raise RuntimeError("Desktop app not available")

    def desktop_main():
        raise RuntimeError("Desktop app not available")

    _desktop_available = False

__all__ = [
    'FastAPIServerManager',
    'FastAPIServer',
    'DesktopApp',
    'desktop_main'
]