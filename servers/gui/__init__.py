"""
GUI package for desktop application components.

This package provides modularized GUI components for the Hugging Face desktop application,
extracted from the original monolithic servers/run.py script.
"""

from .desktop_app import DesktopApp
from .event_handlers import EventHandlers
from .window_manager import WindowManager

__all__ = [
    'DesktopApp',
    'EventHandlers',
    'WindowManager'
]