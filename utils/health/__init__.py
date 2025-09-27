"""
Health monitoring package for HuggingFace GUI.

This package provides comprehensive health monitoring components:
- SystemMonitor: Monitors system resources (CPU, memory, disk)
- ServiceChecker: Checks health and availability of application services
"""

from .system_monitor import SystemMonitor
from .service_checker import ServiceChecker

__all__ = [
    'SystemMonitor',
    'ServiceChecker',
]

# Version information
__version__ = '1.0.0'

# Package metadata
__author__ = 'HuggingFace GUI Team'
__description__ = 'Comprehensive health monitoring components'