"""
Compatibility checking package for HuggingFace GUI.

This package provides modular components for system compatibility checking:
- SystemInfoCollector: Collects comprehensive system information
- RequirementValidator: Validates system requirements
- ReportGenerator: Generates formatted reports and recommendations
- Colors: ANSI color utilities for terminal output
"""

from .system_info_collector import SystemInfoCollector
from .requirement_validator import RequirementValidator
from .report_generator import ReportGenerator, Colors

__all__ = [
    'SystemInfoCollector',
    'RequirementValidator',
    'ReportGenerator',
    'Colors',
]

# Version information
__version__ = '1.0.0'

# Package metadata
__author__ = 'HuggingFace GUI Team'
__description__ = 'Modular system compatibility checking components'