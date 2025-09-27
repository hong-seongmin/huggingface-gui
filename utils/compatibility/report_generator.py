"""
Report Generator for compatibility checking.

This module handles generation of compatibility reports including console output,
recommendations, and file export functionality.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging


class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    PURPLE = '\033[0;35m'
    CYAN = '\033[0;36m'
    WHITE = '\033[1;37m'
    NC = '\033[0m'  # No Color

    @classmethod
    def colorize(cls, text: str, color: str) -> str:
        """Colorize text with ANSI codes."""
        return f"{color}{text}{cls.NC}"


class ReportGenerator:
    """Generates comprehensive compatibility reports."""

    def __init__(self, use_colors: bool = True):
        """Initialize report generator."""
        self.logger = logging.getLogger("ReportGenerator")
        self.use_colors = use_colors

    def generate_console_report(self, validation_results: Dict[str, Any],
                              system_info: Dict[str, Any],
                              requirements: Dict[str, Any]) -> None:
        """Generate formatted console report."""
        self.logger.info("[REPORT] Generating console compatibility report")

        print("\n" + "="*60)
        print(self._colorize("🔍 HuggingFace GUI Compatibility Report", Colors.CYAN))
        print("="*60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # System Information Summary
        self._print_system_summary(system_info)

        # Validation Results
        self._print_validation_results(validation_results)

        # Overall Status
        self._print_overall_status(validation_results.get('overall', False))

        # Recommendations
        recommendations = self.generate_recommendations(validation_results, system_info, requirements)
        if recommendations:
            self._print_recommendations(recommendations)

        print("="*60 + "\n")

    def _print_system_summary(self, system_info: Dict[str, Any]) -> None:
        """Print system information summary."""
        print(self._colorize("📊 System Information", Colors.BLUE))
        print("-" * 30)

        # Basic info
        print(f"Platform: {system_info.get('platform', 'Unknown')}")
        print(f"Python Version: {system_info.get('python_version', 'Unknown')}")
        print(f"Architecture: {system_info.get('architecture', 'Unknown')}")

        # Resources
        memory_gb = system_info.get('memory_gb', 'Unknown')
        disk_free = system_info.get('disk_free_gb', 'Unknown')
        print(f"Memory: {memory_gb}{'GB' if isinstance(memory_gb, (int, float)) else ''}")
        print(f"Free Disk Space: {disk_free}{'GB' if isinstance(disk_free, (int, float)) else ''}")

        # GPU info
        gpu_info = system_info.get('gpu_info', {})
        if gpu_info.get('nvidia_available'):
            gpu_count = gpu_info.get('gpu_count', 0)
            print(f"NVIDIA GPUs: {gpu_count} available")
            if gpu_info.get('gpu_names'):
                for gpu_name in gpu_info['gpu_names'][:2]:  # Show first 2
                    print(f"  - {gpu_name}")

        cuda_available = gpu_info.get('cuda_available', False)
        mps_available = gpu_info.get('mps_available', False)
        gpu_status = []
        if cuda_available:
            gpu_status.append("CUDA")
        if mps_available:
            gpu_status.append("MPS")

        if gpu_status:
            print(f"GPU Acceleration: {', '.join(gpu_status)} available")
        else:
            print("GPU Acceleration: Not available")

        print()

    def _print_validation_results(self, validation_results: Dict[str, Any]) -> None:
        """Print validation results."""
        print(self._colorize("✅ Compatibility Checks", Colors.BLUE))
        print("-" * 30)

        # Python version check
        python_result = validation_results.get('python_version', {})
        if isinstance(python_result, dict):
            status = "✓" if python_result.get('valid') else "✗"
            color = Colors.GREEN if python_result.get('valid') else Colors.RED
            required = python_result.get('required', 'Unknown')
            current = python_result.get('current', 'Unknown')
            print(f"{self._colorize(status, color)} Python Version: {current} (required: {required})")
        elif isinstance(python_result, bool):
            status = "✓" if python_result else "✗"
            color = Colors.GREEN if python_result else Colors.RED
            print(f"{self._colorize(status, color)} Python Version")

        # Package check
        package_result = validation_results.get('package_check', {})
        if isinstance(package_result, dict):
            all_required = package_result.get('all_required_available', False)
            status = "✓" if all_required else "✗"
            color = Colors.GREEN if all_required else Colors.RED
            missing = package_result.get('missing_required', [])
            available = package_result.get('available_required', [])

            print(f"{self._colorize(status, color)} Required Packages: {len(available)} available")
            if missing:
                print(f"  Missing: {', '.join(missing)}")
        elif isinstance(package_result, bool):
            status = "✓" if package_result else "✗"
            color = Colors.GREEN if package_result else Colors.RED
            print(f"{self._colorize(status, color)} Required Packages")

        # Memory check
        memory_result = validation_results.get('memory_check', {})
        if isinstance(memory_result, dict):
            status = "✓" if memory_result.get('valid') else "✗"
            color = Colors.GREEN if memory_result.get('valid') else Colors.RED
            current = memory_result.get('current_gb', 'Unknown')
            required = memory_result.get('required_gb', 'Unknown')
            assumed = " (assumed)" if memory_result.get('assumed') else ""
            print(f"{self._colorize(status, color)} Memory: {current}GB (required: {required}GB){assumed}")
        elif isinstance(memory_result, bool):
            status = "✓" if memory_result else "✗"
            color = Colors.GREEN if memory_result else Colors.RED
            print(f"{self._colorize(status, color)} Memory")

        # Disk space check
        disk_result = validation_results.get('disk_space_check', {})
        if isinstance(disk_result, dict):
            status = "✓" if disk_result.get('valid') else "✗"
            color = Colors.GREEN if disk_result.get('valid') else Colors.RED
            current = disk_result.get('current_gb', 'Unknown')
            required = disk_result.get('required_gb', 'Unknown')
            assumed = " (assumed)" if disk_result.get('assumed') else ""
            print(f"{self._colorize(status, color)} Disk Space: {current}GB free (required: {required}GB){assumed}")
        elif isinstance(disk_result, bool):
            status = "✓" if disk_result else "✗"
            color = Colors.GREEN if disk_result else Colors.RED
            print(f"{self._colorize(status, color)} Disk Space")

        # GPU check (if present)
        gpu_result = validation_results.get('gpu_check', {})
        if gpu_result and gpu_result.get('cuda_required'):
            status = "✓" if gpu_result.get('valid') else "✗"
            color = Colors.GREEN if gpu_result.get('valid') else Colors.RED
            print(f"{self._colorize(status, color)} GPU (CUDA required)")

        print()

    def _print_overall_status(self, overall_success: bool) -> None:
        """Print overall compatibility status."""
        if overall_success:
            print(self._colorize("🎉 Overall Status: COMPATIBLE", Colors.GREEN))
            print("Your system meets all requirements for HuggingFace GUI!")
        else:
            print(self._colorize("⚠️  Overall Status: ISSUES FOUND", Colors.YELLOW))
            print("Some requirements are not met. See recommendations below.")

        print()

    def _print_recommendations(self, recommendations: List[str]) -> None:
        """Print recommendations for fixing issues."""
        print(self._colorize("💡 Recommendations", Colors.BLUE))
        print("-" * 30)

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

        print()

    def generate_recommendations(self, validation_results: Dict[str, Any],
                               system_info: Dict[str, Any],
                               requirements: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        # Python version recommendations
        python_result = validation_results.get('python_version', {})
        if isinstance(python_result, dict) and not python_result.get('valid'):
            required = python_result.get('required', 'unknown')
            recommendations.append(
                f"Upgrade Python to version {required} or higher. "
                f"Visit https://python.org/downloads/ for installation instructions."
            )
        elif isinstance(python_result, bool) and not python_result:
            min_version = requirements.get('python_min_version', (3, 9))
            recommendations.append(
                f"Upgrade Python to version {min_version[0]}.{min_version[1]} or higher."
            )

        # Package recommendations
        package_result = validation_results.get('package_check', {})
        if isinstance(package_result, dict):
            missing = package_result.get('missing_required', [])
            if missing:
                if len(missing) == 1:
                    recommendations.append(f"Install missing package: pip install {missing[0]}")
                else:
                    package_list = ' '.join(missing)
                    recommendations.append(f"Install missing packages: pip install {package_list}")

                # Add alternative installation methods
                recommendations.append(
                    "Alternatively, install all requirements: pip install -r requirements.txt"
                )

        elif isinstance(package_result, bool) and not package_result:
            recommendations.append("Install required Python packages using: pip install -r requirements.txt")

        # Memory recommendations
        memory_result = validation_results.get('memory_check', {})
        if isinstance(memory_result, dict) and not memory_result.get('valid') and not memory_result.get('assumed'):
            current = memory_result.get('current_gb', 0)
            required = memory_result.get('required_gb', 4)
            recommendations.append(
                f"Increase system memory from {current}GB to at least {required}GB for optimal performance."
            )

        # Disk space recommendations
        disk_result = validation_results.get('disk_space_check', {})
        if isinstance(disk_result, dict) and not disk_result.get('valid') and not disk_result.get('assumed'):
            current = disk_result.get('current_gb', 0)
            required = disk_result.get('required_gb', 10)
            recommendations.append(
                f"Free up disk space. Current: {current}GB, Required: {required}GB. "
                f"Consider clearing cache files or moving files to external storage."
            )

        # GPU recommendations
        gpu_result = validation_results.get('gpu_check', {})
        if gpu_result and gpu_result.get('cuda_required') and not gpu_result.get('valid'):
            recommendations.append(
                "Install NVIDIA GPU drivers and CUDA toolkit for GPU acceleration. "
                "Visit https://developer.nvidia.com/cuda-downloads for instructions."
            )

        # General recommendations
        if not recommendations:
            recommendations.append("Your system is fully compatible! No action required.")

        return recommendations

    def export_report(self, validation_results: Dict[str, Any],
                     system_info: Dict[str, Any],
                     requirements: Dict[str, Any],
                     output_path: str) -> bool:
        """Export compatibility report to JSON file."""
        try:
            self.logger.info(f"[REPORT] Exporting report to {output_path}")

            report_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_version': '1.0',
                    'tool': 'HuggingFace GUI Compatibility Checker'
                },
                'system_info': system_info,
                'requirements': requirements,
                'validation_results': validation_results,
                'recommendations': self.generate_recommendations(
                    validation_results, system_info, requirements
                ),
                'overall_compatible': validation_results.get('overall', False)
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"[REPORT] Report exported successfully")
            return True

        except Exception as e:
            self.logger.error(f"[REPORT] Failed to export report: {e}")
            return False

    def _colorize(self, text: str, color: str) -> str:
        """Apply color to text if colors are enabled."""
        if self.use_colors:
            return Colors.colorize(text, color)
        return text

    def log_message(self, level: str, message: str) -> None:
        """Log a colored message to console."""
        color_map = {
            'info': Colors.BLUE,
            'success': Colors.GREEN,
            'warning': Colors.YELLOW,
            'error': Colors.RED
        }

        prefix_map = {
            'info': 'ℹ️ ',
            'success': '✅ ',
            'warning': '⚠️ ',
            'error': '❌ '
        }

        color = color_map.get(level, Colors.NC)
        prefix = prefix_map.get(level, '')

        colored_message = self._colorize(f"{prefix}{message}", color)
        print(colored_message)

        # Also log to Python logger
        if hasattr(self.logger, level):
            getattr(self.logger, level)(message)

    def generate_summary_stats(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the validation results."""
        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        warnings = 0

        # Count different types of checks
        for key, result in validation_results.items():
            if key == 'overall':
                continue

            total_checks += 1

            if isinstance(result, bool):
                if result:
                    passed_checks += 1
                else:
                    failed_checks += 1
            elif isinstance(result, dict):
                if result.get('valid', False):
                    passed_checks += 1
                elif result.get('assumed', False):
                    warnings += 1
                else:
                    failed_checks += 1

        return {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': failed_checks,
            'warnings': warnings,
            'success_rate': round((passed_checks / total_checks * 100) if total_checks > 0 else 0, 1)
        }