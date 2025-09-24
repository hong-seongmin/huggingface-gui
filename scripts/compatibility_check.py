#!/usr/bin/env python3
"""
System Compatibility Check for HuggingFace GUI
Validates system requirements and configuration before running the application.
"""

import os
import sys
import platform
import subprocess
import shutil
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


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


class CompatibilityChecker:
    """System compatibility checker for HuggingFace GUI."""

    def __init__(self):
        self.requirements = {
            'python_min_version': (3, 9),
            'memory_min_gb': 4,
            'disk_space_min_gb': 10,
            'required_packages': [
                'streamlit',
                'transformers',
                'torch',
                'fastapi',
                'uvicorn',
                'huggingface_hub'
            ],
            'optional_packages': [
                'customtkinter',
                'GPUtil',
                'psutil',
                'plotly',
                'pandas'
            ],
            'system_commands': {
                'git': 'Git version control',
                'curl': 'HTTP client for downloads'
            }
        }
        self.results = {
            'system_info': {},
            'checks': {},
            'warnings': [],
            'errors': [],
            'recommendations': []
        }

    def log(self, level: str, message: str) -> None:
        """Log messages with color coding."""
        color_map = {
            'info': Colors.BLUE,
            'success': Colors.GREEN,
            'warning': Colors.YELLOW,
            'error': Colors.RED,
            'header': Colors.PURPLE
        }
        color = color_map.get(level, Colors.NC)
        prefix_map = {
            'info': '[INFO]',
            'success': '[SUCCESS]',
            'warning': '[WARNING]',
            'error': '[ERROR]',
            'header': '[HEADER]'
        }
        prefix = prefix_map.get(level, '')
        print(f"{color}{prefix} {message}{Colors.NC}")

    def get_system_info(self) -> Dict:
        """Collect system information."""
        self.log('info', 'Collecting system information...')

        try:
            # Basic system info
            system_info = {
                'os': platform.system(),
                'os_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'python_version': sys.version,
                'python_executable': sys.executable,
                'platform': platform.platform(),
                'hostname': platform.node()
            }

            # Memory information
            try:
                if platform.system() == 'Linux':
                    with open('/proc/meminfo', 'r') as f:
                        meminfo = f.read()
                    for line in meminfo.split('\n'):
                        if 'MemTotal:' in line:
                            memory_kb = int(line.split()[1])
                            system_info['memory_gb'] = round(memory_kb / 1024 / 1024, 1)
                            break
                elif platform.system() == 'Darwin':  # macOS
                    result = subprocess.run(['sysctl', 'hw.memsize'],
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        memory_bytes = int(result.stdout.split(': ')[1])
                        system_info['memory_gb'] = round(memory_bytes / 1024 / 1024 / 1024, 1)
                elif platform.system() == 'Windows':
                    try:
                        import psutil
                        memory_bytes = psutil.virtual_memory().total
                        system_info['memory_gb'] = round(memory_bytes / 1024 / 1024 / 1024, 1)
                    except ImportError:
                        system_info['memory_gb'] = 'Unknown'
            except Exception:
                system_info['memory_gb'] = 'Unknown'

            # Disk space
            try:
                import shutil
                total, used, free = shutil.disk_usage('.')
                system_info['disk_free_gb'] = round(free / 1024 / 1024 / 1024, 1)
                system_info['disk_total_gb'] = round(total / 1024 / 1024 / 1024, 1)
            except Exception:
                system_info['disk_free_gb'] = 'Unknown'
                system_info['disk_total_gb'] = 'Unknown'

            # GPU information
            system_info['gpu_info'] = self._get_gpu_info()

            # Environment variables
            system_info['env_vars'] = {
                'PATH': os.environ.get('PATH', ''),
                'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                'HF_HOME': os.environ.get('HF_HOME', ''),
                'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', '')
            }

            self.results['system_info'] = system_info
            return system_info

        except Exception as e:
            self.log('error', f'Failed to collect system information: {e}')
            return {}

    def _get_gpu_info(self) -> Dict:
        """Get GPU information."""
        gpu_info = {
            'nvidia_available': False,
            'cuda_available': False,
            'mps_available': False,
            'gpu_count': 0,
            'gpu_names': []
        }

        try:
            # Check NVIDIA GPU
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                                   '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_info['nvidia_available'] = True
                lines = result.stdout.strip().split('\n')
                gpu_info['gpu_count'] = len(lines)
                for line in lines:
                    if line.strip():
                        name, memory = line.split(', ')
                        gpu_info['gpu_names'].append(f"{name.strip()} ({memory.strip()}MB)")
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            pass

        try:
            # Check CUDA availability
            import torch
            if torch.cuda.is_available():
                gpu_info['cuda_available'] = True
                if gpu_info['gpu_count'] == 0:  # If nvidia-smi failed but CUDA is available
                    gpu_info['gpu_count'] = torch.cuda.device_count()
        except ImportError:
            pass

        try:
            # Check Apple Metal Performance Shaders (MPS)
            if platform.system() == 'Darwin':
                import torch
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    gpu_info['mps_available'] = True
        except ImportError:
            pass

        return gpu_info

    def check_python_version(self) -> bool:
        """Check if Python version meets requirements."""
        self.log('info', 'Checking Python version...')

        current_version = sys.version_info[:2]
        required_version = self.requirements['python_min_version']

        if current_version >= required_version:
            self.log('success', f'Python {sys.version.split()[0]} is compatible')
            self.results['checks']['python_version'] = True
            return True
        else:
            self.log('error', f'Python {sys.version.split()[0]} is too old. '
                              f'Minimum required: {required_version[0]}.{required_version[1]}')
            self.results['checks']['python_version'] = False
            self.results['errors'].append(
                f'Upgrade Python to {required_version[0]}.{required_version[1]}+'
            )
            return False

    def check_memory(self) -> bool:
        """Check system memory requirements."""
        self.log('info', 'Checking system memory...')

        memory_gb = self.results['system_info'].get('memory_gb')
        if memory_gb == 'Unknown':
            self.log('warning', 'Could not determine system memory')
            self.results['warnings'].append('Unable to check memory requirements')
            return True

        required_memory = self.requirements['memory_min_gb']
        if isinstance(memory_gb, (int, float)) and memory_gb >= required_memory:
            self.log('success', f'System memory: {memory_gb}GB (sufficient)')
            self.results['checks']['memory'] = True
            return True
        else:
            self.log('warning', f'System has {memory_gb}GB RAM. '
                               f'Minimum {required_memory}GB recommended.')
            self.results['warnings'].append(
                f'Low memory: {memory_gb}GB < {required_memory}GB recommended'
            )
            self.results['recommendations'].append(
                'Consider adding more RAM or using smaller models'
            )
            self.results['checks']['memory'] = False
            return False

    def check_disk_space(self) -> bool:
        """Check available disk space."""
        self.log('info', 'Checking disk space...')

        disk_free = self.results['system_info'].get('disk_free_gb')
        if disk_free == 'Unknown':
            self.log('warning', 'Could not determine disk space')
            self.results['warnings'].append('Unable to check disk space')
            return True

        required_space = self.requirements['disk_space_min_gb']
        if isinstance(disk_free, (int, float)) and disk_free >= required_space:
            self.log('success', f'Available disk space: {disk_free}GB (sufficient)')
            self.results['checks']['disk_space'] = True
            return True
        else:
            self.log('warning', f'Low disk space: {disk_free}GB available. '
                               f'Minimum {required_space}GB recommended.')
            self.results['warnings'].append(
                f'Low disk space: {disk_free}GB < {required_space}GB recommended'
            )
            self.results['recommendations'].append(
                'Free up disk space or move model cache to larger drive'
            )
            self.results['checks']['disk_space'] = False
            return False

    def check_packages(self) -> bool:
        """Check if required Python packages are installed."""
        self.log('info', 'Checking Python packages...')

        missing_required = []
        missing_optional = []

        # Check required packages
        for package in self.requirements['required_packages']:
            try:
                __import__(package)
                self.log('success', f'✓ {package}')
            except ImportError:
                self.log('error', f'✗ {package} (required)')
                missing_required.append(package)

        # Check optional packages
        for package in self.requirements['optional_packages']:
            try:
                __import__(package)
                self.log('success', f'✓ {package} (optional)')
            except ImportError:
                self.log('warning', f'○ {package} (optional)')
                missing_optional.append(package)

        if missing_required:
            self.results['errors'].extend([
                f'Missing required package: {pkg}' for pkg in missing_required
            ])
            self.results['recommendations'].append(
                f'Install missing packages: pip install {" ".join(missing_required)}'
            )
            self.results['checks']['required_packages'] = False
            return False

        if missing_optional:
            self.results['warnings'].extend([
                f'Missing optional package: {pkg}' for pkg in missing_optional
            ])
            self.results['recommendations'].append(
                f'Install optional packages: pip install {" ".join(missing_optional)}'
            )

        self.results['checks']['required_packages'] = True
        return True

    def check_system_commands(self) -> bool:
        """Check if required system commands are available."""
        self.log('info', 'Checking system commands...')

        all_available = True
        for cmd, description in self.requirements['system_commands'].items():
            if shutil.which(cmd):
                self.log('success', f'✓ {cmd} - {description}')
            else:
                self.log('warning', f'○ {cmd} - {description} (not found)')
                self.results['warnings'].append(f'Missing command: {cmd}')
                all_available = False

        self.results['checks']['system_commands'] = all_available
        return all_available

    def check_gpu_acceleration(self) -> bool:
        """Check GPU acceleration capabilities."""
        self.log('info', 'Checking GPU acceleration...')

        gpu_info = self.results['system_info']['gpu_info']

        if gpu_info['cuda_available']:
            self.log('success', '✓ CUDA GPU acceleration available')
            if gpu_info['gpu_names']:
                for gpu in gpu_info['gpu_names']:
                    self.log('info', f'  - {gpu}')
            self.results['checks']['gpu_acceleration'] = True
            return True
        elif gpu_info['mps_available']:
            self.log('success', '✓ Apple MPS acceleration available')
            self.results['checks']['gpu_acceleration'] = True
            return True
        else:
            self.log('warning', '○ No GPU acceleration detected (CPU only)')
            self.results['warnings'].append('No GPU acceleration available')
            self.results['recommendations'].append(
                'Consider using a GPU for better performance with large models'
            )
            self.results['checks']['gpu_acceleration'] = False
            return False

    def check_environment_config(self) -> bool:
        """Check environment configuration."""
        self.log('info', 'Checking environment configuration...')

        project_dir = Path.cwd()
        issues = []

        # Check for .env file
        env_file = project_dir / '.env'
        env_example = project_dir / '.env.example'

        if env_file.exists():
            self.log('success', '✓ .env configuration file found')
        elif env_example.exists():
            self.log('warning', '○ .env file missing, but .env.example exists')
            issues.append('Create .env file from .env.example template')
        else:
            self.log('warning', '○ No environment configuration files found')
            issues.append('Create .env configuration file')

        # Check critical directories
        cache_dir = os.environ.get('HF_MODEL_CACHE_DIR', '/tmp/hf_model_cache')
        if not os.path.exists(cache_dir):
            self.log('warning', f'○ Model cache directory does not exist: {cache_dir}')
            issues.append(f'Create model cache directory: {cache_dir}')

        if issues:
            self.results['warnings'].extend(issues)
            self.results['checks']['environment_config'] = False
            return False

        self.results['checks']['environment_config'] = True
        return True

    def generate_report(self) -> str:
        """Generate a comprehensive compatibility report."""
        report = []
        report.append("=" * 60)
        report.append("🔍 HuggingFace GUI Compatibility Report")
        report.append("=" * 60)
        report.append("")

        # System Information
        report.append("📋 System Information:")
        report.append("-" * 30)
        system_info = self.results['system_info']
        report.append(f"OS: {system_info.get('os', 'Unknown')} {system_info.get('os_version', '')}")
        report.append(f"Architecture: {system_info.get('architecture', 'Unknown')}")
        report.append(f"Python: {system_info.get('python_version', 'Unknown').split()[0]}")
        report.append(f"Memory: {system_info.get('memory_gb', 'Unknown')}GB")
        report.append(f"Disk Space: {system_info.get('disk_free_gb', 'Unknown')}GB free")

        gpu_info = system_info.get('gpu_info', {})
        if gpu_info.get('cuda_available'):
            report.append(f"GPU: CUDA ({gpu_info.get('gpu_count', 0)} devices)")
        elif gpu_info.get('mps_available'):
            report.append("GPU: Apple MPS")
        else:
            report.append("GPU: Not available")
        report.append("")

        # Compatibility Checks
        report.append("✅ Compatibility Checks:")
        report.append("-" * 30)
        checks = self.results['checks']
        for check, passed in checks.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            check_name = check.replace('_', ' ').title()
            report.append(f"{status} - {check_name}")
        report.append("")

        # Warnings
        if self.results['warnings']:
            report.append("⚠️  Warnings:")
            report.append("-" * 30)
            for warning in self.results['warnings']:
                report.append(f"• {warning}")
            report.append("")

        # Errors
        if self.results['errors']:
            report.append("❌ Errors:")
            report.append("-" * 30)
            for error in self.results['errors']:
                report.append(f"• {error}")
            report.append("")

        # Recommendations
        if self.results['recommendations']:
            report.append("💡 Recommendations:")
            report.append("-" * 30)
            for rec in self.results['recommendations']:
                report.append(f"• {rec}")
            report.append("")

        # Overall Status
        all_critical_passed = (
            checks.get('python_version', False) and
            checks.get('required_packages', False)
        )

        report.append("🎯 Overall Status:")
        report.append("-" * 30)
        if all_critical_passed:
            if len(self.results['warnings']) == 0:
                report.append("✅ System is fully compatible - ready to run!")
            else:
                report.append("⚠️  System is compatible with warnings - should work but may have issues")
        else:
            report.append("❌ System has critical issues - please resolve errors before running")

        return "\n".join(report)

    def run_all_checks(self) -> bool:
        """Run all compatibility checks."""
        self.log('header', 'Starting HuggingFace GUI compatibility check...')
        print()

        # Collect system information
        self.get_system_info()

        # Run all checks
        checks = [
            self.check_python_version(),
            self.check_memory(),
            self.check_disk_space(),
            self.check_packages(),
            self.check_system_commands(),
            self.check_gpu_acceleration(),
            self.check_environment_config()
        ]

        print()

        # Generate and display report
        report = self.generate_report()
        print(report)

        # Return overall success (critical checks only)
        return (
            self.results['checks'].get('python_version', False) and
            self.results['checks'].get('required_packages', False)
        )

    def save_report(self, filename: str = 'compatibility_report.json') -> None:
        """Save detailed results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            self.log('success', f'Detailed report saved to {filename}')
        except Exception as e:
            self.log('error', f'Failed to save report: {e}')


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='HuggingFace GUI Compatibility Checker')
    parser.add_argument('--save-report', metavar='FILE',
                       help='Save detailed report to JSON file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Only show errors and final result')
    args = parser.parse_args()

    checker = CompatibilityChecker()

    # Suppress some output in quiet mode
    if args.quiet:
        # Redirect stdout temporarily for checks
        import io
        import contextlib

        old_stdout = sys.stdout
        captured_output = io.StringIO()

        try:
            with contextlib.redirect_stdout(captured_output):
                success = checker.run_all_checks()
        finally:
            sys.stdout = old_stdout

        # Only show the final report
        report = checker.generate_report()
        print(report)
    else:
        success = checker.run_all_checks()

    if args.save_report:
        checker.save_report(args.save_report)

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()