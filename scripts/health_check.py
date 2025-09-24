#!/usr/bin/env python3
"""
Health Check Script for HuggingFace GUI
Performs runtime health checks on the running application and system components.
"""

import os
import sys
import time
import json
import requests
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from datetime import datetime


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


class HealthChecker:
    """Application and system health checker."""

    def __init__(self, streamlit_port: int = 8501, fastapi_port: int = 8000):
        self.streamlit_port = streamlit_port
        self.fastapi_port = fastapi_port
        self.streamlit_url = f"http://localhost:{streamlit_port}"
        self.fastapi_url = f"http://localhost:{fastapi_port}"
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'performance': {},
            'warnings': [],
            'errors': []
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

    def check_streamlit_health(self) -> bool:
        """Check if Streamlit application is running and healthy."""
        self.log('info', 'Checking Streamlit application health...')

        try:
            # Check Streamlit health endpoint
            response = requests.get(f"{self.streamlit_url}/_stcore/health", timeout=10)
            if response.status_code == 200:
                self.log('success', f'Streamlit is running on port {self.streamlit_port}')
                self.results['checks']['streamlit_health'] = True

                # Measure response time
                start_time = time.time()
                requests.get(self.streamlit_url, timeout=10)
                response_time = (time.time() - start_time) * 1000
                self.results['performance']['streamlit_response_time_ms'] = round(response_time, 2)

                if response_time > 5000:  # 5 seconds
                    self.log('warning', f'Streamlit is slow to respond: {response_time:.0f}ms')
                    self.results['warnings'].append('Streamlit response time is high')

                return True
            else:
                self.log('error', f'Streamlit health check failed: HTTP {response.status_code}')
                self.results['checks']['streamlit_health'] = False
                return False

        except requests.exceptions.ConnectionError:
            self.log('error', f'Cannot connect to Streamlit on port {self.streamlit_port}')
            self.results['checks']['streamlit_health'] = False
            self.results['errors'].append('Streamlit is not running or not accessible')
            return False
        except requests.exceptions.Timeout:
            self.log('error', 'Streamlit health check timed out')
            self.results['checks']['streamlit_health'] = False
            self.results['errors'].append('Streamlit is not responding')
            return False
        except Exception as e:
            self.log('error', f'Streamlit health check failed: {str(e)}')
            self.results['checks']['streamlit_health'] = False
            return False

    def check_fastapi_health(self) -> bool:
        """Check if FastAPI server is running and healthy."""
        self.log('info', 'Checking FastAPI server health...')

        try:
            # Check FastAPI root endpoint
            response = requests.get(f"{self.fastapi_url}/", timeout=10)
            if response.status_code == 200:
                self.log('success', f'FastAPI server is running on port {self.fastapi_port}')
                self.results['checks']['fastapi_health'] = True

                # Check specific endpoints
                endpoints_to_check = ['/models', '/system/status']
                for endpoint in endpoints_to_check:
                    try:
                        resp = requests.get(f"{self.fastapi_url}{endpoint}", timeout=5)
                        if resp.status_code == 200:
                            self.log('success', f'✓ {endpoint} endpoint accessible')
                        else:
                            self.log('warning', f'○ {endpoint} endpoint returned {resp.status_code}')
                    except Exception:
                        self.log('warning', f'○ {endpoint} endpoint not accessible')

                return True
            else:
                self.log('error', f'FastAPI server health check failed: HTTP {response.status_code}')
                self.results['checks']['fastapi_health'] = False
                return False

        except requests.exceptions.ConnectionError:
            self.log('warning', f'FastAPI server is not running on port {self.fastapi_port}')
            self.results['checks']['fastapi_health'] = False
            # This is a warning, not an error, as FastAPI might not be started
            return False
        except requests.exceptions.Timeout:
            self.log('error', 'FastAPI server health check timed out')
            self.results['checks']['fastapi_health'] = False
            self.results['errors'].append('FastAPI server is not responding')
            return False
        except Exception as e:
            self.log('error', f'FastAPI health check failed: {str(e)}')
            self.results['checks']['fastapi_health'] = False
            return False

    def check_system_resources(self) -> bool:
        """Check system resource usage."""
        self.log('info', 'Checking system resources...')

        try:
            import psutil

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.results['performance']['cpu_usage_percent'] = cpu_percent

            if cpu_percent > 90:
                self.log('error', f'High CPU usage: {cpu_percent}%')
                self.results['errors'].append(f'High CPU usage: {cpu_percent}%')
            elif cpu_percent > 70:
                self.log('warning', f'Moderate CPU usage: {cpu_percent}%')
                self.results['warnings'].append(f'Moderate CPU usage: {cpu_percent}%')
            else:
                self.log('success', f'CPU usage: {cpu_percent}%')

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / 1024 / 1024 / 1024

            self.results['performance']['memory_usage_percent'] = memory_percent
            self.results['performance']['memory_available_gb'] = round(memory_available_gb, 1)

            if memory_percent > 90:
                self.log('error', f'High memory usage: {memory_percent}%')
                self.results['errors'].append(f'High memory usage: {memory_percent}%')
            elif memory_percent > 75:
                self.log('warning', f'Moderate memory usage: {memory_percent}%')
                self.results['warnings'].append(f'Moderate memory usage: {memory_percent}%')
            else:
                self.log('success', f'Memory usage: {memory_percent}% ({memory_available_gb:.1f}GB available)')

            # Disk usage
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / 1024 / 1024 / 1024

            self.results['performance']['disk_usage_percent'] = round(disk_percent, 1)
            self.results['performance']['disk_free_gb'] = round(disk_free_gb, 1)

            if disk_percent > 95:
                self.log('error', f'Very low disk space: {disk_percent:.1f}% used')
                self.results['errors'].append('Very low disk space')
            elif disk_percent > 85:
                self.log('warning', f'Low disk space: {disk_percent:.1f}% used')
                self.results['warnings'].append('Low disk space')
            else:
                self.log('success', f'Disk usage: {disk_percent:.1f}% ({disk_free_gb:.1f}GB free)')

            self.results['checks']['system_resources'] = True
            return True

        except ImportError:
            self.log('warning', 'psutil not available - cannot check system resources')
            self.results['warnings'].append('System resource monitoring unavailable')
            self.results['checks']['system_resources'] = False
            return False
        except Exception as e:
            self.log('error', f'System resource check failed: {str(e)}')
            self.results['checks']['system_resources'] = False
            return False

    def check_gpu_status(self) -> bool:
        """Check GPU status and usage."""
        self.log('info', 'Checking GPU status...')

        try:
            # Try NVIDIA GPU first
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=10
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []

                for i, line in enumerate(lines):
                    if line.strip():
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 4:
                            gpu_util, mem_util, mem_used, mem_total = parts
                            gpu_data = {
                                'gpu_id': i,
                                'utilization_percent': int(gpu_util),
                                'memory_utilization_percent': int(mem_util),
                                'memory_used_mb': int(mem_used),
                                'memory_total_mb': int(mem_total)
                            }
                            gpu_info.append(gpu_data)

                            self.log('success', f'GPU {i}: {gpu_util}% util, {mem_util}% mem ({mem_used}/{mem_total}MB)')

                            # Check for high usage
                            if int(gpu_util) > 95:
                                self.results['warnings'].append(f'GPU {i} utilization very high: {gpu_util}%')
                            if int(mem_util) > 90:
                                self.results['warnings'].append(f'GPU {i} memory usage high: {mem_util}%')

                self.results['performance']['gpu_info'] = gpu_info
                self.results['checks']['gpu_status'] = True
                return True

        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for Apple MPS
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.log('success', 'Apple MPS GPU acceleration available')
                self.results['performance']['gpu_type'] = 'MPS'
                self.results['checks']['gpu_status'] = True
                return True
        except ImportError:
            pass

        # No GPU found
        self.log('warning', 'No GPU acceleration detected')
        self.results['checks']['gpu_status'] = False
        return False

    def check_model_cache(self) -> bool:
        """Check model cache directory and usage."""
        self.log('info', 'Checking model cache...')

        # Get cache directory from environment or use default
        cache_dir = os.environ.get('HF_MODEL_CACHE_DIR', '/tmp/hf_model_cache')
        cache_path = Path(cache_dir)

        try:
            if not cache_path.exists():
                self.log('warning', f'Model cache directory does not exist: {cache_dir}')
                self.results['warnings'].append('Model cache directory missing')
                self.results['checks']['model_cache'] = False
                return False

            # Calculate cache size
            total_size = 0
            model_count = 0
            for item in cache_path.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                    if item.name == 'config.json':  # Count models by config files
                        model_count += 1

            cache_size_gb = total_size / 1024 / 1024 / 1024
            self.results['performance']['cache_size_gb'] = round(cache_size_gb, 2)
            self.results['performance']['cached_models_count'] = model_count

            self.log('success', f'Model cache: {cache_size_gb:.2f}GB ({model_count} models)')

            # Check if cache is getting large
            if cache_size_gb > 50:
                self.log('warning', f'Large model cache: {cache_size_gb:.2f}GB')
                self.results['warnings'].append('Model cache is large - consider cleanup')
            elif cache_size_gb > 100:
                self.log('error', f'Very large model cache: {cache_size_gb:.2f}GB')
                self.results['errors'].append('Model cache is very large')

            self.results['checks']['model_cache'] = True
            return True

        except Exception as e:
            self.log('error', f'Model cache check failed: {str(e)}')
            self.results['checks']['model_cache'] = False
            return False

    def check_log_files(self) -> bool:
        """Check application log files."""
        self.log('info', 'Checking log files...')

        log_files = [
            'app_debug.log',
            'logs/app.log',
            'logs/error.log'
        ]

        issues_found = False
        for log_file in log_files:
            log_path = Path(log_file)
            if log_path.exists():
                try:
                    size_mb = log_path.stat().st_size / 1024 / 1024
                    if size_mb > 100:  # 100MB
                        self.log('warning', f'Large log file: {log_file} ({size_mb:.1f}MB)')
                        self.results['warnings'].append(f'Large log file: {log_file}')
                        issues_found = True
                    else:
                        self.log('success', f'✓ {log_file} ({size_mb:.1f}MB)')

                    # Check for recent errors
                    try:
                        with open(log_path, 'r') as f:
                            last_lines = f.readlines()[-50:]  # Last 50 lines

                        error_count = sum(1 for line in last_lines if 'ERROR' in line.upper())
                        if error_count > 0:
                            self.log('warning', f'Recent errors found in {log_file}: {error_count}')
                            self.results['warnings'].append(f'Recent errors in {log_file}')
                            issues_found = True
                    except Exception:
                        pass

                except Exception as e:
                    self.log('warning', f'Could not check {log_file}: {str(e)}')

        if not issues_found:
            self.log('success', 'Log files look healthy')

        self.results['checks']['log_files'] = not issues_found
        return not issues_found

    def check_environment_variables(self) -> bool:
        """Check important environment variables."""
        self.log('info', 'Checking environment variables...')

        important_vars = {
            'HF_MODEL_CACHE_DIR': 'Model cache directory',
            'CUDA_VISIBLE_DEVICES': 'GPU device selection',
            'HF_TOKEN': 'HuggingFace authentication token',
            'TRANSFORMERS_OFFLINE': 'Offline mode setting',
            'TOKENIZERS_PARALLELISM': 'Tokenizer parallelism setting'
        }

        warnings_found = False
        for var, description in important_vars.items():
            value = os.environ.get(var)
            if value:
                # Don't log sensitive tokens
                if 'TOKEN' in var:
                    self.log('success', f'✓ {var} is set (hidden)')
                else:
                    self.log('success', f'✓ {var} = {value}')
            else:
                self.log('info', f'○ {var} not set ({description})')

        # Check for potentially problematic settings
        if os.environ.get('TOKENIZERS_PARALLELISM') == 'true':
            self.log('warning', 'TOKENIZERS_PARALLELISM=true may cause issues in some environments')
            self.results['warnings'].append('TOKENIZERS_PARALLELISM may cause conflicts')
            warnings_found = True

        self.results['checks']['environment_variables'] = not warnings_found
        return not warnings_found

    def run_comprehensive_health_check(self) -> bool:
        """Run all health checks."""
        self.log('header', 'Starting comprehensive health check...')
        print()

        checks = [
            ('Streamlit Health', self.check_streamlit_health),
            ('FastAPI Health', self.check_fastapi_health),
            ('System Resources', self.check_system_resources),
            ('GPU Status', self.check_gpu_status),
            ('Model Cache', self.check_model_cache),
            ('Log Files', self.check_log_files),
            ('Environment Variables', self.check_environment_variables)
        ]

        passed_checks = 0
        total_checks = len(checks)

        for check_name, check_func in checks:
            try:
                if check_func():
                    passed_checks += 1
                print()  # Add spacing between checks
            except Exception as e:
                self.log('error', f'Check {check_name} failed with exception: {str(e)}')
                print()

        # Generate summary
        self.log('header', 'Health Check Summary')
        print('=' * 50)

        success_rate = (passed_checks / total_checks) * 100
        self.results['performance']['health_score'] = round(success_rate, 1)

        if success_rate == 100:
            self.log('success', f'🎉 All checks passed! ({passed_checks}/{total_checks})')
            overall_status = True
        elif success_rate >= 75:
            self.log('warning', f'⚠️  Most checks passed ({passed_checks}/{total_checks}) - {success_rate:.1f}%')
            overall_status = True
        else:
            self.log('error', f'❌ Multiple issues detected ({passed_checks}/{total_checks}) - {success_rate:.1f}%')
            overall_status = False

        # Summary of warnings and errors
        if self.results['warnings']:
            print(f"\n{Colors.YELLOW}Warnings:{Colors.NC}")
            for warning in self.results['warnings']:
                print(f"  • {warning}")

        if self.results['errors']:
            print(f"\n{Colors.RED}Errors:{Colors.NC}")
            for error in self.results['errors']:
                print(f"  • {error}")

        return overall_status

    def save_results(self, filename: str = 'health_check_results.json') -> None:
        """Save health check results to JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.log('success', f'Health check results saved to {filename}')
        except Exception as e:
            self.log('error', f'Failed to save results: {str(e)}')


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='HuggingFace GUI Health Checker')
    parser.add_argument('--streamlit-port', type=int, default=8501,
                       help='Streamlit port (default: 8501)')
    parser.add_argument('--fastapi-port', type=int, default=8000,
                       help='FastAPI port (default: 8000)')
    parser.add_argument('--save-results', metavar='FILE',
                       help='Save results to JSON file')
    parser.add_argument('--quick', action='store_true',
                       help='Run only essential checks')
    args = parser.parse_args()

    checker = HealthChecker(args.streamlit_port, args.fastapi_port)

    if args.quick:
        # Quick check - only essential services
        success = (checker.check_streamlit_health() and
                  checker.check_system_resources())
    else:
        # Full comprehensive check
        success = checker.run_comprehensive_health_check()

    if args.save_results:
        checker.save_results(args.save_results)

    print()
    if success:
        print(f"{Colors.GREEN}✅ Health check completed successfully{Colors.NC}")
    else:
        print(f"{Colors.RED}❌ Health check found issues that need attention{Colors.NC}")

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()