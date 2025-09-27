"""
Service Checker for health monitoring.

This module checks the health and availability of application services
including Streamlit UI, FastAPI backend, and related endpoints.
"""

import requests
import time
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class ServiceChecker:
    """Checks health and availability of application services."""

    def __init__(self, streamlit_port: int = 8501, fastapi_port: int = 8000,
                 timeout: int = 10):
        """Initialize service checker with configurable ports and timeout."""
        self.streamlit_port = streamlit_port
        self.fastapi_port = fastapi_port
        self.timeout = timeout

        self.streamlit_url = f"http://localhost:{streamlit_port}"
        self.fastapi_url = f"http://localhost:{fastapi_port}"

        self.logger = logging.getLogger("ServiceChecker")

    def check_all_services(self) -> Dict[str, Any]:
        """Check health of all services."""
        self.logger.info("[SERVICE] Starting comprehensive service health check")

        results = {
            'timestamp': datetime.now().isoformat(),
            'streamlit': self.check_streamlit_health(),
            'fastapi': self.check_fastapi_health(),
            'overall_status': 'unknown',
            'services_running': 0,
            'total_services': 2
        }

        # Calculate overall status
        streamlit_ok = results['streamlit'].get('status') == 'healthy'
        fastapi_ok = results['fastapi'].get('status') == 'healthy'

        running_count = sum([streamlit_ok, fastapi_ok])
        results['services_running'] = running_count

        if running_count == 2:
            results['overall_status'] = 'healthy'
        elif running_count == 1:
            results['overall_status'] = 'partial'
        else:
            results['overall_status'] = 'unhealthy'

        self.logger.info(f"[SERVICE] Service check completed: {results['overall_status']}")
        return results

    def check_streamlit_health(self) -> Dict[str, Any]:
        """Check Streamlit application health."""
        self.logger.info("[SERVICE] Checking Streamlit health")

        result = {
            'service': 'streamlit',
            'url': self.streamlit_url,
            'port': self.streamlit_port,
            'status': 'unknown',
            'response_time_ms': None,
            'error': None,
            'details': {}
        }

        try:
            start_time = time.time()

            # Check if Streamlit is accessible
            response = requests.get(self.streamlit_url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000

            result['response_time_ms'] = round(response_time, 2)

            if response.status_code == 200:
                result['status'] = 'healthy'
                result['details'] = {
                    'http_status': response.status_code,
                    'content_type': response.headers.get('content-type'),
                    'server': response.headers.get('server'),
                    'content_length': len(response.content)
                }

                # Check if it's actually Streamlit content
                if 'streamlit' in response.text.lower() or 'st-emotion-cache' in response.text:
                    result['details']['streamlit_detected'] = True
                else:
                    result['details']['streamlit_detected'] = False
                    result['status'] = 'unhealthy'
                    result['error'] = 'Response does not appear to be from Streamlit'

            else:
                result['status'] = 'unhealthy'
                result['error'] = f'HTTP {response.status_code}'

        except requests.exceptions.ConnectionError:
            result['status'] = 'not_running'
            result['error'] = 'Connection refused'
        except requests.exceptions.Timeout:
            result['status'] = 'timeout'
            result['error'] = f'Request timed out after {self.timeout}s'
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def check_fastapi_health(self) -> Dict[str, Any]:
        """Check FastAPI backend health."""
        self.logger.info("[SERVICE] Checking FastAPI health")

        result = {
            'service': 'fastapi',
            'url': self.fastapi_url,
            'port': self.fastapi_port,
            'status': 'unknown',
            'response_time_ms': None,
            'error': None,
            'details': {},
            'endpoints': {}
        }

        try:
            start_time = time.time()

            # Check root endpoint
            response = requests.get(self.fastapi_url, timeout=self.timeout)
            response_time = (time.time() - start_time) * 1000

            result['response_time_ms'] = round(response_time, 2)

            if response.status_code == 200:
                result['status'] = 'healthy'
                result['details'] = {
                    'http_status': response.status_code,
                    'content_type': response.headers.get('content-type'),
                    'server': response.headers.get('server')
                }

                # Check specific endpoints
                result['endpoints'] = self._check_fastapi_endpoints()

            else:
                result['status'] = 'unhealthy'
                result['error'] = f'HTTP {response.status_code}'

        except requests.exceptions.ConnectionError:
            result['status'] = 'not_running'
            result['error'] = 'Connection refused'
        except requests.exceptions.Timeout:
            result['status'] = 'timeout'
            result['error'] = f'Request timed out after {self.timeout}s'
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def _check_fastapi_endpoints(self) -> Dict[str, Dict[str, Any]]:
        """Check specific FastAPI endpoints."""
        endpoints_to_check = {
            '/docs': 'OpenAPI documentation',
            '/openapi.json': 'OpenAPI specification',
            '/models': 'Model management endpoint',
            '/system/status': 'System status endpoint',
            '/health': 'Health check endpoint'
        }

        endpoint_results = {}

        for endpoint_path, description in endpoints_to_check.items():
            endpoint_result = {
                'path': endpoint_path,
                'description': description,
                'status': 'unknown',
                'response_time_ms': None,
                'error': None
            }

            try:
                start_time = time.time()
                url = f"{self.fastapi_url}{endpoint_path}"
                response = requests.get(url, timeout=5)
                response_time = (time.time() - start_time) * 1000

                endpoint_result['response_time_ms'] = round(response_time, 2)

                if response.status_code == 200:
                    endpoint_result['status'] = 'healthy'
                elif response.status_code == 404:
                    endpoint_result['status'] = 'not_found'
                elif response.status_code >= 500:
                    endpoint_result['status'] = 'server_error'
                else:
                    endpoint_result['status'] = 'client_error'

            except requests.exceptions.ConnectionError:
                endpoint_result['status'] = 'connection_error'
                endpoint_result['error'] = 'Connection refused'
            except requests.exceptions.Timeout:
                endpoint_result['status'] = 'timeout'
                endpoint_result['error'] = 'Request timeout'
            except Exception as e:
                endpoint_result['status'] = 'error'
                endpoint_result['error'] = str(e)

            endpoint_results[endpoint_path] = endpoint_result

        return endpoint_results

    def check_port_availability(self, port: int) -> Dict[str, Any]:
        """Check if a specific port is available or in use."""
        result = {
            'port': port,
            'available': False,
            'process_using': None,
            'error': None
        }

        try:
            import socket

            # Try to bind to the port
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)

            try:
                sock.bind(('localhost', port))
                result['available'] = True
            except OSError:
                result['available'] = False

                # Try to find what process is using the port
                try:
                    import psutil
                    for proc in psutil.process_iter(['pid', 'name', 'connections']):
                        try:
                            for conn in proc.info['connections']:
                                if conn.laddr.port == port:
                                    result['process_using'] = {
                                        'pid': proc.info['pid'],
                                        'name': proc.info['name']
                                    }
                                    break
                            if result['process_using']:
                                break
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                except ImportError:
                    pass  # psutil not available

            finally:
                sock.close()

        except Exception as e:
            result['error'] = str(e)

        return result

    def get_service_processes(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get information about running service processes."""
        processes = {
            'streamlit': [],
            'fastapi': [],
            'uvicorn': [],
            'python': []
        }

        try:
            import psutil

            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
                try:
                    proc_info = proc.info
                    name = proc_info.get('name', '').lower()
                    cmdline = ' '.join(proc_info.get('cmdline', [])).lower()

                    process_data = {
                        'pid': proc_info['pid'],
                        'name': proc_info['name'],
                        'cmdline': proc_info.get('cmdline', []),
                        'cpu_percent': proc_info.get('cpu_percent', 0),
                        'memory_percent': proc_info.get('memory_percent', 0)
                    }

                    # Categorize processes
                    if 'streamlit' in name or 'streamlit' in cmdline:
                        processes['streamlit'].append(process_data)
                    elif 'fastapi' in cmdline or 'uvicorn' in name or 'uvicorn' in cmdline:
                        if 'uvicorn' in name or 'uvicorn' in cmdline:
                            processes['uvicorn'].append(process_data)
                        if 'fastapi' in cmdline:
                            processes['fastapi'].append(process_data)
                    elif name == 'python' or name == 'python3':
                        processes['python'].append(process_data)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except ImportError:
            self.logger.warning("[SERVICE] psutil not available for process checking")

        return processes

    def check_service_dependencies(self) -> Dict[str, Any]:
        """Check if service dependencies are available."""
        dependencies = {
            'python_packages': {},
            'system_commands': {},
            'overall_status': 'unknown'
        }

        # Check Python packages
        packages_to_check = [
            'streamlit', 'fastapi', 'uvicorn', 'requests',
            'transformers', 'torch', 'huggingface_hub'
        ]

        for package in packages_to_check:
            try:
                __import__(package)
                dependencies['python_packages'][package] = {
                    'available': True,
                    'version': self._get_package_version(package)
                }
            except ImportError:
                dependencies['python_packages'][package] = {
                    'available': False,
                    'version': None
                }

        # Check system commands
        commands_to_check = ['python', 'python3', 'pip', 'pip3']

        for command in commands_to_check:
            try:
                result = subprocess.run(
                    [command, '--version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                dependencies['system_commands'][command] = {
                    'available': result.returncode == 0,
                    'version': result.stdout.strip() if result.returncode == 0 else None
                }
            except (subprocess.TimeoutExpired, FileNotFoundError):
                dependencies['system_commands'][command] = {
                    'available': False,
                    'version': None
                }

        # Determine overall status
        all_packages_available = all(
            pkg['available'] for pkg in dependencies['python_packages'].values()
        )
        essential_commands_available = any(
            dependencies['system_commands'].get(cmd, {}).get('available', False)
            for cmd in ['python', 'python3']
        )

        if all_packages_available and essential_commands_available:
            dependencies['overall_status'] = 'healthy'
        elif essential_commands_available:
            dependencies['overall_status'] = 'partial'
        else:
            dependencies['overall_status'] = 'unhealthy'

        return dependencies

    def _get_package_version(self, package_name: str) -> Optional[str]:
        """Get version of an installed package."""
        try:
            module = __import__(package_name)
            # Try different version attributes
            for attr in ['__version__', 'version', 'VERSION']:
                if hasattr(module, attr):
                    return str(getattr(module, attr))
            return 'unknown'
        except Exception:
            return None

    def ping_service(self, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Simple ping check for any service URL."""
        result = {
            'url': url,
            'status': 'unknown',
            'response_time_ms': None,
            'error': None
        }

        try:
            start_time = time.time()
            response = requests.get(url, timeout=timeout)
            response_time = (time.time() - start_time) * 1000

            result['response_time_ms'] = round(response_time, 2)
            result['status'] = 'reachable' if response.status_code == 200 else 'unreachable'

        except requests.exceptions.ConnectionError:
            result['status'] = 'unreachable'
            result['error'] = 'Connection refused'
        except requests.exceptions.Timeout:
            result['status'] = 'timeout'
            result['error'] = f'Request timed out after {timeout}s'
        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result