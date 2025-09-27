"""
System Information Collector for compatibility checking.

This module handles system information collection including platform details,
memory, disk space, GPU information, and environment variables.
"""

import os
import sys
import platform
import subprocess
import shutil
from typing import Dict, Any, Optional
import logging


class SystemInfoCollector:
    """Collects comprehensive system information for compatibility checking."""

    def __init__(self):
        """Initialize system info collector."""
        self.logger = logging.getLogger("SystemInfoCollector")

    def collect_all_info(self) -> Dict[str, Any]:
        """Collect all system information."""
        try:
            self.logger.info("[SYSTEM] Starting system information collection")

            system_info = {
                'python_version': sys.version,
                'python_executable': sys.executable,
                'platform': platform.platform(),
                'hostname': platform.node(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'system': platform.system(),
                'release': platform.release(),
            }

            # Add detailed system information
            system_info.update(self._collect_memory_info())
            system_info.update(self._collect_disk_info())
            system_info.update(self._collect_gpu_info())
            system_info.update(self._collect_environment_info())
            system_info.update(self._collect_python_info())

            self.logger.info("[SYSTEM] System information collection completed")
            return system_info

        except Exception as e:
            self.logger.error(f"[SYSTEM] Failed to collect system information: {e}")
            return {}

    def _collect_memory_info(self) -> Dict[str, Any]:
        """Collect memory information."""
        memory_info = {}

        try:
            system = platform.system()

            if system == 'Linux':
                memory_info.update(self._get_linux_memory())
            elif system == 'Darwin':  # macOS
                memory_info.update(self._get_macos_memory())
            elif system == 'Windows':
                memory_info.update(self._get_windows_memory())
            else:
                memory_info['memory_gb'] = 'Unknown'
                memory_info['memory_method'] = 'unsupported_platform'

        except Exception as e:
            self.logger.warning(f"[SYSTEM] Memory info collection failed: {e}")
            memory_info['memory_gb'] = 'Unknown'
            memory_info['memory_error'] = str(e)

        return memory_info

    def _get_linux_memory(self) -> Dict[str, Any]:
        """Get memory information on Linux systems."""
        memory_info = {}

        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()

            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    memory_kb = int(line.split()[1])
                    memory_info['memory_gb'] = round(memory_kb / 1024 / 1024, 1)
                    memory_info['memory_method'] = 'proc_meminfo'
                    break

        except Exception as e:
            memory_info['memory_gb'] = 'Unknown'
            memory_info['memory_error'] = str(e)

        return memory_info

    def _get_macos_memory(self) -> Dict[str, Any]:
        """Get memory information on macOS systems."""
        memory_info = {}

        try:
            result = subprocess.run(['sysctl', 'hw.memsize'],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                memory_bytes = int(result.stdout.split(': ')[1])
                memory_info['memory_gb'] = round(memory_bytes / 1024 / 1024 / 1024, 1)
                memory_info['memory_method'] = 'sysctl'
            else:
                memory_info['memory_gb'] = 'Unknown'
                memory_info['memory_error'] = 'sysctl_failed'

        except Exception as e:
            memory_info['memory_gb'] = 'Unknown'
            memory_info['memory_error'] = str(e)

        return memory_info

    def _get_windows_memory(self) -> Dict[str, Any]:
        """Get memory information on Windows systems."""
        memory_info = {}

        try:
            import psutil
            memory_bytes = psutil.virtual_memory().total
            memory_info['memory_gb'] = round(memory_bytes / 1024 / 1024 / 1024, 1)
            memory_info['memory_method'] = 'psutil'

        except ImportError:
            memory_info['memory_gb'] = 'Unknown'
            memory_info['memory_error'] = 'psutil_not_available'
        except Exception as e:
            memory_info['memory_gb'] = 'Unknown'
            memory_info['memory_error'] = str(e)

        return memory_info

    def _collect_disk_info(self) -> Dict[str, Any]:
        """Collect disk space information."""
        disk_info = {}

        try:
            total, used, free = shutil.disk_usage('.')
            disk_info['disk_free_gb'] = round(free / 1024 / 1024 / 1024, 1)
            disk_info['disk_total_gb'] = round(total / 1024 / 1024 / 1024, 1)
            disk_info['disk_used_gb'] = round(used / 1024 / 1024 / 1024, 1)
            disk_info['disk_usage_percent'] = round((used / total) * 100, 1)

        except Exception as e:
            self.logger.warning(f"[SYSTEM] Disk info collection failed: {e}")
            disk_info['disk_free_gb'] = 'Unknown'
            disk_info['disk_total_gb'] = 'Unknown'
            disk_info['disk_error'] = str(e)

        return disk_info

    def _collect_gpu_info(self) -> Dict[str, Any]:
        """Collect GPU information."""
        gpu_info = {
            'nvidia_available': False,
            'cuda_available': False,
            'mps_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': []
        }

        # Check NVIDIA GPU
        gpu_info.update(self._check_nvidia_gpu())

        # Check CUDA availability
        gpu_info.update(self._check_cuda_availability())

        # Check Apple Metal Performance Shaders (MPS)
        if platform.system() == 'Darwin':
            gpu_info.update(self._check_mps_availability())

        return {'gpu_info': gpu_info}

    def _check_nvidia_gpu(self) -> Dict[str, Any]:
        """Check for NVIDIA GPUs using nvidia-smi."""
        nvidia_info = {}

        try:
            result = subprocess.run([
                'nvidia-smi',
                '--query-gpu=name,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                nvidia_info['nvidia_available'] = True
                lines = result.stdout.strip().split('\n')
                nvidia_info['gpu_count'] = len([line for line in lines if line.strip()])

                gpu_names = []
                gpu_memory = []
                for line in lines:
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            memory = parts[1].strip()
                            gpu_names.append(f"{name} ({memory}MB)")
                            gpu_memory.append(int(memory))

                nvidia_info['gpu_names'] = gpu_names
                nvidia_info['gpu_memory'] = gpu_memory

        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            nvidia_info['nvidia_available'] = False
            nvidia_info['nvidia_error'] = 'nvidia_smi_not_available'
        except Exception as e:
            nvidia_info['nvidia_available'] = False
            nvidia_info['nvidia_error'] = str(e)

        return nvidia_info

    def _check_cuda_availability(self) -> Dict[str, Any]:
        """Check CUDA availability through PyTorch."""
        cuda_info = {}

        try:
            import torch
            cuda_info['cuda_available'] = torch.cuda.is_available()

            if cuda_info['cuda_available']:
                cuda_info['cuda_device_count'] = torch.cuda.device_count()
                cuda_info['cuda_version'] = torch.version.cuda

                # Get device names if available
                cuda_devices = []
                for i in range(torch.cuda.device_count()):
                    device_name = torch.cuda.get_device_name(i)
                    cuda_devices.append(device_name)
                cuda_info['cuda_devices'] = cuda_devices

        except ImportError:
            cuda_info['cuda_available'] = False
            cuda_info['cuda_error'] = 'torch_not_available'
        except Exception as e:
            cuda_info['cuda_available'] = False
            cuda_info['cuda_error'] = str(e)

        return cuda_info

    def _check_mps_availability(self) -> Dict[str, Any]:
        """Check Apple Metal Performance Shaders (MPS) availability."""
        mps_info = {}

        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                mps_info['mps_available'] = True
                mps_info['mps_built'] = torch.backends.mps.is_built()
            else:
                mps_info['mps_available'] = False
                mps_info['mps_error'] = 'mps_not_supported'

        except ImportError:
            mps_info['mps_available'] = False
            mps_info['mps_error'] = 'torch_not_available'
        except Exception as e:
            mps_info['mps_available'] = False
            mps_info['mps_error'] = str(e)

        return mps_info

    def _collect_environment_info(self) -> Dict[str, Any]:
        """Collect relevant environment variables."""
        env_vars = {
            'PATH': os.environ.get('PATH', ''),
            'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
            'HF_HOME': os.environ.get('HF_HOME', ''),
            'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'TRANSFORMERS_CACHE': os.environ.get('TRANSFORMERS_CACHE', ''),
            'HF_HUB_CACHE': os.environ.get('HF_HUB_CACHE', ''),
            'TOKENIZERS_PARALLELISM': os.environ.get('TOKENIZERS_PARALLELISM', ''),
        }

        # Filter out empty values for cleaner output
        env_vars = {k: v for k, v in env_vars.items() if v}

        return {'env_vars': env_vars}

    def _collect_python_info(self) -> Dict[str, Any]:
        """Collect detailed Python information."""
        python_info = {
            'version_tuple': platform.python_version_tuple(),
            'implementation': platform.python_implementation(),
            'compiler': platform.python_compiler(),
            'build_info': platform.python_build(),
        }

        # Add pip information if available
        try:
            import pip
            python_info['pip_version'] = pip.__version__
        except (ImportError, AttributeError):
            python_info['pip_version'] = 'Unknown'

        # Add site packages location
        try:
            import site
            python_info['site_packages'] = site.getsitepackages()
        except Exception:
            python_info['site_packages'] = 'Unknown'

        return {'python_info': python_info}

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summarized version of system information."""
        full_info = self.collect_all_info()

        summary = {
            'platform': full_info.get('platform', 'Unknown'),
            'python_version': platform.python_version(),
            'memory_gb': full_info.get('memory_gb', 'Unknown'),
            'disk_free_gb': full_info.get('disk_free_gb', 'Unknown'),
            'gpu_available': full_info.get('gpu_info', {}).get('nvidia_available', False),
            'cuda_available': full_info.get('gpu_info', {}).get('cuda_available', False),
        }

        return summary

    def check_system_requirements(self, requirements: Dict[str, Any]) -> Dict[str, bool]:
        """Check if system meets basic requirements."""
        system_info = self.collect_all_info()
        checks = {}

        # Check Python version
        if 'python_min_version' in requirements:
            min_version = requirements['python_min_version']
            current_version = platform.python_version_tuple()
            current_tuple = (int(current_version[0]), int(current_version[1]))
            checks['python_version'] = current_tuple >= min_version

        # Check memory
        if 'memory_min_gb' in requirements:
            memory_gb = system_info.get('memory_gb', 0)
            if isinstance(memory_gb, (int, float)):
                checks['memory'] = memory_gb >= requirements['memory_min_gb']
            else:
                checks['memory'] = True  # Unknown memory, assume OK

        # Check disk space
        if 'disk_space_min_gb' in requirements:
            disk_free = system_info.get('disk_free_gb', 0)
            if isinstance(disk_free, (int, float)):
                checks['disk_space'] = disk_free >= requirements['disk_space_min_gb']
            else:
                checks['disk_space'] = True  # Unknown disk space, assume OK

        return checks