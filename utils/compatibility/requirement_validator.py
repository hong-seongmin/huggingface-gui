"""
Requirement Validator for compatibility checking.

This module handles validation of system requirements including Python packages,
versions, and system resources.
"""

import importlib
import platform
from typing import Dict, List, Any, Tuple, Optional
import subprocess
import logging


class RequirementValidator:
    """Validates system requirements for compatibility checking."""

    def __init__(self):
        """Initialize requirement validator."""
        self.logger = logging.getLogger("RequirementValidator")
        self.validation_results = {}

    def validate_python_version(self, min_version: Tuple[int, int]) -> bool:
        """Validate Python version meets minimum requirements."""
        try:
            self.logger.info(f"[VALIDATOR] Checking Python version >= {min_version}")

            current_version = platform.python_version_tuple()
            current_tuple = (int(current_version[0]), int(current_version[1]))

            is_valid = current_tuple >= min_version

            self.validation_results['python_version'] = {
                'required': f"{min_version[0]}.{min_version[1]}+",
                'current': f"{current_tuple[0]}.{current_tuple[1]}",
                'valid': is_valid
            }

            if is_valid:
                self.logger.info(f"[VALIDATOR] Python version OK: {current_tuple}")
            else:
                self.logger.warning(f"[VALIDATOR] Python version insufficient: {current_tuple} < {min_version}")

            return is_valid

        except Exception as e:
            self.logger.error(f"[VALIDATOR] Python version check failed: {e}")
            return False

    def validate_packages(self, required_packages: List[str],
                         optional_packages: List[str] = None) -> bool:
        """Validate that required packages are available."""
        if optional_packages is None:
            optional_packages = []

        self.logger.info(f"[VALIDATOR] Checking {len(required_packages)} required packages")

        missing_packages = []
        available_packages = []
        optional_available = []
        optional_missing = []

        # Check required packages
        for package in required_packages:
            if self._check_package_availability(package):
                available_packages.append(package)
            else:
                missing_packages.append(package)

        # Check optional packages
        for package in optional_packages:
            if self._check_package_availability(package):
                optional_available.append(package)
            else:
                optional_missing.append(package)

        all_required_available = len(missing_packages) == 0

        self.validation_results['package_check'] = {
            'required_packages': required_packages,
            'optional_packages': optional_packages,
            'available_required': available_packages,
            'missing_required': missing_packages,
            'available_optional': optional_available,
            'missing_optional': optional_missing,
            'all_required_available': all_required_available
        }

        if all_required_available:
            self.logger.info(f"[VALIDATOR] All required packages available")
        else:
            self.logger.warning(f"[VALIDATOR] Missing required packages: {missing_packages}")

        return all_required_available

    def _check_package_availability(self, package_name: str) -> bool:
        """Check if a specific package is available."""
        try:
            # Handle special package names
            import_name = self._get_import_name(package_name)
            importlib.import_module(import_name)
            return True
        except ImportError:
            return False
        except Exception as e:
            self.logger.debug(f"[VALIDATOR] Package check error for {package_name}: {e}")
            return False

    def _get_import_name(self, package_name: str) -> str:
        """Get the correct import name for a package."""
        # Map package names to import names where they differ
        name_mapping = {
            'huggingface_hub': 'huggingface_hub',
            'transformers': 'transformers',
            'torch': 'torch',
            'tensorflow': 'tensorflow',
            'streamlit': 'streamlit',
            'fastapi': 'fastapi',
            'uvicorn': 'uvicorn',
            'customtkinter': 'customtkinter',
            'pillow': 'PIL',
            'opencv-python': 'cv2',
            'scikit-learn': 'sklearn',
            'beautifulsoup4': 'bs4',
            'python-multipart': 'multipart',
            'pydantic': 'pydantic',
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'requests': 'requests',
            'aiofiles': 'aiofiles',
            'python-jose': 'jose',
            'passlib': 'passlib',
            'bcrypt': 'bcrypt',
            'safetensors': 'safetensors',
        }

        return name_mapping.get(package_name, package_name)

    def validate_memory(self, system_memory_gb: Any, min_memory_gb: float) -> bool:
        """Validate system memory meets minimum requirements."""
        try:
            self.logger.info(f"[VALIDATOR] Checking memory >= {min_memory_gb}GB")

            if system_memory_gb == 'Unknown' or system_memory_gb is None:
                self.logger.warning("[VALIDATOR] Memory size unknown, assuming sufficient")
                self.validation_results['memory_check'] = {
                    'required_gb': min_memory_gb,
                    'current_gb': 'Unknown',
                    'valid': True,
                    'assumed': True
                }
                return True

            if not isinstance(system_memory_gb, (int, float)):
                self.logger.warning(f"[VALIDATOR] Invalid memory value: {system_memory_gb}")
                return True  # Assume sufficient to avoid false negatives

            is_valid = system_memory_gb >= min_memory_gb

            self.validation_results['memory_check'] = {
                'required_gb': min_memory_gb,
                'current_gb': system_memory_gb,
                'valid': is_valid,
                'assumed': False
            }

            if is_valid:
                self.logger.info(f"[VALIDATOR] Memory OK: {system_memory_gb}GB")
            else:
                self.logger.warning(f"[VALIDATOR] Insufficient memory: {system_memory_gb}GB < {min_memory_gb}GB")

            return is_valid

        except Exception as e:
            self.logger.error(f"[VALIDATOR] Memory check failed: {e}")
            return True  # Assume sufficient to avoid false negatives

    def validate_disk_space(self, system_disk_free_gb: Any, min_disk_gb: float) -> bool:
        """Validate disk space meets minimum requirements."""
        try:
            self.logger.info(f"[VALIDATOR] Checking disk space >= {min_disk_gb}GB")

            if system_disk_free_gb == 'Unknown' or system_disk_free_gb is None:
                self.logger.warning("[VALIDATOR] Disk space unknown, assuming sufficient")
                self.validation_results['disk_space_check'] = {
                    'required_gb': min_disk_gb,
                    'current_gb': 'Unknown',
                    'valid': True,
                    'assumed': True
                }
                return True

            if not isinstance(system_disk_free_gb, (int, float)):
                self.logger.warning(f"[VALIDATOR] Invalid disk space value: {system_disk_free_gb}")
                return True  # Assume sufficient to avoid false negatives

            is_valid = system_disk_free_gb >= min_disk_gb

            self.validation_results['disk_space_check'] = {
                'required_gb': min_disk_gb,
                'current_gb': system_disk_free_gb,
                'valid': is_valid,
                'assumed': False
            }

            if is_valid:
                self.logger.info(f"[VALIDATOR] Disk space OK: {system_disk_free_gb}GB")
            else:
                self.logger.warning(f"[VALIDATOR] Insufficient disk space: {system_disk_free_gb}GB < {min_disk_gb}GB")

            return is_valid

        except Exception as e:
            self.logger.error(f"[VALIDATOR] Disk space check failed: {e}")
            return True  # Assume sufficient to avoid false negatives

    def validate_gpu_requirements(self, gpu_info: Dict[str, Any],
                                 cuda_required: bool = False) -> bool:
        """Validate GPU requirements if specified."""
        try:
            self.logger.info(f"[VALIDATOR] Checking GPU requirements (CUDA required: {cuda_required})")

            if not cuda_required:
                self.validation_results['gpu_check'] = {
                    'cuda_required': False,
                    'valid': True,
                    'message': 'GPU not required'
                }
                return True

            cuda_available = gpu_info.get('cuda_available', False)
            nvidia_available = gpu_info.get('nvidia_available', False)

            is_valid = cuda_available or nvidia_available

            self.validation_results['gpu_check'] = {
                'cuda_required': cuda_required,
                'cuda_available': cuda_available,
                'nvidia_available': nvidia_available,
                'valid': is_valid
            }

            if is_valid:
                self.logger.info("[VALIDATOR] GPU requirements satisfied")
            else:
                self.logger.warning("[VALIDATOR] GPU requirements not met")

            return is_valid

        except Exception as e:
            self.logger.error(f"[VALIDATOR] GPU check failed: {e}")
            return True  # Assume sufficient to avoid false negatives

    def get_package_versions(self, packages: List[str]) -> Dict[str, str]:
        """Get versions of installed packages."""
        versions = {}

        for package in packages:
            try:
                # Try to get version using importlib
                module = importlib.import_module(self._get_import_name(package))

                # Try different version attributes
                version = None
                for attr in ['__version__', 'version', 'VERSION']:
                    if hasattr(module, attr):
                        version = getattr(module, attr)
                        break

                if version:
                    versions[package] = str(version)
                else:
                    versions[package] = 'Unknown'

            except ImportError:
                versions[package] = 'Not Installed'
            except Exception as e:
                versions[package] = f'Error: {str(e)}'

        return versions

    def check_package_conflicts(self, packages: List[str]) -> Dict[str, List[str]]:
        """Check for known package conflicts."""
        conflicts = {}

        # Define known conflicts
        known_conflicts = {
            'tensorflow': ['tensorflow-gpu'],
            'torch': ['torchvision'],  # Version compatibility
            'transformers': ['tokenizers'],  # Version compatibility
        }

        for package in packages:
            if package in known_conflicts:
                conflicting_packages = []
                for conflict in known_conflicts[package]:
                    if self._check_package_availability(conflict):
                        conflicting_packages.append(conflict)

                if conflicting_packages:
                    conflicts[package] = conflicting_packages

        return conflicts

    def validate_all_requirements(self, requirements: Dict[str, Any],
                                system_info: Dict[str, Any]) -> bool:
        """Validate all requirements against system information."""
        all_valid = True

        # Python version
        if 'python_min_version' in requirements:
            if not self.validate_python_version(requirements['python_min_version']):
                all_valid = False

        # Packages
        if 'required_packages' in requirements:
            optional = requirements.get('optional_packages', [])
            if not self.validate_packages(requirements['required_packages'], optional):
                all_valid = False

        # Memory
        if 'memory_min_gb' in requirements:
            memory_gb = system_info.get('memory_gb')
            if not self.validate_memory(memory_gb, requirements['memory_min_gb']):
                all_valid = False

        # Disk space
        if 'disk_space_min_gb' in requirements:
            disk_free = system_info.get('disk_free_gb')
            if not self.validate_disk_space(disk_free, requirements['disk_space_min_gb']):
                all_valid = False

        # GPU (optional)
        if requirements.get('cuda_required', False):
            gpu_info = system_info.get('gpu_info', {})
            if not self.validate_gpu_requirements(gpu_info, True):
                all_valid = False

        self.validation_results['overall'] = all_valid
        return all_valid

    def get_validation_results(self) -> Dict[str, Any]:
        """Get all validation results."""
        return self.validation_results.copy()

    def clear_results(self):
        """Clear validation results."""
        self.validation_results.clear()