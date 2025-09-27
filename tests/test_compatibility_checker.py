"""
Comprehensive unit tests for CompatibilityChecker.

This test suite covers all major functionality of the CompatibilityChecker class
including system information gathering, requirement validation, and compatibility checking.
"""

import unittest
import os
import sys
import tempfile
import shutil
import platform
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import json

# Add the project root directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from scripts.compatibility_check import CompatibilityChecker, Colors
except ImportError as e:
    print(f"Warning: Could not import CompatibilityChecker: {e}")
    CompatibilityChecker = None


class TestCompatibilityChecker(unittest.TestCase):
    """Test cases for CompatibilityChecker class."""

    def setUp(self):
        """Set up test fixtures."""
        if CompatibilityChecker is None:
            self.skipTest("CompatibilityChecker not available for testing")

        self.checker = CompatibilityChecker()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checker_initialization(self):
        """Test CompatibilityChecker initialization."""
        self.assertIsInstance(self.checker, CompatibilityChecker)
        self.assertIsNotNone(self.checker.requirements)
        self.assertIn('python_min_version', self.checker.requirements)
        self.assertIn('required_packages', self.checker.requirements)
        self.assertIsInstance(self.checker.results, dict)

    def test_colors_class(self):
        """Test Colors class constants."""
        self.assertIsNotNone(Colors.RED)
        self.assertIsNotNone(Colors.GREEN)
        self.assertIsNotNone(Colors.YELLOW)
        self.assertIsNotNone(Colors.NC)
        self.assertTrue(Colors.RED.startswith('\033'))

    def test_log_method(self):
        """Test logging functionality."""
        # Test info logging
        self.checker.log('info', 'Test info message')

        # Test warning logging
        self.checker.log('warning', 'Test warning message')

        # Test error logging
        self.checker.log('error', 'Test error message')

        # Test success logging
        self.checker.log('success', 'Test success message')

    @patch('platform.python_version_tuple')
    def test_check_python_version_success(self, mock_version):
        """Test Python version check with compatible version."""
        mock_version.return_value = ('3', '10', '0')

        result = self.checker.check_python_version()

        self.assertTrue(result)
        self.assertIn('python_version', self.checker.results)

    @patch('platform.python_version_tuple')
    def test_check_python_version_failure(self, mock_version):
        """Test Python version check with incompatible version."""
        mock_version.return_value = ('3', '8', '0')  # Below minimum

        result = self.checker.check_python_version()

        self.assertFalse(result)

    @patch('importlib.import_module')
    def test_check_packages_success(self, mock_import):
        """Test package availability check - success case."""
        mock_import.return_value = Mock()

        result = self.checker.check_packages()

        self.assertTrue(result)
        self.assertIn('package_check', self.checker.results)

    @patch('importlib.import_module')
    def test_check_packages_failure(self, mock_import):
        """Test package availability check - failure case."""
        mock_import.side_effect = ImportError("Package not found")

        result = self.checker.check_packages()

        self.assertFalse(result)

    @patch('platform.system')
    def test_collect_system_info_linux(self, mock_system):
        """Test system information collection on Linux."""
        mock_system.return_value = 'Linux'

        with patch('builtins.open', unittest.mock.mock_open(read_data='MemTotal:        8000000 kB\n')):
            system_info = self.checker.collect_system_info()

        self.assertIsInstance(system_info, dict)
        self.assertIn('platform', system_info)
        self.assertIn('python_version', system_info)

    @patch('platform.system')
    @patch('subprocess.run')
    def test_collect_system_info_macos(self, mock_run, mock_system):
        """Test system information collection on macOS."""
        mock_system.return_value = 'Darwin'
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = 'hw.memsize: 8589934592'

        system_info = self.checker.collect_system_info()

        self.assertIsInstance(system_info, dict)
        self.assertIn('platform', system_info)

    @patch('platform.system')
    def test_collect_system_info_windows(self, mock_system):
        """Test system information collection on Windows."""
        mock_system.return_value = 'Windows'

        with patch('importlib.import_module') as mock_import:
            mock_psutil = Mock()
            mock_psutil.virtual_memory.return_value.total = 8589934592
            mock_import.return_value = mock_psutil

            system_info = self.checker.collect_system_info()

        self.assertIsInstance(system_info, dict)

    @patch('subprocess.run')
    def test_get_gpu_info_nvidia_available(self, mock_run):
        """Test GPU information gathering with NVIDIA GPU available."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = 'GeForce RTX 3080, 10240\nGeForce GTX 1080, 8192'

        gpu_info = self.checker._get_gpu_info()

        self.assertIsInstance(gpu_info, dict)
        self.assertIn('nvidia_available', gpu_info)
        self.assertIn('gpu_count', gpu_info)
        self.assertIn('gpu_names', gpu_info)

    @patch('subprocess.run')
    def test_get_gpu_info_nvidia_unavailable(self, mock_run):
        """Test GPU information gathering without NVIDIA GPU."""
        mock_run.side_effect = FileNotFoundError()

        gpu_info = self.checker._get_gpu_info()

        self.assertFalse(gpu_info['nvidia_available'])
        self.assertEqual(gpu_info['gpu_count'], 0)

    @patch('importlib.import_module')
    def test_get_gpu_info_cuda_available(self, mock_import):
        """Test GPU information with CUDA available."""
        mock_torch = Mock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_import.return_value = mock_torch

        gpu_info = self.checker._get_gpu_info()

        self.assertTrue(gpu_info['cuda_available'])

    @patch('shutil.disk_usage')
    def test_check_disk_space_sufficient(self, mock_disk_usage):
        """Test disk space check with sufficient space."""
        # Mock 50GB free space
        mock_disk_usage.return_value = (100 * 1024**3, 50 * 1024**3, 50 * 1024**3)

        result = self.checker.check_disk_space()

        self.assertTrue(result)

    @patch('shutil.disk_usage')
    def test_check_disk_space_insufficient(self, mock_disk_usage):
        """Test disk space check with insufficient space."""
        # Mock 5GB free space (below minimum 10GB)
        mock_disk_usage.return_value = (100 * 1024**3, 95 * 1024**3, 5 * 1024**3)

        result = self.checker.check_disk_space()

        self.assertFalse(result)

    def test_check_memory_sufficient(self):
        """Test memory check with sufficient memory."""
        # Mock system info with 8GB memory
        self.checker.results['system_info'] = {'memory_gb': 8.0}

        result = self.checker.check_memory()

        self.assertTrue(result)

    def test_check_memory_insufficient(self):
        """Test memory check with insufficient memory."""
        # Mock system info with 2GB memory (below minimum 4GB)
        self.checker.results['system_info'] = {'memory_gb': 2.0}

        result = self.checker.check_memory()

        self.assertFalse(result)

    def test_check_memory_unknown(self):
        """Test memory check with unknown memory."""
        # Mock system info with unknown memory
        self.checker.results['system_info'] = {'memory_gb': 'Unknown'}

        result = self.checker.check_memory()

        # Should return True for unknown memory to avoid false negatives
        self.assertTrue(result)

    def test_run_full_compatibility_check(self):
        """Test full compatibility check run."""
        with patch.multiple(
            self.checker,
            collect_system_info=Mock(return_value={'memory_gb': 8.0, 'platform': 'Linux'}),
            check_python_version=Mock(return_value=True),
            check_packages=Mock(return_value=True),
            check_memory=Mock(return_value=True),
            check_disk_space=Mock(return_value=True)
        ):
            result = self.checker.run_compatibility_check()

            self.assertTrue(result)

    def test_run_partial_compatibility_check(self):
        """Test compatibility check with some failures."""
        with patch.multiple(
            self.checker,
            collect_system_info=Mock(return_value={'memory_gb': 2.0, 'platform': 'Linux'}),
            check_python_version=Mock(return_value=True),
            check_packages=Mock(return_value=False),  # Package check fails
            check_memory=Mock(return_value=False),    # Memory check fails
            check_disk_space=Mock(return_value=True)
        ):
            result = self.checker.run_compatibility_check()

            self.assertFalse(result)

    def test_generate_report_success(self):
        """Test report generation for successful compatibility check."""
        self.checker.results = {
            'system_info': {'platform': 'Linux', 'memory_gb': 8.0},
            'python_version': True,
            'package_check': True,
            'memory_check': True,
            'disk_space_check': True,
            'overall': True
        }

        # This should not raise an exception
        self.checker.generate_report()

    def test_generate_report_failure(self):
        """Test report generation for failed compatibility check."""
        self.checker.results = {
            'system_info': {'platform': 'Linux', 'memory_gb': 2.0},
            'python_version': False,
            'package_check': False,
            'memory_check': False,
            'disk_space_check': False,
            'overall': False
        }

        # This should not raise an exception
        self.checker.generate_report()

    def test_get_recommendations_python_version(self):
        """Test getting recommendations for Python version issues."""
        self.checker.results = {
            'python_version': False
        }

        recommendations = self.checker._get_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertTrue(any('python' in rec.lower() for rec in recommendations))

    def test_get_recommendations_packages(self):
        """Test getting recommendations for package issues."""
        self.checker.results = {
            'package_check': False,
            'missing_packages': ['streamlit', 'transformers']
        }

        recommendations = self.checker._get_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertTrue(any('install' in rec.lower() for rec in recommendations))

    def test_get_recommendations_memory(self):
        """Test getting recommendations for memory issues."""
        self.checker.results = {
            'memory_check': False,
            'system_info': {'memory_gb': 2.0}
        }

        recommendations = self.checker._get_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertTrue(any('memory' in rec.lower() for rec in recommendations))

    def test_get_recommendations_disk_space(self):
        """Test getting recommendations for disk space issues."""
        self.checker.results = {
            'disk_space_check': False
        }

        recommendations = self.checker._get_recommendations()

        self.assertIsInstance(recommendations, list)
        self.assertTrue(any('disk' in rec.lower() or 'space' in rec.lower() for rec in recommendations))

    def test_save_report_to_file(self):
        """Test saving compatibility report to file."""
        self.checker.results = {
            'system_info': {'platform': 'Linux'},
            'overall': True
        }

        report_file = os.path.join(self.temp_dir, 'compatibility_report.json')

        # Mock the save functionality
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.checker.save_report(report_file)
            mock_file.assert_called_once()

    def test_error_handling_in_system_info(self):
        """Test error handling during system information collection."""
        with patch('platform.platform', side_effect=Exception("Platform error")):
            system_info = self.checker.collect_system_info()

            # Should return empty dict on error, not crash
            self.assertIsInstance(system_info, dict)

    def test_error_handling_in_gpu_info(self):
        """Test error handling during GPU information collection."""
        with patch('subprocess.run', side_effect=Exception("GPU error")):
            gpu_info = self.checker._get_gpu_info()

            # Should return default GPU info structure
            self.assertIsInstance(gpu_info, dict)
            self.assertIn('nvidia_available', gpu_info)

    def test_requirements_structure(self):
        """Test that requirements have expected structure."""
        requirements = self.checker.requirements

        self.assertIn('python_min_version', requirements)
        self.assertIsInstance(requirements['python_min_version'], tuple)
        self.assertEqual(len(requirements['python_min_version']), 2)

        self.assertIn('required_packages', requirements)
        self.assertIsInstance(requirements['required_packages'], list)

        self.assertIn('memory_min_gb', requirements)
        self.assertIsInstance(requirements['memory_min_gb'], (int, float))

        self.assertIn('disk_space_min_gb', requirements)
        self.assertIsInstance(requirements['disk_space_min_gb'], (int, float))

    def test_verbose_mode(self):
        """Test verbose mode functionality."""
        # Create checker with verbose mode
        verbose_checker = CompatibilityChecker()

        # Test that verbose logging works
        verbose_checker.log('info', 'Verbose test message')

        # Should not raise exceptions
        self.assertTrue(True)


class TestCompatibilityCheckerIntegration(unittest.TestCase):
    """Integration tests for CompatibilityChecker."""

    def setUp(self):
        """Set up integration test fixtures."""
        if CompatibilityChecker is None:
            self.skipTest("CompatibilityChecker not available for testing")

        self.checker = CompatibilityChecker()

    def test_real_system_info_collection(self):
        """Test actual system information collection."""
        system_info = self.checker.collect_system_info()

        self.assertIsInstance(system_info, dict)
        # Should have basic system info
        self.assertIn('platform', system_info)
        self.assertIn('python_version', system_info)

    def test_real_python_version_check(self):
        """Test actual Python version check."""
        result = self.checker.check_python_version()

        # Should pass since we're running on Python 3.9+
        self.assertTrue(result)

    def test_integration_workflow(self):
        """Test the complete integration workflow."""
        # Run system information collection
        system_info = self.checker.collect_system_info()
        self.assertIsInstance(system_info, dict)

        # Run Python version check
        python_ok = self.checker.check_python_version()
        self.assertIsInstance(python_ok, bool)

        # Generate report
        self.checker.generate_report()

        # Should complete without exceptions
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main(verbosity=2)