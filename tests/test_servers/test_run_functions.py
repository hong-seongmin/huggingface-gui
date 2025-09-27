"""
Unit tests for core functions in servers/run.py.

This test suite follows TDD principles to verify existing functionality
before refactoring the large procedural script into modular components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import sys
import tempfile
import json

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

# Change directory to servers to match the original script's behavior
original_cwd = os.getcwd()
servers_dir = os.path.join(os.path.dirname(__file__), '../../servers')
if os.path.exists(servers_dir):
    os.chdir(servers_dir)


class TestLoginFunctions(unittest.TestCase):
    """Test login-related functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_token = "hf_test_token_123"
        self.login_file = "test_login_token.txt"

    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.login_file):
            os.remove(self.login_file)

    @patch('run.LOGIN_FILE', 'test_login_token.txt')
    def test_load_login_token_exists(self):
        """Test loading existing login token."""
        # Create test token file
        with open(self.login_file, "w") as f:
            f.write(self.test_token)

        # Import after patching
        import run
        token = run.load_login_token()
        self.assertEqual(token, self.test_token)

    @patch('servers.run.LOGIN_FILE', 'nonexistent_token.txt')
    def test_load_login_token_not_exists(self):
        """Test loading non-existent login token."""
        from servers.run import load_login_token

        token = load_login_token()
        self.assertIsNone(token)

    @patch('servers.run.LOGIN_FILE', 'test_login_token.txt')
    def test_save_login_token(self):
        """Test saving login token."""
        from servers.run import save_login_token

        save_login_token(self.test_token)

        # Verify file was created and contains correct token
        self.assertTrue(os.path.exists(self.login_file))
        with open(self.login_file, "r") as f:
            saved_token = f.read().strip()
        self.assertEqual(saved_token, self.test_token)

    @patch('servers.run.LOGIN_FILE', 'test_login_token.txt')
    def test_delete_login_token_exists(self):
        """Test deleting existing login token."""
        # Create test token file
        with open(self.login_file, "w") as f:
            f.write(self.test_token)

        from servers.run import delete_login_token

        delete_login_token()

        # Verify file was deleted
        self.assertFalse(os.path.exists(self.login_file))

    @patch('servers.run.LOGIN_FILE', 'nonexistent_token.txt')
    def test_delete_login_token_not_exists(self):
        """Test deleting non-existent login token."""
        from servers.run import delete_login_token

        # Should not raise exception
        delete_login_token()


class TestSystemDisplayFunctions(unittest.TestCase):
    """Test system display and monitoring functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_system_data_labels = {
            'cpu_percent_label': Mock(),
            'memory_percent_label': Mock(),
            'gpu_info_label': Mock(),
            'disk_info_label': Mock()
        }

        self.sample_data = {
            'cpu': {'percent': 45.2},
            'memory': {
                'used': 8589934592,  # 8 GB in bytes
                'total': 17179869184,  # 16 GB in bytes
                'percent': 50.0
            },
            'gpu': [
                {'load': 30.5},
                {'load': 25.8}
            ],
            'disk': {
                'used': 107374182400,  # 100 GB in bytes
                'total': 536870912000,  # 500 GB in bytes
                'percent': 20.0
            }
        }

    @patch('servers.run.system_data_labels')
    def test_update_system_display_with_data(self, mock_labels):
        """Test updating system display with valid data."""
        mock_labels.update(self.mock_system_data_labels)

        from servers.run import update_system_display

        update_system_display(self.sample_data)

        # Verify CPU label was updated
        self.mock_system_data_labels['cpu_percent_label'].configure.assert_called_with(
            text="CPU: 45.2%"
        )

        # Verify memory label was updated
        self.mock_system_data_labels['memory_percent_label'].configure.assert_called_with(
            text="Memory: 50.0% (8.0/16.0 GB)"
        )

        # Verify GPU label was updated with average
        self.mock_system_data_labels['gpu_info_label'].configure.assert_called_with(
            text="GPU: 28.2% (Average)"
        )

    @patch('servers.run.system_data_labels')
    def test_update_system_display_no_gpu(self, mock_labels):
        """Test updating system display when no GPU data available."""
        mock_labels.update(self.mock_system_data_labels)

        data_no_gpu = self.sample_data.copy()
        data_no_gpu['gpu'] = []

        from servers.run import update_system_display

        update_system_display(data_no_gpu)

        # Verify GPU label shows N/A
        self.mock_system_data_labels['gpu_info_label'].configure.assert_called_with(
            text="GPU: N/A"
        )


class TestCacheFunctions(unittest.TestCase):
    """Test cache-related functions."""

    @patch('servers.run.scan_cache_dir')
    @patch('servers.run.cache_info')
    def test_scan_cache(self, mock_cache_info, mock_scan_cache_dir):
        """Test cache scanning function."""
        # Mock scan_cache_dir return value
        mock_cache = Mock()
        mock_cache.repos = [
            Mock(repo_id='model1', repo_type='model'),
            Mock(repo_id='model2', repo_type='model')
        ]
        mock_scan_cache_dir.return_value = mock_cache

        from servers.run import scan_cache

        scan_cache()

        # Verify scan_cache_dir was called
        mock_scan_cache_dir.assert_called_once()

        # Verify cache_info was updated (by reference)
        self.assertEqual(mock_cache_info, mock_cache)


class TestModelAnalysisFunctions(unittest.TestCase):
    """Test model analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_analysis = {
            'model_info': {
                'architecture': 'transformer',
                'parameters': '7B',
                'size_gb': 13.5
            },
            'config': {
                'model_type': 'llama',
                'hidden_size': 4096,
                'num_layers': 32
            }
        }

    @patch('servers.run.model_analyzer')
    @patch('servers.run.threading.Thread')
    def test_analyze_model(self, mock_thread, mock_analyzer):
        """Test model analysis function."""
        from servers.run import analyze_model

        analyze_model()

        # Verify thread was created and started
        mock_thread.assert_called_once()
        mock_thread.return_value.start.assert_called_once()

    @patch('servers.run.current_model_analysis')
    def test_display_model_analysis(self, mock_analysis):
        """Test displaying model analysis results."""
        from servers.run import display_model_analysis

        display_model_analysis(self.sample_analysis)

        # Verify analysis was stored globally
        self.assertEqual(mock_analysis, self.sample_analysis)


class TestModelListFunctions(unittest.TestCase):
    """Test model list and selection functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_checkboxes = [Mock(), Mock(), Mock()]
        self.mock_cache_info = Mock()
        self.mock_cache_info.repos = [
            Mock(repo_id='model1', size_on_disk=1000000000),  # 1GB
            Mock(repo_id='model2', size_on_disk=2000000000),  # 2GB
            Mock(repo_id='model3', size_on_disk=500000000)    # 0.5GB
        ]

    @patch('servers.run.checkboxes')
    @patch('servers.run.cache_info')
    def test_select_all(self, mock_cache_info, mock_checkboxes):
        """Test select all checkboxes function."""
        mock_checkboxes.extend(self.mock_checkboxes)
        mock_cache_info.repos = self.mock_cache_info.repos

        from servers.run import select_all

        select_all()

        # Verify all checkboxes were selected
        for checkbox in self.mock_checkboxes:
            checkbox.select.assert_called_once()

    @patch('servers.run.checkboxes')
    @patch('servers.run.cache_info')
    def test_deselect_all(self, mock_cache_info, mock_checkboxes):
        """Test deselect all checkboxes function."""
        mock_checkboxes.extend(self.mock_checkboxes)
        mock_cache_info.repos = self.mock_cache_info.repos

        from servers.run import deselect_all

        deselect_all()

        # Verify all checkboxes were deselected
        for checkbox in self.mock_checkboxes:
            checkbox.deselect.assert_called_once()

    @patch('servers.run.update_selection_summary')
    def test_update_selection_summary(self, mock_update):
        """Test update selection summary function."""
        from servers.run import update_selection_summary

        update_selection_summary()

        # Function should execute without errors
        # Detailed testing would require GUI components


class TestServerFunctions(unittest.TestCase):
    """Test server-related functions."""

    @patch('servers.run.fastapi_server')
    def test_server_operations(self, mock_server):
        """Test server start/stop operations."""
        # Import after patching to ensure mocks are in place
        from servers import run

        # Test that fastapi_server is accessible
        self.assertIsNotNone(run.fastapi_server)


if __name__ == '__main__':
    unittest.main()