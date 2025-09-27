"""
Unit tests for FastAPI server components.

This test suite follows TDD principles to verify FastAPI endpoints and server functionality
before refactoring the large fastapi_server.py into modular components.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import asyncio
from fastapi.testclient import TestClient
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


class TestFastAPIServerModels(unittest.TestCase):
    """Test Pydantic model validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.valid_prediction_request = {
            "text": "Hello world",
            "max_length": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "do_sample": True
        }

        self.valid_model_load_request = {
            "model_name": "test-model",
            "model_path": "/path/to/model"
        }

    def test_prediction_request_valid(self):
        """Test valid PredictionRequest creation."""
        try:
            from servers.fastapi_server import PredictionRequest
            request = PredictionRequest(**self.valid_prediction_request)
            self.assertEqual(request.text, "Hello world")
            self.assertEqual(request.max_length, 100)
            self.assertEqual(request.temperature, 0.8)
        except ImportError:
            self.skipTest("FastAPI server module not available")

    def test_prediction_request_minimal(self):
        """Test PredictionRequest with minimal fields."""
        try:
            from servers.fastapi_server import PredictionRequest
            request = PredictionRequest(text="Test text")
            self.assertEqual(request.text, "Test text")
            self.assertEqual(request.max_length, 512)  # default value
            self.assertEqual(request.temperature, 1.0)  # default value
        except ImportError:
            self.skipTest("FastAPI server module not available")

    def test_model_load_request_valid(self):
        """Test valid ModelLoadRequest creation."""
        try:
            from servers.fastapi_server import ModelLoadRequest
            request = ModelLoadRequest(**self.valid_model_load_request)
            self.assertEqual(request.model_name, "test-model")
            self.assertEqual(request.model_path, "/path/to/model")
        except ImportError:
            self.skipTest("FastAPI server module not available")


class TestJSONRepairMiddleware(unittest.TestCase):
    """Test JSON repair middleware functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.middleware = None
        try:
            from servers.fastapi_server import JSONRepairMiddleware
            self.middleware = JSONRepairMiddleware(app=None)
        except ImportError:
            pass

    def test_repair_json_valid(self):
        """Test JSON repair with valid JSON."""
        if not self.middleware:
            self.skipTest("JSONRepairMiddleware not available")

        valid_json = '{"text": "Hello world"}'
        repaired = self.middleware.repair_json(valid_json)
        self.assertEqual(repaired, valid_json)

    def test_repair_json_single_quotes(self):
        """Test JSON repair with single quotes."""
        if not self.middleware:
            self.skipTest("JSONRepairMiddleware not available")

        invalid_json = "{'text': 'Hello world'}"
        repaired = self.middleware.repair_json(invalid_json)
        # Should convert to double quotes
        self.assertIn('"text"', repaired)
        self.assertIn('"Hello world"', repaired)

    def test_repair_json_trailing_comma(self):
        """Test JSON repair with trailing comma."""
        if not self.middleware:
            self.skipTest("JSONRepairMiddleware not available")

        invalid_json = '{"text": "Hello world", "extra": true,}'
        repaired = self.middleware.repair_json(invalid_json)
        # Should remove trailing comma
        self.assertNotIn(',}', repaired)

    def test_extract_text_fallback(self):
        """Test text extraction fallback."""
        if not self.middleware:
            self.skipTest("JSONRepairMiddleware not available")

        invalid_text = "Just some text without JSON structure"
        fallback = self.middleware.extract_text_fallback(invalid_text)
        self.assertIn("text", fallback)


class TestFastAPIServerEndpoints(unittest.TestCase):
    """Test FastAPI server endpoint functionality."""

    def setUp(self):
        """Set up test fixtures with mock model manager."""
        self.mock_model_manager = Mock()
        self.mock_model_manager.get_loaded_models.return_value = {
            "test-model": {"status": "loaded", "type": "text-generation"}
        }
        self.mock_model_manager.is_model_loaded.return_value = True
        self.mock_model_manager.get_model_info.return_value = {
            "name": "test-model",
            "type": "text-generation",
            "status": "loaded"
        }

    def test_server_creation(self):
        """Test FastAPI server instance creation."""
        try:
            from servers.fastapi_server import FastAPIServer
            server = FastAPIServer(self.mock_model_manager)
            self.assertIsNotNone(server)
            self.assertEqual(server.host, "127.0.0.1")
            self.assertEqual(server.port, 8000)
        except ImportError:
            self.skipTest("FastAPIServer not available")

    def test_app_creation(self):
        """Test FastAPI app creation."""
        try:
            from servers.fastapi_server import FastAPIServer
            server = FastAPIServer(self.mock_model_manager)
            app = server.create_app()
            self.assertIsNotNone(app)
        except ImportError:
            self.skipTest("FastAPIServer not available")

    @patch('servers.fastapi_server.MultiModelManager')
    def test_health_endpoint(self, mock_model_manager_class):
        """Test health check endpoint."""
        try:
            from servers.fastapi_server import FastAPIServer

            mock_model_manager_class.return_value = self.mock_model_manager
            server = FastAPIServer(self.mock_model_manager)
            app = server.create_app()

            client = TestClient(app)
            response = client.get("/health")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "healthy")

        except ImportError:
            self.skipTest("FastAPIServer not available")

    @patch('servers.fastapi_server.MultiModelManager')
    def test_models_list_endpoint(self, mock_model_manager_class):
        """Test models list endpoint."""
        try:
            from servers.fastapi_server import FastAPIServer

            mock_model_manager_class.return_value = self.mock_model_manager
            server = FastAPIServer(self.mock_model_manager)
            app = server.create_app()

            client = TestClient(app)
            response = client.get("/models")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("loaded_models", data)

        except ImportError:
            self.skipTest("FastAPIServer not available")

    @patch('servers.fastapi_server.MultiModelManager')
    def test_model_info_endpoint(self, mock_model_manager_class):
        """Test individual model info endpoint."""
        try:
            from servers.fastapi_server import FastAPIServer

            mock_model_manager_class.return_value = self.mock_model_manager
            server = FastAPIServer(self.mock_model_manager)
            app = server.create_app()

            client = TestClient(app)
            response = client.get("/models/test-model")

            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["name"], "test-model")

        except ImportError:
            self.skipTest("FastAPIServer not available")


class TestFastAPIServerManager(unittest.TestCase):
    """Test FastAPI server manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_model_manager = Mock()

    def test_server_start_stop(self):
        """Test server start and stop functionality."""
        try:
            from servers.fastapi_server import FastAPIServer

            server = FastAPIServer(self.mock_model_manager)

            # Test that server can be created without starting
            self.assertIsNotNone(server)
            self.assertFalse(server.running)

            # Note: We don't actually start the server in tests to avoid port conflicts

        except ImportError:
            self.skipTest("FastAPIServer not available")

    def test_server_configuration(self):
        """Test server configuration options."""
        try:
            from servers.fastapi_server import FastAPIServer

            # Test custom host and port
            server = FastAPIServer(self.mock_model_manager, host="0.0.0.0", port=9000)
            self.assertEqual(server.host, "0.0.0.0")
            self.assertEqual(server.port, 9000)

        except ImportError:
            self.skipTest("FastAPIServer not available")


class TestPredictionEndpoints(unittest.TestCase):
    """Test prediction-related endpoints."""

    def setUp(self):
        """Set up test fixtures for predictions."""
        self.mock_model_manager = Mock()

        # Mock successful prediction
        self.mock_model_manager.predict.return_value = {
            "success": True,
            "result": "Generated text response",
            "model_name": "test-model"
        }

    @patch('servers.fastapi_server.MultiModelManager')
    def test_prediction_endpoint(self, mock_model_manager_class):
        """Test prediction endpoint functionality."""
        try:
            from servers.fastapi_server import FastAPIServer

            mock_model_manager_class.return_value = self.mock_model_manager
            server = FastAPIServer(self.mock_model_manager)
            app = server.create_app()

            client = TestClient(app)

            prediction_data = {
                "text": "Test input text",
                "max_length": 50,
                "temperature": 0.8
            }

            response = client.post("/models/test-model/predict", json=prediction_data)

            # Should not fail with 500 error
            self.assertNotEqual(response.status_code, 500)

        except ImportError:
            self.skipTest("FastAPIServer not available")


if __name__ == '__main__':
    unittest.main()