"""
Refactored FastAPI server using modular architecture.

This module now serves as a thin compatibility layer that uses the new
modular API components from the servers.api package. The original 893-line
monolithic implementation has been broken down into specialized modules
for better maintainability and separation of concerns.

Original file backed up as servers/fastapi_server_original.py
"""

from typing import Dict, Any
import logging

# Import the new modular components
from servers.api import (
    PredictionRequest,
    ModelLoadRequest,
    FastAPIServerManager,
    JSONRepairMiddleware,
    RequestLoggingMiddleware,
    CORSMiddleware
)


class FastAPIServer:
    """
    Backward compatibility wrapper for FastAPI server functionality.

    This class maintains the original API while delegating to the new
    modular FastAPIServerManager implementation.
    """

    def __init__(self, model_manager, host: str = "127.0.0.1", port: int = 8000):
        """Initialize FastAPI server with backward compatibility."""
        self.model_manager = model_manager
        self.host = host
        self.default_port = port

        # Import device manager if available
        try:
            from device_manager import device_manager
            self.device_manager = device_manager
        except ImportError:
            self.device_manager = None

        # Initialize the new modular server manager
        self.server_manager = FastAPIServerManager(
            model_manager=model_manager,
            device_manager=self.device_manager,
            host=host,
            port=port
        )

        # Maintain backward compatibility properties
        self.app = self.server_manager.app
        self.servers = self.server_manager.servers
        self.model_ports = self.server_manager.model_ports
        self.pipeline_cache = self.server_manager.pipeline_cache
        self.logger = self.server_manager.logger
        self.server_thread = self.server_manager.server_thread
        self.server = self.server_manager.server
        self.running = self.server_manager.running

    def create_app(self, target_models=None):
        """Create FastAPI application - delegates to server manager."""
        return self.server_manager.create_app(target_models)

    def start_server(self, model_ports=None):
        """Start FastAPI server - delegates to server manager."""
        return self.server_manager.start_server(model_ports)

    def stop_server(self, port=None):
        """Stop FastAPI server - delegates to server manager."""
        return self.server_manager.stop_server(port)

    def is_running(self, port=None) -> bool:
        """Check if server is running - delegates to server manager."""
        return self.server_manager.is_running(port)

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information - delegates to server manager."""
        return self.server_manager.get_server_info()

    def clear_pipeline_cache(self):
        """Clear pipeline cache - delegates to server manager."""
        return self.server_manager.pipeline_cache.clear()

    def get_running_ports(self):
        """Get running ports - delegates to server manager."""
        return self.server_manager.get_running_ports()

    def get_model_server_port(self, model_name: str):
        """Get model server port - backward compatibility method."""
        # Check model_ports dictionary
        if model_name in self.model_ports:
            port = self.model_ports[model_name]
            if self.is_running(port):
                return port

        # Check active servers
        for port, server_info in self.servers.items():
            if server_info.get('running', False):
                models = server_info.get('models', [])
                if model_name in models:
                    return port

        # Check default server
        if self.running:
            loaded_models = self.model_manager.get_loaded_models()
            if model_name in loaded_models:
                return self.default_port

        return None


# Backward compatibility: Re-export the Pydantic models
# These were originally defined in the monolithic file
PredictionRequest = PredictionRequest
ModelLoadRequest = ModelLoadRequest

# Backward compatibility: Re-export the middleware
# These were originally defined in the monolithic file
JSONRepairMiddleware = JSONRepairMiddleware

# Convenience functions for backward compatibility
def create_api_server(model_manager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """Create FastAPI server - backward compatibility function."""
    return FastAPIServer(model_manager, host, port)


def start_background_server(model_manager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """Start background server - backward compatibility function."""
    server = create_api_server(model_manager, host, port)
    server.start_server()
    return server


# Logging setup for backward compatibility
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("FastAPI server module loaded with modular architecture")
logger.info("Original implementation backed up as fastapi_server_original.py")
logger.info("New architecture: 893 lines → ~150 lines (83% reduction)")