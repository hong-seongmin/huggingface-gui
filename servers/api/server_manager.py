"""
FastAPI server manager for centralized server operations.

This module contains the main FastAPI server class and management functionality,
extracted and refactored from the original fastapi_server.py file.
"""

from fastapi import FastAPI
import uvicorn
import threading
import asyncio
import socket
import subprocess
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .models_router import ModelsRouter
from .prediction_router import PredictionRouter
from .middleware import JSONRepairMiddleware, RequestLoggingMiddleware, CORSMiddleware


class FastAPIServerManager:
    """Main FastAPI server manager class."""

    def __init__(self, model_manager, device_manager=None, host: str = "127.0.0.1", port: int = 8000):
        """Initialize FastAPI server manager."""
        self.model_manager = model_manager
        self.device_manager = device_manager
        self.host = host
        self.default_port = port

        # Multi-server management
        self.servers = {}  # {port: {'app': app, 'thread': thread, 'running': bool}}
        self.model_ports = {}  # {model_name: port}

        # Pipeline cache
        self.pipeline_cache = {}

        # Logger setup
        self.logger = logging.getLogger("FastAPIServerManager")

        # Default server (for backward compatibility)
        self.app = self.create_app()
        self.server_thread = None
        self.server = None
        self.running = False

    def create_app(self, target_models=None) -> FastAPI:
        """Create FastAPI application instance."""
        app = FastAPI(
            title="HuggingFace Model API",
            description="API for managing and using HuggingFace models",
            version="1.0.0"
        )

        # Add middleware
        app.add_middleware(JSONRepairMiddleware, logger=self.logger)
        app.add_middleware(RequestLoggingMiddleware, logger=self.logger)
        app.add_middleware(CORSMiddleware)

        # Setup routes
        self._setup_routes(app, target_models)
        return app

    def _setup_routes(self, app: FastAPI, target_models=None):
        """Setup API routes."""

        # Root endpoint
        @app.get("/")
        async def root():
            return {
                "message": "HuggingFace Model API",
                "version": "1.0.0",
                "docs": "/docs",
                "models_loaded": len(self.model_manager.get_loaded_models())
            }

        # Health check endpoint
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": len(self.model_manager.get_loaded_models())
            }

        # System endpoints
        @app.get("/system/status")
        async def get_system_status():
            """Get system status information."""
            return self.model_manager.get_system_summary()

        @app.get("/system/memory")
        async def get_memory_info():
            """Get memory usage information."""
            return self.model_manager.get_memory_info()

        @app.get("/system/metrics")
        async def get_system_metrics():
            """Get real-time system metrics for monitoring."""
            try:
                # Import streamlit to access session state
                import streamlit as st

                # Try to get SystemMonitor from session state
                system_monitor = None
                if hasattr(st, 'session_state') and 'system_monitor' in st.session_state:
                    system_monitor = st.session_state['system_monitor']

                if system_monitor is None:
                    # Create a temporary SystemMonitor instance
                    from system_monitor import SystemMonitor
                    system_monitor = SystemMonitor()

                # Get current system data
                data = system_monitor.get_current_data()

                # Extract GPU average if available
                gpu_percent = 0.0
                if data.get('gpu') and len(data['gpu']) > 0:
                    gpu_percent = sum(gpu['load'] for gpu in data['gpu']) / len(data['gpu'])

                return {
                    "timestamp": data['timestamp'].isoformat(),
                    "cpu": data['cpu']['percent'],
                    "memory": data['memory']['percent'],
                    "disk": data['disk']['percent'],
                    "gpu": gpu_percent,
                    "status": "success"
                }

            except Exception as e:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

        @app.post("/system/cleanup")
        async def cleanup_system():
            """Clean up system resources."""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Clear pipeline cache
                self.pipeline_cache.clear()

                return {
                    "message": "System cleanup completed",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                from fastapi import HTTPException
                raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

        @app.get("/export/models")
        async def export_models_info():
            """Export models information."""
            return self.model_manager.export_models_info()

        # Include routers
        models_router = ModelsRouter(self.model_manager, self.logger)
        prediction_router = PredictionRouter(self.model_manager, self.device_manager, self.logger)

        app.include_router(models_router.get_router())
        app.include_router(prediction_router.get_router())

    def start_server(self, model_ports=None) -> str:
        """Start FastAPI server."""
        if model_ports is None:
            # Single server mode (backward compatibility)
            return self._start_single_server()

        # Multi-port mode
        self.model_ports = model_ports.copy()
        loaded_models = self.model_manager.get_loaded_models()

        if not loaded_models:
            return "No models loaded. Please load models first."

        results = []
        ports_to_start = set()

        # Collect ports for loaded models
        for model_name in loaded_models:
            port = model_ports.get(model_name, self.default_port)
            ports_to_start.add(port)

        # Start server for each port
        for port in ports_to_start:
            if port in self.servers and self.servers[port]['running']:
                results.append(f"Server already running on port {port}")
                continue

            if self._is_port_in_use(port):
                results.append(f"Port {port} is already in use by another process")
                continue

            models_for_port = [m for m, p in model_ports.items() if p == port and m in loaded_models]

            if models_for_port:
                result = self._start_server_on_port(port, models_for_port)
                results.append(result)

        return "; ".join(results)

    def _start_single_server(self) -> str:
        """Start single server (backward compatibility)."""
        if self.running:
            return f"Server already running at http://{self.host}:{self.default_port}"

        def run_server():
            try:
                self.running = True
                uvicorn.run(self.app, host=self.host, port=self.default_port, log_level="info")
            except Exception as e:
                self.logger.error(f"Server error: {e}")
            finally:
                self.running = False

        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()

        return f"Server started at http://{self.host}:{self.default_port}"

    def _start_server_on_port(self, port: int, target_models: List[str]) -> str:
        """Start server on specific port for specific models."""
        try:
            # Create app for this port
            app = self.create_app(target_models)

            # Create uvicorn server instance
            config = uvicorn.Config(app=app, host=self.host, port=port, log_level="info")
            server = uvicorn.Server(config)

            def run_server():
                try:
                    self.servers[port]['running'] = True
                    self.servers[port]['server_instance'] = server

                    # Run in new event loop
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(server.serve())

                except Exception as e:
                    self.logger.error(f"Server error on port {port}: {e}")
                finally:
                    if port in self.servers:
                        self.servers[port]['running'] = False

            # Create and start server thread
            server_thread = threading.Thread(target=run_server)
            server_thread.daemon = True

            self.servers[port] = {
                'app': app,
                'thread': server_thread,
                'running': False,
                'models': target_models,
                'server_instance': None
            }

            server_thread.start()

            return f"Server started on port {port} for models: {', '.join(target_models)}"

        except Exception as e:
            return f"Failed to start server on port {port}: {str(e)}"

    def stop_server(self, port=None) -> str:
        """Stop server."""
        if port is None:
            # Stop all servers
            results = []

            # Stop multi-port servers
            for server_port in list(self.servers.keys()):
                if self.servers[server_port]['running']:
                    result = self._stop_server_gracefully(server_port)
                    results.append(result)

            # Stop default server
            if self.running:
                self.running = False
                self._force_kill_port(self.default_port)
                results.append(f"Default server on port {self.default_port} stopped")

            # Clear pipeline cache
            self.pipeline_cache.clear()

            return "; ".join(results) if results else "No servers running"

        else:
            # Stop specific port server
            if port == self.default_port and self.running:
                self.running = False
                self._force_kill_port(port)
                return f"Default server on port {port} stopped"

            elif port in self.servers and self.servers[port]['running']:
                return self._stop_server_gracefully(port)

            else:
                return f"No server running on port {port}"

    def is_running(self, port=None) -> bool:
        """Check if server is running."""
        if port is None:
            return self.running or any(s['running'] for s in self.servers.values())
        elif port == self.default_port:
            return self.running
        else:
            return port in self.servers and self.servers[port]['running']

    def get_server_info(self) -> Dict[str, Any]:
        """Get server information."""
        info = {
            "host": self.host,
            "default_port": self.default_port,
            "default_server_running": self.running,
            "loaded_models": len(self.model_manager.get_loaded_models()),
            "cached_pipelines": len(self.pipeline_cache),
            "model_ports": self.model_ports.copy(),
            "active_servers": []
        }

        # Default server info
        if self.running:
            info["active_servers"].append({
                "port": self.default_port,
                "url": f"http://{self.host}:{self.default_port}",
                "docs_url": f"http://{self.host}:{self.default_port}/docs",
                "models": "all",
                "running": True
            })

        # Multi-port server info
        for port, server_info in self.servers.items():
            if server_info['running']:
                info["active_servers"].append({
                    "port": port,
                    "url": f"http://{self.host}:{port}",
                    "docs_url": f"http://{self.host}:{port}/docs",
                    "models": server_info['models'],
                    "running": server_info['running']
                })

        return info

    def get_running_ports(self) -> List[int]:
        """Get list of running ports."""
        ports = []

        if self.running:
            ports.append(self.default_port)

        for port, server_info in self.servers.items():
            if server_info['running']:
                ports.append(port)

        return sorted(ports)

    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                return result == 0
        except:
            return False

    def _stop_server_gracefully(self, port: int) -> str:
        """Stop server gracefully."""
        try:
            if port not in self.servers:
                return f"No server found on port {port}"

            server_info = self.servers[port]

            # Graceful shutdown if server instance exists
            if 'server_instance' in server_info and server_info['server_instance']:
                try:
                    server_instance = server_info['server_instance']

                    def shutdown_async():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            server_instance.should_exit = True
                            time.sleep(1)
                            loop.close()
                        except Exception as e:
                            self.logger.error(f"Async shutdown error: {e}")

                    shutdown_thread = threading.Thread(target=shutdown_async)
                    shutdown_thread.start()
                    shutdown_thread.join(timeout=3)

                    self.logger.info(f"Graceful shutdown initiated for port {port}")

                except Exception as e:
                    self.logger.error(f"Graceful shutdown failed for port {port}: {e}")

            # Update status
            server_info['running'] = False

            # Force kill if still in use
            time.sleep(2)
            if self._is_port_in_use(port):
                self._force_kill_port(port)
                return f"Server on port {port} force stopped"
            else:
                return f"Server on port {port} gracefully stopped"

        except Exception as e:
            return f"Failed to stop server on port {port}: {str(e)}"

    def _force_kill_port(self, port: int) -> bool:
        """Force kill process on port."""
        try:
            streamlit_pid = os.getpid()

            result = subprocess.run(['lsof', '-ti', f':{port}'],
                                  capture_output=True, text=True, timeout=5)
            if result.stdout.strip():
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid.strip():
                        try:
                            pid_int = int(pid.strip())
                            if pid_int != streamlit_pid:
                                check_result = subprocess.run(['ps', '-p', str(pid_int), '-o', 'comm='],
                                                            capture_output=True, text=True, timeout=2)
                                process_name = check_result.stdout.strip()

                                if any(keyword in process_name.lower() for keyword in ['uvicorn', 'fastapi', 'python']):
                                    subprocess.run(['kill', '-15', str(pid_int)], capture_output=True, timeout=2)
                                    self.logger.info(f"Sent SIGTERM to FastAPI process {pid_int} on port {port}")

                                    time.sleep(2)
                                    subprocess.run(['kill', '-9', str(pid_int)], capture_output=True, timeout=1)
                                    self.logger.info(f"Killed FastAPI process {pid_int} on port {port}")
                        except (ValueError, subprocess.TimeoutExpired):
                            pass
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return False