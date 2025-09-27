"""
Model management API router for FastAPI server.

This module contains router endpoints for model-related operations,
extracted from the original fastapi_server.py file.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, List
from datetime import datetime
import logging

from .models import ModelLoadRequest, ModelInfo, ModelsListResponse, ModelTasksResponse, ModelAnalysisResponse, ErrorResponse


class ModelsRouter:
    """Router class for model management endpoints."""

    def __init__(self, model_manager, logger=None):
        """Initialize models router."""
        self.model_manager = model_manager
        self.logger = logger or logging.getLogger("ModelsRouter")
        self.router = APIRouter(prefix="/models", tags=["models"])
        self._setup_routes()

    def _setup_routes(self):
        """Setup model management routes."""

        @self.router.get("/")
        async def list_models():
            """Get list of loaded models."""
            try:
                loaded_models = self.model_manager.get_loaded_models()
                all_models = list(self.model_manager.models.keys())
                models_status = self.model_manager.get_all_models_status()

                # Calculate memory usage
                memory_info = self.model_manager.get_memory_info()

                return ModelsListResponse(
                    loaded_models={
                        name: ModelInfo(
                            name=info.name,
                            path=info.path,
                            type=getattr(info, 'model_type', None),
                            status=info.status,
                            loaded_at=info.load_time.isoformat() if info.load_time else None,
                            memory_usage=info.memory_usage,
                            device=getattr(info, 'device', None)
                        ) for name, info in models_status.items() if info.status == "loaded"
                    },
                    total_count=len(loaded_models),
                    memory_usage=memory_info
                )

            except Exception as e:
                self.logger.error(f"Error listing models: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

        @self.router.get("/{model_name}")
        async def get_model_info(model_name: str):
            """Get information about a specific model."""
            try:
                model_info = self.model_manager.get_model_info(model_name)
                if not model_info:
                    raise HTTPException(status_code=404, detail="Model not found")

                available_tasks = self.model_manager.get_available_tasks(model_name)

                return {
                    "name": model_info.name,
                    "path": model_info.path,
                    "status": model_info.status,
                    "memory_usage": model_info.memory_usage,
                    "load_time": model_info.load_time.isoformat() if model_info.load_time else None,
                    "config_analysis": model_info.config_analysis,
                    "available_tasks": available_tasks,
                    "device": getattr(model_info, 'device', None),
                    "model_type": getattr(model_info, 'model_type', None)
                }

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting model info for {model_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

        @self.router.post("/load")
        async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
            """Load a model asynchronously."""
            try:
                def load_callback(model_name: str, success: bool, message: str):
                    """Callback function for model loading."""
                    if success:
                        self.logger.info(f"Model {model_name} loaded successfully via API")
                    else:
                        self.logger.error(f"Model {model_name} failed to load via API: {message}")

                background_tasks.add_task(
                    self.model_manager.load_model_async,
                    request.model_name,
                    request.model_path,
                    load_callback
                )

                return {
                    "message": f"Model {request.model_name} loading started",
                    "model_name": request.model_name,
                    "model_path": request.model_path,
                    "status": "loading",
                    "timestamp": datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.error(f"Error initiating model load: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to start model loading: {str(e)}")

        @self.router.post("/{model_name}/unload")
        async def unload_model(model_name: str):
            """Unload a specific model."""
            try:
                success = self.model_manager.unload_model(model_name)
                if success:
                    return {
                        "message": f"Model {model_name} unloaded successfully",
                        "model_name": model_name,
                        "status": "unloaded",
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    raise HTTPException(status_code=404, detail="Model not found or already unloaded")

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error unloading model {model_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

        @self.router.get("/{model_name}/tasks")
        async def get_model_tasks(model_name: str):
            """Get supported tasks for a specific model."""
            try:
                tasks = self.model_manager.get_available_tasks(model_name)
                if not tasks:
                    raise HTTPException(status_code=404, detail="Model not found or no tasks available")

                return ModelTasksResponse(
                    model_name=model_name,
                    supported_tasks=tasks,
                    task_info={
                        "count": len(tasks),
                        "primary_task": tasks[0] if tasks else None
                    }
                )

            except HTTPException:
                raise
            except Exception as e:
                self.logger.error(f"Error getting tasks for model {model_name}: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to get model tasks: {str(e)}")

        @self.router.post("/{model_name}/analyze")
        async def analyze_model(model_name: str, model_path: str):
            """Analyze a model without loading it."""
            try:
                analysis = self.model_manager.analyze_model(model_path)

                return ModelAnalysisResponse(
                    model_name=model_name,
                    analysis=analysis,
                    success=True
                )

            except Exception as e:
                self.logger.error(f"Error analyzing model {model_name}: {e}")
                return ModelAnalysisResponse(
                    model_name=model_name,
                    analysis={},
                    success=False,
                    error=str(e)
                )

    def get_router(self) -> APIRouter:
        """Get the configured router instance."""
        return self.router