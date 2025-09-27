"""
Pydantic models for FastAPI request/response validation.

This module contains all data models used in the FastAPI server for request validation,
response formatting, and data serialization.
"""

from pydantic import BaseModel, ValidationError
from typing import Dict, Any, List, Optional


class PredictionRequest(BaseModel):
    """Request model for text prediction/generation."""
    text: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    do_sample: Optional[bool] = False
    return_all_scores: Optional[bool] = False
    aggregation_strategy: Optional[str] = "simple"

    class Config:
        """Pydantic configuration."""
        extra = "forbid"  # Don't allow extra fields
        validate_assignment = True


class ModelLoadRequest(BaseModel):
    """Request model for loading a model."""
    model_name: str
    model_path: str

    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True


class ModelInfo(BaseModel):
    """Model information response."""
    name: str
    path: Optional[str] = None
    type: Optional[str] = None
    status: str
    loaded_at: Optional[str] = None
    memory_usage: Optional[Dict[str, Any]] = None
    device: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    model_name: str
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelsListResponse(BaseModel):
    """Response model for models list."""
    loaded_models: Dict[str, ModelInfo]
    total_count: int
    memory_usage: Optional[Dict[str, Any]] = None


class HealthCheckResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: Optional[str] = None
    models_count: Optional[int] = None
    system_info: Optional[Dict[str, Any]] = None


class ModelTasksResponse(BaseModel):
    """Model tasks response."""
    model_name: str
    supported_tasks: List[str]
    task_info: Optional[Dict[str, Any]] = None


class ModelAnalysisResponse(BaseModel):
    """Model analysis response."""
    model_name: str
    analysis: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    status_code: int
    timestamp: str


class SystemStatusResponse(BaseModel):
    """System status response."""
    cpu_usage: Optional[float] = None
    memory_usage: Optional[Dict[str, Any]] = None
    gpu_usage: Optional[Dict[str, Any]] = None
    disk_usage: Optional[Dict[str, Any]] = None
    uptime: Optional[str] = None


class ExportResponse(BaseModel):
    """Model export response."""
    models: Dict[str, Dict[str, Any]]
    export_timestamp: str
    total_models: int