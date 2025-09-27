"""
API package for FastAPI server components.

This package provides modularized API components for the Hugging Face FastAPI server,
extracted from the original monolithic servers/fastapi_server.py file.
"""

from .models import PredictionRequest, ModelLoadRequest, ModelInfo, ModelsListResponse, PredictionResponse
from .middleware import JSONRepairMiddleware, RequestLoggingMiddleware, CORSMiddleware
from .models_router import ModelsRouter
from .prediction_router import PredictionRouter
from .server_manager import FastAPIServerManager

__all__ = [
    'PredictionRequest',
    'ModelLoadRequest',
    'ModelInfo',
    'ModelsListResponse',
    'PredictionResponse',
    'JSONRepairMiddleware',
    'RequestLoggingMiddleware',
    'CORSMiddleware',
    'ModelsRouter',
    'PredictionRouter',
    'FastAPIServerManager'
]