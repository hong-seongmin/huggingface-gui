from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import threading
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from model_manager import MultiModelManager
from transformers import pipeline
import asyncio

class PredictionRequest(BaseModel):
    text: str
    max_length: Optional[int] = 512
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = 50
    do_sample: Optional[bool] = False

class ModelLoadRequest(BaseModel):
    model_name: str
    model_path: str

class FastAPIServer:
    def __init__(self, model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000):
        self.app = FastAPI(
            title="HuggingFace Model API",
            description="API for managing and using HuggingFace models",
            version="1.0.0"
        )
        self.model_manager = model_manager
        self.host = host
        self.port = port
        self.server_thread = None
        self.server = None
        self.running = False
        
        # 파이프라인 캐시
        self.pipeline_cache = {}
        
        self.setup_routes()
    
    def setup_routes(self):
        """API 라우트 설정"""
        
        @self.app.get("/")
        async def root():
            return {
                "message": "HuggingFace Model API",
                "version": "1.0.0",
                "docs": "/docs",
                "models_loaded": len(self.model_manager.get_loaded_models())
            }
        
        @self.app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": len(self.model_manager.get_loaded_models())
            }
        
        @self.app.get("/models")
        async def list_models():
            """로드된 모델 목록 반환"""
            return {
                "loaded_models": self.model_manager.get_loaded_models(),
                "all_models": list(self.model_manager.models.keys()),
                "models_status": self.model_manager.get_all_models_status()
            }
        
        @self.app.get("/models/{model_name}")
        async def get_model_info(model_name: str):
            """특정 모델 정보 반환"""
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info:
                raise HTTPException(status_code=404, detail="Model not found")
            
            return {
                "name": model_info.name,
                "path": model_info.path,
                "status": model_info.status,
                "memory_usage": model_info.memory_usage,
                "load_time": model_info.load_time.isoformat() if model_info.load_time else None,
                "config_analysis": model_info.config_analysis,
                "available_tasks": self.model_manager.get_available_tasks(model_name)
            }
        
        @self.app.post("/models/load")
        async def load_model(request: ModelLoadRequest, background_tasks: BackgroundTasks):
            """모델 로드"""
            def load_callback(model_name: str, success: bool, message: str):
                if success:
                    print(f"Model {model_name} loaded successfully via API")
                else:
                    print(f"Model {model_name} failed to load via API: {message}")
            
            background_tasks.add_task(
                self.model_manager.load_model_async,
                request.model_name,
                request.model_path,
                load_callback
            )
            
            return {
                "message": f"Model {request.model_name} loading started",
                "model_name": request.model_name,
                "model_path": request.model_path
            }
        
        @self.app.post("/models/{model_name}/unload")
        async def unload_model(model_name: str):
            """모델 언로드"""
            success = self.model_manager.unload_model(model_name)
            if success:
                # 파이프라인 캐시에서 제거
                if model_name in self.pipeline_cache:
                    del self.pipeline_cache[model_name]
                return {"message": f"Model {model_name} unloaded successfully"}
            else:
                raise HTTPException(status_code=404, detail="Model not found or already unloaded")
        
        @self.app.post("/models/{model_name}/predict")
        async def predict(model_name: str, request: PredictionRequest):
            """모델 예측"""
            # 모델 로드 상태 확인
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info or model_info.status != "loaded":
                raise HTTPException(status_code=404, detail="Model not found or not loaded")
            
            # 모델과 토크나이저 가져오기
            model_tokenizer = self.model_manager.get_model_for_inference(model_name)
            if not model_tokenizer:
                raise HTTPException(status_code=500, detail="Failed to get model for inference")
            
            model, tokenizer = model_tokenizer
            
            # 지원하는 태스크 확인
            available_tasks = self.model_manager.get_available_tasks(model_name)
            if not available_tasks:
                raise HTTPException(status_code=400, detail="No supported tasks found for this model")
            
            # 첫 번째 지원 태스크 사용
            task = available_tasks[0]
            
            try:
                # 파이프라인 생성 또는 캐시에서 가져오기
                cache_key = f"{model_name}_{task}"
                if cache_key not in self.pipeline_cache:
                    self.pipeline_cache[cache_key] = pipeline(
                        task=task,
                        model=model,
                        tokenizer=tokenizer,
                        return_full_text=False if task == "text-generation" else True
                    )
                
                pipe = self.pipeline_cache[cache_key]
                
                # 태스크별 파라미터 설정
                if task == "text-generation":
                    result = pipe(
                        request.text,
                        max_length=request.max_length,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        top_k=request.top_k,
                        do_sample=request.do_sample,
                        pad_token_id=tokenizer.eos_token_id
                    )
                else:
                    result = pipe(request.text)
                
                return {
                    "model_name": model_name,
                    "task": task,
                    "input": request.text,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
        
        @self.app.get("/models/{model_name}/tasks")
        async def get_model_tasks(model_name: str):
            """모델이 지원하는 태스크 목록"""
            tasks = self.model_manager.get_available_tasks(model_name)
            if not tasks:
                raise HTTPException(status_code=404, detail="Model not found or no tasks available")
            
            return {
                "model_name": model_name,
                "supported_tasks": tasks
            }
        
        @self.app.post("/models/{model_name}/analyze")
        async def analyze_model(model_name: str, model_path: str):
            """모델 분석 (로드 없이)"""
            try:
                analysis = self.model_manager.analyze_model(model_path)
                return {
                    "model_name": model_name,
                    "model_path": model_path,
                    "analysis": analysis
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
        
        @self.app.get("/system/status")
        async def get_system_status():
            """시스템 상태 정보"""
            return self.model_manager.get_system_summary()
        
        @self.app.get("/system/memory")
        async def get_memory_info():
            """메모리 사용량 정보"""
            return self.model_manager.get_memory_info()
        
        @self.app.post("/system/cleanup")
        async def cleanup_system():
            """시스템 정리 (GPU 캐시 등)"""
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # 파이프라인 캐시 정리
                self.pipeline_cache.clear()
                
                return {
                    "message": "System cleanup completed",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
        
        @self.app.get("/export/models")
        async def export_models_info():
            """모델 정보 내보내기"""
            return self.model_manager.export_models_info()
    
    def start_server(self):
        """FastAPI 서버 시작"""
        if self.running:
            return f"Server already running at http://{self.host}:{self.port}"
        
        def run_server():
            try:
                self.running = True
                uvicorn.run(self.app, host=self.host, port=self.port, log_level="info")
            except Exception as e:
                print(f"Server error: {e}")
            finally:
                self.running = False
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return f"Server started at http://{self.host}:{self.port}"
    
    def stop_server(self):
        """서버 중지"""
        if not self.running:
            return "Server is not running"
        
        self.running = False
        # 서버 종료는 uvicorn의 제한으로 완전한 종료가 어려움
        # 실제 프로덕션에서는 별도의 프로세스 관리 필요
        return "Server stop requested"
    
    def is_running(self) -> bool:
        """서버 실행 상태 확인"""
        return self.running
    
    def get_server_info(self) -> Dict:
        """서버 정보 반환"""
        return {
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "url": f"http://{self.host}:{self.port}",
            "docs_url": f"http://{self.host}:{self.port}/docs",
            "loaded_models": len(self.model_manager.get_loaded_models()),
            "cached_pipelines": len(self.pipeline_cache)
        }
    
    def clear_pipeline_cache(self):
        """파이프라인 캐시 정리"""
        self.pipeline_cache.clear()
    
    def get_available_endpoints(self) -> List[Dict]:
        """사용 가능한 엔드포인트 목록"""
        endpoints = []
        for route in self.app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                endpoints.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name if hasattr(route, 'name') else None
                })
        return endpoints

# 편의 함수들
def create_api_server(model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """FastAPI 서버 생성"""
    return FastAPIServer(model_manager, host, port)

def start_background_server(model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """백그라운드에서 서버 시작"""
    server = create_api_server(model_manager, host, port)
    server.start_server()
    return server