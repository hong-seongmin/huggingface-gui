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
import torch

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
        self.model_manager = model_manager
        self.host = host
        self.default_port = port
        
        # 멀티 서버 관리
        self.servers = {}  # {port: {'app': app, 'thread': thread, 'running': bool}}
        self.model_ports = {}  # {model_name: port}
        
        # 파이프라인 캐시
        self.pipeline_cache = {}
        
        # 기본 서버 (하위 호환성을 위해 유지)
        self.app = self.create_app()
        self.server_thread = None
        self.server = None
        self.running = False
    
    def create_app(self, target_models=None):
        """FastAPI 앱 생성 (특정 모델들을 위한 서버 또는 전체 서버)"""
        app = FastAPI(
            title="HuggingFace Model API",
            description="API for managing and using HuggingFace models",
            version="1.0.0"
        )
        
        self.setup_routes(app, target_models)
        return app
    
    def setup_routes(self, app, target_models=None):
        """API 라우트 설정"""
        
        @app.get("/")
        async def root():
            return {
                "message": "HuggingFace Model API",
                "version": "1.0.0",
                "docs": "/docs",
                "models_loaded": len(self.model_manager.get_loaded_models())
            }
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "models_loaded": len(self.model_manager.get_loaded_models())
            }
        
        @app.get("/models")
        async def list_models():
            """로드된 모델 목록 반환"""
            loaded_models = self.model_manager.get_loaded_models()
            # 특정 모델들만 서빙하는 서버인 경우 필터링
            if target_models:
                loaded_models = [m for m in loaded_models if m in target_models]
            
            return {
                "loaded_models": loaded_models,
                "all_models": list(self.model_manager.models.keys()),
                "models_status": self.model_manager.get_all_models_status(),
                "target_models": target_models
            }
        
        @app.get("/models/{model_name}")
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
        
        @app.post("/models/load")
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
        
        @app.post("/models/{model_name}/unload")
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
        
        @app.post("/models/{model_name}/predict")
        async def predict(model_name: str, request: PredictionRequest):
            """모델 예측"""
            # 모델 로드 상태 확인
            model_info = self.model_manager.get_model_info(model_name)
            if not model_info or model_info.status != "loaded":
                raise HTTPException(status_code=404, detail="Model not found or not loaded")
            
            # 특정 모델들만 서빙하는 서버인 경우 체크
            if target_models and model_name not in target_models:
                raise HTTPException(status_code=404, detail="Model not available on this server")
            
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
                # 텍스트 분류의 경우 파이프라인 우회하여 직접 처리
                if task == "text-classification":
                    try:
                        # 직접 모델 추론 (파이프라인 우회)
                        inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True)
                        
                        # GPU/CPU 디바이스 일치 확인
                        try:
                            device = next(model.parameters()).device
                            inputs = {k: v.to(device) for k, v in inputs.items()}
                        except StopIteration:
                            # 모델에 파라미터가 없는 경우 CPU 사용
                            device = torch.device('cpu')
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                            if hasattr(outputs, 'logits'):
                                import torch.nn.functional as F
                                probs = F.softmax(outputs.logits, dim=-1)
                                predicted_class = torch.argmax(outputs.logits, dim=-1)
                                
                                # 모델의 label 매핑 가져오기
                                if hasattr(model.config, 'id2label'):
                                    id2label = model.config.id2label
                                    predicted_label = id2label.get(predicted_class.item(), f"LABEL_{predicted_class.item()}")
                                else:
                                    predicted_label = f"LABEL_{predicted_class.item()}"
                                
                                result = [{
                                    "label": predicted_label,
                                    "score": probs[0][predicted_class.item()].item()
                                }]
                            else:
                                raise Exception(f"Model output does not contain logits: {outputs}")
                    except Exception as e:
                        raise HTTPException(
                            status_code=500, 
                            detail=f"Direct model inference failed: {str(e)}"
                        )
                else:
                    # 다른 태스크의 경우 파이프라인 사용
                    cache_key = f"{model_name}_{task}"
                    if cache_key not in self.pipeline_cache:
                        # 안전한 파이프라인 생성 - 각 태스크별로 적절한 파라미터 사용
                        try:
                            if task == "text-generation":
                                self.pipeline_cache[cache_key] = pipeline(
                                    task=task,
                                    model=model,
                                    tokenizer=tokenizer,
                                    return_full_text=False
                                )
                            elif task == "feature-extraction":
                                # 임베딩 모델용 파이프라인
                                self.pipeline_cache[cache_key] = pipeline(
                                    task=task,
                                    model=model,
                                    tokenizer=tokenizer
                                )
                            else:
                                # 기타 태스크는 기본 파라미터만 사용
                                self.pipeline_cache[cache_key] = pipeline(
                                    task=task,
                                    model=model,
                                    tokenizer=tokenizer
                                )
                        except Exception as e:
                            # 파이프라인 생성 실패 시 더 단순한 방법 시도
                            print(f"Pipeline creation failed for {task}: {e}")
                            try:
                                # 가장 기본적인 형태로 재시도
                                self.pipeline_cache[cache_key] = pipeline(
                                    task,
                                    model=model,
                                    tokenizer=tokenizer
                                )
                            except Exception as e2:
                                raise HTTPException(
                                    status_code=500, 
                                    detail=f"Failed to create pipeline for {task}: {str(e2)}"
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
                        # 기타 태스크 (feature-extraction 등)
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
        
        @app.get("/models/{model_name}/tasks")
        async def get_model_tasks(model_name: str):
            """모델이 지원하는 태스크 목록"""
            tasks = self.model_manager.get_available_tasks(model_name)
            if not tasks:
                raise HTTPException(status_code=404, detail="Model not found or no tasks available")
            
            return {
                "model_name": model_name,
                "supported_tasks": tasks
            }
        
        @app.post("/models/{model_name}/analyze")
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
        
        @app.get("/system/status")
        async def get_system_status():
            """시스템 상태 정보"""
            return self.model_manager.get_system_summary()
        
        @app.get("/system/memory")
        async def get_memory_info():
            """메모리 사용량 정보"""
            return self.model_manager.get_memory_info()
        
        @app.post("/system/cleanup")
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
        
        @app.get("/export/models")
        async def export_models_info():
            """모델 정보 내보내기"""
            return self.model_manager.export_models_info()
    
    def start_server(self, model_ports=None):
        """FastAPI 서버 시작 (멀티 포트 지원)"""
        if model_ports is None:
            # 기본 단일 서버 모드 (하위 호환성)
            return self._start_single_server()
        
        # 멀티 포트 모드
        self.model_ports = model_ports.copy()
        loaded_models = self.model_manager.get_loaded_models()
        
        if not loaded_models:
            return "No models loaded. Please load models first."
        
        results = []
        ports_to_start = set()
        
        # 로드된 모델들의 포트 수집
        for model_name in loaded_models:
            port = model_ports.get(model_name, self.default_port)
            ports_to_start.add(port)
        
        # 각 포트별로 서버 시작
        for port in ports_to_start:
            if port in self.servers and self.servers[port]['running']:
                results.append(f"Server already running on port {port}")
                continue
            
            # 해당 포트에서 서빙할 모델들 찾기
            models_for_port = [m for m, p in model_ports.items() if p == port and m in loaded_models]
            
            if models_for_port:
                result = self._start_server_on_port(port, models_for_port)
                results.append(result)
        
        return "; ".join(results)
    
    def _start_single_server(self):
        """단일 서버 시작 (하위 호환성)"""
        if self.running:
            return f"Server already running at http://{self.host}:{self.default_port}"
        
        def run_server():
            try:
                self.running = True
                uvicorn.run(self.app, host=self.host, port=self.default_port, log_level="info")
            except Exception as e:
                print(f"Server error: {e}")
            finally:
                self.running = False
        
        self.server_thread = threading.Thread(target=run_server)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        return f"Server started at http://{self.host}:{self.default_port}"
    
    def _start_server_on_port(self, port, target_models):
        """특정 포트에서 특정 모델들을 위한 서버 시작"""
        try:
            # 해당 포트의 앱 생성
            app = self.create_app(target_models)
            
            def run_server():
                try:
                    self.servers[port]['running'] = True
                    uvicorn.run(app, host=self.host, port=port, log_level="info")
                except Exception as e:
                    print(f"Server error on port {port}: {e}")
                finally:
                    if port in self.servers:
                        self.servers[port]['running'] = False
            
            # 서버 정보 저장
            server_thread = threading.Thread(target=run_server)
            server_thread.daemon = True
            
            self.servers[port] = {
                'app': app,
                'thread': server_thread,
                'running': False,
                'models': target_models
            }
            
            server_thread.start()
            
            return f"Server started on port {port} for models: {', '.join(target_models)}"
            
        except Exception as e:
            return f"Failed to start server on port {port}: {str(e)}"
    
    def stop_server(self, port=None):
        """서버 중지 (특정 포트 또는 전체)"""
        if port is None:
            # 모든 서버 중지
            results = []
            
            # 멀티 포트 서버들 중지
            for server_port in list(self.servers.keys()):
                if self.servers[server_port]['running']:
                    self.servers[server_port]['running'] = False
                    results.append(f"Server on port {server_port} stop requested")
            
            # 기본 서버 중지
            if self.running:
                self.running = False
                results.append(f"Default server on port {self.default_port} stop requested")
            
            return "; ".join(results) if results else "No servers running"
        
        else:
            # 특정 포트 서버 중지
            if port == self.default_port and self.running:
                self.running = False
                return f"Default server on port {port} stop requested"
            
            elif port in self.servers and self.servers[port]['running']:
                self.servers[port]['running'] = False
                return f"Server on port {port} stop requested"
            
            else:
                return f"No server running on port {port}"
    
    def is_running(self, port=None) -> bool:
        """서버 실행 상태 확인"""
        if port is None:
            # 하나라도 실행 중이면 True
            return self.running or any(s['running'] for s in self.servers.values())
        
        elif port == self.default_port:
            return self.running
        
        else:
            return port in self.servers and self.servers[port]['running']
    
    def get_server_info(self) -> Dict:
        """서버 정보 반환 (멀티 포트 지원)"""
        info = {
            "host": self.host,
            "default_port": self.default_port,
            "default_server_running": self.running,
            "loaded_models": len(self.model_manager.get_loaded_models()),
            "cached_pipelines": len(self.pipeline_cache),
            "model_ports": self.model_ports.copy(),
            "active_servers": []
        }
        
        # 기본 서버 정보
        if self.running:
            info["active_servers"].append({
                "port": self.default_port,
                "url": f"http://{self.host}:{self.default_port}",
                "docs_url": f"http://{self.host}:{self.default_port}/docs",
                "models": "all",
                "running": True
            })
        
        # 멀티 포트 서버들 정보
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
    
    def clear_pipeline_cache(self):
        """파이프라인 캐시 정리"""
        self.pipeline_cache.clear()
    
    def get_available_endpoints(self, port=None) -> List[Dict]:
        """사용 가능한 엔드포인트 목록"""
        if port is None or port == self.default_port:
            app = self.app
        elif port in self.servers:
            app = self.servers[port]['app']
        else:
            return []
        
        endpoints = []
        for route in app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                endpoints.append({
                    "path": route.path,
                    "methods": list(route.methods),
                    "name": route.name if hasattr(route, 'name') else None
                })
        return endpoints
    
    def get_running_ports(self) -> List[int]:
        """실행 중인 포트 목록 반환"""
        ports = []
        
        if self.running:
            ports.append(self.default_port)
        
        for port, server_info in self.servers.items():
            if server_info['running']:
                ports.append(port)
        
        return sorted(ports)
    
    def get_model_server_port(self, model_name: str) -> int:
        """특정 모델이 실행 중인 포트 반환"""
        if model_name in self.model_ports:
            port = self.model_ports[model_name]
            if self.is_running(port):
                return port
        
        # 기본 서버에서 실행 중인지 확인
        if self.running:
            return self.default_port
        
        return None

# 편의 함수들
def create_api_server(model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """FastAPI 서버 생성"""
    return FastAPIServer(model_manager, host, port)

def start_background_server(model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """백그라운드에서 서버 시작"""
    server = create_api_server(model_manager, host, port)
    server.start_server()
    return server