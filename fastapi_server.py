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
import logging
from device_manager import device_manager

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
        
        # 로거 설정
        self.logger = logging.getLogger("FastAPIServer")
        
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
            # 특정 모델들만 서빙하는 서버인 경우 필터링 (임시로 비활성화)
            # if target_models:
            #     loaded_models = [m for m in loaded_models if m in target_models]
            
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
            
            # 특정 모델들만 서빙하는 서버인 경우 체크 (임시로 비활성화)
            # if target_models and model_name not in target_models:
            #     raise HTTPException(status_code=404, detail="Model not available on this server")
            
            # 모델과 토크나이저 가져오기
            model_tokenizer = self.model_manager.get_model_for_inference(model_name)
            if not model_tokenizer:
                raise HTTPException(status_code=500, detail="Failed to get model for inference")
            
            model, tokenizer = model_tokenizer
            
            # 지원하는 태스크 확인
            available_tasks = self.model_manager.get_available_tasks(model_name)
            if not available_tasks:
                raise HTTPException(status_code=400, detail="No supported tasks found for this model")
            
            # 첫 번째 지원 태스크 사용, 단 sentiment 관련 모델은 강제로 text-classification 사용
            if "sentiment" in model_name.lower() or "classifier" in model_name.lower():
                task = "text-classification"
                print(f"[DEBUG] Forcing text-classification task for {model_name}")
            else:
                task = available_tasks[0]
            
            try:
                # 통합 추론 엔진 - 모든 태스크에 대해 직접 모델 추론 사용
                result = self._unified_inference(model, tokenizer, request.text, task)
                
                
                
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
            # 기존 서버 상태 확인
            if port in self.servers and self.servers[port]['running']:
                results.append(f"Server already running on port {port}")
                continue
            
            # 포트 사용 가능 여부 확인
            if self._is_port_in_use(port):
                results.append(f"Port {port} is already in use by another process")
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
            import uvicorn
            import asyncio
            
            # 해당 포트의 앱 생성
            app = self.create_app(target_models)
            
            # uvicorn 서버 인스턴스 생성
            config = uvicorn.Config(app=app, host=self.host, port=port, log_level="info")
            server = uvicorn.Server(config)
            
            def run_server():
                try:
                    self.servers[port]['running'] = True
                    self.servers[port]['server_instance'] = server
                    
                    # 새로운 이벤트 루프에서 서버 실행
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(server.serve())
                    
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
                'models': target_models,
                'server_instance': None
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
                    result = self._stop_server_gracefully(server_port)
                    results.append(result)
            
            # 기본 서버 중지
            if self.running:
                self.running = False
                self._force_kill_port(self.default_port)
                results.append(f"Default server on port {self.default_port} stopped")
            
            # 파이프라인 캐시 정리
            self.clear_pipeline_cache()
            
            return "; ".join(results) if results else "No servers running"
        
        else:
            # 특정 포트 서버 중지
            if port == self.default_port and self.running:
                self.running = False
                self._force_kill_port(port)
                return f"Default server on port {port} stopped"
            
            elif port in self.servers and self.servers[port]['running']:
                return self._stop_server_gracefully(port)
            
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
        # 1. model_ports 딕셔너리에서 확인
        if model_name in self.model_ports:
            port = self.model_ports[model_name]
            if self.is_running(port):
                return port
        
        # 2. 활성 서버에서 해당 모델을 찾기 (동기화 보완)
        for port, server_info in self.servers.items():
            if server_info.get('running', False):
                models = server_info.get('models', [])
                if model_name in models:
                    return port
        
        # 3. 기본 서버에서 실행 중인지 확인
        if self.running:
            # 로드된 모델 목록에 해당 모델이 있는지 확인
            loaded_models = self.model_manager.get_loaded_models()
            if model_name in loaded_models:
                return self.default_port
        
        return None
    
    def _force_kill_port(self, port):
        """FastAPI 서버만 안전하게 중지 (Streamlit은 유지)"""
        try:
            import subprocess
            import os
            
            # 현재 Streamlit 프로세스 PID 보호
            streamlit_pid = os.getpid()
            
            # lsof로 포트 사용 프로세스 찾기
            try:
                result = subprocess.run(['lsof', '-ti', f':{port}'], 
                                      capture_output=True, text=True, timeout=5)
                if result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        if pid.strip():
                            try:
                                pid_int = int(pid.strip())
                                # Streamlit 메인 프로세스는 건드리지 않음
                                if pid_int != streamlit_pid:
                                    # 프로세스 정보 확인
                                    check_result = subprocess.run(['ps', '-p', str(pid_int), '-o', 'comm='], 
                                                                capture_output=True, text=True, timeout=2)
                                    process_name = check_result.stdout.strip()
                                    
                                    # uvicorn이나 FastAPI 관련 프로세스만 종료
                                    if any(keyword in process_name.lower() for keyword in ['uvicorn', 'fastapi', 'python']):
                                        # 먼저 SIGTERM으로 graceful shutdown 시도
                                        subprocess.run(['kill', '-15', str(pid_int)], capture_output=True, timeout=2)
                                        print(f"Sent SIGTERM to FastAPI process {pid_int} on port {port}")
                                        
                                        # 2초 후에도 살아있으면 SIGKILL
                                        import time
                                        time.sleep(2)
                                        subprocess.run(['kill', '-9', str(pid_int)], capture_output=True, timeout=1)
                                        print(f"Killed FastAPI process {pid_int} on port {port}")
                            except (ValueError, subprocess.TimeoutExpired):
                                pass
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass
                
        except Exception as e:
            print(f"Error stopping FastAPI server on port {port}: {e}")
        
        return False
    
    def _stop_server_gracefully(self, port):
        """uvicorn 서버 인스턴스를 사용한 graceful shutdown"""
        try:
            if port not in self.servers:
                return f"No server found on port {port}"
            
            server_info = self.servers[port]
            
            # 서버 인스턴스가 있으면 graceful shutdown
            if 'server_instance' in server_info and server_info['server_instance']:
                try:
                    server_instance = server_info['server_instance']
                    
                    # 비동기적으로 서버 중지 요청
                    import asyncio
                    import threading
                    
                    def shutdown_async():
                        try:
                            # 새로운 이벤트 루프에서 shutdown 실행
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            
                            # 서버 종료 신호
                            server_instance.should_exit = True
                            
                            # 잠시 대기
                            import time
                            time.sleep(1)
                            
                            loop.close()
                        except Exception as e:
                            print(f"Async shutdown error: {e}")
                    
                    # 별도 스레드에서 shutdown 실행
                    shutdown_thread = threading.Thread(target=shutdown_async)
                    shutdown_thread.start()
                    shutdown_thread.join(timeout=3)
                    
                    print(f"Graceful shutdown initiated for port {port}")
                    
                except Exception as e:
                    print(f"Graceful shutdown failed for port {port}: {e}")
            
            # 상태 업데이트
            server_info['running'] = False
            
            # 포트가 여전히 사용 중이면 강제 종료
            import time
            time.sleep(2)  # graceful shutdown 완료 대기
            
            if self._is_port_in_use(port):
                self._force_kill_port(port)
                return f"Server on port {port} force stopped"
            else:
                return f"Server on port {port} gracefully stopped"
                
        except Exception as e:
            return f"Failed to stop server on port {port}: {str(e)}"
    
    def _is_port_in_use(self, port):
        """포트가 사용 중인지 확인"""
        import socket
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                result = s.connect_ex(('localhost', port))
                return result == 0  # 연결 성공하면 포트 사용 중
        except:
            return False
    
    def _unified_inference(self, model: Any, tokenizer: Any, text: str, task: str) -> Any:
        """모든 태스크에 대한 통합 추론 엔진"""
        try:
            # 디바이스 일관성 보장
            model, tokenizer = device_manager.ensure_device_consistency(model, tokenizer)
            
            # 입력 토큰화
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            
            # 입력을 모델과 같은 디바이스로 이동
            inputs = device_manager.prepare_inputs(inputs, model)
            
            # 모델 디바이스 일관성 검증
            if not device_manager.validate_device_consistency(model):
                raise Exception("Model device consistency validation failed")
            
            with torch.no_grad():
                outputs = model(**inputs)
                
                if task == "text-classification":
                    return self._process_classification_output(outputs, model)
                elif task == "feature-extraction":
                    return self._process_embedding_output(outputs, inputs)
                elif task == "text-generation":
                    return self._process_generation_output(outputs, tokenizer)
                else:
                    # 기본 처리: 출력을 그대로 반환
                    if hasattr(outputs, 'last_hidden_state'):
                        return outputs.last_hidden_state.cpu().numpy().tolist()
                    elif hasattr(outputs, 'logits'):
                        return outputs.logits.cpu().numpy().tolist()
                    else:
                        return str(outputs)
                        
        except Exception as e:
            self.logger.error(f"통합 추론 실패: {e}")
            raise Exception(f"Unified inference failed: {str(e)}")
    
    def _process_classification_output(self, outputs: Any, model: Any) -> List[Dict[str, Any]]:
        """텍스트 분류 출력 처리"""
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
            
            return [{
                "label": predicted_label,
                "score": probs[0][predicted_class.item()].item()
            }]
        else:
            raise Exception(f"Classification model output does not contain logits: {outputs}")
    
    def _process_embedding_output(self, outputs: Any, inputs: Dict[str, torch.Tensor]) -> List[List[float]]:
        """임베딩 출력 처리"""
        if hasattr(outputs, 'last_hidden_state'):
            # 평균 풀링 사용
            attention_mask = inputs.get('attention_mask')
            if attention_mask is not None:
                # 마스크된 토큰 제외하고 평균
                masked_embeddings = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                pooled_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            else:
                # 단순 평균
                pooled_embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return pooled_embeddings.cpu().numpy().tolist()
        elif hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output.cpu().numpy().tolist()
        else:
            raise Exception(f"Embedding model output does not contain expected fields: {outputs}")
    
    def _process_generation_output(self, outputs: Any, tokenizer: Any) -> List[Dict[str, str]]:
        """텍스트 생성 출력 처리"""
        if hasattr(outputs, 'logits'):
            # 다음 토큰 예측
            next_token_id = torch.argmax(outputs.logits[0, -1, :]).item()
            next_token = tokenizer.decode([next_token_id])
            return [{"generated_text": next_token}]
        else:
            raise Exception(f"Generation model output does not contain logits: {outputs}")

# 편의 함수들
def create_api_server(model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """FastAPI 서버 생성"""
    return FastAPIServer(model_manager, host, port)

def start_background_server(model_manager: MultiModelManager, host: str = "127.0.0.1", port: int = 8000) -> FastAPIServer:
    """백그라운드에서 서버 시작"""
    server = create_api_server(model_manager, host, port)
    server.start_server()
    return server