"""
병렬 모델 로딩 시스템 - 멀티스레딩으로 극한 성능 달성
"""
import os
import time
import threading
import concurrent.futures
from typing import Dict, Any, Optional, Tuple, List
import torch
import logging
from fast_tensor_loader import fast_loader

class ParallelModelLoader:
    """병렬 처리로 모델 로딩 시간을 극단적으로 단축하는 로더"""
    
    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(8, os.cpu_count())
        self.logger = logging.getLogger("ParallelLoader")
        self.logger.setLevel(logging.INFO)
        
        # 성능 최적화를 위한 설정
        self._setup_parallel_environment()
    
    def _setup_parallel_environment(self):
        """병렬 처리 환경 최적화"""
        # CPU 코어 최대 활용
        cpu_count = os.cpu_count()
        
        # 환경 변수로만 설정 (PyTorch 스레드 설정은 이미 초기화됨)
        os.environ["OMP_NUM_THREADS"] = str(cpu_count)
        os.environ["MKL_NUM_THREADS"] = str(cpu_count)
        
        # PyTorch 스레드 설정은 안전하게 시도
        try:
            torch.set_num_threads(cpu_count)
            torch.set_num_interop_threads(cpu_count)
        except RuntimeError as e:
            self.logger.warning(f"[PARALLEL] PyTorch 스레드 설정 실패 (이미 초기화됨): {e}")
        
        self.logger.info(f"[PARALLEL] 최대 워커: {self.max_workers}, CPU 코어: {cpu_count}")
    
    def load_model_and_tokenizer_parallel(self, model_path: str, device: str = "cpu") -> Tuple[Any, Any, float]:
        """
        모델과 토크나이저를 병렬로 로딩
        
        Returns:
            (model, tokenizer, total_load_time)
        """
        start_time = time.time()
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                
                # 병렬 작업 제출
                self.logger.info("[PARALLEL] 병렬 로딩 시작...")
                
                # 1. 모델 로딩 작업
                model_future = executor.submit(self._load_model_task, model_path, device)
                
                # 2. 토크나이저 로딩 작업
                tokenizer_future = executor.submit(self._load_tokenizer_task, model_path)
                
                # 3. 설정 및 메타데이터 로딩 작업
                metadata_future = executor.submit(self._load_metadata_task, model_path)
                
                # 결과 수집
                self.logger.info("[PARALLEL] 작업 결과 수집 중...")
                
                # 모델 로딩 결과
                model, model_time = model_future.result()
                self.logger.info(f"[PARALLEL] 모델 로딩 완료: {model_time:.2f}초")
                
                # 토크나이저 로딩 결과
                tokenizer, tokenizer_time = tokenizer_future.result()
                self.logger.info(f"[PARALLEL] 토크나이저 로딩 완료: {tokenizer_time:.2f}초")
                
                # 메타데이터 결과
                metadata = metadata_future.result()
                self.logger.info("[PARALLEL] 메타데이터 로딩 완료")
                
                # 모델 최종 설정
                if model and metadata:
                    self._apply_metadata_to_model(model, metadata)
                
                total_time = time.time() - start_time
                self.logger.info(f"[PARALLEL] 전체 병렬 로딩 완료: {total_time:.2f}초")
                
                # 성능 통계 출력
                sequential_time = model_time + tokenizer_time
                speedup = sequential_time / total_time if total_time > 0 else 1.0
                self.logger.info(f"[PARALLEL] 병렬 처리 가속: {speedup:.1f}배")
                
                return model, tokenizer, total_time
                
        except Exception as e:
            self.logger.error(f"[PARALLEL] 병렬 로딩 실패: {e}")
            return None, None, 0.0
    
    def _load_model_task(self, model_path: str, device: str) -> Tuple[Any, float]:
        """모델 로딩 작업 (병렬 실행)"""
        thread_id = threading.current_thread().ident
        self.logger.info(f"[PARALLEL-{thread_id}] 모델 로딩 시작")
        
        try:
            # Fast tensor loader 사용
            model, load_time = fast_loader.load_model_ultra_fast(model_path, device)
            
            if model is None:
                # 폴백: 일반 로딩
                self.logger.warning(f"[PARALLEL-{thread_id}] 빠른 로딩 실패, 일반 로딩으로 전환")
                model, load_time = self._fallback_model_loading(model_path, device)
            
            return model, load_time
            
        except Exception as e:
            self.logger.error(f"[PARALLEL-{thread_id}] 모델 로딩 실패: {e}")
            return None, 0.0
    
    def _load_tokenizer_task(self, model_path: str) -> Tuple[Any, float]:
        """토크나이저 로딩 작업 (병렬 실행)"""
        thread_id = threading.current_thread().ident
        self.logger.info(f"[PARALLEL-{thread_id}] 토크나이저 로딩 시작")
        
        try:
            # Fast tokenizer loader 사용
            tokenizer, load_time = fast_loader.load_tokenizer_fast(model_path)
            return tokenizer, load_time
            
        except Exception as e:
            self.logger.error(f"[PARALLEL-{thread_id}] 토크나이저 로딩 실패: {e}")
            return None, 0.0
    
    def _load_metadata_task(self, model_path: str) -> Dict:
        """메타데이터 로딩 작업 (병렬 실행)"""
        thread_id = threading.current_thread().ident
        self.logger.info(f"[PARALLEL-{thread_id}] 메타데이터 로딩 시작")
        
        try:
            import json
            
            metadata = {}
            
            # config.json 읽기
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    metadata['config'] = json.load(f)
            
            # 다른 메타데이터 파일들 읽기
            for filename in ['tokenizer_config.json', 'special_tokens_map.json']:
                filepath = os.path.join(model_path, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, 'r') as f:
                            metadata[filename.replace('.json', '')] = json.load(f)
                    except:
                        pass
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"[PARALLEL-{thread_id}] 메타데이터 로딩 실패: {e}")
            return {}
    
    def _apply_metadata_to_model(self, model, metadata: Dict):
        """메타데이터를 모델에 적용"""
        try:
            if 'config' in metadata and hasattr(model, 'config'):
                # 추가 설정 적용
                config_dict = metadata['config']
                
                # 중요한 설정들 적용
                if hasattr(model.config, 'update'):
                    model.config.update(config_dict)
            
            self.logger.info("[PARALLEL] 메타데이터 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"[PARALLEL] 메타데이터 적용 실패: {e}")
    
    def _fallback_model_loading(self, model_path: str, device: str) -> Tuple[Any, float]:
        """폴백 모델 로딩 (최적화된 transformers 사용)"""
        start_time = time.time()
        
        try:
            from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
            
            # 설정 확인
            config = AutoConfig.from_pretrained(model_path)
            is_classification = (
                hasattr(config, 'architectures') and 
                config.architectures and
                any('Classification' in arch for arch in config.architectures)
            )
            
            # 최적화된 로딩 설정
            load_kwargs = {
                "trust_remote_code": True,
                "local_files_only": True,
                "torch_dtype": torch.float32,
                "low_cpu_mem_usage": True,
                "use_safetensors": True
            }
            
            # 모델 로딩
            if is_classification:
                model = AutoModelForSequenceClassification.from_pretrained(model_path, **load_kwargs)
            else:
                model = AutoModel.from_pretrained(model_path, **load_kwargs)
            
            model = model.to(device)
            model.eval()
            
            load_time = time.time() - start_time
            self.logger.info(f"[PARALLEL] 폴백 로딩 완료: {load_time:.2f}초")
            
            return model, load_time
            
        except Exception as e:
            self.logger.error(f"[PARALLEL] 폴백 로딩 실패: {e}")
            return None, 0.0
    
    def load_with_chunked_processing(self, model_path: str, device: str = "cpu", chunk_size: int = 10) -> Tuple[Any, Any, float]:
        """청크 단위 병렬 처리로 메모리 효율적인 로딩"""
        start_time = time.time()
        
        try:
            self.logger.info(f"[PARALLEL] 청크 크기 {chunk_size}로 병렬 로딩 시작")
            
            # 청크 단위로 처리할 작업들
            tasks = [
                ("config", lambda: self._load_metadata_task(model_path)),
                ("tensors", lambda: self._load_tensors_chunked(model_path, chunk_size)),
                ("tokenizer", lambda: self._load_tokenizer_task(model_path))
            ]
            
            results = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(tasks), self.max_workers)) as executor:
                future_to_task = {
                    executor.submit(task_func): task_name 
                    for task_name, task_func in tasks
                }
                
                for future in concurrent.futures.as_completed(future_to_task):
                    task_name = future_to_task[future]
                    try:
                        results[task_name] = future.result()
                        self.logger.info(f"[PARALLEL] {task_name} 완료")
                    except Exception as e:
                        self.logger.error(f"[PARALLEL] {task_name} 실패: {e}")
                        results[task_name] = None
            
            # 결과 조합
            model = self._assemble_model_from_chunks(results, device)
            tokenizer = results.get('tokenizer', (None, 0))[0]
            
            total_time = time.time() - start_time
            self.logger.info(f"[PARALLEL] 청크 처리 완료: {total_time:.2f}초")
            
            return model, tokenizer, total_time
            
        except Exception as e:
            self.logger.error(f"[PARALLEL] 청크 처리 실패: {e}")
            return None, None, 0.0
    
    def _load_tensors_chunked(self, model_path: str, chunk_size: int) -> Dict:
        """텐서를 청크 단위로 로딩"""
        try:
            from safetensors import safe_open
            
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(safetensors_path):
                return {}
            
            tensor_dict = {}
            
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                
                # 청크 단위로 처리
                for i in range(0, len(keys), chunk_size):
                    chunk_keys = keys[i:i + chunk_size]
                    
                    for key in chunk_keys:
                        tensor_dict[key] = f.get_tensor(key)
                    
                    self.logger.info(f"[PARALLEL] 텐서 청크 {i//chunk_size + 1} 로딩 완료")
            
            return tensor_dict
            
        except Exception as e:
            self.logger.error(f"[PARALLEL] 청크 텐서 로딩 실패: {e}")
            return {}
    
    def _assemble_model_from_chunks(self, results: Dict, device: str) -> Any:
        """청크 결과로부터 모델 조립"""
        try:
            config_data = results.get('config', {})
            tensor_data = results.get('tensors', {})
            
            if not config_data or not tensor_data:
                return None
            
            # 빠른 모델 조립
            model, _ = fast_loader.load_model_ultra_fast(model_path="", device=device)
            
            if model and tensor_data:
                # 텐서 할당
                fast_loader._assign_tensors_direct(model, tensor_data, device)
            
            return model
            
        except Exception as e:
            self.logger.error(f"[PARALLEL] 모델 조립 실패: {e}")
            return None

# 전역 인스턴스
parallel_loader = ParallelModelLoader()