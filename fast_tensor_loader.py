"""
직접 텐서 로딩 엔진 - transformers 우회로 극한 성능 달성
"""
import os
import json
import time
import torch
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

class FastTensorLoader:
    """직접 safetensors를 로딩하여 모델을 재구성하는 극한 최적화 로더"""
    
    def __init__(self):
        self.logger = logging.getLogger("FastTensorLoader")
        self.logger.setLevel(logging.INFO)
    
    def load_model_ultra_fast(self, model_path: str, device: str = "cpu") -> Tuple[Any, float]:
        """
        Ultra-fast 모델 로딩 - transformers 완전 우회
        
        Returns:
            (model, load_time_seconds)
        """
        start_time = time.time()
        
        try:
            # 1. 설정 파일 로딩 (빠름)
            config_path = os.path.join(model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.logger.info(f"[FAST] 설정 로딩 완료: {time.time() - start_time:.2f}초")
            
            # 2. safetensors 직접 로딩 (극한 최적화)
            tensor_dict = self._load_safetensors_direct(model_path)
            self.logger.info(f"[FAST] 텐서 로딩 완료: {time.time() - start_time:.2f}초")
            
            # 3. 빈 모델 객체 생성 (가벼움)
            model = self._create_empty_model(config, device)
            self.logger.info(f"[FAST] 빈 모델 생성 완료: {time.time() - start_time:.2f}초")
            
            # 4. 텐서 직접 할당 (매우 빠름)
            self._assign_tensors_direct(model, tensor_dict, device)
            self.logger.info(f"[FAST] 텐서 할당 완료: {time.time() - start_time:.2f}초")
            
            # 5. 모델 최적화
            model.eval()
            if hasattr(model, 'config'):
                model.config = self._dict_to_config(config)
            
            load_time = time.time() - start_time
            self.logger.info(f"[FAST] 전체 로딩 완료: {load_time:.2f}초")
            
            return model, load_time
            
        except Exception as e:
            self.logger.error(f"[FAST] 빠른 로딩 실패: {e}")
            # 실패시 None 반환하여 폴백 로딩 사용
            return None, 0.0
    
    def _load_safetensors_direct(self, model_path: str) -> Dict[str, torch.Tensor]:
        """safetensors 파일을 직접 로딩"""
        try:
            from safetensors import safe_open
            
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if not os.path.exists(safetensors_path):
                raise FileNotFoundError(f"safetensors 파일 없음: {safetensors_path}")
            
            tensor_dict = {}
            
            # 직접 메모리 매핑으로 빠른 로딩
            with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensor_dict[key] = f.get_tensor(key)
            
            self.logger.info(f"[FAST] {len(tensor_dict)}개 텐서 로딩됨")
            return tensor_dict
            
        except ImportError:
            self.logger.error("[FAST] safetensors 라이브러리 없음")
            raise
        except Exception as e:
            self.logger.error(f"[FAST] safetensors 로딩 실패: {e}")
            raise
    
    def _create_empty_model(self, config: Dict, device: str):
        """설정 기반으로 빈 모델 객체 생성"""
        try:
            # 모델 타입에 따른 동적 import
            model_type = config.get("model_type", "")
            architectures = config.get("architectures", [])
            
            # Classification 모델 감지
            if any("Classification" in arch for arch in architectures):
                from transformers import AutoModelForSequenceClassification, AutoConfig
                
                # 안전한 config 객체 생성
                try:
                    config_obj = AutoConfig.for_model(model_type, **config)
                except:
                    # 폴백: 직접 config 클래스 생성
                    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                    if model_type in CONFIG_MAPPING:
                        config_class = CONFIG_MAPPING[model_type]
                        config_obj = config_class(**config)
                    else:
                        # 최후 폴백: 기본 AutoConfig
                        config_obj = AutoConfig(**config)
                
                # 빈 모델 생성 (가중치 초기화 없이)
                with torch.no_grad():
                    model = AutoModelForSequenceClassification.from_config(
                        config_obj,
                        torch_dtype=torch.float32
                    )
                
                self.logger.info("[FAST] Classification 모델 생성됨")
                
            else:
                from transformers import AutoModel, AutoConfig
                
                # 안전한 config 객체 생성
                try:
                    config_obj = AutoConfig.for_model(model_type, **config)
                except:
                    # 폴백: 직접 config 클래스 생성
                    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
                    if model_type in CONFIG_MAPPING:
                        config_class = CONFIG_MAPPING[model_type]
                        config_obj = config_class(**config)
                    else:
                        # 최후 폴백: 기본 AutoConfig
                        config_obj = AutoConfig(**config)
                
                with torch.no_grad():
                    model = AutoModel.from_config(
                        config_obj,
                        torch_dtype=torch.float32
                    )
                
                self.logger.info("[FAST] 일반 모델 생성됨")
            
            return model.to(device)
            
        except Exception as e:
            self.logger.error(f"[FAST] 모델 생성 실패: {e}")
            raise
    
    def _assign_tensors_direct(self, model, tensor_dict: Dict[str, torch.Tensor], device: str):
        """텐서를 모델에 직접 할당"""
        try:
            assigned_count = 0
            
            # 모델의 state_dict와 매칭하여 할당
            model_state_dict = model.state_dict()
            
            with torch.no_grad():
                for param_name, param_tensor in model_state_dict.items():
                    if param_name in tensor_dict:
                        loaded_tensor = tensor_dict[param_name].to(device)
                        
                        # 직접 텐서 데이터 복사
                        param_tensor.data.copy_(loaded_tensor)
                        assigned_count += 1
                    else:
                        self.logger.warning(f"[FAST] 텐서 누락: {param_name}")
            
            self.logger.info(f"[FAST] {assigned_count}/{len(model_state_dict)} 텐서 할당됨")
            
            if assigned_count == 0:
                raise ValueError("할당된 텐서가 없음")
                
        except Exception as e:
            self.logger.error(f"[FAST] 텐서 할당 실패: {e}")
            raise
    
    def _dict_to_config(self, config_dict: Dict) -> Any:
        """딕셔너리를 Config 객체로 변환"""
        try:
            from transformers import AutoConfig
            # from_dict 대신 from_pretrained_dict 사용
            if hasattr(AutoConfig, 'from_pretrained_dict'):
                return AutoConfig.from_pretrained_dict(config_dict)
            else:
                # 임시 객체 생성 후 속성 설정
                config = AutoConfig()
                for key, value in config_dict.items():
                    setattr(config, key, value)
                return config
        except Exception as e:
            self.logger.warning(f"[FAST] Config 변환 실패: {e}")
            return None
    
    def load_tokenizer_fast(self, model_path: str) -> Tuple[Any, float]:
        """토크나이저 빠른 로딩"""
        start_time = time.time()
        
        try:
            from transformers import AutoTokenizer
            
            # 최적화된 토크나이저 로딩
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                use_fast=True,
                local_files_only=True,
                trust_remote_code=True
            )
            
            load_time = time.time() - start_time
            self.logger.info(f"[FAST] 토크나이저 로딩: {load_time:.2f}초")
            
            return tokenizer, load_time
            
        except Exception as e:
            self.logger.error(f"[FAST] 토크나이저 로딩 실패: {e}")
            return None, 0.0
    
    def validate_model(self, model, tokenizer) -> bool:
        """로딩된 모델 검증"""
        try:
            # 간단한 추론 테스트
            if tokenizer is None:
                return False
                
            test_text = "Hello world"
            inputs = tokenizer(test_text, return_tensors="pt")
            
            with torch.no_grad():
                if hasattr(model, 'forward'):
                    outputs = model(**inputs)
                    self.logger.info("[FAST] 모델 검증 성공")
                    return True
                else:
                    self.logger.warning("[FAST] forward 메서드 없음")
                    return False
                    
        except Exception as e:
            self.logger.error(f"[FAST] 모델 검증 실패: {e}")
            return False

# 전역 인스턴스
fast_loader = FastTensorLoader()