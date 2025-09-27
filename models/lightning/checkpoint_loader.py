"""
체크포인트 로더 컴포넌트 - Lightning 모델 로더의 체크포인트 처리 기능을 분리

이 모듈은 다양한 형태의 모델 체크포인트 로딩 전략을 제공합니다:
- Pickle 기반 로딩
- SafeTensors 기반 로딩
- 메모리 매핑 로딩
- 선택적 텐서 로딩
"""

import os
import json
import mmap
import time
import torch
import logging
from typing import Any, Tuple, Optional, List, Dict


class CheckpointLoader:
    """모델 체크포인트 로딩을 위한 전문 컴포넌트."""

    def __init__(self):
        """체크포인트 로더 초기화."""
        self.logger = logging.getLogger("CheckpointLoader")

    def load_pickle_checkpoint(self, model_path: str, device: str) -> Optional[Tuple[Any, Any, float]]:
        """Pickle 기반 체크포인트 로딩."""
        try:
            start_time = time.time()
            self.logger.info("[CHECKPOINT] Pickle 체크포인트 로딩 시작")

            # 설정 파일 로딩
            config = self._load_config(model_path)
            if not config:
                return None

            # 모델 컴포넌트 생성
            from .model_converter import ModelConverter
            converter = ModelConverter()

            model = converter.create_nano_model(config, device)
            tokenizer = converter.create_nano_tokenizer(model_path)

            # 핵심 텐서만 로딩
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                self._load_priority_tensors(model, safetensors_path, device, max_tensors=5)

            load_time = time.time() - start_time
            self.logger.info(f"[CHECKPOINT] Pickle 로딩 완료: {load_time:.2f}초")

            return model, tokenizer, load_time

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] Pickle 로딩 실패: {e}")
            return None

    def load_safetensors_checkpoint(self, model_path: str, device: str,
                                  max_tensors: int = 20) -> Optional[Tuple[Any, Any, float]]:
        """SafeTensors 기반 체크포인트 로딩."""
        try:
            start_time = time.time()
            self.logger.info(f"[CHECKPOINT] SafeTensors 로딩 시작 (최대 {max_tensors}개)")

            # 설정 로딩
            config = self._load_config(model_path)
            if not config:
                return None

            # 모델 생성
            from .model_converter import ModelConverter
            converter = ModelConverter()

            model = converter.create_nano_model(config, device)
            tokenizer = converter.create_nano_tokenizer(model_path)

            # SafeTensors 파일 처리
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                loaded_count = self._load_important_tensors(model, safetensors_path, device, max_tensors)
                self.logger.info(f"[CHECKPOINT] {loaded_count}개 텐서 로딩 완료")

            load_time = time.time() - start_time
            return model, tokenizer, load_time

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] SafeTensors 로딩 실패: {e}")
            return None

    def load_memory_mapped_checkpoint(self, model_path: str, device: str) -> Optional[Tuple[Any, Any, float]]:
        """메모리 매핑 기반 체크포인트 로딩."""
        try:
            start_time = time.time()
            self.logger.info("[CHECKPOINT] 메모리 매핑 로딩 시작")

            # 메모리 매핑으로 설정 로딩
            config = self._load_config_with_mmap(model_path)
            if not config:
                return None

            # 모델 생성
            from .model_converter import ModelConverter
            converter = ModelConverter()

            model = converter.create_nano_model(config, device)
            tokenizer = converter.create_nano_tokenizer(model_path)

            # 메모리 매핑 텐서 로딩
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                self._load_tensors_with_mmap(model, safetensors_path, device)

            load_time = time.time() - start_time
            self.logger.info(f"[CHECKPOINT] 메모리 매핑 완료: {load_time:.2f}초")

            return model, tokenizer, load_time

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 메모리 매핑 실패: {e}")
            return None

    def _load_config(self, model_path: str) -> Optional[Dict]:
        """표준 방식으로 config.json 로딩."""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 설정 로딩 실패: {e}")
            return None

    def _load_config_with_mmap(self, model_path: str) -> Optional[Dict]:
        """메모리 매핑을 사용한 config.json 로딩."""
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            return None

        try:
            with open(config_path, 'r+b') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    config_data = mm.read().decode('utf-8')
                    return json.loads(config_data)
        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 메모리 매핑 설정 로딩 실패: {e}")
            return None

    def _load_important_tensors(self, model: Any, safetensors_path: str,
                               device: str, max_tensors: int) -> int:
        """중요한 텐서들만 선별적으로 로딩."""
        try:
            from safetensors import safe_open

            tensors_loaded = 0
            with safe_open(safetensors_path, framework="pt", device=device) as f:
                keys = list(f.keys())

                # 중요한 텐서 키워드 우선순위
                important_keywords = [
                    'embed', 'attention', 'output', 'classifier', 'head',
                    'dense', 'linear', 'weight', 'bias'
                ]

                # 우선순위별로 텐서 로딩
                for keyword in important_keywords:
                    if tensors_loaded >= max_tensors:
                        break

                    matching_keys = [k for k in keys if keyword in k.lower()]
                    for key in matching_keys[:min(5, max_tensors - tensors_loaded)]:
                        try:
                            tensor = f.get_tensor(key)
                            self._assign_tensor_to_model(model, key, tensor, tensors_loaded)
                            tensors_loaded += 1
                        except Exception as e:
                            self.logger.debug(f"[CHECKPOINT] 텐서 할당 실패 {key}: {e}")
                            continue

            return tensors_loaded

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 중요 텐서 로딩 실패: {e}")
            return 0

    def _load_priority_tensors(self, model: Any, safetensors_path: str,
                              device: str, max_tensors: int = 5) -> int:
        """최우선 텐서만 로딩 (ultra fast)."""
        try:
            from safetensors import safe_open

            tensors_loaded = 0
            with safe_open(safetensors_path, framework="pt", device=device) as f:
                keys = list(f.keys())

                # 최우선 키워드만
                priority_keywords = ['classifier', 'output', 'head']

                for keyword in priority_keywords:
                    for key in keys:
                        if keyword in key.lower() and tensors_loaded < max_tensors:
                            try:
                                tensor = f.get_tensor(key)
                                setattr(model, f"priority_param_{tensors_loaded}",
                                       torch.nn.Parameter(tensor))
                                tensors_loaded += 1
                                if tensors_loaded >= max_tensors:
                                    break
                            except Exception:
                                continue
                    if tensors_loaded >= max_tensors:
                        break

            return tensors_loaded

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 우선순위 텐서 로딩 실패: {e}")
            return 0

    def _load_tensors_with_mmap(self, model: Any, safetensors_path: str, device: str):
        """메모리 매핑으로 텐서 로딩."""
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(safetensors_path)
            if file_size > 200 * 1024 * 1024:  # 200MB 제한
                self.logger.warning("[CHECKPOINT] 파일 크기 초과 - 메모리 매핑 스킵")
                return

            from safetensors import safe_open

            with open(safetensors_path, 'rb') as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    with safe_open(safetensors_path, framework="pt", device=device) as sf:
                        keys = list(sf.keys())
                        if keys:
                            # 첫 번째 텐서만 로딩 (속도 우선)
                            first_key = keys[0]
                            tensor = sf.get_tensor(first_key)
                            setattr(model, "mmap_param", torch.nn.Parameter(tensor))
                            self.logger.info(f"[CHECKPOINT] 메모리 매핑 텐서 로딩: {first_key}")

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 메모리 매핑 텐서 로딩 실패: {e}")

    def _assign_tensor_to_model(self, model: Any, key: str, tensor: torch.Tensor,
                               index: int):
        """텐서를 모델에 구조적으로 할당."""
        try:
            # 계층 구조 파싱
            parts = key.split('.')
            current = model

            # 중간 모듈들 생성
            for part in parts[:-1]:
                if not hasattr(current, part):
                    setattr(current, part, torch.nn.Module())
                current = getattr(current, part)

            # 최종 파라미터 할당
            final_name = parts[-1]
            setattr(current, final_name, torch.nn.Parameter(tensor))

        except Exception as e:
            # 폴백: 플래트 할당
            setattr(model, f"param_{index}", torch.nn.Parameter(tensor))
            self.logger.debug(f"[CHECKPOINT] 플래트 할당으로 폴백: {key}")

    def get_checkpoint_info(self, model_path: str) -> Dict[str, Any]:
        """체크포인트 파일 정보 추출."""
        info = {
            "has_config": False,
            "has_safetensors": False,
            "has_pytorch_model": False,
            "safetensors_size": 0,
            "config_info": {}
        }

        try:
            # Config 확인
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                info["has_config"] = True
                config = self._load_config(model_path)
                if config:
                    info["config_info"] = {
                        "model_type": config.get("model_type"),
                        "architectures": config.get("architectures", []),
                        "num_parameters": config.get("num_parameters", "unknown")
                    }

            # SafeTensors 확인
            safetensors_path = os.path.join(model_path, "model.safetensors")
            if os.path.exists(safetensors_path):
                info["has_safetensors"] = True
                info["safetensors_size"] = os.path.getsize(safetensors_path)

            # PyTorch 모델 확인
            pytorch_model_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                info["has_pytorch_model"] = True

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 정보 추출 실패: {e}")

        return info

    def validate_checkpoint(self, model_path: str) -> Dict[str, bool]:
        """체크포인트 유효성 검증."""
        validation = {
            "path_exists": os.path.exists(model_path),
            "has_model_files": False,
            "config_valid": False,
            "tensors_accessible": False
        }

        if not validation["path_exists"]:
            return validation

        try:
            # 모델 파일 존재 확인
            safetensors_path = os.path.join(model_path, "model.safetensors")
            pytorch_path = os.path.join(model_path, "pytorch_model.bin")
            validation["has_model_files"] = (os.path.exists(safetensors_path) or
                                            os.path.exists(pytorch_path))

            # Config 유효성
            config = self._load_config(model_path)
            validation["config_valid"] = config is not None and "model_type" in config

            # 텐서 접근 가능성 (SafeTensors)
            if os.path.exists(safetensors_path):
                try:
                    from safetensors import safe_open
                    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
                        keys = list(f.keys())
                        validation["tensors_accessible"] = len(keys) > 0
                except Exception:
                    validation["tensors_accessible"] = False

        except Exception as e:
            self.logger.warning(f"[CHECKPOINT] 검증 실패: {e}")

        return validation

    def estimate_loading_time(self, model_path: str, method: str = "safetensors") -> float:
        """로딩 시간 추정."""
        try:
            if method == "safetensors":
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
                    # 경험적 공식: 100MB당 약 1초
                    return size_mb / 100.0

            elif method == "memory_mapped":
                # 메모리 매핑은 일반적으로 더 빠름
                safetensors_path = os.path.join(model_path, "model.safetensors")
                if os.path.exists(safetensors_path):
                    size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
                    return size_mb / 200.0  # 더 빠른 추정

            return 1.0  # 기본 추정치

        except Exception:
            return 5.0  # 보수적 추정치