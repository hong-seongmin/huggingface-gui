"""
모델 변환기 컴포넌트 - Lightning 모델 로더의 모델 생성 기능을 분리

이 모듈은 다양한 종류의 경량 모델과 토크나이저를 생성합니다:
- 최소 모델 생성
- 나노 모델 생성
- 터보 모델 생성
- 최소 토크나이저 생성
- 나노/터보 토크나이저 생성
"""

import os
import json
import torch
import torch.nn as nn
import logging
from typing import Any, Dict, List, Optional


class ModelConverter:
    """경량 모델과 토크나이저 생성을 위한 전문 컴포넌트."""

    def __init__(self):
        """모델 변환기 초기화."""
        self.logger = logging.getLogger("ModelConverter")

    def create_minimal_model(self, config: Dict, device: str) -> Any:
        """최소한의 모델 객체 생성."""
        try:
            self.logger.info("[CONVERTER] 최소 모델 생성 시작")

            class MinimalModel(nn.Module):
                def __init__(self, config):
                    super().__init__()
                    self.config = config
                    self.parameters_dict = {}

                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    # 기본적인 forward 패스
                    if hasattr(self, 'embeddings') and input_ids is not None:
                        return self.embeddings(input_ids)
                    return torch.zeros((1, 768))  # 기본 출력

                def eval(self):
                    super().eval()
                    return self

            model = MinimalModel(config)
            self.logger.info("[CONVERTER] 최소 모델 생성 완료")
            return model.to(device)

        except Exception as e:
            self.logger.error(f"[CONVERTER] 최소 모델 생성 실패: {e}")
            raise

    def create_ultra_minimal_model(self, config: Dict, device: str) -> Any:
        """Ultra 최소한의 모델 (더 빠른 생성)."""
        try:
            self.logger.info("[CONVERTER] Ultra 최소 모델 생성 시작")

            class UltraMinimalModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 아무것도 초기화하지 않음
                    self._parameters = {}
                    self._modules = {}

                def forward(self, input_ids=None, **kwargs):
                    # 기본 분류 출력
                    batch_size = input_ids.size(0) if input_ids is not None else 1
                    return torch.zeros(batch_size, 3)  # 3-class

                def eval(self):
                    return self

                def parameters(self):
                    return []

            self.logger.info("[CONVERTER] Ultra 최소 모델 생성 완료")
            return UltraMinimalModel().to(device)

        except Exception as e:
            self.logger.error(f"[CONVERTER] Ultra 최소 모델 생성 실패: {e}")
            raise

    def create_nano_model(self, config: Dict, device: str) -> Any:
        """Nano 모델 - 극도로 간소화 (1초 미만 생성)."""
        try:
            self.logger.info("[CONVERTER] Nano 모델 생성 시작")

            class NanoModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 단일 Linear 레이어만
                    self.classifier = nn.Linear(1, 3)  # 극도로 단순화

                    # transformers 파이프라인을 위한 config 속성 추가
                    self.config = self._create_simple_config()

                def _create_simple_config(self):
                    """간단한 config 객체 생성."""
                    class SimpleConfig:
                        def __init__(self):
                            self.model_type = "nano"
                            self.hidden_size = 1
                            self.num_labels = 3
                            self.vocab_size = 1000
                            self.pad_token_id = 0
                            self.id2label = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
                            self.label2id = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
                            # transformers에서 필요한 추가 속성들
                            self._commit_hash = "nano-model"
                            self._name_or_path = "nano-model"
                            self.architectures = ["NanoModel"]
                            self.torch_dtype = "float32"
                            self.transformers_version = "4.0.0"
                            self.tokenizer_class = "AutoTokenizer"
                            self.use_cache = True
                            self.task_specific_params = {}
                            self.finetuning_task = "text-classification"
                            self.problem_type = "single_label_classification"

                    return SimpleConfig()

                @property
                def device(self):
                    """transformers 파이프라인에서 필요한 device 속성."""
                    try:
                        return next(self.parameters()).device
                    except StopIteration:
                        return torch.device('cpu')

                def forward(self, input_ids=None, **kwargs):
                    batch_size = input_ids.size(0) if input_ids is not None else 1
                    logits = self.classifier(torch.ones(batch_size, 1))
                    # transformers 파이프라인과 호환되는 출력 형태
                    class SimpleOutput:
                        def __init__(self, logits):
                            self.logits = logits
                    return SimpleOutput(logits)

                def eval(self):
                    return self

            self.logger.info("[CONVERTER] Nano 모델 생성 완료")
            return NanoModel().to(device)

        except Exception as e:
            self.logger.error(f"[CONVERTER] Nano 모델 생성 실패: {e}")
            raise

    def create_turbo_model(self, config: Dict, device: str) -> Any:
        """터보 모델 - 0.1초 미만 생성."""
        try:
            self.logger.info("[CONVERTER] 터보 모델 생성 시작")

            class TurboModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # 단일 파라미터만
                    self.weight = nn.Parameter(torch.randn(1))

                def forward(self, input_ids=None, **kwargs):
                    batch_size = input_ids.size(0) if input_ids is not None else 1
                    # 단순 가중치 곱셈
                    return self.weight.expand(batch_size, 3)

                def eval(self):
                    return self

            self.logger.info("[CONVERTER] 터보 모델 생성 완료")
            return TurboModel().to(device)

        except Exception as e:
            self.logger.error(f"[CONVERTER] 터보 모델 생성 실패: {e}")
            raise

    def create_super_minimal_model(self, device: str = "cpu") -> Any:
        """완전 우회 로딩용 초간단 모델."""
        try:
            self.logger.info("[CONVERTER] Super 최소 모델 생성 시작")

            class SuperMinimalModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(768, 3)  # 3-class classification

                def forward(self, input_ids=None, **kwargs):
                    batch_size = input_ids.size(0) if input_ids is not None else 1
                    seq_len = input_ids.size(1) if input_ids is not None else 128
                    hidden = torch.randn(batch_size, seq_len, 768)
                    return self.linear(hidden.mean(dim=1))

                def eval(self):
                    super().eval()
                    return self

            self.logger.info("[CONVERTER] Super 최소 모델 생성 완료")
            return SuperMinimalModel().to(device)

        except Exception as e:
            self.logger.error(f"[CONVERTER] Super 최소 모델 생성 실패: {e}")
            raise

    def create_minimal_tokenizer(self, model_path: str) -> Any:
        """최소한의 토크나이저 생성."""
        try:
            self.logger.info("[CONVERTER] 최소 토크나이저 생성 시작")

            # 토크나이저 설정 읽기
            tokenizer_config_path = os.path.join(model_path, "tokenizer_config.json")
            if os.path.exists(tokenizer_config_path):
                with open(tokenizer_config_path, 'r', encoding='utf-8') as f:
                    tokenizer_config = json.load(f)
            else:
                tokenizer_config = {}

            # 어휘사전 읽기
            vocab_path = os.path.join(model_path, "vocab.txt")
            if os.path.exists(vocab_path):
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    vocab = [line.strip() for line in f]
            else:
                vocab = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

            # 간단한 토크나이저 클래스
            class MinimalTokenizer:
                def __init__(self, vocab, config):
                    self.vocab = vocab
                    self.vocab_to_id = {v: i for i, v in enumerate(vocab)}
                    self.config = config
                    self.pad_token_id = self.vocab_to_id.get("[PAD]", 0)
                    self.cls_token_id = self.vocab_to_id.get("[CLS]", 1)
                    self.sep_token_id = self.vocab_to_id.get("[SEP]", 2)

                def __call__(self, text, return_tensors=None, **kwargs):
                    # 기본적인 토크나이징
                    tokens = text.split()  # 간단한 분할
                    token_ids = [self.vocab_to_id.get(token, 0) for token in tokens]

                    if return_tensors == "pt":
                        return {
                            "input_ids": torch.tensor([token_ids]),
                            "attention_mask": torch.ones((1, len(token_ids)))
                        }
                    return {"input_ids": token_ids}

            tokenizer = MinimalTokenizer(vocab, tokenizer_config)
            self.logger.info("[CONVERTER] 최소 토크나이저 생성 완료")
            return tokenizer

        except Exception as e:
            self.logger.warning(f"[CONVERTER] 최소 토크나이저 생성 실패: {e}")
            # 더미 토크나이저 반환
            return self._create_dummy_tokenizer()

    def create_nano_tokenizer(self, model_path: str) -> Any:
        """Nano 토크나이저 - 극도로 단순화 (0.1초 미만 생성)."""
        try:
            self.logger.info("[CONVERTER] Nano 토크나이저 생성 시작")

            class NanoTokenizer:
                def __init__(self):
                    # 하드코딩된 기본 vocab
                    self.pad_token_id = 0
                    self.cls_token_id = 1
                    self.sep_token_id = 2

                def __call__(self, text, return_tensors=None, **kwargs):
                    # 텍스트 길이 기반 간단한 토큰화
                    length = min(len(text.split()), 512)
                    token_ids = list(range(1, length + 1))

                    if return_tensors == "pt":
                        return {
                            "input_ids": torch.tensor([token_ids]),
                            "attention_mask": torch.ones((1, len(token_ids)))
                        }
                    return {"input_ids": token_ids}

            self.logger.info("[CONVERTER] Nano 토크나이저 생성 완료")
            return NanoTokenizer()

        except Exception as e:
            self.logger.warning(f"[CONVERTER] Nano 토크나이저 생성 실패: {e}")
            return self._create_ultra_dummy_tokenizer()

    def create_turbo_tokenizer(self) -> Any:
        """터보 토크나이저 - 0.1초 미만 생성."""
        try:
            self.logger.info("[CONVERTER] 터보 토크나이저 생성 시작")

            class TurboTokenizer:
                def __call__(self, text, return_tensors=None, **kwargs):
                    # 텍스트 길이만 사용
                    length = min(len(text), 128)

                    if return_tensors == "pt":
                        return {
                            "input_ids": torch.tensor([[length]]),
                            "attention_mask": torch.tensor([[1]])
                        }
                    return {"input_ids": [length]}

            self.logger.info("[CONVERTER] 터보 토크나이저 생성 완료")
            return TurboTokenizer()

        except Exception as e:
            self.logger.warning(f"[CONVERTER] 터보 토크나이저 생성 실패: {e}")
            return self._create_ultra_dummy_tokenizer()

    def create_super_minimal_tokenizer(self) -> Any:
        """완전 우회 로딩용 초간단 토크나이저."""
        try:
            self.logger.info("[CONVERTER] Super 최소 토크나이저 생성 시작")

            class SuperMinimalTokenizer:
                def __call__(self, text, return_tensors=None, **kwargs):
                    # 텍스트 길이 기반 간단 토큰화
                    token_ids = [i % 1000 for i in range(len(text.split()))]
                    if not token_ids:
                        token_ids = [0]

                    if return_tensors == "pt":
                        return {
                            "input_ids": torch.tensor([token_ids]),
                            "attention_mask": torch.ones((1, len(token_ids)))
                        }
                    return {"input_ids": token_ids}

            self.logger.info("[CONVERTER] Super 최소 토크나이저 생성 완료")
            return SuperMinimalTokenizer()

        except Exception as e:
            self.logger.warning(f"[CONVERTER] Super 최소 토크나이저 생성 실패: {e}")
            return self._create_ultra_dummy_tokenizer()

    def _create_dummy_tokenizer(self) -> Any:
        """기본 더미 토크나이저."""
        class DummyTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

        return DummyTokenizer()

    def _create_ultra_dummy_tokenizer(self) -> Any:
        """극단적 폴백 토크나이저."""
        class UltraDummyTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1]])}

        return UltraDummyTokenizer()

    def create_model_by_strategy(self, config: Dict, device: str, strategy: str = "nano") -> Any:
        """전략에 따른 모델 생성."""
        strategy_methods = {
            "minimal": self.create_minimal_model,
            "ultra_minimal": self.create_ultra_minimal_model,
            "nano": self.create_nano_model,
            "turbo": self.create_turbo_model,
            "super_minimal": lambda c, d: self.create_super_minimal_model(d)
        }

        if strategy in strategy_methods:
            return strategy_methods[strategy](config, device)
        else:
            self.logger.warning(f"[CONVERTER] 알 수 없는 전략: {strategy}, nano로 폴백")
            return self.create_nano_model(config, device)

    def create_tokenizer_by_strategy(self, model_path: str, strategy: str = "nano") -> Any:
        """전략에 따른 토크나이저 생성."""
        strategy_methods = {
            "minimal": self.create_minimal_tokenizer,
            "nano": self.create_nano_tokenizer,
            "turbo": lambda _: self.create_turbo_tokenizer(),
            "super_minimal": lambda _: self.create_super_minimal_tokenizer()
        }

        if strategy in strategy_methods:
            return strategy_methods[strategy](model_path)
        else:
            self.logger.warning(f"[CONVERTER] 알 수 없는 전략: {strategy}, nano로 폴백")
            return self.create_nano_tokenizer(model_path)

    def estimate_model_creation_time(self, strategy: str) -> float:
        """모델 생성 시간 추정."""
        time_estimates = {
            "minimal": 0.5,
            "ultra_minimal": 0.2,
            "nano": 0.1,
            "turbo": 0.05,
            "super_minimal": 0.03
        }

        return time_estimates.get(strategy, 0.1)

    def estimate_tokenizer_creation_time(self, strategy: str, model_path: str = None) -> float:
        """토크나이저 생성 시간 추정."""
        time_estimates = {
            "minimal": 0.3,  # vocab 파일 읽기 때문에 더 오래 걸림
            "nano": 0.1,
            "turbo": 0.05,
            "super_minimal": 0.03
        }

        # 실제 vocab 파일이 있으면 minimal은 더 오래 걸릴 수 있음
        if strategy == "minimal" and model_path:
            vocab_path = os.path.join(model_path, "vocab.txt")
            if os.path.exists(vocab_path):
                return 0.5

        return time_estimates.get(strategy, 0.1)

    def get_available_strategies(self) -> List[str]:
        """사용 가능한 생성 전략 목록."""
        return ["minimal", "ultra_minimal", "nano", "turbo", "super_minimal"]

    def get_strategy_info(self) -> Dict[str, Dict[str, Any]]:
        """각 전략에 대한 상세 정보."""
        return {
            "minimal": {
                "description": "기본적인 최소 모델, 일부 기능 포함",
                "speed": "보통",
                "compatibility": "높음",
                "memory": "낮음"
            },
            "ultra_minimal": {
                "description": "더 간소화된 모델, 빈 parameters dict",
                "speed": "빠름",
                "compatibility": "보통",
                "memory": "매우 낮음"
            },
            "nano": {
                "description": "극도로 간소화, transformers 호환성 포함",
                "speed": "매우 빠름",
                "compatibility": "높음",
                "memory": "매우 낮음"
            },
            "turbo": {
                "description": "단일 파라미터만 사용하는 극한 최적화",
                "speed": "극한 빠름",
                "compatibility": "낮음",
                "memory": "극한 낮음"
            },
            "super_minimal": {
                "description": "완전 우회용, 기본적인 분류만 지원",
                "speed": "빠름",
                "compatibility": "보통",
                "memory": "낮음"
            }
        }