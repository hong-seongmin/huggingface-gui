"""
Main code snippet generators for different model types and tasks.
"""
from typing import Dict, List, Optional, Any
from .snippets import CodeSnippet, SnippetType, SnippetTemplate
from .optimization import InferenceOptimizer
from .datasets import DatasetBuilder, get_sample_data_for_task
from core.logging_config import get_logger

logger = get_logger(__name__)


class LoadingSnippetTemplate(SnippetTemplate):
    """Template for model loading snippets."""

    def generate(self) -> CodeSnippet:
        """Generate model loading snippet."""
        imports = self._get_base_imports() + self._get_task_specific_imports()

        code = f"""
{chr(10).join(imports)}

# Model loading
model_id = {self._format_model_id()}
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = {self.model_class}.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)

print(f"Model loaded successfully: {{model_id}}")
print(f"Model device: {{next(model.parameters()).device}}")
"""

        return CodeSnippet(
            title=f"{self.model_class} 모델 로딩",
            description=f"{self.task_type} 태스크를 위한 {self.model_class} 모델 로딩",
            code=code.strip(),
            language="python",
            snippet_type=SnippetType.LOADING,
            requirements=["transformers", "torch"],
            notes=[
                "모델이 클 경우 충분한 메모리가 필요합니다",
                "GPU가 있는 경우 자동으로 GPU에 로드됩니다",
                "trust_remote_code=True는 커스텀 모델에만 사용하세요"
            ]
        )


class InferenceSnippetTemplate(SnippetTemplate):
    """Template for model inference snippets."""

    def generate(self) -> CodeSnippet:
        """Generate model inference snippet."""
        sample_text = self._get_sample_text()
        processing_code = self._get_task_specific_processing()

        code = f"""
# 기본 추론 예제
sample_text = "{sample_text}"

# 토큰화
inputs = tokenizer(
    sample_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

# GPU로 이동 (사용 가능한 경우)
if torch.cuda.is_available() and next(model.parameters()).is_cuda:
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}

# 추론 실행
with torch.no_grad():
    outputs = model(**inputs)

{processing_code}
"""

        return CodeSnippet(
            title=f"{self.task_type.title()} 추론",
            description=f"{self.task_type} 태스크를 위한 기본 추론 예제",
            code=code.strip(),
            language="python",
            snippet_type=SnippetType.INFERENCE,
            requirements=["transformers", "torch"],
            notes=[
                "torch.no_grad()로 그래디언트 계산을 비활성화합니다",
                "배치 처리를 위해 여러 텍스트를 함께 처리할 수 있습니다"
            ]
        )

    def _get_sample_text(self) -> str:
        """Get sample text for the task type."""
        samples = {
            'text-classification': "이 제품은 정말 훌륭합니다!",
            'token-classification': "김철수는 서울에서 일합니다.",
            'question-answering': "파리는 프랑스의 수도입니다.",
            'text-generation': "인공지능의 미래는",
            'summarization': "긴 문서의 내용을 요약해주세요...",
            'translation': "Hello, how are you today?",
            'feature-extraction': "이 문장의 벡터 표현을 추출합니다."
        }
        return samples.get(self.task_type, "샘플 텍스트")

    def _get_task_specific_processing(self) -> str:
        """Get task-specific output processing code."""
        processing = {
            'text-classification': """
# 분류 결과 처리
logits = outputs.logits
probabilities = torch.nn.functional.softmax(logits, dim=-1)
predicted_class = torch.argmax(probabilities, dim=-1)

print(f"예측 클래스: {predicted_class.item()}")
print(f"확률 분포: {probabilities.squeeze().tolist()}")
""",
            'token-classification': """
# 토큰 분류 결과 처리
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)

# 토큰별 예측 결과
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
for token, pred in zip(tokens, predictions[0]):
    print(f"{token}: {pred.item()}")
""",
            'feature-extraction': """
# 특성 벡터 추출
embeddings = outputs.last_hidden_state
pooled_output = embeddings.mean(dim=1)  # 평균 풀링

print(f"임베딩 차원: {embeddings.shape}")
print(f"풀링된 벡터 크기: {pooled_output.shape}")
"""
        }
        return processing.get(self.task_type, "# 결과 출력\nprint(outputs)")


class OptimizationSnippetTemplate(SnippetTemplate):
    """Template for optimization snippets."""

    def generate(self) -> CodeSnippet:
        """Generate optimization snippet."""
        code = """
import torch
from torch.cuda.amp import autocast
import time

# 메모리 최적화 설정
def optimize_memory():
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.cuda.empty_cache()

    print("메모리 최적화 설정 완료")

optimize_memory()

# 최적화된 추론 함수
@torch.no_grad()
def optimized_inference(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 자동 혼합 정밀도 사용
    if torch.cuda.is_available():
        with autocast():
            outputs = model(**inputs)
    else:
        outputs = model(**inputs)

    return outputs

# 성능 벤치마크
def benchmark_inference(text, model, tokenizer, num_runs=100):
    times = []

    for _ in range(num_runs):
        start_time = time.time()
        _ = optimized_inference(text, model, tokenizer)
        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    print(f"평균 추론 시간: {avg_time*1000:.2f}ms")
    print(f"초당 처리량: {1/avg_time:.1f} samples/sec")

    return avg_time

# 사용 예제
sample_text = "최적화 테스트를 위한 샘플 텍스트입니다."
avg_time = benchmark_inference(sample_text, model, tokenizer)
"""

        return CodeSnippet(
            title="성능 최적화",
            description="모델 추론 성능 최적화 및 벤치마킹",
            code=code.strip(),
            language="python",
            snippet_type=SnippetType.OPTIMIZATION,
            requirements=["transformers", "torch"],
            notes=[
                "autocast는 GPU에서만 효과적입니다",
                "메모리 최적화는 큰 모델에서 특히 중요합니다",
                "벤치마크는 워밍업 없이 진행되므로 실제 성능과 다를 수 있습니다"
            ]
        )


class FineTuningSnippetTemplate(SnippetTemplate):
    """Template for fine-tuning snippets."""

    def __init__(self, model_id: str, task_type: str = None, model_class: str = None):
        super().__init__(model_id, task_type, model_class)
        self.dataset_builder = None

    def generate(self) -> CodeSnippet:
        """Generate fine-tuning snippet."""
        sample_data_code = self._get_sample_data_code()

        code = f"""
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

{sample_data_code}

# 훈련 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='steps',
    eval_steps=500,
    save_strategy='steps',
    save_steps=1000,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

    return {{
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }}

# 트레이너 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 훈련 시작
print("파인튜닝 시작...")
trainer.train()

# 모델 저장
trainer.save_model('./fine_tuned_model')
print("파인튜닝 완료 및 모델 저장됨")
"""

        return CodeSnippet(
            title="모델 파인튜닝",
            description=f"{self.task_type} 태스크를 위한 모델 파인튜닝",
            code=code.strip(),
            language="python",
            snippet_type=SnippetType.FINE_TUNING,
            requirements=["transformers", "torch", "scikit-learn"],
            notes=[
                "충분한 GPU 메모리가 필요합니다",
                "배치 크기는 메모리에 맞게 조정하세요",
                "검증 데이터로 오버피팅을 모니터링하세요"
            ]
        )

    def _get_sample_data_code(self) -> str:
        """Get sample dataset creation code."""
        return """
# 커스텀 데이터셋 클래스
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 샘플 데이터 (실제 데이터로 교체하세요)
train_texts = [
    "이 제품은 훌륭합니다.",
    "서비스가 별로입니다.",
    "가격이 적당합니다.",
    "품질이 좋습니다."
]
train_labels = [1, 0, 1, 1]  # 1: 긍정, 0: 부정

val_texts = ["만족스러운 구매였습니다.", "다시는 사지 않겠습니다."]
val_labels = [1, 0]

# 데이터셋 생성
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)
"""


class CodeSnippetGenerator:
    """Main code snippet generator."""

    def __init__(self):
        """Initialize code snippet generator."""
        from models.model_type_detector import ModelTypeDetector
        self.model_detector = ModelTypeDetector()

    def generate_snippets(self, model_name: str, model_path: str, model_id: str = None) -> Dict[str, CodeSnippet]:
        """
        Generate all types of snippets for a model.

        Args:
            model_name: Name of the model
            model_path: Path to the model
            model_id: HuggingFace model ID (optional)

        Returns:
            Dictionary of snippets by type
        """
        try:
            # Detect model type
            task_type, model_class, analysis_info = self.model_detector.detect_model_type(model_name, model_path)

            # Determine model ID
            if not model_id:
                model_id = model_name if self._is_huggingface_model_id(model_name) else model_path

            snippets = {}

            # Generate different types of snippets
            templates = {
                "loading": LoadingSnippetTemplate(model_id, task_type, model_class),
                "inference": InferenceSnippetTemplate(model_id, task_type, model_class),
                "optimization": OptimizationSnippetTemplate(model_id, task_type, model_class),
                "fine_tuning": FineTuningSnippetTemplate(model_id, task_type, model_class)
            }

            for snippet_type, template in templates.items():
                try:
                    snippets[snippet_type] = template.generate()
                except Exception as e:
                    logger.error(f"Failed to generate {snippet_type} snippet: {e}")

            return snippets

        except Exception as e:
            logger.error(f"Snippet generation failed: {e}")
            return {}

    def _is_huggingface_model_id(self, name: str) -> bool:
        """Check if name is a HuggingFace model ID."""
        return '/' in name and not name.startswith('/') and '\\' not in name

    def export_snippets(self, snippets: Dict[str, CodeSnippet], format: str = "python") -> str:
        """
        Export snippets to a single file.

        Args:
            snippets: Dictionary of snippets
            format: Export format ("python", "jupyter", "markdown")

        Returns:
            Formatted export string
        """
        if format == "python":
            return self._export_as_python(snippets)
        elif format == "jupyter":
            return self._export_as_jupyter(snippets)
        elif format == "markdown":
            return self._export_as_markdown(snippets)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_as_python(self, snippets: Dict[str, CodeSnippet]) -> str:
        """Export as Python file."""
        lines = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            '"""',
            "Generated code snippets for HuggingFace model",
            '"""',
            ""
        ]

        for name, snippet in snippets.items():
            lines.extend([
                f"# {snippet.title}",
                f"# {snippet.description}",
                "",
                snippet.code,
                "",
                ""
            ])

        return "\n".join(lines)

    def _export_as_markdown(self, snippets: Dict[str, CodeSnippet]) -> str:
        """Export as Markdown file."""
        lines = ["# HuggingFace Model Code Snippets", ""]

        for name, snippet in snippets.items():
            lines.extend([
                f"## {snippet.title}",
                "",
                snippet.description,
                "",
                f"```{snippet.language}",
                snippet.code,
                "```",
                "",
                "**Requirements:**",
                "",
                "```bash",
                "\n".join(f"pip install {req}" for req in snippet.requirements),
                "```",
                "",
                "**Notes:**",
                "",
                "\n".join(f"- {note}" for note in snippet.notes),
                "",
                "---",
                ""
            ])

        return "\n".join(lines)

    def _export_as_jupyter(self, snippets: Dict[str, CodeSnippet]) -> str:
        """Export as Jupyter notebook JSON."""
        import json

        cells = []

        # Title cell
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": ["# HuggingFace Model Code Snippets\n"]
        })

        # Snippet cells
        for name, snippet in snippets.items():
            # Markdown cell for description
            cells.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {snippet.title}\n\n{snippet.description}\n"]
            })

            # Code cell
            cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [snippet.code.split('\n')]
            })

        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        return json.dumps(notebook, indent=2)


# Backward compatibility
def extract_features(texts, model, tokenizer):
    """Extract features from texts (backward compatibility function)."""
    from .datasets import extract_features as _extract_features
    return _extract_features(texts, model, tokenizer)