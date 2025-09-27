"""
Dataset utilities for model training and fine-tuning.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Union
import numpy as np
from core.logging_config import get_logger

logger = get_logger(__name__)


class TextDataset(Dataset):
    """Dataset for text data with tokenization."""

    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        """
        Initialize text dataset.

        Args:
            texts: List of text strings
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

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
            'text': text
        }


class CustomClassificationDataset(Dataset):
    """Dataset for text classification tasks."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize classification dataset.

        Args:
            texts: List of text strings
            labels: List of label integers
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

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


class CustomTokenClassificationDataset(Dataset):
    """Dataset for token classification tasks (NER, POS tagging, etc.)."""

    def __init__(self, texts: List[str], labels: List[List[int]], tokenizer, max_length: int = 512):
        """
        Initialize token classification dataset.

        Args:
            texts: List of text strings
            labels: List of label sequences (one per text)
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        if len(texts) != len(labels):
            raise ValueError("Number of texts and labels must match")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            is_split_into_words=False
        )

        # Adjust labels to match tokenized length
        tokenized_labels = [-100] * self.max_length  # -100 is ignored in loss computation
        for i, label in enumerate(labels[:self.max_length]):
            if i < len(tokenized_labels):
                tokenized_labels[i] = label

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(tokenized_labels, dtype=torch.long)
        }


class ImageTextDataset(Dataset):
    """Dataset for image-text tasks."""

    def __init__(self, image_paths: List[str], texts: List[str], processor, transform=None):
        """
        Initialize image-text dataset.

        Args:
            image_paths: List of image file paths
            texts: List of text strings
            processor: Image processor
            transform: Optional image transforms
        """
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.transform = transform

        if len(image_paths) != len(texts):
            raise ValueError("Number of images and texts must match")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            from PIL import Image

            image_path = self.image_paths[idx]
            text = self.texts[idx]

            # Load image
            image = Image.open(image_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            # Process with processor
            inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)

            return {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten(),
                'pixel_values': inputs['pixel_values'].squeeze(),
                'text': text,
                'image_path': image_path
            }

        except Exception as e:
            logger.error(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return dummy data on error
            return {
                'input_ids': torch.zeros(512, dtype=torch.long),
                'attention_mask': torch.zeros(512, dtype=torch.long),
                'pixel_values': torch.zeros(3, 224, 224),
                'text': text,
                'image_path': image_path
            }


class AudioDataset(Dataset):
    """Dataset for audio processing tasks."""

    def __init__(self, audio_paths: List[str], texts: Optional[List[str]] = None,
                 processor=None, sample_rate: int = 16000):
        """
        Initialize audio dataset.

        Args:
            audio_paths: List of audio file paths
            texts: Optional list of transcription texts
            processor: Audio processor
            sample_rate: Target sample rate
        """
        self.audio_paths = audio_paths
        self.texts = texts or [None] * len(audio_paths)
        self.processor = processor
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        try:
            import librosa

            audio_path = self.audio_paths[idx]
            text = self.texts[idx]

            # Load audio
            audio, _ = librosa.load(audio_path, sr=self.sample_rate)

            if self.processor:
                inputs = self.processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
                return {
                    'input_values': inputs['input_values'].squeeze(),
                    'text': text,
                    'audio_path': audio_path
                }
            else:
                return {
                    'audio': torch.tensor(audio, dtype=torch.float32),
                    'text': text,
                    'audio_path': audio_path
                }

        except Exception as e:
            logger.error(f"Error loading audio {self.audio_paths[idx]}: {e}")
            # Return dummy data on error
            return {
                'input_values': torch.zeros(16000),  # 1 second of silence
                'text': text,
                'audio_path': audio_path
            }


def create_dataloader(dataset: Dataset, batch_size: int = 8, shuffle: bool = True,
                     num_workers: int = 0) -> DataLoader:
    """
    Create a DataLoader from a dataset.

    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def extract_features(texts: List[str], model, tokenizer, batch_size: int = 8) -> np.ndarray:
    """
    Extract features from texts using a model.

    Args:
        texts: List of input texts
        model: Model for feature extraction
        tokenizer: Tokenizer for the model
        batch_size: Batch size for processing

    Returns:
        Feature matrix as numpy array
    """
    try:
        dataset = TextDataset(texts, tokenizer)
        dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=False)

        features = []
        model.eval()

        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }

                # Move to device if model is on GPU
                if next(model.parameters()).is_cuda:
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                outputs = model(**inputs)

                # Extract features (usually from last hidden state)
                if hasattr(outputs, 'last_hidden_state'):
                    # Use [CLS] token or mean pooling
                    batch_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
                elif hasattr(outputs, 'pooler_output'):
                    batch_features = outputs.pooler_output
                else:
                    # Fallback to first output
                    batch_features = outputs[0][:, 0, :]

                features.append(batch_features.cpu().numpy())

        return np.concatenate(features, axis=0)

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return np.array([])


def compute_metrics(eval_pred):
    """
    Compute metrics for evaluation.

    Args:
        eval_pred: Evaluation predictions from trainer

    Returns:
        Dictionary of metrics
    """
    try:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    except Exception as e:
        logger.error(f"Metrics computation failed: {e}")
        return {'accuracy': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}


class DatasetBuilder:
    """Builder class for creating datasets."""

    def __init__(self, tokenizer, max_length: int = 512):
        """
        Initialize dataset builder.

        Args:
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def build_text_dataset(self, texts: List[str]) -> TextDataset:
        """Build a text dataset."""
        return TextDataset(texts, self.tokenizer, self.max_length)

    def build_classification_dataset(self, texts: List[str], labels: List[int]) -> CustomClassificationDataset:
        """Build a classification dataset."""
        return CustomClassificationDataset(texts, labels, self.tokenizer, self.max_length)

    def build_token_classification_dataset(self, texts: List[str],
                                         labels: List[List[int]]) -> CustomTokenClassificationDataset:
        """Build a token classification dataset."""
        return CustomTokenClassificationDataset(texts, labels, self.tokenizer, self.max_length)

    def build_image_text_dataset(self, image_paths: List[str], texts: List[str],
                                processor) -> ImageTextDataset:
        """Build an image-text dataset."""
        return ImageTextDataset(image_paths, texts, processor)

    def build_audio_dataset(self, audio_paths: List[str], texts: Optional[List[str]] = None,
                           processor=None) -> AudioDataset:
        """Build an audio dataset."""
        return AudioDataset(audio_paths, texts, processor)


def get_sample_data_for_task(task_type: str) -> Dict[str, Any]:
    """
    Get sample data for different task types.

    Args:
        task_type: Type of task

    Returns:
        Dictionary with sample data
    """
    sample_data = {
        'text-classification': {
            'train_texts': [
                "이 제품은 훌륭합니다.",
                "서비스가 별로입니다.",
                "가격이 적당합니다.",
                "품질이 좋습니다."
            ],
            'train_labels': [1, 0, 1, 1],  # 1: positive, 0: negative
            'val_texts': [
                "만족스러운 구매였습니다.",
                "다시는 사지 않겠습니다."
            ],
            'val_labels': [1, 0]
        },
        'token-classification': {
            'train_texts': [
                "김철수는 서울에서 일합니다.",
                "이영희는 부산 출신입니다."
            ],
            'train_labels': [
                [1, 2, 0, 3, 0, 0, 0],  # B-PER, I-PER, O, B-LOC, O, O, O
                [1, 2, 0, 3, 0, 0]      # B-PER, I-PER, O, B-LOC, O, O
            ]
        },
        'question-answering': {
            'contexts': [
                "파리는 프랑스의 수도입니다.",
                "도쿄는 일본의 수도입니다."
            ],
            'questions': [
                "프랑스의 수도는 어디인가요?",
                "일본의 수도는 어디인가요?"
            ],
            'answers': [
                {"answer_start": 0, "text": "파리"},
                {"answer_start": 0, "text": "도쿄"}
            ]
        }
    }

    return sample_data.get(task_type, {})


# Legacy class alias for backward compatibility
CustomDataset = CustomClassificationDataset