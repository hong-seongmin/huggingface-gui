"""
Configuration analyzer for model analysis.

This module provides specialized analysis for model configuration files,
extracted from the original monolithic model_analyzer.py.
"""

import json
import os
from typing import Dict, Any, List


class ConfigAnalyzer:
    """Analyzer for model configuration files."""

    def __init__(self):
        """Initialize config analyzer."""
        pass

    def analyze_config(self, file_path: str) -> Dict[str, Any]:
        """Analyze config.json file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return {
            'model_type': config.get('model_type', 'unknown'),
            'architectures': config.get('architectures', []),
            'vocab_size': config.get('vocab_size', 0),
            'hidden_size': config.get('hidden_size', 0),
            'num_hidden_layers': config.get('num_hidden_layers', 0),
            'num_attention_heads': config.get('num_attention_heads', 0),
            'max_position_embeddings': config.get('max_position_embeddings', 0),
            'intermediate_size': config.get('intermediate_size', 0),
            'hidden_act': config.get('hidden_act', 'unknown'),
            'hidden_dropout_prob': config.get('hidden_dropout_prob', None),
            'attention_probs_dropout_prob': config.get('attention_probs_dropout_prob', None),
            'initializer_range': config.get('initializer_range', None),
            'layer_norm_eps': config.get('layer_norm_eps', None),
            'task_specific_params': config.get('task_specific_params', {}),
            'supported_tasks': self.infer_tasks_from_config(config),
            'model_parameters': self.estimate_parameters(config),
            'full_config': config
        }

    def analyze_generation_config(self, file_path: str) -> Dict[str, Any]:
        """Analyze generation_config.json file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return {
            'max_length': config.get('max_length', 0),
            'max_new_tokens': config.get('max_new_tokens', 0),
            'temperature': config.get('temperature', 1.0),
            'top_p': config.get('top_p', 1.0),
            'top_k': config.get('top_k', 50),
            'do_sample': config.get('do_sample', False),
            'pad_token_id': config.get('pad_token_id', None),
            'eos_token_id': config.get('eos_token_id', None),
            'repetition_penalty': config.get('repetition_penalty', 1.0),
            'no_repeat_ngram_size': config.get('no_repeat_ngram_size', 0),
            'bad_words_ids': config.get('bad_words_ids', []),
            'force_words_ids': config.get('force_words_ids', []),
            'full_config': config
        }

    def infer_tasks_from_config(self, config: Dict, model_name: str = "", model_path: str = "") -> List[str]:
        """Infer supported tasks from model configuration."""
        tasks = []
        model_type = config.get('model_type', '').lower()
        architectures = config.get('architectures', [])

        # Architecture-based task inference
        for arch in architectures:
            arch_lower = arch.lower()

            # Classification tasks
            if any(keyword in arch_lower for keyword in [
                'classification', 'classifier', 'forsequenceclassification'
            ]):
                if 'token' in arch_lower:
                    tasks.append('token-classification')
                else:
                    tasks.append('text-classification')

            # Question Answering
            elif any(keyword in arch_lower for keyword in [
                'questionanswering', 'forquestionanswering'
            ]):
                tasks.append('question-answering')

            # Generation tasks
            elif any(keyword in arch_lower for keyword in [
                'lmhead', 'causal', 'generation', 'gpt'
            ]):
                tasks.append('text-generation')

            # Conditional generation (seq2seq)
            elif any(keyword in arch_lower for keyword in [
                'conditionalgeneration', 'forconditionalgeneration', 'seq2seq'
            ]):
                tasks.append('text2text-generation')

            # Masked language modeling
            elif any(keyword in arch_lower for keyword in [
                'maskedlm', 'formaskedlm'
            ]):
                tasks.append('fill-mask')

            # Multiple choice
            elif any(keyword in arch_lower for keyword in [
                'multiplechoice', 'formultiplechoice'
            ]):
                tasks.append('multiple-choice')

        # Model type-based inference
        if model_type:
            if model_type in ['bert', 'distilbert', 'roberta', 'electra']:
                if not tasks:  # If no tasks inferred from architecture
                    tasks.extend(['fill-mask', 'text-classification', 'token-classification'])

            elif model_type in ['gpt', 'gpt2', 'gpt-neo', 'gpt-j']:
                if 'text-generation' not in tasks:
                    tasks.append('text-generation')

            elif model_type in ['t5', 'bart', 'pegasus']:
                if 'text2text-generation' not in tasks:
                    tasks.append('text2text-generation')

            elif model_type in ['clip', 'blip']:
                tasks.append('feature-extraction')
                tasks.append('image-to-text')

            elif model_type in ['whisper']:
                tasks.append('automatic-speech-recognition')

            elif model_type in ['wav2vec2']:
                tasks.append('audio-classification')
                tasks.append('automatic-speech-recognition')

        # Model name-based inference (fallback)
        if model_name:
            name_lower = model_name.lower()

            if any(keyword in name_lower for keyword in ['sentiment', 'emotion']):
                if 'text-classification' not in tasks:
                    tasks.append('text-classification')

            elif any(keyword in name_lower for keyword in ['ner', 'named-entity']):
                if 'token-classification' not in tasks:
                    tasks.append('token-classification')

            elif any(keyword in name_lower for keyword in ['qa', 'question']):
                if 'question-answering' not in tasks:
                    tasks.append('question-answering')

            elif any(keyword in name_lower for keyword in ['summarization', 'summary']):
                if 'summarization' not in tasks:
                    tasks.append('summarization')

            elif any(keyword in name_lower for keyword in ['translation', 'translate']):
                if 'translation' not in tasks:
                    tasks.append('translation')

        # Default fallback
        if not tasks:
            if model_type in ['bert', 'distilbert', 'roberta']:
                tasks.append('feature-extraction')
            elif model_type in ['gpt', 'gpt2']:
                tasks.append('text-generation')
            else:
                tasks.append('feature-extraction')

        return list(set(tasks))  # Remove duplicates

    def estimate_parameters(self, config: Dict) -> int:
        """Estimate model parameters from configuration."""
        vocab_size = config.get('vocab_size', 0)
        hidden_size = config.get('hidden_size', 0)
        num_layers = config.get('num_hidden_layers', 0)
        num_heads = config.get('num_attention_heads', 0)
        intermediate_size = config.get('intermediate_size', 0)
        max_pos = config.get('max_position_embeddings', 0)

        if not all([vocab_size, hidden_size, num_layers]):
            return 0

        # Embedding parameters
        embedding_params = vocab_size * hidden_size  # Token embeddings
        if max_pos > 0:
            embedding_params += max_pos * hidden_size  # Position embeddings

        # Transformer layer parameters
        layer_params = 0
        if num_heads > 0:
            # Self-attention: Q, K, V projections + output projection
            layer_params += 4 * hidden_size * hidden_size

        if intermediate_size > 0:
            # Feed-forward: two linear layers
            layer_params += hidden_size * intermediate_size + intermediate_size * hidden_size
        else:
            # Fallback: assume 4x expansion
            layer_params += hidden_size * (4 * hidden_size) + (4 * hidden_size) * hidden_size

        # Layer normalization parameters (2 per layer typically)
        layer_params += 2 * 2 * hidden_size  # 2 LayerNorms with 2 parameters each

        # Total parameters
        total_params = embedding_params + (layer_params * num_layers)

        # Add output/classification head if applicable
        architectures = config.get('architectures', [])
        for arch in architectures:
            if 'Classification' in arch:
                num_labels = config.get('num_labels', 2)
                total_params += hidden_size * num_labels
            elif 'LMHead' in arch or 'Generation' in arch:
                total_params += vocab_size * hidden_size  # Language modeling head

        return int(total_params)

    def get_model_size_category(self, parameters: int) -> str:
        """Categorize model size based on parameter count."""
        if parameters < 1_000_000:
            return "tiny"
        elif parameters < 10_000_000:
            return "small"
        elif parameters < 100_000_000:
            return "medium"
        elif parameters < 1_000_000_000:
            return "large"
        elif parameters < 10_000_000_000:
            return "extra-large"
        else:
            return "massive"

    def analyze_model_efficiency(self, config: Dict) -> Dict[str, Any]:
        """Analyze model efficiency characteristics."""
        hidden_size = config.get('hidden_size', 0)
        num_layers = config.get('num_hidden_layers', 0)
        num_heads = config.get('num_attention_heads', 0)
        intermediate_size = config.get('intermediate_size', 0)
        max_pos = config.get('max_position_embeddings', 0)

        efficiency = {}

        # Compute attention efficiency
        if hidden_size and num_heads:
            head_dim = hidden_size // num_heads
            efficiency['head_dimension'] = head_dim
            efficiency['attention_efficiency'] = 'efficient' if head_dim in [32, 64, 128] else 'suboptimal'

        # Compute feed-forward ratio
        if hidden_size and intermediate_size:
            ff_ratio = intermediate_size / hidden_size
            efficiency['ff_expansion_ratio'] = ff_ratio
            efficiency['ff_efficiency'] = 'standard' if 3 <= ff_ratio <= 5 else 'non-standard'

        # Layer-to-parameter ratio
        parameters = self.estimate_parameters(config)
        if parameters and num_layers:
            params_per_layer = parameters / num_layers
            efficiency['parameters_per_layer'] = int(params_per_layer)

        # Context length analysis
        if max_pos:
            efficiency['max_sequence_length'] = max_pos
            if max_pos <= 512:
                efficiency['context_category'] = 'short'
            elif max_pos <= 2048:
                efficiency['context_category'] = 'medium'
            elif max_pos <= 8192:
                efficiency['context_category'] = 'long'
            else:
                efficiency['context_category'] = 'very_long'

        return efficiency