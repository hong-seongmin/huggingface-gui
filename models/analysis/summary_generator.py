"""
Summary generator for model analysis.

This module provides comprehensive summary generation functionality,
extracted from the original monolithic model_analyzer.py.
"""

from typing import Dict, Any, List


class SummaryGenerator:
    """Generator for comprehensive model summaries and recommendations."""

    def __init__(self):
        """Initialize summary generator."""
        pass

    def generate_model_summary(self, analysis_results: Dict, model_name: str = "", model_path: str = "") -> Dict[str, Any]:
        """Generate comprehensive model summary."""
        summary = {
            'model_name': model_name,
            'model_path': model_path,
            'model_type': 'unknown',
            'architecture': [],
            'parameters': 0,
            'file_size_mb': 0,
            'supported_tasks': [],
            'max_sequence_length': 0,
            'vocab_size': 0,
            'tokenizer_type': 'unknown',
            'generation_config': {},
            'embedding_info': {},
            'capabilities': [],
            'usage_examples': {},
            'performance_characteristics': {}
        }

        # Extract config information
        if 'config.json' in analysis_results:
            config_analysis = analysis_results['config.json']
            summary.update({
                'model_type': config_analysis.get('model_type', 'unknown'),
                'architecture': config_analysis.get('architectures', []),
                'parameters': config_analysis.get('model_parameters', 0),
                'supported_tasks': config_analysis.get('supported_tasks', []),
                'max_sequence_length': config_analysis.get('max_position_embeddings', 0),
                'vocab_size': config_analysis.get('vocab_size', 0)
            })

        # Extract tokenizer information
        if 'tokenizer_config.json' in analysis_results:
            tokenizer_config = analysis_results['tokenizer_config.json']
            summary['tokenizer_type'] = tokenizer_config.get('tokenizer_class', 'unknown')
            if tokenizer_config.get('model_max_length', 0) > 0:
                summary['max_sequence_length'] = tokenizer_config.get('model_max_length', 0)

        # Extract weight file information
        weight_files = ['pytorch_model.bin', 'model.safetensors']
        for weight_file in weight_files:
            if weight_file in analysis_results:
                weight_analysis = analysis_results[weight_file]
                if 'error' not in weight_analysis:
                    summary['file_size_mb'] = weight_analysis.get('file_size_mb', 0)
                    if weight_analysis.get('total_parameters', 0) > summary['parameters']:
                        summary['parameters'] = weight_analysis.get('total_parameters', 0)
                    break

        # Extract generation config
        if 'generation_config.json' in analysis_results:
            summary['generation_config'] = analysis_results['generation_config.json']

        # Generate embedding info
        summary['embedding_info'] = self._extract_embedding_info(summary, analysis_results)

        # Analyze model capabilities
        summary['capabilities'] = self._analyze_model_capabilities(summary, analysis_results)

        # Generate usage examples
        if summary['supported_tasks']:
            summary['usage_examples'] = self._generate_usage_examples(
                summary['supported_tasks'],
                summary['model_type'],
                analysis_results
            )

        # Generate performance characteristics
        summary['performance_characteristics'] = self._analyze_performance_characteristics(summary, analysis_results)

        return summary

    def generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []

        # Missing files recommendations
        missing_files = analysis.get('files_missing', [])
        critical_missing = []

        for missing_file in missing_files:
            if missing_file in ['config.json']:
                critical_missing.append(missing_file)
            elif missing_file in ['tokenizer_config.json', 'tokenizer.json', 'vocab.txt']:
                if not any(f in analysis.get('files_found', []) for f in ['tokenizer.json', 'vocab.txt']):
                    critical_missing.append('tokenizer files')

        if critical_missing:
            recommendations.append(f"Missing critical files: {', '.join(critical_missing)}")

        # Model size recommendations
        analysis_results = analysis.get('analysis_results', {})
        if 'config.json' in analysis_results:
            config = analysis_results['config.json']
            params = config.get('model_parameters', 0)

            if params > 10_000_000_000:  # > 10B parameters
                recommendations.append("Very large model (>10B params) - consider using model parallelism or quantization")
            elif params > 1_000_000_000:  # > 1B parameters
                recommendations.append("Large model (>1B params) - may require significant computational resources")

        # Task-specific recommendations
        model_summary = analysis.get('model_summary', {})
        supported_tasks = model_summary.get('supported_tasks', [])

        if not supported_tasks:
            recommendations.append("No clear task inference - manual task specification may be required")
        elif len(supported_tasks) == 1:
            recommendations.append(f"Single-task model optimized for {supported_tasks[0]}")
        else:
            recommendations.append(f"Multi-task model supporting {len(supported_tasks)} different tasks")

        # Performance recommendations
        max_length = model_summary.get('max_sequence_length', 0)
        if max_length > 4096:
            recommendations.append(f"Long context model ({max_length} tokens) - memory usage scales quadratically")
        elif max_length < 512:
            recommendations.append(f"Short context model ({max_length} tokens) - may truncate long inputs")

        return recommendations

    def _extract_embedding_info(self, summary: Dict, analysis_results: Dict) -> Dict[str, Any]:
        """Extract embedding-related information."""
        embedding_info = {}

        if 'config.json' in analysis_results:
            config = analysis_results['config.json']

            embedding_info.update({
                'hidden_size': config.get('hidden_size', 0),
                'vocab_size': config.get('vocab_size', 0),
                'max_position_embeddings': config.get('max_position_embeddings', 0),
                'embedding_dimension': config.get('hidden_size', 0)
            })

            # Calculate embedding parameters
            vocab_size = config.get('vocab_size', 0)
            hidden_size = config.get('hidden_size', 0)
            max_pos = config.get('max_position_embeddings', 0)

            if vocab_size and hidden_size:
                embedding_params = vocab_size * hidden_size
                if max_pos:
                    embedding_params += max_pos * hidden_size
                embedding_info['embedding_parameters'] = embedding_params

        return embedding_info

    def _analyze_model_capabilities(self, summary: Dict, analysis_results: Dict) -> List[str]:
        """Analyze and list model capabilities."""
        capabilities = []

        model_type = summary.get('model_type', '').lower()
        supported_tasks = summary.get('supported_tasks', [])

        # Task-based capabilities
        task_capabilities = {
            'text-classification': 'Text classification and sentiment analysis',
            'token-classification': 'Named entity recognition and part-of-speech tagging',
            'text-generation': 'Text generation and completion',
            'text2text-generation': 'Text-to-text transformation and translation',
            'question-answering': 'Question answering and reading comprehension',
            'fill-mask': 'Masked language modeling and text completion',
            'summarization': 'Text summarization',
            'translation': 'Language translation',
            'feature-extraction': 'Feature extraction and embeddings'
        }

        for task in supported_tasks:
            if task in task_capabilities:
                capabilities.append(task_capabilities[task])

        # Model type specific capabilities
        if model_type in ['bert', 'distilbert', 'roberta']:
            capabilities.append('Bidirectional context understanding')
            capabilities.append('Rich text representations')
        elif model_type in ['gpt', 'gpt2', 'gpt-neo']:
            capabilities.append('Autoregressive text generation')
            capabilities.append('Creative writing and completion')
        elif model_type in ['t5', 'bart']:
            capabilities.append('Encoder-decoder architecture')
            capabilities.append('Flexible text transformation')

        # Parameter-based capabilities
        parameters = summary.get('parameters', 0)
        if parameters > 1_000_000_000:
            capabilities.append('High-capacity model with extensive knowledge')
        elif parameters > 100_000_000:
            capabilities.append('Medium-capacity model suitable for most tasks')

        # Context length capabilities
        max_length = summary.get('max_sequence_length', 0)
        if max_length > 2048:
            capabilities.append('Long-form text processing')
        elif max_length > 512:
            capabilities.append('Document-level understanding')

        return list(set(capabilities))

    def _generate_usage_examples(self, tasks: List[str], model_type: str, analysis_results: Dict = None) -> Dict[str, Dict]:
        """Generate usage examples for supported tasks."""
        examples = {}

        for task in tasks:
            if task == 'text-classification':
                examples[task] = {
                    'description': 'Classify text into predefined categories',
                    'code_example': '''from transformers import pipeline
classifier = pipeline("text-classification", model="model_name")
result = classifier("This movie is amazing!")
print(result)''',
                    'sample_input': 'This movie is amazing!',
                    'expected_output': '[{"label": "POSITIVE", "score": 0.95}]'
                }

            elif task == 'text-generation':
                examples[task] = {
                    'description': 'Generate text continuation from a prompt',
                    'code_example': '''from transformers import pipeline
generator = pipeline("text-generation", model="model_name")
result = generator("Once upon a time", max_length=50)
print(result)''',
                    'sample_input': 'Once upon a time',
                    'expected_output': 'Once upon a time, in a distant land...'
                }

            elif task == 'token-classification':
                examples[task] = {
                    'description': 'Identify and classify individual tokens (NER, POS tagging)',
                    'code_example': '''from transformers import pipeline
ner = pipeline("token-classification", model="model_name")
result = ner("John works at Google in California")
print(result)''',
                    'sample_input': 'John works at Google in California',
                    'expected_output': '[{"entity": "B-PER", "word": "John", "score": 0.99}]'
                }

            elif task == 'question-answering':
                examples[task] = {
                    'description': 'Answer questions based on given context',
                    'code_example': '''from transformers import pipeline
qa = pipeline("question-answering", model="model_name")
result = qa(question="What is AI?", context="AI is artificial intelligence...")
print(result)''',
                    'sample_input': {'question': 'What is AI?', 'context': 'AI is artificial intelligence...'},
                    'expected_output': '{"answer": "artificial intelligence", "score": 0.85}'
                }

            elif task == 'fill-mask':
                examples[task] = {
                    'description': 'Fill in masked tokens in text',
                    'code_example': '''from transformers import pipeline
fill_mask = pipeline("fill-mask", model="model_name")
result = fill_mask("The capital of France is [MASK].")
print(result)''',
                    'sample_input': 'The capital of France is [MASK].',
                    'expected_output': '[{"token_str": "Paris", "score": 0.92}]'
                }

            elif task == 'feature-extraction':
                examples[task] = {
                    'description': 'Extract feature embeddings from text',
                    'code_example': '''from transformers import pipeline
feature_extractor = pipeline("feature-extraction", model="model_name")
embeddings = feature_extractor("Sample text for embedding")
print(embeddings.shape)''',
                    'sample_input': 'Sample text for embedding',
                    'expected_output': 'torch.Size([1, sequence_length, hidden_size])'
                }

        return examples

    def _analyze_performance_characteristics(self, summary: Dict, analysis_results: Dict) -> Dict[str, Any]:
        """Analyze model performance characteristics."""
        perf = {}

        # Model size impact
        parameters = summary.get('parameters', 0)
        if parameters > 0:
            if parameters < 50_000_000:
                perf['size_category'] = 'lightweight'
                perf['inference_speed'] = 'fast'
                perf['memory_usage'] = 'low'
            elif parameters < 500_000_000:
                perf['size_category'] = 'medium'
                perf['inference_speed'] = 'moderate'
                perf['memory_usage'] = 'moderate'
            else:
                perf['size_category'] = 'large'
                perf['inference_speed'] = 'slow'
                perf['memory_usage'] = 'high'

        # Context length impact
        max_length = summary.get('max_sequence_length', 0)
        if max_length > 0:
            if max_length <= 512:
                perf['context_efficiency'] = 'high'
            elif max_length <= 2048:
                perf['context_efficiency'] = 'moderate'
            else:
                perf['context_efficiency'] = 'low'
                perf['memory_scaling'] = 'quadratic'

        # Architecture-specific characteristics
        model_type = summary.get('model_type', '').lower()
        if model_type in ['bert', 'distilbert', 'roberta']:
            perf['parallelizable'] = False
            perf['bidirectional'] = True
        elif model_type in ['gpt', 'gpt2']:
            perf['parallelizable'] = True
            perf['autoregressive'] = True
        elif model_type in ['t5', 'bart']:
            perf['encoder_decoder'] = True
            perf['versatile'] = True

        return perf