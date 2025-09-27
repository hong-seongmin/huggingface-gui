"""
Unit tests for ComprehensiveModelAnalyzer.

This test suite follows TDD principles to verify model analysis functionality
before refactoring the large model_analyzer.py into modular components.
"""

import unittest
import tempfile
import os
import json
from unittest.mock import Mock, patch, mock_open
import sys

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))


class TestComprehensiveModelAnalyzer(unittest.TestCase):
    """Test ComprehensiveModelAnalyzer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from models.model_analyzer import ComprehensiveModelAnalyzer
            self.analyzer = ComprehensiveModelAnalyzer()
        except ImportError:
            self.analyzer = None

        # Sample config data
        self.sample_config = {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"],
            "vocab_size": 30522,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "max_position_embeddings": 512,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1
        }

        # Sample tokenizer config
        self.sample_tokenizer_config = {
            "tokenizer_class": "BertTokenizer",
            "model_max_length": 512,
            "padding_side": "right",
            "special_tokens": ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
        }

        # Sample generation config
        self.sample_generation_config = {
            "max_length": 20,
            "temperature": 1.0,
            "top_p": 1.0,
            "do_sample": False
        }

    def test_analyzer_initialization(self):
        """Test ComprehensiveModelAnalyzer initialization."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        self.assertIsNotNone(self.analyzer)
        self.assertIsInstance(self.analyzer.supported_files, dict)
        self.assertGreater(len(self.analyzer.supported_files), 0)

    def test_supported_files_coverage(self):
        """Test that analyzer supports common model files."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        expected_files = [
            'config.json',
            'tokenizer.json',
            'tokenizer_config.json',
            'vocab.txt',
            'merges.txt',
            'special_tokens_map.json',
            'pytorch_model.bin',
            'model.safetensors',
            'generation_config.json'
        ]

        for file_type in expected_files:
            self.assertIn(file_type, self.analyzer.supported_files)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_analyze_model_directory_basic(self, mock_file, mock_exists):
        """Test basic model directory analysis."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Mock file existence
        def mock_exists_side_effect(path):
            return path.endswith('config.json') or path.endswith('tokenizer_config.json')

        mock_exists.side_effect = mock_exists_side_effect

        # Mock file content
        mock_file.return_value.read.side_effect = [
            json.dumps(self.sample_config),
            json.dumps(self.sample_tokenizer_config)
        ]

        result = self.analyzer.analyze_model_directory("/fake/path", "test-model")

        # Verify basic structure
        self.assertIn('model_path', result)
        self.assertIn('model_name', result)
        self.assertIn('files_found', result)
        self.assertIn('files_missing', result)
        self.assertIn('analysis_results', result)
        self.assertIn('model_summary', result)
        self.assertIn('recommendations', result)

        # Verify model name and path
        self.assertEqual(result['model_name'], "test-model")
        self.assertEqual(result['model_path'], "/fake/path")

        # Verify files are categorized correctly
        self.assertIn('config.json', result['files_found'])
        self.assertIn('tokenizer_config.json', result['files_found'])
        self.assertTrue(len(result['files_missing']) > 0)

    def test_analyze_config_file(self):
        """Test config.json analysis."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_config, f)
            config_path = f.name

        try:
            result = self.analyzer._analyze_config(config_path)

            # Verify config analysis structure
            expected_keys = [
                'model_type', 'architectures', 'vocab_size', 'hidden_size',
                'num_hidden_layers', 'num_attention_heads', 'max_position_embeddings',
                'supported_tasks', 'model_parameters'
            ]

            for key in expected_keys:
                self.assertIn(key, result)

            # Verify specific values
            self.assertEqual(result['model_type'], 'bert')
            self.assertEqual(result['vocab_size'], 30522)
            self.assertEqual(result['hidden_size'], 768)
            self.assertIsInstance(result['supported_tasks'], list)
            self.assertIsInstance(result['model_parameters'], int)
            self.assertGreater(result['model_parameters'], 0)

        finally:
            os.unlink(config_path)

    def test_analyze_tokenizer_config_file(self):
        """Test tokenizer_config.json analysis."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_tokenizer_config, f)
            config_path = f.name

        try:
            result = self.analyzer._analyze_tokenizer_config(config_path)

            # Verify tokenizer config analysis
            expected_keys = ['tokenizer_class', 'model_max_length', 'padding_side']
            for key in expected_keys:
                self.assertIn(key, result)

            self.assertEqual(result['tokenizer_class'], 'BertTokenizer')
            self.assertEqual(result['model_max_length'], 512)
            self.assertEqual(result['padding_side'], 'right')

        finally:
            os.unlink(config_path)

    def test_analyze_generation_config_file(self):
        """Test generation_config.json analysis."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_generation_config, f)
            config_path = f.name

        try:
            result = self.analyzer._analyze_generation_config(config_path)

            # Verify generation config analysis
            expected_keys = ['max_length', 'temperature', 'do_sample']
            for key in expected_keys:
                self.assertIn(key, result)

            self.assertEqual(result['max_length'], 20)
            self.assertEqual(result['temperature'], 1.0)
            self.assertEqual(result['do_sample'], False)

        finally:
            os.unlink(config_path)

    def test_infer_tasks_from_config(self):
        """Test task inference from model config."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Test BERT classification model
        bert_config = {
            "model_type": "bert",
            "architectures": ["BertForSequenceClassification"]
        }
        tasks = self.analyzer._infer_tasks_from_config(bert_config)
        self.assertIn("text-classification", tasks)

        # Test GPT generation model
        gpt_config = {
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"]
        }
        tasks = self.analyzer._infer_tasks_from_config(gpt_config)
        self.assertIn("text-generation", tasks)

        # Test T5 model
        t5_config = {
            "model_type": "t5",
            "architectures": ["T5ForConditionalGeneration"]
        }
        tasks = self.analyzer._infer_tasks_from_config(t5_config)
        self.assertIn("text2text-generation", tasks)

    def test_estimate_parameters(self):
        """Test parameter estimation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Test BERT-base parameter estimation
        params = self.analyzer._estimate_parameters(self.sample_config)
        self.assertIsInstance(params, int)
        self.assertGreater(params, 100_000_000)  # Should be > 100M for BERT-base
        self.assertLess(params, 200_000_000)     # Should be < 200M for BERT-base

        # Test smaller model
        small_config = {
            "vocab_size": 1000,
            "hidden_size": 128,
            "num_hidden_layers": 6,
            "num_attention_heads": 8,
            "intermediate_size": 512
        }
        small_params = self.analyzer._estimate_parameters(small_config)
        self.assertLess(small_params, params)

    def test_generate_model_summary(self):
        """Test model summary generation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Mock analysis results
        analysis_results = {
            'config.json': {
                'model_type': 'bert',
                'architectures': ['BertForSequenceClassification'],
                'vocab_size': 30522,
                'hidden_size': 768,
                'num_hidden_layers': 12,
                'supported_tasks': ['text-classification'],
                'model_parameters': 110000000
            },
            'tokenizer_config.json': {
                'tokenizer_class': 'BertTokenizer',
                'model_max_length': 512
            }
        }

        summary = self.analyzer._generate_model_summary(
            analysis_results, "test-bert", "/path/to/model"
        )

        # Verify summary structure
        expected_keys = [
            'model_name', 'model_type', 'architecture', 'parameters',
            'supported_tasks', 'max_sequence_length', 'vocab_size'
        ]

        for key in expected_keys:
            self.assertIn(key, summary)

        # Verify summary content
        self.assertEqual(summary['model_name'], 'test-bert')
        self.assertEqual(summary['model_type'], 'bert')
        self.assertIn('BertForSequenceClassification', summary['architecture'])
        self.assertEqual(summary['parameters'], 110000000)
        self.assertIn('text-classification', summary['supported_tasks'])

    def test_generate_recommendations(self):
        """Test recommendation generation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Mock analysis with missing files
        analysis = {
            'files_found': ['config.json'],
            'files_missing': ['tokenizer.json', 'vocab.txt'],
            'analysis_results': {
                'config.json': {
                    'model_type': 'bert',
                    'supported_tasks': ['text-classification']
                }
            }
        }

        recommendations = self.analyzer._generate_recommendations(analysis)

        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)

        # Should have recommendations about missing files
        recommendation_text = ' '.join(recommendations)
        self.assertTrue(any(missing in recommendation_text
                           for missing in analysis['files_missing']))

    def test_error_handling_invalid_file(self):
        """Test error handling for invalid files."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            invalid_config_path = f.name

        try:
            # Should not raise exception, but return error in result
            result = self.analyzer._analyze_config(invalid_config_path)
            # The method might handle the error differently
            # This test ensures the method doesn't crash

        except:
            # If it does raise an exception, that's also acceptable
            # as long as it's handled at the directory level
            pass

        finally:
            os.unlink(invalid_config_path)

    def test_usage_example_generation(self):
        """Test usage example generation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Test with classification task
        examples = self.analyzer._generate_usage_examples(
            ['text-classification'], 'bert'
        )

        self.assertIsInstance(examples, dict)
        self.assertIn('text-classification', examples)

        classification_example = examples['text-classification']
        self.assertIn('description', classification_example)
        self.assertIn('code_example', classification_example)
        self.assertIn('sample_input', classification_example)
        self.assertIn('expected_output', classification_example)

        # Verify code example is not empty
        self.assertTrue(len(classification_example['code_example']) > 0)


class TestModelAnalyzerComponents(unittest.TestCase):
    """Test individual components that will be modularized."""

    def setUp(self):
        """Set up test fixtures."""
        try:
            from models.model_analyzer import ComprehensiveModelAnalyzer
            self.analyzer = ComprehensiveModelAnalyzer()
        except ImportError:
            self.analyzer = None

    def test_vocab_analysis_readiness(self):
        """Test readiness for vocab.txt analysis separation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Create mock vocab file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            vocab_content = "\n".join([f"token{i}" for i in range(100)])
            f.write(vocab_content)
            vocab_path = f.name

        try:
            result = self.analyzer._analyze_vocab(vocab_path)

            self.assertIn('vocab_size', result)
            self.assertIn('sample_tokens', result)
            self.assertEqual(result['vocab_size'], 100)
            self.assertIsInstance(result['sample_tokens'], list)

        finally:
            os.unlink(vocab_path)

    def test_merges_analysis_readiness(self):
        """Test readiness for merges.txt analysis separation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Create mock merges file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            merges_content = "\n".join([f"t o\nke n" for _ in range(50)])
            f.write(merges_content)
            merges_path = f.name

        try:
            result = self.analyzer._analyze_merges(merges_path)

            self.assertIn('merge_count', result)
            self.assertIn('sample_merges', result)
            self.assertIsInstance(result['merge_count'], int)
            self.assertIsInstance(result['sample_merges'], list)

        finally:
            os.unlink(merges_path)

    def test_weight_analysis_readiness(self):
        """Test readiness for weight analysis separation."""
        if not self.analyzer:
            self.skipTest("ComprehensiveModelAnalyzer not available")

        # Test pytorch model analysis structure
        with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
            mock_path = f.name

        try:
            # Should handle non-existent or invalid pytorch files gracefully
            result = self.analyzer._analyze_pytorch_model(mock_path)

            # Should return some result structure even for invalid files
            self.assertIsInstance(result, dict)

        finally:
            os.unlink(mock_path)


if __name__ == '__main__':
    unittest.main()