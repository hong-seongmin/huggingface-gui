"""
Comprehensive unit tests for ModelTypeDetector.

This test suite covers all major functionality of the ModelTypeDetector class
including model type detection, configuration analysis, pattern matching,
architecture analysis, and caching mechanisms.
"""

import unittest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import sys

# Add the project root directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

try:
    from models.model_type_detector import ModelTypeDetector
except ImportError as e:
    print(f"Warning: Could not import ModelTypeDetector: {e}")
    ModelTypeDetector = None


class TestModelTypeDetector(unittest.TestCase):
    """Test cases for ModelTypeDetector class."""

    def setUp(self):
        """Set up test fixtures."""
        if ModelTypeDetector is None:
            self.skipTest("ModelTypeDetector not available for testing")

        self.detector = ModelTypeDetector()
        self.temp_dir = tempfile.mkdtemp()

        # Create test model directories
        self.test_model_dirs = {
            'bert_classification': os.path.join(self.temp_dir, 'bert_classification'),
            'bert_base': os.path.join(self.temp_dir, 'bert_base'),
            'gpt2_generation': os.path.join(self.temp_dir, 'gpt2_generation'),
            't5_seq2seq': os.path.join(self.temp_dir, 't5_seq2seq'),
            'deberta_ner': os.path.join(self.temp_dir, 'deberta_ner'),
        }

        for model_dir in self.test_model_dirs.values():
            os.makedirs(model_dir, exist_ok=True)

    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_config_file(self, model_dir: str, config_data: Dict[str, Any]):
        """Helper method to create a config.json file in a model directory."""
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        return config_path

    def test_detector_initialization(self):
        """Test ModelTypeDetector initialization."""
        self.assertIsInstance(self.detector, ModelTypeDetector)
        self.assertIsNotNone(self.detector.architecture_to_task)
        self.assertIsNotNone(self.detector.name_patterns)
        self.assertIsNotNone(self.detector.task_to_model_class)
        self.assertEqual(len(self.detector.detection_cache), 0)

    def test_detect_model_type_with_cache(self):
        """Test model type detection with caching."""
        model_name = "test_model"
        model_path = self.test_model_dirs['bert_classification']

        # Create config file
        config_data = {
            "architectures": ["BertForSequenceClassification"],
            "model_type": "bert",
            "num_labels": 2
        }
        self.create_config_file(model_path, config_data)

        # First call should perform detection
        task_type, model_class, analysis_info = self.detector.detect_model_type(
            model_name, model_path
        )

        self.assertEqual(task_type, "text-classification")
        self.assertIn("AutoModelForSequenceClassification", model_class)
        self.assertGreater(analysis_info["confidence"], 0.8)

        # Second call should use cache
        task_type2, model_class2, analysis_info2 = self.detector.detect_model_type(
            model_name, model_path
        )

        self.assertEqual(task_type, task_type2)
        self.assertEqual(model_class, model_class2)
        self.assertIn("cache", analysis_info2["detection_method"])

    def test_analyze_config_sequence_classification(self):
        """Test config analysis for sequence classification models."""
        config_data = {
            "architectures": ["BertForSequenceClassification"],
            "model_type": "bert",
            "num_labels": 3,
            "id2label": {"0": "negative", "1": "neutral", "2": "positive"},
            "label2id": {"negative": 0, "neutral": 1, "positive": 2}
        }

        config_path = self.create_config_file(
            self.test_model_dirs['bert_classification'], config_data
        )

        task_type, model_class, confidence = self.detector._analyze_config(
            self.test_model_dirs['bert_classification']
        )

        self.assertEqual(task_type, "text-classification")
        self.assertEqual(model_class, "AutoModelForSequenceClassification")
        self.assertGreater(confidence, 0.9)

    def test_analyze_config_token_classification(self):
        """Test config analysis for token classification models."""
        config_data = {
            "architectures": ["BertForTokenClassification"],
            "model_type": "bert",
            "num_labels": 9,
            "id2label": {
                "0": "O", "1": "B-PER", "2": "I-PER",
                "3": "B-LOC", "4": "I-LOC", "5": "B-ORG",
                "6": "I-ORG", "7": "B-MISC", "8": "I-MISC"
            }
        }

        config_path = self.create_config_file(
            self.test_model_dirs['deberta_ner'], config_data
        )

        task_type, model_class, confidence = self.detector._analyze_config(
            self.test_model_dirs['deberta_ner']
        )

        self.assertEqual(task_type, "token-classification")
        self.assertEqual(model_class, "AutoModelForTokenClassification")
        self.assertGreater(confidence, 0.9)

    def test_analyze_config_text_generation(self):
        """Test config analysis for text generation models."""
        config_data = {
            "architectures": ["GPT2LMHeadModel"],
            "model_type": "gpt2",
            "vocab_size": 50257
        }

        config_path = self.create_config_file(
            self.test_model_dirs['gpt2_generation'], config_data
        )

        task_type, model_class, confidence = self.detector._analyze_config(
            self.test_model_dirs['gpt2_generation']
        )

        self.assertEqual(task_type, "text-generation")
        self.assertEqual(model_class, "AutoModelForCausalLM")
        self.assertGreater(confidence, 0.9)

    def test_analyze_config_seq2seq(self):
        """Test config analysis for sequence-to-sequence models."""
        config_data = {
            "architectures": ["T5ForConditionalGeneration"],
            "model_type": "t5",
            "decoder_start_token_id": 0
        }

        config_path = self.create_config_file(
            self.test_model_dirs['t5_seq2seq'], config_data
        )

        task_type, model_class, confidence = self.detector._analyze_config(
            self.test_model_dirs['t5_seq2seq']
        )

        self.assertEqual(task_type, "text2text-generation")
        self.assertEqual(model_class, "AutoModelForSeq2SeqLM")
        self.assertGreater(confidence, 0.9)

    def test_analyze_config_missing_file(self):
        """Test config analysis with missing config.json file."""
        non_existent_path = os.path.join(self.temp_dir, 'non_existent_model')

        task_type, model_class, confidence = self.detector._analyze_config(
            non_existent_path
        )

        self.assertIsNone(task_type)
        self.assertIsNone(model_class)
        self.assertEqual(confidence, 0.0)

    def test_analyze_model_name_patterns(self):
        """Test model name pattern analysis."""
        test_cases = [
            ("bert-base-sentiment-analysis", "text-classification"),
            ("distilbert-base-ner", "token-classification"),
            ("gpt2-medium", "text-generation"),
            ("bge-large-en-v1.5", "feature-extraction"),
            ("t5-base-finetuned-squad", "question-answering"),
        ]

        for model_name, expected_task in test_cases:
            with self.subTest(model_name=model_name):
                task_type, confidence = self.detector._analyze_model_name(model_name)
                self.assertEqual(task_type, expected_task)
                self.assertGreater(confidence, 0.0)

    def test_fallback_detection_multitask(self):
        """Test fallback detection for multitask models."""
        test_cases = [
            ("multitask-bert-base", "text-classification", "AutoModelForSequenceClassification"),
            ("multitask-deberta-ner", "token-classification", "AutoModelForTokenClassification"),
            ("multi-task-classifier", "text-classification", "AutoModelForSequenceClassification"),
        ]

        for model_name, expected_task, expected_class in test_cases:
            with self.subTest(model_name=model_name):
                task_type, model_class = self.detector._fallback_detection(model_name)
                self.assertEqual(task_type, expected_task)
                self.assertEqual(model_class, expected_class)

    def test_fallback_detection_specific_patterns(self):
        """Test fallback detection for specific model patterns."""
        test_cases = [
            ("deberta-v3-base-ner", "token-classification", "AutoModelForTokenClassification"),
            ("electra-small-discriminator", "text-classification", "AutoModelForSequenceClassification"),
            ("sentence-transformers-all-MiniLM", "feature-extraction", "AutoModel"),
            ("llama2-7b-chat", "text-generation", "AutoModelForCausalLM"),
        ]

        for model_name, expected_task, expected_class in test_cases:
            with self.subTest(model_name=model_name):
                task_type, model_class = self.detector._fallback_detection(model_name)
                self.assertEqual(task_type, expected_task)
                self.assertEqual(model_class, expected_class)

    def test_determine_task_from_architecture(self):
        """Test task determination from architecture names."""
        mock_config = Mock()
        mock_config.id2label = {"0": "negative", "1": "positive"}
        mock_config.label2id = {"negative": 0, "positive": 1}

        test_cases = [
            ("BertForSequenceClassification", "text-classification", 0.95),
            ("DistilBertForTokenClassification", "token-classification", 0.95),
            ("RobertaForQuestionAnswering", "question-answering", 0.95),
            ("GPT2LMHeadModel", "text-generation", 0.95),
            ("T5ForConditionalGeneration", "text2text-generation", 0.95),
        ]

        for architecture, expected_task, expected_confidence in test_cases:
            with self.subTest(architecture=architecture):
                task_type, confidence = self.detector._determine_task_from_architecture(
                    architecture, mock_config, "test_model"
                )
                self.assertEqual(task_type, expected_task)
                self.assertGreaterEqual(confidence, expected_confidence)

    def test_extract_task_from_architecture(self):
        """Test task extraction from architecture patterns."""
        mock_config = {
            "id2label": {"0": "O", "1": "B-PER", "2": "I-PER"},
            "num_labels": 3
        }

        test_cases = [
            ("CustomForSequenceClassification", "text-classification"),
            ("CustomForTokenClassification", "token-classification"),
            ("CustomForQuestionAnswering", "question-answering"),
            ("CustomForCausalLM", "text-generation"),
            ("CustomModel", "feature-extraction"),  # base model
        ]

        for architecture, expected_task in test_cases:
            with self.subTest(architecture=architecture):
                task_type, confidence = self.detector._extract_task_from_architecture(
                    architecture, mock_config
                )
                if expected_task:
                    self.assertEqual(task_type, expected_task)
                    self.assertGreater(confidence, 0.0)

    def test_infer_task_from_config_details(self):
        """Test task inference from detailed config analysis."""
        # Test NER labels detection
        ner_config = {
            "id2label": {"0": "O", "1": "B-PER", "2": "I-PER", "3": "B-LOC"},
            "label2id": {"O": 0, "B-PER": 1, "I-PER": 2, "B-LOC": 3},
            "num_labels": 4
        }

        task_type, confidence = self.detector._infer_task_from_config_details(
            "BertModel", ner_config
        )
        self.assertEqual(task_type, "token-classification")
        self.assertGreater(confidence, 0.9)

        # Test sentiment classification labels
        sentiment_config = {
            "id2label": {"0": "negative", "1": "positive"},
            "label2id": {"negative": 0, "positive": 1},
            "num_labels": 2
        }

        task_type, confidence = self.detector._infer_task_from_config_details(
            "BertModel", sentiment_config
        )
        self.assertEqual(task_type, "text-classification")
        self.assertGreater(confidence, 0.8)

    def test_get_transformers_class_name(self):
        """Test transformers class name mapping."""
        test_cases = [
            ("AutoModel", "AutoModel"),
            ("AutoModelForSequenceClassification", "AutoModelForSequenceClassification"),
            ("AutoModelForCausalLM", "AutoModelForCausalLM"),
            ("AutoModelForSeq2SeqLM", "AutoModelForSeq2SeqLM"),
            ("AutoModelForTokenClassification", "AutoModelForTokenClassification"),
            ("UnknownModel", "AutoModel"),  # fallback
        ]

        for input_class, expected_class in test_cases:
            with self.subTest(input_class=input_class):
                result = self.detector.get_transformers_class_name(input_class)
                self.assertEqual(result, expected_class)

    def test_get_model_specific_class(self):
        """Test model-specific class determination."""
        test_cases = [
            ("AutoModelForSequenceClassification", "bert", "BertForSequenceClassification"),
            ("AutoModelForSequenceClassification", "distilbert", "DistilBertForSequenceClassification"),
            ("AutoModelForSequenceClassification", "roberta", "RobertaForSequenceClassification"),
            ("AutoModel", "bert", "BertModel"),
            ("AutoModel", "distilbert", "DistilBertModel"),
            ("AutoModel", "unknown", "AutoModel"),  # fallback
        ]

        for model_class, model_type, expected_class in test_cases:
            with self.subTest(model_class=model_class, model_type=model_type):
                result = self.detector.get_model_specific_class(model_class, model_type)
                self.assertEqual(result, expected_class)

    def test_cache_operations(self):
        """Test cache operations (clear and stats)."""
        # Add some entries to cache
        self.detector.detection_cache["test1"] = ("text-classification", "AutoModelForSequenceClassification")
        self.detector.detection_cache["test2"] = ("text-generation", "AutoModelForCausalLM")

        # Test cache stats
        stats = self.detector.get_cache_stats()
        self.assertEqual(stats["cache_size"], 2)
        self.assertIn("test1", stats["cached_models"])
        self.assertIn("test2", stats["cached_models"])

        # Test cache clear
        self.detector.clear_cache()
        self.assertEqual(len(self.detector.detection_cache), 0)

        stats_after_clear = self.detector.get_cache_stats()
        self.assertEqual(stats_after_clear["cache_size"], 0)
        self.assertEqual(len(stats_after_clear["cached_models"]), 0)

    @patch('models.model_type_detector.AutoConfig')
    def test_analyze_with_autoconfig_success(self, mock_autoconfig):
        """Test successful AutoConfig analysis."""
        # Mock AutoConfig.from_pretrained
        mock_config = Mock()
        mock_config.model_type = "bert"
        mock_config.architectures = ["BertForSequenceClassification"]
        mock_autoconfig.from_pretrained.return_value = mock_config

        # Mock model_database.get_model_info
        with patch('models.model_type_detector.model_database') as mock_db:
            mock_model_info = Mock()
            mock_model_info.primary_tasks = [Mock(value="text-classification")]
            mock_db.get_model_info.return_value = mock_model_info

            task_type, model_class, confidence = self.detector._analyze_with_autoconfig(
                self.test_model_dirs['bert_classification'], "bert-base-uncased"
            )

            self.assertEqual(task_type, "text-classification")
            self.assertIn("AutoModel", model_class)
            self.assertGreater(confidence, 0.8)

    @patch('models.model_type_detector.AutoConfig')
    def test_analyze_with_autoconfig_failure(self, mock_autoconfig):
        """Test AutoConfig analysis failure handling."""
        # Mock AutoConfig to raise exception
        mock_autoconfig.from_pretrained.side_effect = Exception("Mock error")

        task_type, model_class, confidence = self.detector._analyze_with_autoconfig(
            self.test_model_dirs['bert_classification'], "bert-base-uncased"
        )

        self.assertIsNone(task_type)
        self.assertIsNone(model_class)
        self.assertEqual(confidence, 0.0)

    def test_comprehensive_detection_flow(self):
        """Test the complete detection flow with various scenarios."""
        # Test 1: Perfect config-based detection
        config_data = {
            "architectures": ["RobertaForSequenceClassification"],
            "model_type": "roberta",
            "num_labels": 2
        }
        model_path = self.test_model_dirs['bert_classification']
        self.create_config_file(model_path, config_data)

        task_type, model_class, analysis_info = self.detector.detect_model_type(
            "roberta-sentiment", model_path
        )

        self.assertEqual(task_type, "text-classification")
        self.assertIn("config_analysis", analysis_info["detection_method"])
        self.assertGreater(analysis_info["confidence"], 0.9)

        # Test 2: Name pattern fallback
        task_type, model_class, analysis_info = self.detector.detect_model_type(
            "bge-large-en-v1.5", "/non/existent/path"
        )

        self.assertEqual(task_type, "feature-extraction")
        self.assertIn("name_pattern", analysis_info["detection_method"])

    def test_architecture_to_task_mapping_completeness(self):
        """Test that architecture to task mapping covers major model types."""
        expected_architectures = [
            "BertForSequenceClassification",
            "DistilBertForSequenceClassification",
            "RobertaForSequenceClassification",
            "DebertaForSequenceClassification",
            "ElectraForSequenceClassification",
            "BertForTokenClassification",
            "GPT2LMHeadModel",
            "T5ForConditionalGeneration",
            "BartForConditionalGeneration",
        ]

        for arch in expected_architectures:
            with self.subTest(architecture=arch):
                self.assertIn(arch, self.detector.architecture_to_task)

    def test_name_patterns_coverage(self):
        """Test that name patterns cover all major task types."""
        expected_tasks = [
            "text-classification",
            "token-classification",
            "text-generation",
            "translation",
            "summarization",
            "question-answering",
            "feature-extraction",
            "text2text-generation"
        ]

        for task in expected_tasks:
            with self.subTest(task=task):
                self.assertIn(task, self.detector.name_patterns)
                self.assertGreater(len(self.detector.name_patterns[task]), 0)


class TestModelTypeDetectorIntegration(unittest.TestCase):
    """Integration tests for ModelTypeDetector with realistic scenarios."""

    def setUp(self):
        """Set up integration test fixtures."""
        if ModelTypeDetector is None:
            self.skipTest("ModelTypeDetector not available for testing")

        self.detector = ModelTypeDetector()
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def create_realistic_model_config(self, model_dir: str, model_type: str, task_type: str):
        """Create realistic model configurations for different scenarios."""
        configs = {
            "bert-sentiment": {
                "architectures": ["BertForSequenceClassification"],
                "model_type": "bert",
                "num_labels": 3,
                "id2label": {"0": "negative", "1": "neutral", "2": "positive"},
                "label2id": {"negative": 0, "neutral": 1, "positive": 2},
                "hidden_size": 768,
                "vocab_size": 30522
            },
            "distilbert-ner": {
                "architectures": ["DistilBertForTokenClassification"],
                "model_type": "distilbert",
                "num_labels": 9,
                "id2label": {
                    "0": "O", "1": "B-PER", "2": "I-PER",
                    "3": "B-LOC", "4": "I-LOC", "5": "B-ORG",
                    "6": "I-ORG", "7": "B-MISC", "8": "I-MISC"
                },
                "hidden_size": 768,
                "vocab_size": 30522
            },
            "gpt2-generation": {
                "architectures": ["GPT2LMHeadModel"],
                "model_type": "gpt2",
                "vocab_size": 50257,
                "n_positions": 1024,
                "n_embd": 768,
                "n_head": 12,
                "n_layer": 12
            }
        }

        config_key = f"{model_type}-{task_type}"
        if config_key in configs:
            config_path = os.path.join(model_dir, 'config.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(configs[config_key], f, indent=2)

    def test_realistic_bert_sentiment_model(self):
        """Test detection of realistic BERT sentiment analysis model."""
        model_dir = os.path.join(self.temp_dir, 'bert_sentiment')
        os.makedirs(model_dir, exist_ok=True)

        self.create_realistic_model_config(model_dir, "bert", "sentiment")

        task_type, model_class, analysis_info = self.detector.detect_model_type(
            "bert-base-uncased-sentiment", model_dir
        )

        self.assertEqual(task_type, "text-classification")
        self.assertEqual(model_class, "AutoModelForSequenceClassification")
        self.assertGreater(analysis_info["confidence"], 0.9)
        self.assertIn("config_analysis", analysis_info["detection_method"])

    def test_realistic_distilbert_ner_model(self):
        """Test detection of realistic DistilBERT NER model."""
        model_dir = os.path.join(self.temp_dir, 'distilbert_ner')
        os.makedirs(model_dir, exist_ok=True)

        self.create_realistic_model_config(model_dir, "distilbert", "ner")

        task_type, model_class, analysis_info = self.detector.detect_model_type(
            "distilbert-base-cased-ner", model_dir
        )

        self.assertEqual(task_type, "token-classification")
        self.assertEqual(model_class, "AutoModelForTokenClassification")
        self.assertGreater(analysis_info["confidence"], 0.9)

    def test_realistic_gpt2_generation_model(self):
        """Test detection of realistic GPT-2 generation model."""
        model_dir = os.path.join(self.temp_dir, 'gpt2_generation')
        os.makedirs(model_dir, exist_ok=True)

        self.create_realistic_model_config(model_dir, "gpt2", "generation")

        task_type, model_class, analysis_info = self.detector.detect_model_type(
            "gpt2-medium", model_dir
        )

        self.assertEqual(task_type, "text-generation")
        self.assertEqual(model_class, "AutoModelForCausalLM")
        self.assertGreater(analysis_info["confidence"], 0.9)


if __name__ == '__main__':
    unittest.main(verbosity=2)