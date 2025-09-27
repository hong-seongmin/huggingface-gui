"""
Tokenizer analyzer for model analysis.

This module provides specialized analysis for tokenizer files,
extracted from the original monolithic model_analyzer.py.
"""

import json
import os
from typing import Dict, Any


class TokenizerAnalyzer:
    """Analyzer for tokenizer-related files."""

    def __init__(self):
        """Initialize tokenizer analyzer."""
        pass

    def analyze_tokenizer(self, file_path: str) -> Dict[str, Any]:
        """Analyze tokenizer.json file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)

        vocab_data = tokenizer_data.get('model', {}).get('vocab', {})

        return {
            'version': tokenizer_data.get('version', 'unknown'),
            'model_type': tokenizer_data.get('model', {}).get('type', 'unknown'),
            'vocab_size': len(vocab_data),
            'special_tokens': tokenizer_data.get('special_tokens', {}),
            'pre_tokenizer': tokenizer_data.get('pre_tokenizer', {}),
            'post_processor': tokenizer_data.get('post_processor', {}),
            'decoder': tokenizer_data.get('decoder', {}),
            'vocab_analysis': self._analyze_vocab_content(vocab_data)
        }

    def analyze_tokenizer_config(self, file_path: str) -> Dict[str, Any]:
        """Analyze tokenizer_config.json file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        return {
            'tokenizer_class': config.get('tokenizer_class', 'unknown'),
            'model_max_length': config.get('model_max_length', 0),
            'padding_side': config.get('padding_side', 'unknown'),
            'truncation_side': config.get('truncation_side', 'unknown'),
            'special_tokens': {k: v for k, v in config.items() if 'token' in k.lower()},
            'clean_up_tokenization_spaces': config.get('clean_up_tokenization_spaces', None),
            'full_config': config
        }

    def analyze_vocab(self, file_path: str) -> Dict[str, Any]:
        """Analyze vocab.txt file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_lines = f.readlines()

        vocab_size = len(vocab_lines)
        special_tokens = []
        regular_tokens = []

        for line in vocab_lines:
            token = line.strip()
            if token.startswith('[') and token.endswith(']'):
                special_tokens.append(token)
            else:
                regular_tokens.append(token)

        return {
            'vocab_size': vocab_size,
            'special_tokens_found': special_tokens,
            'special_tokens_count': len(special_tokens),
            'regular_tokens_count': len(regular_tokens),
            'sample_tokens': [line.strip() for line in vocab_lines[:10]],
            'token_length_stats': self._analyze_token_lengths(regular_tokens[:1000])
        }

    def analyze_merges(self, file_path: str) -> Dict[str, Any]:
        """Analyze merges.txt file (for BPE tokenizers)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            merges = f.readlines()

        # Filter out header lines
        merge_rules = [line.strip() for line in merges if ' ' in line.strip()]

        return {
            'merge_count': len(merge_rules),
            'sample_merges': merge_rules[:10],
            'merge_patterns': self._analyze_merge_patterns(merge_rules[:100])
        }

    def analyze_special_tokens(self, file_path: str) -> Dict[str, Any]:
        """Analyze special_tokens_map.json file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            special_tokens = json.load(f)

        return {
            'special_tokens_map': special_tokens,
            'token_count': len(special_tokens),
            'token_types': list(special_tokens.keys())
        }

    def _analyze_vocab_content(self, vocab_data: Dict[str, int]) -> Dict[str, Any]:
        """Analyze vocabulary content for insights."""
        if not vocab_data:
            return {}

        # Get token statistics
        tokens = list(vocab_data.keys())
        token_lengths = [len(token) for token in tokens]

        # Count different token types
        special_count = sum(1 for token in tokens if token.startswith('[') and token.endswith(']'))
        subword_count = sum(1 for token in tokens if token.startswith('##'))
        unicode_count = sum(1 for token in tokens if any(ord(char) > 127 for char in token))

        return {
            'total_tokens': len(tokens),
            'avg_token_length': sum(token_lengths) / len(token_lengths) if token_lengths else 0,
            'max_token_length': max(token_lengths) if token_lengths else 0,
            'min_token_length': min(token_lengths) if token_lengths else 0,
            'special_tokens_count': special_count,
            'subword_tokens_count': subword_count,
            'unicode_tokens_count': unicode_count,
            'sample_tokens': tokens[:10]
        }

    def _analyze_token_lengths(self, tokens: list) -> Dict[str, Any]:
        """Analyze token length distribution."""
        if not tokens:
            return {}

        lengths = [len(token) for token in tokens]

        return {
            'avg_length': sum(lengths) / len(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'length_distribution': self._get_length_distribution(lengths)
        }

    def _get_length_distribution(self, lengths: list) -> Dict[str, int]:
        """Get distribution of token lengths."""
        distribution = {}
        for length in lengths:
            distribution[str(length)] = distribution.get(str(length), 0) + 1
        return dict(sorted(distribution.items(), key=lambda x: int(x[0])))

    def _analyze_merge_patterns(self, merge_rules: list) -> Dict[str, Any]:
        """Analyze BPE merge patterns."""
        if not merge_rules:
            return {}

        # Count character-level vs subword-level merges
        char_merges = 0
        subword_merges = 0

        for rule in merge_rules:
            if ' ' in rule:
                parts = rule.split()
                if len(parts) >= 2:
                    if len(parts[0]) == 1 and len(parts[1]) == 1:
                        char_merges += 1
                    else:
                        subword_merges += 1

        return {
            'character_level_merges': char_merges,
            'subword_level_merges': subword_merges,
            'total_rules_analyzed': len(merge_rules)
        }

    def get_tokenizer_recommendations(self, analysis_results: Dict[str, Any]) -> list:
        """Generate recommendations based on tokenizer analysis."""
        recommendations = []

        # Analyze vocab size
        vocab_size = 0
        if 'tokenizer.json' in analysis_results:
            vocab_size = analysis_results['tokenizer.json'].get('vocab_size', 0)
        elif 'vocab.txt' in analysis_results:
            vocab_size = analysis_results['vocab.txt'].get('vocab_size', 0)

        if vocab_size > 0:
            if vocab_size < 10000:
                recommendations.append(f"Small vocabulary ({vocab_size} tokens) - may limit model expressiveness")
            elif vocab_size > 100000:
                recommendations.append(f"Very large vocabulary ({vocab_size} tokens) - may increase memory usage")

        # Check for missing tokenizer components
        if 'tokenizer_config.json' not in analysis_results:
            recommendations.append("Missing tokenizer_config.json - may cause compatibility issues")

        if 'tokenizer.json' not in analysis_results and 'vocab.txt' not in analysis_results:
            recommendations.append("No vocabulary file found - tokenizer may not work properly")

        # Analyze special tokens
        for file_type, analysis in analysis_results.items():
            if 'special_tokens' in analysis:
                special_tokens = analysis['special_tokens']
                if not special_tokens:
                    recommendations.append(f"No special tokens defined in {file_type}")

        return recommendations