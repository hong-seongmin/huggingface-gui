"""
Weight analyzer for model analysis.

This module provides specialized analysis for model weight files,
extracted from the original monolithic model_analyzer.py.
"""

import os
from typing import Dict, Any


class WeightAnalyzer:
    """Analyzer for model weight files."""

    def __init__(self):
        """Initialize weight analyzer."""
        pass

    def analyze_pytorch_model(self, file_path: str) -> Dict[str, Any]:
        """Analyze pytorch_model.bin file."""
        try:
            import torch
            model_data = torch.load(file_path, map_location='cpu')

            total_params = sum(p.numel() for p in model_data.values() if hasattr(p, 'numel'))
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            # Analyze parameter distribution
            param_info = self._analyze_parameter_distribution(model_data)

            return {
                'file_size_mb': file_size,
                'total_parameters': total_params,
                'parameter_keys': list(model_data.keys())[:10],  # First 10 keys
                'dtype_info': str(next(iter(model_data.values())).dtype) if model_data else 'unknown',
                'parameter_distribution': param_info,
                'layer_analysis': self._analyze_layer_structure(model_data)
            }
        except Exception as e:
            return {'error': str(e)}

    def analyze_safetensors(self, file_path: str) -> Dict[str, Any]:
        """Analyze model.safetensors file."""
        try:
            from safetensors import safe_open

            metadata = {}
            tensor_count = 0
            total_size = 0
            tensor_info = []

            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for key in f.keys():
                    tensor_count += 1
                    tensor = f.get_tensor(key)
                    total_size += tensor.numel()

                    # Collect tensor information
                    if len(tensor_info) < 10:  # Limit to first 10 for performance
                        tensor_info.append({
                            'name': key,
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'numel': tensor.numel()
                        })

            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            return {
                'file_size_mb': file_size,
                'tensor_count': tensor_count,
                'total_parameters': total_size,
                'metadata': metadata,
                'tensor_info': tensor_info,
                'format': 'safetensors',
                'efficiency_score': self._calculate_efficiency_score(file_size, total_size)
            }
        except Exception as e:
            return {'error': str(e)}

    def _analyze_parameter_distribution(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the distribution of parameters across different components."""
        distribution = {
            'embeddings': 0,
            'attention': 0,
            'feed_forward': 0,
            'layer_norm': 0,
            'output_head': 0,
            'other': 0
        }

        for key, tensor in model_data.items():
            if not hasattr(tensor, 'numel'):
                continue

            param_count = tensor.numel()
            key_lower = key.lower()

            # Categorize parameters
            if any(keyword in key_lower for keyword in ['embed', 'word_embeddings', 'position_embeddings']):
                distribution['embeddings'] += param_count
            elif any(keyword in key_lower for keyword in ['attention', 'attn', 'self_attn', 'multi_head']):
                distribution['attention'] += param_count
            elif any(keyword in key_lower for keyword in ['feed_forward', 'ffn', 'mlp', 'fc', 'dense']):
                distribution['feed_forward'] += param_count
            elif any(keyword in key_lower for keyword in ['layer_norm', 'layernorm', 'ln']):
                distribution['layer_norm'] += param_count
            elif any(keyword in key_lower for keyword in ['classifier', 'lm_head', 'output', 'head']):
                distribution['output_head'] += param_count
            else:
                distribution['other'] += param_count

        # Calculate percentages
        total = sum(distribution.values())
        if total > 0:
            percentages = {k: (v / total) * 100 for k, v in distribution.items()}
            distribution['percentages'] = percentages

        return distribution

    def _analyze_layer_structure(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the layer structure of the model."""
        layers = {}
        layer_pattern_counts = {}

        for key in model_data.keys():
            # Extract layer numbers and patterns
            key_parts = key.split('.')

            # Look for layer patterns
            for i, part in enumerate(key_parts):
                if part.isdigit():
                    layer_num = int(part)
                    if i > 0:
                        layer_type = key_parts[i-1]
                        if layer_type not in layer_pattern_counts:
                            layer_pattern_counts[layer_type] = set()
                        layer_pattern_counts[layer_type].add(layer_num)

        # Convert sets to counts
        layer_structure = {}
        for layer_type, layer_nums in layer_pattern_counts.items():
            layer_structure[layer_type] = {
                'count': len(layer_nums),
                'max_layer': max(layer_nums) if layer_nums else 0
            }

        return layer_structure

    def _calculate_efficiency_score(self, file_size_mb: float, total_params: int) -> Dict[str, Any]:
        """Calculate efficiency metrics for the model weights."""
        if total_params == 0:
            return {'score': 0, 'category': 'unknown'}

        # Calculate bits per parameter
        file_size_bits = file_size_mb * 1024 * 1024 * 8
        bits_per_param = file_size_bits / total_params

        # Categorize efficiency
        if bits_per_param <= 8:
            category = 'excellent'  # INT8 or better
            score = 95
        elif bits_per_param <= 16:
            category = 'good'  # FP16
            score = 80
        elif bits_per_param <= 32:
            category = 'standard'  # FP32
            score = 65
        else:
            category = 'inefficient'
            score = 40

        return {
            'bits_per_parameter': round(bits_per_param, 2),
            'score': score,
            'category': category,
            'compression_ratio': round(32 / bits_per_param, 2) if bits_per_param > 0 else 1
        }

    def compare_weight_formats(self, pytorch_path: str = None, safetensors_path: str = None) -> Dict[str, Any]:
        """Compare different weight file formats if both are available."""
        comparison = {}

        if pytorch_path and os.path.exists(pytorch_path):
            comparison['pytorch'] = self.analyze_pytorch_model(pytorch_path)

        if safetensors_path and os.path.exists(safetensors_path):
            comparison['safetensors'] = self.analyze_safetensors(safetensors_path)

        # Add comparison metrics if both formats are available
        if 'pytorch' in comparison and 'safetensors' in comparison:
            pt_data = comparison['pytorch']
            st_data = comparison['safetensors']

            if 'error' not in pt_data and 'error' not in st_data:
                comparison['format_comparison'] = {
                    'size_difference_mb': pt_data['file_size_mb'] - st_data['file_size_mb'],
                    'parameter_count_match': pt_data['total_parameters'] == st_data['total_parameters'],
                    'recommended_format': 'safetensors' if st_data['file_size_mb'] <= pt_data['file_size_mb'] else 'pytorch'
                }

        return comparison

    def get_weight_file_recommendations(self, analysis_results: Dict[str, Any]) -> list:
        """Generate recommendations based on weight file analysis."""
        recommendations = []

        for file_type, analysis in analysis_results.items():
            if 'error' in analysis:
                recommendations.append(f"Could not analyze {file_type}: {analysis['error']}")
                continue

            # File size recommendations
            file_size = analysis.get('file_size_mb', 0)
            if file_size > 1000:  # > 1GB
                recommendations.append(f"{file_type}: Large model ({file_size:.1f}MB) - consider using quantization")
            elif file_size > 5000:  # > 5GB
                recommendations.append(f"{file_type}: Very large model ({file_size:.1f}MB) - strongly consider quantization or sharding")

            # Efficiency recommendations
            if 'efficiency_score' in analysis:
                efficiency = analysis['efficiency_score']
                if efficiency['category'] == 'inefficient':
                    recommendations.append(f"{file_type}: Inefficient storage ({efficiency['bits_per_parameter']} bits/param) - consider using FP16 or quantization")

            # Format recommendations
            if file_type == 'pytorch' and file_size > 100:
                recommendations.append("Consider converting to SafeTensors format for better security and loading speed")

        return recommendations