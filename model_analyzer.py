import json
import os
from pathlib import Path
from typing import Dict, Any, List

class ComprehensiveModelAnalyzer:
    def __init__(self):
        self.supported_files = {
            'config.json': self._analyze_config,
            'tokenizer.json': self._analyze_tokenizer,
            'tokenizer_config.json': self._analyze_tokenizer_config,
            'vocab.txt': self._analyze_vocab,
            'merges.txt': self._analyze_merges,
            'special_tokens_map.json': self._analyze_special_tokens,
            'pytorch_model.bin': self._analyze_pytorch_model,
            'model.safetensors': self._analyze_safetensors,
            'generation_config.json': self._analyze_generation_config
        }
    
    def analyze_model_directory(self, model_path: str) -> Dict[str, Any]:
        """모델 디렉토리 전체 분석"""
        analysis = {
            'model_path': model_path,
            'files_found': [],
            'files_missing': [],
            'analysis_results': {},
            'model_summary': {},
            'recommendations': []
        }
        
        # 파일 존재 여부 확인
        for filename in self.supported_files.keys():
            file_path = os.path.join(model_path, filename)
            if os.path.exists(file_path):
                analysis['files_found'].append(filename)
                try:
                    analysis['analysis_results'][filename] = self.supported_files[filename](file_path)
                except Exception as e:
                    analysis['analysis_results'][filename] = {'error': str(e)}
            else:
                analysis['files_missing'].append(filename)
        
        # 모델 요약 생성
        analysis['model_summary'] = self._generate_model_summary(analysis['analysis_results'])
        
        # 권장사항 생성
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _analyze_config(self, file_path: str) -> Dict[str, Any]:
        """config.json 분석"""
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
            'supported_tasks': self._infer_tasks_from_config(config),
            'model_parameters': self._estimate_parameters(config),
            'full_config': config  # 전체 설정 저장
        }
    
    def _analyze_tokenizer(self, file_path: str) -> Dict[str, Any]:
        """tokenizer.json 분석"""
        with open(file_path, 'r', encoding='utf-8') as f:
            tokenizer_data = json.load(f)
        
        return {
            'version': tokenizer_data.get('version', 'unknown'),
            'model_type': tokenizer_data.get('model', {}).get('type', 'unknown'),
            'vocab_size': len(tokenizer_data.get('model', {}).get('vocab', {})),
            'special_tokens': tokenizer_data.get('special_tokens', {}),
            'pre_tokenizer': tokenizer_data.get('pre_tokenizer', {}),
            'post_processor': tokenizer_data.get('post_processor', {}),
            'decoder': tokenizer_data.get('decoder', {})
        }
    
    def _analyze_tokenizer_config(self, file_path: str) -> Dict[str, Any]:
        """tokenizer_config.json 분석"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return {
            'tokenizer_class': config.get('tokenizer_class', 'unknown'),
            'model_max_length': config.get('model_max_length', 0),
            'padding_side': config.get('padding_side', 'unknown'),
            'truncation_side': config.get('truncation_side', 'unknown'),
            'special_tokens': {k: v for k, v in config.items() if 'token' in k.lower()},
            'clean_up_tokenization_spaces': config.get('clean_up_tokenization_spaces', None)
        }
    
    def _analyze_vocab(self, file_path: str) -> Dict[str, Any]:
        """vocab.txt 분석"""
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_lines = f.readlines()
        
        vocab_size = len(vocab_lines)
        special_tokens = []
        
        for line in vocab_lines[:100]:  # 처음 100개만 체크
            token = line.strip()
            if token.startswith('[') and token.endswith(']'):
                special_tokens.append(token)
        
        return {
            'vocab_size': vocab_size,
            'special_tokens_found': special_tokens,
            'sample_tokens': [line.strip() for line in vocab_lines[:10]]
        }
    
    def _analyze_merges(self, file_path: str) -> Dict[str, Any]:
        """merges.txt 분석 (BPE 토크나이저용)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            merges = f.readlines()
        
        return {
            'num_merges': len(merges),
            'sample_merges': [line.strip() for line in merges[:10]]
        }
    
    def _analyze_special_tokens(self, file_path: str) -> Dict[str, Any]:
        """special_tokens_map.json 분석"""
        with open(file_path, 'r', encoding='utf-8') as f:
            special_tokens = json.load(f)
        
        return special_tokens
    
    def _analyze_pytorch_model(self, file_path: str) -> Dict[str, Any]:
        """pytorch_model.bin 분석"""
        try:
            import torch
            model_data = torch.load(file_path, map_location='cpu')
            
            total_params = sum(p.numel() for p in model_data.values() if hasattr(p, 'numel'))
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            return {
                'file_size_mb': file_size,
                'total_parameters': total_params,
                'parameter_keys': list(model_data.keys())[:10],  # 첫 10개 키만
                'dtype_info': str(next(iter(model_data.values())).dtype) if model_data else 'unknown'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_safetensors(self, file_path: str) -> Dict[str, Any]:
        """model.safetensors 분석"""
        try:
            from safetensors import safe_open
            
            metadata = {}
            tensor_count = 0
            total_size = 0
            
            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                for key in f.keys():
                    tensor_count += 1
                    tensor = f.get_tensor(key)
                    total_size += tensor.numel()
            
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            
            return {
                'file_size_mb': file_size,
                'tensor_count': tensor_count,
                'total_parameters': total_size,
                'metadata': metadata
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_generation_config(self, file_path: str) -> Dict[str, Any]:
        """generation_config.json 분석"""
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
            'eos_token_id': config.get('eos_token_id', None)
        }
    
    def _generate_model_summary(self, analysis_results: Dict) -> Dict[str, Any]:
        """분석 결과를 바탕으로 모델 요약 생성"""
        summary = {
            'model_type': 'unknown',
            'total_parameters': 0,
            'model_size_mb': 0,
            'supported_tasks': [],
            'tokenizer_type': 'unknown',
            'max_sequence_length': 0,
            'vocabulary_size': 0,
            'usage_examples': {},
            'detailed_config': {}
        }
        
        # config.json에서 정보 추출
        if 'config.json' in analysis_results:
            config_data = analysis_results['config.json']
            summary['model_type'] = config_data.get('model_type', 'unknown')
            summary['supported_tasks'] = config_data.get('supported_tasks', [])
            summary['max_sequence_length'] = config_data.get('max_position_embeddings', 0)
            summary['vocabulary_size'] = config_data.get('vocab_size', 0)
            summary['total_parameters'] = config_data.get('model_parameters', 0)
            
            # 상세 설정 정보
            summary['detailed_config'] = {
                'architecture': config_data.get('architectures', ['N/A'])[0] if config_data.get('architectures') else 'N/A',
                'hidden_size': config_data.get('hidden_size', 'N/A'),
                'num_attention_heads': config_data.get('num_attention_heads', 'N/A'),
                'num_hidden_layers': config_data.get('num_hidden_layers', 'N/A'),
                'max_position_embeddings': config_data.get('max_position_embeddings', 'N/A'),
                'dropout': config_data.get('hidden_dropout_prob', config_data.get('dropout', 'N/A')),
                'initializer_range': config_data.get('initializer_range', 'N/A'),
                'layer_norm_eps': config_data.get('layer_norm_eps', 'N/A'),
                'intermediate_size': config_data.get('intermediate_size', 'N/A'),
                'attention_dropout': config_data.get('attention_probs_dropout_prob', 'N/A'),
                'activation_function': config_data.get('hidden_act', 'N/A')
            }
        
        # 모델 파일 크기 정보
        for file_key in ['pytorch_model.bin', 'model.safetensors']:
            if file_key in analysis_results:
                file_data = analysis_results[file_key]
                summary['model_size_mb'] = file_data.get('file_size_mb', 0)
                if 'total_parameters' in file_data and file_data['total_parameters'] > 0:
                    summary['total_parameters'] = file_data['total_parameters']
                break
        
        # 토크나이저 정보
        if 'tokenizer_config.json' in analysis_results:
            tokenizer_data = analysis_results['tokenizer_config.json']
            summary['tokenizer_type'] = tokenizer_data.get('tokenizer_class', 'unknown')
            summary['detailed_config']['tokenizer_max_length'] = tokenizer_data.get('model_max_length', 'N/A')
            summary['detailed_config']['padding_side'] = tokenizer_data.get('padding_side', 'N/A')
            summary['detailed_config']['truncation_side'] = tokenizer_data.get('truncation_side', 'N/A')
        
        # 생성 설정 정보
        if 'generation_config.json' in analysis_results:
            gen_data = analysis_results['generation_config.json']
            summary['detailed_config']['generation_max_length'] = gen_data.get('max_length', 'N/A')
            summary['detailed_config']['generation_temperature'] = gen_data.get('temperature', 'N/A')
            summary['detailed_config']['generation_top_p'] = gen_data.get('top_p', 'N/A')
            summary['detailed_config']['generation_top_k'] = gen_data.get('top_k', 'N/A')
        
        # 사용 예시 생성 (분석 결과 포함)
        summary['usage_examples'] = self._generate_usage_examples(summary['supported_tasks'], summary['model_type'], analysis_results)
        
        return summary
    
    def _generate_usage_examples(self, tasks: List[str], model_type: str, analysis_results: Dict = None) -> Dict[str, Dict]:
        """태스크별 사용 예시 생성 (발견된 파일 정보 기반)"""
        examples = {}
        
        # 분석 결과에서 파라미터 추출
        max_length = 512
        special_tokens = {}
        tokenizer_class = "AutoTokenizer"
        
        if analysis_results:
            # tokenizer_config.json에서 정보 추출
            if 'tokenizer_config.json' in analysis_results:
                tokenizer_data = analysis_results['tokenizer_config.json']
                max_length = tokenizer_data.get('model_max_length', 512)
                tokenizer_class = tokenizer_data.get('tokenizer_class', 'AutoTokenizer')
            
            # special_tokens_map.json에서 특수 토큰 정보 추출
            if 'special_tokens_map.json' in analysis_results:
                special_tokens = analysis_results['special_tokens_map.json']
        
        for task in tasks:
            if task == 'text-classification':
                examples[task] = {
                    'description': '텍스트 분류 (감정 분석, 스팸 감지 등)',
                    'example_code': f'''from transformers import pipeline
classifier = pipeline("text-classification", model="model_name")
result = classifier("I love this product!", max_length={max_length}, truncation=True)
print(result)''',
                    'example_input': '"I love this product!"',
                    'expected_output': '[{"label": "POSITIVE", "score": 0.9998}]',
                    'parameters': {
                        'max_length': max_length,
                        'truncation': True,
                        'padding': True,
                        'special_tokens': special_tokens
                    }
                }
            elif task == 'token-classification':
                examples[task] = {
                    'description': '토큰 분류 (NER, POS 태깅 등)',
                    'example_code': f'''from transformers import pipeline
ner = pipeline("ner", model="model_name", aggregation_strategy="simple")
result = ner("My name is John and I live in Seoul.", max_length={max_length}, truncation=True)
print(result)''',
                    'example_input': '"My name is John and I live in Seoul."',
                    'expected_output': '[{"entity_group": "PER", "score": 0.99, "word": "John", "start": 11, "end": 15}]',
                    'parameters': {
                        'max_length': max_length,
                        'aggregation_strategy': 'simple',
                        'truncation': True,
                        'special_tokens': special_tokens
                    }
                }
            elif task == 'question-answering':
                examples[task] = {
                    'description': '질문 답변 (문맥 기반 답변 생성)',
                    'example_code': '''from transformers import pipeline
qa = pipeline("question-answering", model="model_name")
result = qa(question="What is AI?", context="AI is artificial intelligence...")
print(result)''',
                    'example_input': 'question="What is AI?", context="AI is artificial intelligence..."',
                    'expected_output': '{"answer": "artificial intelligence", "score": 0.95, "start": 6, "end": 28}'
                }
            elif task == 'text-generation':
                examples[task] = {
                    'description': '텍스트 생성 (문장 완성, 창작 등)',
                    'example_code': '''from transformers import pipeline
generator = pipeline("text-generation", model="model_name")
result = generator("The future of AI is", max_length=50)
print(result)''',
                    'example_input': '"The future of AI is"',
                    'expected_output': '[{"generated_text": "The future of AI is bright and full of possibilities..."}]'
                }
            elif task == 'fill-mask':
                examples[task] = {
                    'description': '빈칸 채우기 (마스크된 토큰 예측)',
                    'example_code': '''from transformers import pipeline
fill_mask = pipeline("fill-mask", model="model_name")
result = fill_mask("The weather is [MASK] today.")
print(result)''',
                    'example_input': '"The weather is [MASK] today."',
                    'expected_output': '[{"sequence": "The weather is nice today.", "score": 0.8, "token": 1234}]'
                }
            elif task == 'text2text-generation':
                examples[task] = {
                    'description': '텍스트 간 변환 (번역, 요약 등)',
                    'example_code': '''from transformers import pipeline
generator = pipeline("text2text-generation", model="model_name")
result = generator("translate English to Korean: Hello world")
print(result)''',
                    'example_input': '"translate English to Korean: Hello world"',
                    'expected_output': '[{"generated_text": "안녕하세요 세계"}]'
                }
        
        return examples
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """분석 결과를 바탕으로 권장사항 생성"""
        recommendations = []
        
        # 필수 파일 누락 체크
        essential_files = ['config.json']
        for file in essential_files:
            if file in analysis['files_missing']:
                recommendations.append(f"Essential file {file} is missing")
        
        # 모델 크기 관련 권장사항
        if 'model_summary' in analysis:
            model_size = analysis['model_summary'].get('model_size_mb', 0)
            if model_size > 1000:  # 1GB 이상
                recommendations.append("Large model detected. Consider using model sharding or quantization")
            
            params = analysis['model_summary'].get('total_parameters', 0)
            if params > 1000000000:  # 1B 이상
                recommendations.append("Model has over 1B parameters. GPU with sufficient VRAM recommended")
        
        return recommendations
    
    def _infer_tasks_from_config(self, config: Dict) -> List[str]:
        """config에서 지원 가능한 태스크 추론"""
        architectures = config.get('architectures', [])
        tasks = []
        
        for arch in architectures:
            if 'ForSequenceClassification' in arch:
                tasks.append('text-classification')
            elif 'ForTokenClassification' in arch:
                tasks.append('token-classification')
            elif 'ForQuestionAnswering' in arch:
                tasks.append('question-answering')
            elif 'ForCausalLM' in arch:
                tasks.append('text-generation')
            elif 'ForMaskedLM' in arch:
                tasks.append('fill-mask')
            elif 'ForConditionalGeneration' in arch:
                tasks.append('text2text-generation')
        
        return tasks
    
    def _estimate_parameters(self, config: Dict) -> int:
        """config에서 파라미터 수 추정"""
        hidden_size = config.get('hidden_size', 0)
        num_layers = config.get('num_hidden_layers', 0)
        vocab_size = config.get('vocab_size', 0)
        
        if hidden_size and num_layers and vocab_size:
            # 간단한 추정 공식 (실제와 다를 수 있음)
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (hidden_size * hidden_size * 4)  # 간단한 추정
            return embedding_params + layer_params
        
        return 0