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
            'detailed_config': {},
            'embedding_info': {},  # 임베딩 모델 전용 정보
            'model_capabilities': []  # 모델 능력 리스트
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
        
        # 임베딩 모델 정보 추출
        if 'feature-extraction' in summary['supported_tasks']:
            summary['embedding_info'] = self._extract_embedding_info(summary, analysis_results)
        
        # 모델 능력 분석
        summary['model_capabilities'] = self._analyze_model_capabilities(summary, analysis_results)
        
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
                    'example_code': f'''# 1. 직접 사용
from transformers import pipeline
classifier = pipeline("text-classification", model="model_name")
result = classifier("I love this product!", max_length={max_length}, truncation=True)
print(result)

# 2. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/classify", 
                        json={{"text": "I love this product!"}})
print(response.json())

# 3. 서버 API 사용 (내부 IP)
import requests
response = requests.post("http://172.28.177.205:8000/classify", 
                        json={{"text": "I love this product!"}})
print(response.json())''',
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
                    'example_code': f'''# 1. 직접 사용
from transformers import pipeline
ner = pipeline("ner", model="model_name", aggregation_strategy="simple")
result = ner("My name is John and I live in Seoul.", max_length={max_length}, truncation=True)
print(result)

# 2. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/ner", 
                        json={{"text": "My name is John and I live in Seoul."}})
print(response.json())

# 3. 서버 API 사용 (내부 IP)
import requests
response = requests.post("http://172.28.177.205:8000/ner", 
                        json={{"text": "My name is John and I live in Seoul."}})
print(response.json())''',
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
                    'example_code': '''# 1. 직접 사용
from transformers import pipeline
qa = pipeline("question-answering", model="model_name")
result = qa(question="What is AI?", context="AI is artificial intelligence...")
print(result)

# 2. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/qa", 
                        json={"question": "What is AI?", "context": "AI is artificial intelligence..."})
print(response.json())

# 3. 서버 API 사용 (내부 IP)
import requests
response = requests.post("http://172.28.177.205:8000/qa", 
                        json={"question": "What is AI?", "context": "AI is artificial intelligence..."})
print(response.json())''',
                    'example_input': 'question="What is AI?", context="AI is artificial intelligence..."',
                    'expected_output': '{"answer": "artificial intelligence", "score": 0.95, "start": 6, "end": 28}'
                }
            elif task == 'text-generation':
                examples[task] = {
                    'description': '텍스트 생성 (문장 완성, 창작 등)',
                    'example_code': '''# 1. 직접 사용
from transformers import pipeline
generator = pipeline("text-generation", model="model_name")
result = generator("The future of AI is", max_length=50)
print(result)

# 2. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/generate", 
                        json={"text": "The future of AI is", "max_length": 50})
print(response.json())

# 3. 서버 API 사용 (내부 IP)
import requests
response = requests.post("http://172.28.177.205:8000/generate", 
                        json={"text": "The future of AI is", "max_length": 50})
print(response.json())''',
                    'example_input': '"The future of AI is"',
                    'expected_output': '[{"generated_text": "The future of AI is bright and full of possibilities..."}]'
                }
            elif task == 'fill-mask':
                examples[task] = {
                    'description': '빈칸 채우기 (마스크된 토큰 예측)',
                    'example_code': '''# 1. 직접 사용
from transformers import pipeline
fill_mask = pipeline("fill-mask", model="model_name")
result = fill_mask("The weather is [MASK] today.")
print(result)

# 2. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/fill-mask", 
                        json={"text": "The weather is [MASK] today."})
print(response.json())

# 3. 서버 API 사용 (내부 IP)
import requests
response = requests.post("http://172.28.177.205:8000/fill-mask", 
                        json={"text": "The weather is [MASK] today."})
print(response.json())''',
                    'example_input': '"The weather is [MASK] today."',
                    'expected_output': '[{"sequence": "The weather is nice today.", "score": 0.8, "token": 1234}]'
                }
            elif task == 'text2text-generation':
                examples[task] = {
                    'description': '텍스트 간 변환 (번역, 요약 등)',
                    'example_code': '''# 1. 직접 사용
from transformers import pipeline
generator = pipeline("text2text-generation", model="model_name")
result = generator("translate English to Korean: Hello world")
print(result)

# 2. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/text2text", 
                        json={"text": "translate English to Korean: Hello world"})
print(response.json())

# 3. 서버 API 사용 (내부 IP)
import requests
response = requests.post("http://172.28.177.205:8000/text2text", 
                        json={"text": "translate English to Korean: Hello world"})
print(response.json())''',
                    'example_input': '"translate English to Korean: Hello world"',
                    'expected_output': '[{"generated_text": "안녕하세요 세계"}]'
                }
            elif task == 'feature-extraction':
                examples[task] = {
                    'description': '텍스트 임베딩 생성 (의미적 유사도, 검색, 클러스터링)',
                    'example_code': f'''# 1. 직접 사용 (임베딩 생성)
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("model_name")
model = AutoModel.from_pretrained("model_name")

def get_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, 
                      max_length={max_length}, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS 토큰 또는 평균 풀링 사용
        embeddings = outputs.last_hidden_state[:, 0, :]  # CLS 토큰
        # 또는 평균 풀링: embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# 사용 예시
texts = ["안녕하세요", "Hello world", "こんにちは"]
embeddings = get_embeddings(texts)
print(f"임베딩 크기: {{embeddings.shape}}")

# 2. 유사도 계산
def calculate_similarity(text1, text2):
    emb1 = get_embeddings([text1])
    emb2 = get_embeddings([text2])
    similarity = F.cosine_similarity(emb1, emb2)
    return similarity.item()

similarity = calculate_similarity("안녕하세요", "Hello")
print(f"유사도: {{similarity:.4f}}")

# 3. 서버 API 사용 (로컬)
import requests
response = requests.post("http://127.0.0.1:8000/models/model_name/predict", 
                        json={{"text": "Hello world"}})
print(response.json())

# 4. 의미적 검색 시스템 구축
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "인공지능은 미래 기술입니다",
    "머신러닝은 데이터 과학의 핵심입니다", 
    "자연어 처리는 AI의 중요한 분야입니다"
]

# 문서 임베딩 생성
doc_embeddings = get_embeddings(documents)

# 검색 쿼리
query = "AI 기술에 대해 알고 싶습니다"
query_embedding = get_embeddings([query])

# 유사도 계산 및 랭킹
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
ranked_docs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

print("검색 결과:")
for idx, score in ranked_docs:
    print(f"문서 {{idx+1}}: {{documents[idx]}} (유사도: {{score:.4f}})")''',
                    'example_input': '"Hello world"',
                    'expected_output': 'Array shape: (1, 1024) - 1024차원 임베딩 벡터',
                    'parameters': {
                        'max_length': max_length,
                        'padding': True,
                        'truncation': True,
                        'return_tensors': 'pt',
                        'pooling_method': 'cls_token or mean_pooling',
                        'similarity_function': 'cosine_similarity',
                        'special_tokens': special_tokens
                    }
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
        model_type = config.get('model_type', '').lower()
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
            elif arch in ['XLMRobertaModel', 'BertModel', 'RobertaModel', 'DistilBertModel', 'ElectraModel']:
                # 임베딩 모델 감지
                tasks.append('feature-extraction')
        
        # 모델 타입 기반 추가 감지
        if not tasks:
            if any(embedding_keyword in model_type for embedding_keyword in ['embed', 'bge', 'sentence', 'retrieval']):
                tasks.append('feature-extraction')
            elif 'bert' in model_type or 'roberta' in model_type:
                tasks.append('feature-extraction')
        
        return tasks
    
    def _extract_embedding_info(self, summary: Dict, analysis_results: Dict) -> Dict[str, Any]:
        """임베딩 모델 전용 정보 추출"""
        embedding_info = {
            'embedding_dimension': summary['detailed_config'].get('hidden_size', 'N/A'),
            'max_sequence_length': summary['max_sequence_length'],
            'pooling_method': 'mean',  # 기본값, 실제로는 모델에 따라 다름
            'similarity_function': 'cosine',  # 기본값
            'multi_lingual': False,
            'supported_languages': [],
            'use_cases': []
        }
        
        # 모델 타입 기반 임베딩 정보 추론
        model_type = summary['model_type'].lower()
        model_name = summary.get('model_name', '').lower()
        
        # 다국어 지원 감지
        if any(keyword in model_type for keyword in ['xlm', 'multilingual', 'mbert']) or \
           any(keyword in model_name for keyword in ['multilingual', 'bge-m3', 'xlm']):
            embedding_info['multi_lingual'] = True
            embedding_info['supported_languages'] = ['english', 'chinese', 'japanese', 'korean', 'multilingual']
        
        # BGE-M3 특화 정보
        if 'bge-m3' in model_name or 'bge-m3' in model_type:
            embedding_info.update({
                'model_family': 'BGE (Beijing Academy of Artificial Intelligence)',
                'specialization': 'Multi-lingual dense retrieval',
                'pooling_method': 'cls_pooling',
                'similarity_function': 'cosine',
                'max_sequence_length': 8192,
                'supported_languages': ['english', 'chinese', 'japanese', 'korean', 'multilingual'],
                'use_cases': [
                    'Semantic search',
                    'Document retrieval',
                    'Cross-lingual similarity',
                    'Text clustering',
                    'Recommendation systems'
                ]
            })
        
        # 사용 사례 추가
        if not embedding_info['use_cases']:
            embedding_info['use_cases'] = [
                'Sentence similarity',
                'Semantic search',
                'Text clustering',
                'Information retrieval'
            ]
        
        return embedding_info
    
    def _analyze_model_capabilities(self, summary: Dict, analysis_results: Dict) -> List[str]:
        """모델 능력 분석"""
        capabilities = []
        
        # 기본 능력
        if summary['supported_tasks']:
            for task in summary['supported_tasks']:
                if task == 'text-classification':
                    capabilities.append('텍스트 분류 (감정 분석, 스팸 감지 등)')
                elif task == 'feature-extraction':
                    capabilities.append('텍스트 임베딩 생성')
                    capabilities.append('의미적 유사도 계산')
                elif task == 'text-generation':
                    capabilities.append('텍스트 생성')
                elif task == 'question-answering':
                    capabilities.append('질문 답변')
                elif task == 'fill-mask':
                    capabilities.append('빈 칸 채우기')
        
        # 모델 크기 기반 능력
        if summary['total_parameters'] > 1000000000:  # 1B 이상
            capabilities.append('대규모 언어 모델 (복잡한 추론 가능)')
        elif summary['total_parameters'] > 100000000:  # 100M 이상
            capabilities.append('중대형 모델 (일반적인 NLP 태스크 처리)')
        
        # 시퀀스 길이 기반 능력
        if summary['max_sequence_length'] > 4000:
            capabilities.append('긴 문서 처리 가능')
        elif summary['max_sequence_length'] > 1000:
            capabilities.append('중간 길이 텍스트 처리')
        
        # 임베딩 모델 특화 능력
        if 'feature-extraction' in summary['supported_tasks']:
            capabilities.append('의미적 검색 시스템 구축')
            capabilities.append('문서 클러스터링')
            capabilities.append('추천 시스템 구축')
            
            # 다국어 지원 확인
            if summary.get('embedding_info', {}).get('multi_lingual', False):
                capabilities.append('다국어 텍스트 처리')
                capabilities.append('교차 언어 검색')
        
        return capabilities
    
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