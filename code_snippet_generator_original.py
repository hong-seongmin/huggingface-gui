"""
코드 스니펫 생성 엔진
감지된 모델 정보를 바탕으로 최적화된 코드 스니펫을 자동 생성
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from model_database import model_database, ModelCategory, TaskType, ModelTypeInfo
from model_type_detector import ModelTypeDetector

class SnippetType(Enum):
    """스니펫 타입 분류"""
    LOADING = "loading"
    INFERENCE = "inference"
    BATCH_INFERENCE = "batch_inference"
    FINE_TUNING = "fine_tuning"
    OPTIMIZATION = "optimization"
    DEPLOYMENT = "deployment"

@dataclass
class CodeSnippet:
    """코드 스니펫 정보"""
    title: str
    description: str
    code: str
    language: str
    snippet_type: SnippetType
    requirements: List[str]
    notes: List[str]
    
class CodeSnippetGenerator:
    """코드 스니펫 생성기"""
    
    def __init__(self):
        self.model_detector = ModelTypeDetector()
        
    def generate_snippets(self, model_name: str, model_path: str, model_id: str = None) -> Dict[str, CodeSnippet]:
        """모델에 대한 모든 타입의 스니펫 생성"""
        
        # 모델 타입 감지
        task_type, model_class, analysis_info = self.model_detector.detect_model_type(model_name, model_path)
        
        # 모델 정보 가져오기
        detected_model_type = self._extract_model_type_from_path(model_path)
        model_info = model_database.get_model_info(detected_model_type) if detected_model_type else None
        
        # 모델 ID 결정 (HuggingFace Hub ID 또는 로컬 경로)
        if not model_id:
            model_id = model_name if self._is_huggingface_model_id(model_name) else model_path
        
        snippets = {}
        
        # 로딩 스니펫
        snippets["loading"] = self._generate_loading_snippet(
            model_id, task_type, model_class, model_info
        )
        
        # 추론 스니펫
        snippets["inference"] = self._generate_inference_snippet(
            model_id, task_type, model_class, model_info
        )
        
        # 배치 추론 스니펫
        snippets["batch_inference"] = self._generate_batch_inference_snippet(
            model_id, task_type, model_class, model_info
        )
        
        # 최적화 스니펫
        snippets["optimization"] = self._generate_optimization_snippet(
            model_id, task_type, model_class, model_info
        )
        
        # 파인튜닝 스니펫 (해당하는 경우)
        if self._supports_fine_tuning(task_type):
            snippets["fine_tuning"] = self._generate_fine_tuning_snippet(
                model_id, task_type, model_class, model_info
            )
        
        return snippets
    
    def _generate_loading_snippet(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """모델 로딩 스니펫 생성"""
        
        # 모델 카테고리별 최적화된 로딩 코드
        if model_info and model_info.category == ModelCategory.MULTIMODAL:
            return self._generate_multimodal_loading_snippet(model_id, task_type, model_class, model_info)
        elif model_info and model_info.category == ModelCategory.VISION:
            return self._generate_vision_loading_snippet(model_id, task_type, model_class, model_info)
        elif model_info and model_info.category == ModelCategory.AUDIO:
            return self._generate_audio_loading_snippet(model_id, task_type, model_class, model_info)
        else:
            return self._generate_text_loading_snippet(model_id, task_type, model_class, model_info)
    
    def _generate_text_loading_snippet(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """텍스트 모델 로딩 스니펫"""
        
        # 모델 클래스 결정
        auto_model_class = model_class if model_class.startswith("Auto") else "AutoModel"
        
        # 최적화 파라미터 결정
        optimization_params = self._get_optimization_parameters(model_info)
        
        code = f'''import torch
from transformers import {auto_model_class}, AutoTokenizer

# 모델과 토크나이저 로드
model_name = "{model_id}"

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

# 모델 로드
model = {auto_model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.{optimization_params['dtype']},
    device_map="{optimization_params['device_map']}"
)

# 평가 모드로 설정
model.eval()

print(f"모델 로드 완료: {{model.__class__.__name__}}")
print(f"모델 파라미터 수: {{model.num_parameters():,}}")'''

        requirements = ["torch", "transformers"]
        if optimization_params.get('quantization'):
            requirements.append("bitsandbytes")
        
        notes = [
            f"태스크 타입: {task_type}",
            f"모델 클래스: {auto_model_class}",
            "GPU 사용 시 더 빠른 추론 가능",
            "trust_remote_code=True로 커스텀 코드 허용"
        ]
        
        if model_info:
            notes.append(f"모델 카테고리: {model_info.category.value}")
            notes.append(f"지원 태스크: {', '.join([task.value for task in model_info.primary_tasks])}")
        
        return CodeSnippet(
            title=f"{model_id} 모델 로딩",
            description=f"{task_type} 태스크를 위한 {model_id} 모델 로딩 코드",
            code=code,
            language="python",
            snippet_type=SnippetType.LOADING,
            requirements=requirements,
            notes=notes
        )
    
    def _generate_inference_snippet(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """추론 스니펫 생성"""
        
        # 태스크별 추론 코드 생성
        if task_type == "text-classification":
            return self._generate_text_classification_inference(model_id, model_class, model_info)
        elif task_type == "token-classification":
            return self._generate_token_classification_inference(model_id, model_class, model_info)
        elif task_type == "text-generation":
            return self._generate_text_generation_inference(model_id, model_class, model_info)
        elif task_type == "question-answering":
            return self._generate_question_answering_inference(model_id, model_class, model_info)
        elif task_type == "feature-extraction":
            return self._generate_feature_extraction_inference(model_id, model_class, model_info)
        else:
            return self._generate_generic_inference(model_id, task_type, model_class, model_info)
    
    def _generate_text_classification_inference(self, model_id: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """텍스트 분류 추론 스니펫"""
        
        code = f'''import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 모델과 토크나이저 로드 (위의 로딩 코드 참조)
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 입력 텍스트
text = "이 영화는 정말 재미있고 감동적이었습니다."

# 토큰화
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

# 추론 실행
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# 결과 처리
predicted_class_id = predictions.argmax().item()
confidence = predictions.max().item()

# 레이블 매핑 (있는 경우)
if hasattr(model.config, 'id2label'):
    label = model.config.id2label[predicted_class_id]
    print(f"예측 결과: {{label}} (신뢰도: {{confidence:.3f}})")
else:
    print(f"예측 클래스: {{predicted_class_id}} (신뢰도: {{confidence:.3f}})")

# 모든 클래스 확률 출력
for i, prob in enumerate(predictions[0]):
    if hasattr(model.config, 'id2label'):
        label = model.config.id2label[i]
        print(f"{{label}}: {{prob:.3f}}")
    else:
        print(f"클래스 {{i}}: {{prob:.3f}}")'''

        return CodeSnippet(
            title="텍스트 분류 추론",
            description=f"{model_id} 모델을 사용한 텍스트 분류 추론",
            code=code,
            language="python",
            snippet_type=SnippetType.INFERENCE,
            requirements=["torch", "transformers"],
            notes=[
                "입력 텍스트를 적절히 수정하여 사용",
                "max_length는 모델의 최대 시퀀스 길이에 맞게 조정",
                "GPU 사용 시 .to('cuda') 추가",
                "배치 처리를 위해서는 batch_inference 스니펫 참조"
            ]
        )
    
    def _generate_token_classification_inference(self, model_id: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """토큰 분류 추론 스니펫"""
        
        code = f'''import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

# 모델과 토크나이저 로드
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 입력 텍스트
text = "삼성전자는 한국의 대표적인 전자 기업입니다."

# 토큰화
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

# 추론 실행
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

# 결과 처리
predicted_token_class_ids = outputs.logits.argmax(dim=-1)

# 토큰별 예측 결과 출력
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
predictions_list = predicted_token_class_ids[0].tolist()

print("토큰별 예측 결과:")
for token, prediction in zip(tokens, predictions_list):
    if hasattr(model.config, 'id2label'):
        label = model.config.id2label[prediction]
        print(f"{{token:>15}} -> {{label}}")
    else:
        print(f"{{token:>15}} -> {{prediction}}")

# 엔티티 추출 (BIO 태깅인 경우)
if hasattr(model.config, 'id2label'):
    entities = []
    current_entity = None
    current_tokens = []
    
    for i, (token, pred_id) in enumerate(zip(tokens, predictions_list)):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        label = model.config.id2label[pred_id]
        
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_entity, ' '.join(current_tokens)))
            current_entity = label[2:]
            current_tokens = [token]
        elif label.startswith('I-') and current_entity:
            current_tokens.append(token)
        else:
            if current_entity:
                entities.append((current_entity, ' '.join(current_tokens)))
                current_entity = None
                current_tokens = []
    
    if current_entity:
        entities.append((current_entity, ' '.join(current_tokens)))
    
    print("\\n추출된 엔티티:")
    for entity_type, entity_text in entities:
        print(f"{{entity_type}}: {{entity_text}}")'''

        return CodeSnippet(
            title="토큰 분류 (NER) 추론",
            description=f"{model_id} 모델을 사용한 토큰 분류 및 개체명 인식",
            code=code,
            language="python",
            snippet_type=SnippetType.INFERENCE,
            requirements=["torch", "transformers"],
            notes=[
                "BIO 태깅 방식의 NER 모델에 최적화",
                "엔티티 추출 로직은 모델의 레이블 체계에 따라 수정 필요",
                "한국어 모델의 경우 토큰화 결과 확인 필요",
                "실제 사용 시 후처리 로직 추가 권장"
            ]
        )
    
    def _generate_text_generation_inference(self, model_id: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """텍스트 생성 추론 스니펫"""
        
        code = f'''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 패딩 토큰 설정 (필요한 경우)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 입력 텍스트 (프롬프트)
prompt = "인공지능의 미래에 대해 설명해주세요."

# 토큰화
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# 생성 파라미터 설정
generation_config = {{
    "max_new_tokens": 200,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9,
    "top_k": 50,
    "repetition_penalty": 1.1,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}}

# 텍스트 생성
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        **generation_config
    )

# 결과 디코딩
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# 프롬프트 부분 제거하여 생성된 텍스트만 추출
if generated_text.startswith(prompt):
    generated_text = generated_text[len(prompt):].strip()

print("생성된 텍스트:")
print(generated_text)

# 스트리밍 생성 (실시간 출력)
print("\\n스트리밍 생성:")
inputs = tokenizer(prompt, return_tensors="pt")
    
with torch.no_grad():
    for i in range(generation_config["max_new_tokens"]):
        outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=generation_config["temperature"],
            do_sample=generation_config["do_sample"],
            top_p=generation_config["top_p"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        new_token = tokenizer.decode(outputs[0][-1], skip_special_tokens=True)
        print(new_token, end="", flush=True)
        
        if outputs[0][-1] == tokenizer.eos_token_id:
            break
            
        inputs = {{"input_ids": outputs, "attention_mask": torch.ones_like(outputs)}}'''

        return CodeSnippet(
            title="텍스트 생성 추론",
            description=f"{model_id} 모델을 사용한 텍스트 생성",
            code=code,
            language="python",
            snippet_type=SnippetType.INFERENCE,
            requirements=["torch", "transformers"],
            notes=[
                "생성 파라미터는 모델과 태스크에 따라 조정",
                "temperature: 창의성 제어 (0.1~2.0)",
                "top_p: 다음 토큰 선택 범위 제어",
                "repetition_penalty: 반복 방지",
                "GPU 메모리 부족 시 device_map 조정"
            ]
        )
    
    def _generate_feature_extraction_inference(self, model_id: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """특성 추출 추론 스니펫"""
        
        code = f'''import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

# 모델과 토크나이저 로드
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 입력 텍스트
texts = [
    "이 영화는 정말 재미있었습니다.",
    "날씨가 좋아서 기분이 좋습니다.",
    "AI 기술의 발전이 놀랍습니다."
]

# 특성 추출 함수
def extract_features(texts, model, tokenizer):
    # 토큰화
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # 추론 실행
    with torch.no_grad():
        outputs = model(**inputs)
        
        # 다양한 특성 추출 방법
        features = {{}}
        
        # 1. [CLS] 토큰 특성 (문장 수준)
        features['cls_features'] = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        # 2. 평균 풀링 (모든 토큰의 평균)
        attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size())
        sum_embeddings = torch.sum(outputs.last_hidden_state * attention_mask, dim=1)
        sum_mask = torch.sum(attention_mask, dim=1)
        features['mean_features'] = (sum_embeddings / sum_mask).cpu().numpy()
        
        # 3. 최대 풀링
        features['max_features'] = torch.max(outputs.last_hidden_state, dim=1)[0].cpu().numpy()
        
        # 4. 전체 시퀀스 특성 (토큰별)
        features['sequence_features'] = outputs.last_hidden_state.cpu().numpy()
        
        return features

# 특성 추출 실행
features = extract_features(texts, model, tokenizer)

# 결과 출력
print("특성 추출 결과:")
for feature_type, feature_vectors in features.items():
    print(f"{{feature_type}}: {{feature_vectors.shape}}")

# 유사도 계산 예제 (코사인 유사도)
from sklearn.metrics.pairwise import cosine_similarity

print("\\n코사인 유사도 매트릭스 ([CLS] 토큰 기준):")
similarity_matrix = cosine_similarity(features['cls_features'])
print(similarity_matrix)

# 가장 유사한 문장 쌍 찾기
max_similarity = 0
best_pair = None
for i in range(len(texts)):
    for j in range(i+1, len(texts)):
        sim = similarity_matrix[i][j]
        if sim > max_similarity:
            max_similarity = sim
            best_pair = (i, j)

if best_pair:
    print(f"\\n가장 유사한 문장 쌍 (유사도: {{max_similarity:.3f}}):")
    print(f"1: {{texts[best_pair[0]]}}")
    print(f"2: {{texts[best_pair[1]]}}")

# 특성 벡터 저장
np.save('extracted_features.npy', features['cls_features'])
print("\\n특성 벡터를 'extracted_features.npy'에 저장했습니다.")'''

        return CodeSnippet(
            title="특성 추출 (임베딩)",
            description=f"{model_id} 모델을 사용한 텍스트 특성 추출",
            code=code,
            language="python",
            snippet_type=SnippetType.INFERENCE,
            requirements=["torch", "transformers", "numpy", "scikit-learn"],
            notes=[
                "다양한 풀링 방법 제공 (CLS, 평균, 최대)",
                "유사도 계산 및 클러스터링에 활용 가능",
                "임베딩 벡터는 검색, 추천 시스템에 활용",
                "배치 처리로 성능 최적화 가능"
            ]
        )
    
    def _generate_batch_inference_snippet(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """배치 추론 스니펫"""
        
        code = f'''import torch
from transformers import {model_class}, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# 커스텀 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
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
        return {{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'text': text
        }}

# 모델과 토크나이저 로드
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 배치 추론할 텍스트 데이터
texts = [
    "이 제품은 정말 훌륭합니다.",
    "서비스가 매우 불만족스럽습니다.",
    "가격 대비 성능이 좋습니다.",
    "배송이 너무 느려서 실망했습니다.",
    "품질이 기대보다 훨씬 좋네요."
    # ... 더 많은 텍스트 추가 가능
]

# 데이터셋 및 데이터로더 생성
dataset = TextDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

# 배치 추론 실행
results = []
model.eval()

with torch.no_grad():
    for batch in tqdm(dataloader, desc="배치 추론 진행"):
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # 태스크별 결과 처리
        if "{task_type}" == "text-classification":
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_classes = predictions.argmax(dim=-1).cpu().numpy()
            confidences = predictions.max(dim=-1)[0].cpu().numpy()
            
            for i, (pred_class, confidence, text) in enumerate(zip(predicted_classes, confidences, batch['text'])):
                results.append({{
                    'text': text,
                    'predicted_class': pred_class,
                    'confidence': confidence,
                    'label': model.config.id2label.get(pred_class, f"Class {{pred_class}}") if hasattr(model.config, 'id2label') else f"Class {{pred_class}}"
                }})
        
        elif "{task_type}" == "feature-extraction":
            # CLS 토큰 특성 추출
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            for i, (feature, text) in enumerate(zip(features, batch['text'])):
                results.append({{
                    'text': text,
                    'features': feature,
                    'feature_dim': feature.shape[0]
                }})
        
        # 다른 태스크들도 필요에 따라 추가...

# 결과 출력
print(f"총 {{len(results)}}개 텍스트 처리 완료")
for i, result in enumerate(results[:5]):  # 처음 5개만 출력
    print(f"{{i+1}}. {{result['text'][:30]}}...")
    if 'label' in result:
        print(f"   예측: {{result['label']}} (신뢰도: {{result['confidence']:.3f}})")
    elif 'features' in result:
        print(f"   특성 차원: {{result['feature_dim']}}")
    print()

# 결과를 파일로 저장
import json
with open('batch_results.json', 'w', encoding='utf-8') as f:
    # NumPy 배열을 JSON 호환 형태로 변환
    json_results = []
    for result in results:
        json_result = result.copy()
        if 'features' in json_result:
            json_result['features'] = json_result['features'].tolist()
        json_results.append(json_result)
    
    json.dump(json_results, f, ensure_ascii=False, indent=2)

print("결과를 'batch_results.json'에 저장했습니다.")'''

        return CodeSnippet(
            title="배치 추론",
            description=f"{model_id} 모델을 사용한 대용량 배치 추론",
            code=code,
            language="python",
            snippet_type=SnippetType.BATCH_INFERENCE,
            requirements=["torch", "transformers", "numpy", "tqdm"],
            notes=[
                "배치 크기는 GPU 메모리에 맞게 조정",
                "대용량 데이터 처리 시 메모리 효율성 중요",
                "progress bar로 진행 상황 모니터링",
                "결과를 JSON 형태로 저장하여 재사용 가능"
            ]
        )
    
    def _generate_optimization_snippet(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """최적화 스니펫"""
        
        optimization_params = self._get_optimization_parameters(model_info)
        
        code = f'''import torch
from transformers import {model_class}, AutoTokenizer
import time

# 성능 최적화된 모델 로딩
model_name = "{model_id}"

# 1. 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 2. 최적화된 모델 로드
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.{optimization_params['dtype']},  # 메모리 효율성
    device_map="{optimization_params['device_map']}",  # 자동 디바이스 매핑
    low_cpu_mem_usage=True,  # CPU 메모리 사용량 감소
    use_safetensors=True     # 안전한 텐서 로딩
)

# 3. 컴파일 최적화 (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    model = torch.compile(model, mode="reduce-overhead")
    print("모델 컴파일 최적화 적용됨")

# 4. 추론 모드 설정
model.eval()
model.requires_grad_(False)

# 5. 메모리 사용량 확인
if torch.cuda.is_available():
    print(f"GPU 메모리 사용량: {{torch.cuda.memory_allocated() / 1024**3:.2f}}GB")
    print(f"GPU 메모리 캐시: {{torch.cuda.memory_reserved() / 1024**3:.2f}}GB")

# 6. 벤치마킹 함수
def benchmark_inference(model, tokenizer, text, num_runs=100):
    \"\"\"추론 성능 벤치마크\"\"\"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # GPU로 이동 (필요한 경우)
    if torch.cuda.is_available():
        inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    # 워밍업
    with torch.no_grad():
        for _ in range(10):
            _ = model(**inputs)
    
    # 실제 벤치마킹
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            outputs = model(**inputs)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time, outputs

# 7. 성능 테스트
test_text = "이것은 성능 테스트를 위한 샘플 텍스트입니다."
avg_time, outputs = benchmark_inference(model, tokenizer, test_text)

print(f"평균 추론 시간: {{avg_time*1000:.2f}}ms")
print(f"초당 처리 가능: {{1/avg_time:.1f}} 샘플/초")

# 8. 메모리 최적화 팁
def optimize_memory():
    \"\"\"메모리 사용량 최적화\"\"\"
    # 그래디언트 계산 비활성화
    torch.set_grad_enabled(False)
    
    # 자동 mixed precision (AMP) 사용
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # 메모리 정리
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("메모리 최적화 설정 완료")

optimize_memory()

# 9. 추론 최적화 예제
@torch.no_grad()
def optimized_inference(text):
    \"\"\"최적화된 추론 함수\"\"\"
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # GPU로 이동
    if torch.cuda.is_available():
        inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    # 자동 mixed precision으로 추론
    with torch.cuda.amp.autocast():
        outputs = model(**inputs)
    
    return outputs

# 사용 예제
result = optimized_inference("최적화된 추론 테스트 텍스트")
print(f"출력 형태: {{result.logits.shape if hasattr(result, 'logits') else 'N/A'}}")

# 10. 성능 모니터링
print("\\n=== 성능 요약 ===")
print(f"모델: {{model.__class__.__name__}}")
print(f"파라미터 수: {{sum(p.numel() for p in model.parameters()):,}}")
print(f"데이터 타입: {{optimization_params['dtype']}}")
print(f"디바이스: {{next(model.parameters()).device}}")
if torch.cuda.is_available():
    print(f"GPU 메모리: {{torch.cuda.memory_allocated() / 1024**3:.2f}}GB")'''

        return CodeSnippet(
            title="성능 최적화",
            description=f"{model_id} 모델의 추론 성능 최적화",
            code=code,
            language="python",
            snippet_type=SnippetType.OPTIMIZATION,
            requirements=["torch", "transformers"],
            notes=[
                "PyTorch 2.0+ torch.compile 사용 권장",
                "GPU 메모리 부족 시 device_map 조정",
                "float16 사용으로 메모리 사용량 절반 감소",
                "배치 크기 조정으로 처리량 최적화",
                "프로덕션 환경에서는 TorchServe 등 고려"
            ]
        )
    
    def _generate_multimodal_loading_snippet(self, model_id: str, task_type: str, model_class: str, model_info: ModelTypeInfo) -> CodeSnippet:
        """멀티모달 모델 로딩 스니펫"""
        
        code = f'''import torch
from transformers import {model_class}, AutoTokenizer, AutoProcessor
from PIL import Image
import requests

# 멀티모달 모델 로딩
model_name = "{model_id}"

# 프로세서 또는 토크나이저 로드 (모델에 따라 다름)
try:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    print("프로세서 로드 완료")
except:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("토크나이저 로드 완료")

# 모델 로드
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 평가 모드 설정
model.eval()

print(f"멀티모달 모델 로드 완료: {{model.__class__.__name__}}")

# 이미지 로드 예제
def load_image_from_url(url):
    return Image.open(requests.get(url, stream=True).raw)

# 샘플 이미지와 텍스트
image_url = "https://example.com/sample_image.jpg"
text_prompt = "이 이미지에서 무엇을 볼 수 있나요?"

# 이미지 로드
try:
    image = load_image_from_url(image_url)
    print("이미지 로드 성공")
except:
    print("이미지 로드 실패 - 로컬 이미지 경로를 사용하세요")
    # image = Image.open("path/to/your/image.jpg")'''

        return CodeSnippet(
            title="멀티모달 모델 로딩",
            description=f"{model_id} 멀티모달 모델 로딩",
            code=code,
            language="python",
            snippet_type=SnippetType.LOADING,
            requirements=["torch", "transformers", "Pillow", "requests"],
            notes=[
                "AutoProcessor 사용 권장 (이미지+텍스트 처리)",
                "이미지 전처리 파이프라인 내장",
                "메모리 사용량이 큰 편이므로 GPU 필수",
                "이미지 해상도에 따라 성능 차이 발생"
            ]
        )
    
    def _generate_vision_loading_snippet(self, model_id: str, task_type: str, model_class: str, model_info: ModelTypeInfo) -> CodeSnippet:
        """비전 모델 로딩 스니펫"""
        
        code = f'''import torch
from transformers import {model_class}, AutoImageProcessor
from PIL import Image
import requests

# 비전 모델과 이미지 프로세서 로드
model_name = "{model_id}"

# 이미지 프로세서 로드
image_processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)

# 모델 로드
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 평가 모드 설정
model.eval()

print(f"비전 모델 로드 완료: {{model.__class__.__name__}}")
print(f"이미지 크기: {{image_processor.size}}")

# 이미지 로드 및 전처리 함수
def load_and_preprocess_image(image_path_or_url):
    # URL 또는 로컬 파일 로드
    if image_path_or_url.startswith(('http://', 'https://')):
        image = Image.open(requests.get(image_path_or_url, stream=True).raw)
    else:
        image = Image.open(image_path_or_url)
    
    # 이미지 전처리
    inputs = image_processor(image, return_tensors="pt")
    return inputs, image

# 샘플 이미지 로드
sample_image_url = "https://example.com/sample.jpg"
# 또는 로컬 파일: sample_image_path = "path/to/image.jpg"

print("이미지 로드 및 전처리 준비 완료")'''

        return CodeSnippet(
            title="비전 모델 로딩",
            description=f"{model_id} 비전 모델 로딩",
            code=code,
            language="python",
            snippet_type=SnippetType.LOADING,
            requirements=["torch", "transformers", "Pillow", "requests"],
            notes=[
                "AutoImageProcessor로 이미지 전처리 자동화",
                "이미지 크기 조정 및 정규화 자동 처리",
                "배치 처리 시 동일한 크기로 리사이징",
                "GPU 메모리 사용량 고려하여 배치 크기 조정"
            ]
        )
    
    def _generate_audio_loading_snippet(self, model_id: str, task_type: str, model_class: str, model_info: ModelTypeInfo) -> CodeSnippet:
        """오디오 모델 로딩 스니펫"""
        
        code = f'''import torch
from transformers import {model_class}, AutoFeatureExtractor
import librosa
import numpy as np

# 오디오 모델과 특성 추출기 로드
model_name = "{model_id}"

# 특성 추출기 로드
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name, trust_remote_code=True)

# 모델 로드
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 평가 모드 설정
model.eval()

print(f"오디오 모델 로드 완료: {{model.__class__.__name__}}")
print(f"샘플링 레이트: {{feature_extractor.sampling_rate}}")

# 오디오 로드 및 전처리 함수
def load_and_preprocess_audio(audio_path):
    # 오디오 파일 로드
    audio, sr = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
    
    # 특성 추출
    inputs = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt"
    )
    
    return inputs, audio

# 샘플 오디오 파일 경로
sample_audio_path = "path/to/audio.wav"

print("오디오 로드 및 전처리 준비 완료")
print("librosa를 사용하여 오디오 파일을 로드합니다.")'''

        return CodeSnippet(
            title="오디오 모델 로딩",
            description=f"{model_id} 오디오 모델 로딩",
            code=code,
            language="python",
            snippet_type=SnippetType.LOADING,
            requirements=["torch", "transformers", "librosa", "numpy"],
            notes=[
                "AutoFeatureExtractor로 오디오 특성 추출",
                "샘플링 레이트 맞춤 자동 처리",
                "librosa 필요 (오디오 파일 로딩)",
                "긴 오디오는 청크 단위로 처리 권장"
            ]
        )
    
    def _generate_generic_inference(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """일반적인 추론 스니펫"""
        
        code = f'''import torch
from transformers import {model_class}, AutoTokenizer

# 모델과 토크나이저 로드
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.float16
)

# 입력 데이터 준비
input_text = "여기에 입력 텍스트를 넣으세요."

# 토큰화
inputs = tokenizer(
    input_text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

# 추론 실행
with torch.no_grad():
    outputs = model(**inputs)

# 결과 처리
print(f"출력 형태: {{outputs.logits.shape if hasattr(outputs, 'logits') else 'N/A'}}")
print(f"태스크: {task_type}")

# 태스크별 후처리 (필요에 따라 구현)
if hasattr(outputs, 'logits'):
    logits = outputs.logits
    print(f"로짓 값: {{logits}}")
    
    # 확률 변환 (분류 태스크인 경우)
    if logits.shape[-1] > 1:
        probs = torch.softmax(logits, dim=-1)
        print(f"확률 분포: {{probs}}")'''

        return CodeSnippet(
            title=f"{task_type} 추론",
            description=f"{model_id} 모델을 사용한 {task_type} 추론",
            code=code,
            language="python",
            snippet_type=SnippetType.INFERENCE,
            requirements=["torch", "transformers"],
            notes=[
                f"태스크: {task_type}",
                "입력 데이터를 태스크에 맞게 수정 필요",
                "출력 형태는 모델과 태스크에 따라 다름",
                "후처리 로직 추가 구현 필요"
            ]
        )
    
    def _generate_fine_tuning_snippet(self, model_id: str, task_type: str, model_class: str, model_info: Optional[ModelTypeInfo]) -> CodeSnippet:
        """파인튜닝 스니펫"""
        
        code = f'''import torch
from transformers import {model_class}, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
import numpy as np

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
        
        return {{
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }}

# 모델과 토크나이저 로드
model_name = "{model_id}"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = {model_class}.from_pretrained(
    model_name,
    trust_remote_code=True,
    num_labels=2  # 분류 클래스 수에 맞게 조정
)

# 샘플 데이터 (실제 데이터로 교체)
train_texts = [
    "이 제품은 훌륭합니다.",
    "서비스가 별로입니다.",
    "가격이 적당합니다.",
    "품질이 좋습니다."
]
train_labels = [1, 0, 1, 1]  # 1: 긍정, 0: 부정

val_texts = [
    "만족스러운 구매였습니다.",
    "다시는 사지 않겠습니다."
]
val_labels = [1, 0]

# 데이터셋 생성
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

# 훈련 설정
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    fp16=True,  # 메모리 절약
    gradient_accumulation_steps=2
)

# 평가 지표 함수
def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
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

# 트레이너 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# 파인튜닝 실행
print("파인튜닝 시작...")
trainer.train()

# 모델 저장
model.save_pretrained('./fine_tuned_model')
tokenizer.save_pretrained('./fine_tuned_model')

# 평가
eval_results = trainer.evaluate()
print(f"평가 결과: {{eval_results}}")

# 파인튜닝된 모델 테스트
test_text = "이 제품 정말 좋네요!"
inputs = tokenizer(test_text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = predictions.argmax().item()
    confidence = predictions.max().item()

print(f"테스트 결과: 클래스 {{predicted_class}}, 신뢰도: {{confidence:.3f}}")'''

        return CodeSnippet(
            title="모델 파인튜닝",
            description=f"{model_id} 모델 파인튜닝",
            code=code,
            language="python",
            snippet_type=SnippetType.FINE_TUNING,
            requirements=["torch", "transformers", "numpy", "scikit-learn"],
            notes=[
                "실제 데이터로 train_texts, train_labels 교체",
                "num_labels를 데이터셋의 클래스 수에 맞게 조정",
                "배치 크기와 학습률 등 하이퍼파라미터 튜닝",
                "검증 데이터로 과적합 방지",
                "GPU 메모리 부족 시 배치 크기 감소"
            ]
        )
    
    def _get_optimization_parameters(self, model_info: Optional[ModelTypeInfo]) -> Dict[str, Any]:
        """모델별 최적화 파라미터 반환"""
        
        # 기본 파라미터
        params = {
            "dtype": "float16",
            "device_map": "auto",
            "quantization": False
        }
        
        if model_info:
            # 모델 카테고리별 최적화
            if model_info.category == ModelCategory.TEXT:
                params["dtype"] = "float16"
                params["device_map"] = "auto"
            elif model_info.category == ModelCategory.VISION:
                params["dtype"] = "float16"
                params["device_map"] = "auto"
            elif model_info.category == ModelCategory.MULTIMODAL:
                params["dtype"] = "float16"
                params["device_map"] = "auto"
            elif model_info.category == ModelCategory.AUDIO:
                params["dtype"] = "float16"
                params["device_map"] = "auto"
            
            # 모델 크기에 따른 최적화 (추정)
            model_size_keywords = ["large", "xl", "xxl", "7b", "13b", "30b"]
            if any(keyword in model_info.model_type.lower() for keyword in model_size_keywords):
                params["quantization"] = True
                params["dtype"] = "float16"
        
        return params
    
    def _extract_model_type_from_path(self, model_path: str) -> Optional[str]:
        """모델 경로에서 모델 타입 추출"""
        try:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get("model_type")
        except:
            pass
        return None
    
    def _is_huggingface_model_id(self, model_name: str) -> bool:
        """HuggingFace Hub 모델 ID인지 확인"""
        return '/' in model_name and not model_name.startswith('/')
    
    def _supports_fine_tuning(self, task_type: str) -> bool:
        """파인튜닝 지원 여부 확인"""
        fine_tuning_tasks = [
            "text-classification",
            "token-classification", 
            "question-answering",
            "text-generation"
        ]
        return task_type in fine_tuning_tasks

# 전역 인스턴스
snippet_generator = CodeSnippetGenerator()