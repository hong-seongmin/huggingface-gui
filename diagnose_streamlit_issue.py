#!/usr/bin/env python3
"""
Streamlit 환경에서 모델 로딩 문제 진단
"""
import requests
import json
import time

def test_streamlit_model_loading():
    """Streamlit 앱의 상태 확인"""
    print("=== Streamlit 앱 모델 로딩 상태 진단 ===")
    
    try:
        # Streamlit 앱 URL
        base_url = "http://localhost:8501"
        
        # 1. 앱 상태 확인
        print("1. Streamlit 앱 연결 확인...")
        response = requests.get(f"{base_url}/healthz", timeout=5)
        if response.status_code == 200:
            print("   ✅ Streamlit 연결 성공")
        else:
            print("   ❌ Streamlit 연결 실패")
            return False
        
        # 2. 현재 세션에서 모델 상태 확인 (가능하다면)
        print("2. 현재 모델 로딩 상태 확인...")
        print("   (직접 세션 접근 불가, 다른 방법 필요)")
        
        # 3. FastAPI 서버 상태 확인
        print("3. FastAPI 서버 상태 확인...")
        try:
            fastapi_response = requests.get("http://localhost:8000/health", timeout=5)
            if fastapi_response.status_code == 200:
                print("   ✅ FastAPI 서버 실행 중")
            else:
                print("   ❌ FastAPI 서버 응답 없음")
        except requests.exceptions.RequestException:
            print("   ⚠️ FastAPI 서버 미실행 (정상일 수 있음)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        return False

def suggest_solutions():
    """해결책 제안"""
    print("\n=== 문제 해결 방안 ===")
    
    print("1. 메모리 기반 해결책:")
    print("   - 더 작은 모델 사용 (예: distilbert-base-uncased)")
    print("   - CPU 전용 로딩 (GPU 메모리 문제)")
    print("   - float16 정밀도 사용")
    
    print("\n2. 코드 수정 해결책:")
    print("   - 타임아웃 추가")
    print("   - 스레드 풀 사용")
    print("   - 동기화 메커니즘 개선")
    
    print("\n3. 환경 기반 해결책:")
    print("   - Streamlit 재시작")
    print("   - Python 프로세스 재시작")
    print("   - 캐시 클리어")
    
    print("\n4. 대안 모델:")
    print("   - cardiffnlp/twitter-roberta-base-sentiment-latest")
    print("   - nlptown/bert-base-multilingual-uncased-sentiment")
    print("   - j-hartmann/emotion-english-distilroberta-base")

def create_simple_model_loader():
    """간단한 모델 로더 생성"""
    print("\n=== 간단한 모델 로더 생성 ===")
    
    simple_loader_code = '''
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import time

@st.cache_resource
def load_sentiment_model():
    """캐시된 모델 로더"""
    try:
        model_id = "tabularisai/multilingual-sentiment-analysis"
        
        # 토크나이저 먼저 로딩
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 모델 로딩 (간단한 방법)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        return model, tokenizer
    except Exception as e:
        st.error(f"모델 로딩 실패: {e}")
        return None, None

def classify_text(text):
    """텍스트 분류"""
    model, tokenizer = load_sentiment_model()
    if model is None or tokenizer is None:
        return "모델 로딩 실패"
    
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=-1).item()
    confidence = torch.softmax(outputs.logits, dim=-1).max().item()
    
    labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
    return f"{labels[predicted_class]} (신뢰도: {confidence:.2f})"

# Streamlit UI
st.title("간단한 감정 분석")
text_input = st.text_area("분석할 텍스트 입력:")
if st.button("분석"):
    if text_input:
        result = classify_text(text_input)
        st.write(f"결과: {result}")
    else:
        st.warning("텍스트를 입력해주세요.")
'''
    
    with open('/home/hong/code/huggingface-gui/simple_sentiment_app.py', 'w') as f:
        f.write(simple_loader_code)
    
    print("간단한 앱 생성 완료: simple_sentiment_app.py")
    print("실행 방법: streamlit run simple_sentiment_app.py")

def main():
    print("Streamlit 모델 로딩 문제 진단 시작\n")
    
    # 1. Streamlit 상태 확인
    streamlit_ok = test_streamlit_model_loading()
    
    # 2. 해결책 제안
    suggest_solutions()
    
    # 3. 간단한 대안 생성
    create_simple_model_loader()
    
    print("\n=== 추천 조치 ===")
    if streamlit_ok:
        print("1. 현재 Streamlit 앱을 재시작해보세요")
        print("2. 간단한 앱(simple_sentiment_app.py)을 테스트해보세요")
        print("3. model_manager.py의 타임아웃 설정을 조정해보세요")
    else:
        print("1. Streamlit 앱을 다시 시작해주세요")
        print("2. 포트 충돌 확인")
        print("3. 가상환경 활성화 확인")

if __name__ == "__main__":
    main()