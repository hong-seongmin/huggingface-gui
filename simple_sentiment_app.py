
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
