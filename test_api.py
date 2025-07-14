#!/usr/bin/env python3
"""
API 테스트 스크립트
"""
import requests
import json

def test_api(port, model_name, test_text):
    """API 테스트 함수"""
    base_url = f"http://localhost:{port}"
    
    print(f"🔍 Testing API on port {port} for model '{model_name}'")
    print(f"📝 Test text: '{test_text}'")
    print("-" * 60)
    
    # 1. Health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # 2. Models endpoint
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        if response.status_code == 200:
            models_data = response.json()
            print(f"✅ Models endpoint: Found {len(models_data.get('loaded_models', []))} loaded models")
            print(f"   Loaded models: {models_data.get('loaded_models', [])}")
        else:
            print(f"❌ Models endpoint: {response.status_code}")
    except Exception as e:
        print(f"❌ Models endpoint failed: {e}")
    
    # 3. Model prediction
    try:
        predict_url = f"{base_url}/models/{model_name}/predict"
        payload = {"text": test_text}
        
        response = requests.post(
            predict_url, 
            json=payload, 
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful:")
            print(f"   Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            try:
                error_detail = response.json()
                print(f"   Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"   Error text: {response.text}")
    except Exception as e:
        print(f"❌ Prediction request failed: {e}")
    
    print("\n" + "=" * 60 + "\n")

def main():
    """메인 함수"""
    print("🚀 API Testing Started")
    print("=" * 60)
    
    # Test cases
    test_cases = [
        {
            "port": 8002,
            "model_name": "multilingual-sentiment-analysis", 
            "test_text": "This product is amazing! I highly recommend it."
        },
        {
            "port": 8002,
            "model_name": "multilingual-sentiment-analysis",
            "test_text": "이 제품 정말 좋아요! 완전 추천합니다."
        },
        {
            "port": 8003,
            "model_name": "bge-m3",
            "test_text": "Hello world, this is a test sentence for embedding generation."
        },
        {
            "port": 8003,
            "model_name": "bge-m3", 
            "test_text": "안녕하세요, 임베딩 생성을 위한 테스트 문장입니다."
        }
    ]
    
    for test_case in test_cases:
        test_api(**test_case)
    
    print("🏁 API Testing Completed")

if __name__ == "__main__":
    main()