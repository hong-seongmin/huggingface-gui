#!/usr/bin/env python3
"""
간단한 API 테스트
"""
import requests
import json

def test_server_status():
    """서버 상태 확인"""
    ports = [8000, 8002, 8003]
    
    for port in ports:
        try:
            # Health check
            response = requests.get(f"http://localhost:{port}/health", timeout=3)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Port {port}: Running (models: {health_data.get('models_loaded', 0)})")
            else:
                print(f"❌ Port {port}: HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Port {port}: Not accessible ({e})")
    
    print()

def test_models_endpoint(port):
    """모델 엔드포인트 테스트"""
    try:
        response = requests.get(f"http://localhost:{port}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"📋 Port {port} Models:")
            print(f"   Loaded: {data.get('loaded_models', [])}")
            print(f"   All: {data.get('all_models', [])}")
            print(f"   Target: {data.get('target_models', [])}")
            return data.get('loaded_models', [])
        else:
            print(f"❌ Port {port} models endpoint: HTTP {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Port {port} models endpoint error: {e}")
        return []

def test_prediction(port, model_name, text):
    """예측 테스트"""
    try:
        url = f"http://localhost:{port}/models/{model_name}/predict"
        payload = {"text": text}
        
        response = requests.post(url, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Port {port} {model_name}: Prediction successful")
            print(f"   Result: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"❌ Port {port} {model_name}: HTTP {response.status_code}")
            try:
                error = response.json()
                print(f"   Error: {error}")
            except:
                print(f"   Error text: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Port {port} {model_name} prediction error: {e}")
        return False

def main():
    print("🚀 Simple API Test")
    print("=" * 50)
    
    # 1. 서버 상태 확인
    test_server_status()
    
    # 2. 각 포트의 모델 목록 확인
    available_models = {}
    for port in [8000, 8002, 8003]:
        models = test_models_endpoint(port)
        if models:
            available_models[port] = models
    
    print()
    
    # 3. 예측 테스트
    test_cases = [
        ("This product is amazing!", "sentiment"),
        ("Hello world for embedding", "embedding")
    ]
    
    for port, models in available_models.items():
        for model in models:
            for text, task_type in test_cases:
                print(f"\n🧪 Testing {model} on port {port}")
                test_prediction(port, model, text)
    
    print("\n🏁 Test completed")

if __name__ == "__main__":
    main()