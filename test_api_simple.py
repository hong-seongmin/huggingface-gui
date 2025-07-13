#!/usr/bin/env python3
"""
간단한 API 테스트 스크립트
"""
import requests
import json

def test_api():
    try:
        url = "http://localhost:8000/models/multilingual-sentiment-analysis/predict"
        data = {"text": "I love this product!"}
        
        print(f"Testing URL: {url}")
        print(f"Data: {data}")
        
        response = requests.post(url, json=data, timeout=10)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Result: {result}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_api()