#!/usr/bin/env python3
"""
모델 로딩 문제 디버깅 스크립트
"""
import threading
import time
import psutil
import torch
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

def check_system_resources():
    """시스템 리소스 확인"""
    print("=== 시스템 리소스 확인 ===")
    
    # 메모리
    memory = psutil.virtual_memory()
    print(f"총 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"사용 중: {memory.used / (1024**3):.1f}GB ({memory.percent:.1f}%)")
    print(f"사용 가능: {memory.available / (1024**3):.1f}GB")
    
    # GPU
    if torch.cuda.is_available():
        print(f"CUDA 사용 가능: Yes")
        for i in range(torch.cuda.device_count()):
            total = torch.cuda.get_device_properties(i).total_memory
            allocated = torch.cuda.memory_allocated(i)
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  총: {total / (1024**3):.1f}GB, 할당: {allocated / (1024**3):.1f}GB")
    else:
        print("CUDA 사용 불가능")
    
    print()

def test_simple_loading():
    """간단한 모델 로딩 테스트"""
    print("=== 간단한 모델 로딩 테스트 ===")
    model_id = "tabularisai/multilingual-sentiment-analysis"
    
    try:
        print("1. Config 로딩...")
        config = AutoConfig.from_pretrained(model_id)
        print(f"   완료: {config.architectures}")
        
        print("2. 토크나이저 로딩...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"   완료 ({time.time() - start_time:.1f}초)")
        
        print("3. 모델 로딩...")
        start_time = time.time()
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id,
            trust_remote_code=True
        )
        print(f"   완료 ({time.time() - start_time:.1f}초)")
        
        print("4. 간단한 추론 테스트...")
        inputs = tokenizer("This is a test", return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"   완료, 출력 크기: {outputs.logits.shape}")
        
        print("✅ 모든 테스트 성공!")
        return True
        
    except Exception as e:
        print(f"❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_threaded_loading():
    """스레드에서 모델 로딩 테스트"""
    print("=== 스레드 모델 로딩 테스트 ===")
    
    result = {"success": False, "error": None, "model": None, "tokenizer": None}
    
    def load_in_thread():
        try:
            print("[스레드] 모델 로딩 시작...")
            model_id = "tabularisai/multilingual-sentiment-analysis"
            
            print("[스레드] 토크나이저 로딩...")
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            print("[스레드] 모델 로딩...")
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            result["model"] = model
            result["tokenizer"] = tokenizer
            result["success"] = True
            print("[스레드] 로딩 완료!")
            
        except Exception as e:
            result["error"] = str(e)
            print(f"[스레드] 오류: {e}")
            import traceback
            traceback.print_exc()
    
    # 스레드 시작
    thread = threading.Thread(target=load_in_thread)
    thread.daemon = True
    thread.start()
    
    # 타임아웃으로 대기
    timeout = 60  # 60초
    start_time = time.time()
    
    while thread.is_alive() and (time.time() - start_time) < timeout:
        print(f"대기 중... ({time.time() - start_time:.1f}초)")
        time.sleep(2)
    
    if thread.is_alive():
        print("❌ 타임아웃! 스레드가 완료되지 않음")
        return False
    
    if result["success"]:
        print("✅ 스레드 로딩 성공!")
        return True
    else:
        print(f"❌ 스레드 로딩 실패: {result['error']}")
        return False

def test_memory_pressure():
    """메모리 압박 상황에서 로딩 테스트"""
    print("=== 메모리 압박 상황 테스트 ===")
    
    # 현재 메모리 상태
    initial_memory = psutil.virtual_memory()
    print(f"초기 메모리 사용률: {initial_memory.percent:.1f}%")
    
    if initial_memory.percent > 80:
        print("⚠️ 메모리 사용률이 이미 높음 (80% 이상)")
        return False
    
    # 메모리 압박 생성 (1GB 할당)
    memory_hog = []
    try:
        print("메모리 압박 생성 중...")
        for i in range(10):  # 100MB씩 10번
            data = bytearray(100 * 1024 * 1024)  # 100MB
            memory_hog.append(data)
            current_memory = psutil.virtual_memory()
            print(f"  할당 {(i+1)*100}MB, 사용률: {current_memory.percent:.1f}%")
            
            if current_memory.percent > 85:
                print("메모리 사용률 85% 도달, 중단")
                break
        
        # 압박 상황에서 모델 로딩 시도
        print("압박 상황에서 모델 로딩 시도...")
        return test_simple_loading()
        
    finally:
        # 메모리 해제
        print("메모리 해제 중...")
        del memory_hog
        import gc
        gc.collect()

def main():
    print("모델 로딩 문제 진단 시작\n")
    
    # 1. 시스템 리소스 확인
    check_system_resources()
    
    # 2. 간단한 로딩 테스트
    simple_result = test_simple_loading()
    print()
    
    # 3. 스레드 로딩 테스트
    thread_result = test_threaded_loading()
    print()
    
    # 4. 메모리 압박 테스트
    memory_result = test_memory_pressure()
    print()
    
    # 결과 요약
    print("=== 진단 결과 요약 ===")
    print(f"간단한 로딩: {'✅ 성공' if simple_result else '❌ 실패'}")
    print(f"스레드 로딩: {'✅ 성공' if thread_result else '❌ 실패'}")
    print(f"메모리 압박 테스트: {'✅ 성공' if memory_result else '❌ 실패'}")
    
    if simple_result and not thread_result:
        print("\n🔍 추정 원인: 스레드 환경에서의 문제")
        print("   - GIL (Global Interpreter Lock) 문제")
        print("   - 스레드 간 리소스 충돌")
        print("   - 메모리 공유 문제")
    
    elif not simple_result:
        print("\n🔍 추정 원인: 기본적인 모델 로딩 문제")
        print("   - 메모리 부족")
        print("   - 네트워크 연결 문제")
        print("   - 의존성 문제")

if __name__ == "__main__":
    main()