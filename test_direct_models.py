#!/usr/bin/env python3
"""
직접 모델 테스트 스크립트 - 로드된 모델들을 직접 테스트
"""
import sys
import os
sys.path.append('/home/hong/code/huggingface-gui')

from model_manager import MultiModelManager

def test_loaded_models():
    """로드된 모델들을 직접 테스트"""
    # Model manager 인스턴스 생성
    manager = MultiModelManager()
    
    # 현재 로드된 모델들 확인
    loaded_models = manager.get_loaded_models()
    print(f"🔍 로드된 모델들: {loaded_models}")
    
    if not loaded_models:
        print("❌ 로드된 모델이 없습니다.")
        return
    
    for model_name in loaded_models:
        print(f"\n🧪 {model_name} 테스트 중...")
        
        # 모델 정보 확인
        model_info = manager.get_model_info(model_name)
        if not model_info:
            print(f"❌ {model_name} 정보를 찾을 수 없습니다.")
            continue
            
        print(f"   상태: {model_info.status}")
        print(f"   메모리 사용량: {model_info.memory_usage:.1f}MB")
        
        # 추론용 모델과 토크나이저 가져오기
        model_tokenizer = manager.get_model_for_inference(model_name)
        if not model_tokenizer:
            print(f"❌ {model_name} 추론용 객체를 가져올 수 없습니다.")
            continue
            
        model, tokenizer = model_tokenizer
        print(f"   모델 타입: {type(model)}")
        print(f"   토크나이저 타입: {type(tokenizer)}")
        
        # 지원하는 태스크 확인
        available_tasks = manager.get_available_tasks(model_name)
        print(f"   지원 태스크: {available_tasks}")
        
        # 간단한 추론 테스트
        try:
            test_text = "This is a test sentence."
            
            if hasattr(tokenizer, '__call__'):
                # 토크나이저 테스트
                tokens = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
                print(f"   ✅ 토크나이저 작동: input_ids shape = {tokens['input_ids'].shape}")
                
                # 모델 추론 테스트  
                if hasattr(model, 'forward') or hasattr(model, '__call__'):
                    with torch.no_grad():
                        output = model(**tokens)
                        if hasattr(output, 'logits'):
                            print(f"   ✅ 모델 추론 성공: logits shape = {output.logits.shape}")
                        else:
                            print(f"   ✅ 모델 추론 성공: output type = {type(output)}")
                else:
                    print(f"   ❌ 모델에 forward 메서드가 없습니다.")
            else:
                print(f"   ❌ 토크나이저가 호출 가능하지 않습니다.")
                
        except Exception as e:
            print(f"   ❌ 추론 테스트 실패: {e}")
    
    print(f"\n🏁 모델 테스트 완료")

if __name__ == "__main__":
    import torch
    test_loaded_models()