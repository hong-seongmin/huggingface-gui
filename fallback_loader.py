#!/usr/bin/env python3
"""
대체 모델 로더 - transformers가 멈출 때 사용
"""

import subprocess
import sys
import json
import tempfile
import os

def load_model_subprocess(model_path, model_name):
    """별도 프로세스에서 모델 로딩"""
    
    # 임시 스크립트 생성
    script_content = f'''
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import pickle
import sys
import os

try:
    print("SUBPROCESS: 모델 로딩 시작")
    
    # 환경 최적화
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    
    model_path = "{model_path}"
    
    # 설정 확인
    config = AutoConfig.from_pretrained(model_path)
    is_classification = hasattr(config, 'architectures') and config.architectures and any('Classification' in arch for arch in config.architectures)
    
    print(f"SUBPROCESS: Classification 모델: {{is_classification}}")
    
    if is_classification:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32
        )
    else:
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            torch_dtype=torch.float32
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("SUBPROCESS: 모델 로딩 완료")
    
    # 결과를 임시 파일에 저장
    with open("/tmp/model_load_result.json", "w") as f:
        json.dump({{"success": True, "message": "모델 로딩 성공"}}, f)
        
except Exception as e:
    print(f"SUBPROCESS: 오류 - {{e}}")
    with open("/tmp/model_load_result.json", "w") as f:
        json.dump({{"success": False, "error": str(e)}}, f)
'''
    
    # 임시 파일에 스크립트 저장
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        script_path = f.name
    
    try:
        print(f"[FALLBACK] 별도 프로세스에서 모델 로딩 시도...")
        
        # 서브프로세스로 실행 (30초 타임아웃)
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        print(f"[FALLBACK] 서브프로세스 완료. 반환 코드: {result.returncode}")
        print(f"[FALLBACK] 출력: {result.stdout}")
        
        if result.stderr:
            print(f"[FALLBACK] 에러: {result.stderr}")
        
        # 결과 파일 확인
        if os.path.exists("/tmp/model_load_result.json"):
            with open("/tmp/model_load_result.json", "r") as f:
                load_result = json.load(f)
            
            os.remove("/tmp/model_load_result.json")  # 정리
            
            if load_result["success"]:
                print(f"[FALLBACK] 모델 로딩 성공!")
                return True, "서브프로세스에서 로딩 성공"
            else:
                return False, f"서브프로세스 오류: {load_result['error']}"
        else:
            return False, "결과 파일을 찾을 수 없음"
            
    except subprocess.TimeoutExpired:
        print(f"[FALLBACK] 서브프로세스 타임아웃 (30초)")
        return False, "서브프로세스 타임아웃"
    except Exception as e:
        print(f"[FALLBACK] 서브프로세스 실행 오류: {e}")
        return False, f"서브프로세스 실행 오류: {e}"
    finally:
        # 임시 스크립트 정리
        if os.path.exists(script_path):
            os.remove(script_path)

if __name__ == "__main__":
    # 테스트
    success, message = load_model_subprocess(
        "/home/hong/.cache/huggingface/hub/models--tabularisai--multilingual-sentiment-analysis/snapshots/f0bcb3b4493d5be7da88fa86f0e0bfbd670a9e97",
        "test"
    )
    print(f"결과: {success}, {message}")