#!/usr/bin/env python3
"""
Hugging Face GUI Demo Script
이 스크립트는 새로운 기능들을 프로그래밍 방식으로 테스트하는 데모입니다.
"""

import time
import threading
from model_manager import MultiModelManager
from system_monitor import SystemMonitor
from fastapi_server import FastAPIServer
from model_analyzer import ComprehensiveModelAnalyzer

def demo_model_analyzer():
    """모델 분석 기능 데모"""
    print("🔍 모델 분석 기능 데모")
    print("=" * 50)
    
    analyzer = ComprehensiveModelAnalyzer()
    
    # 예시 모델 경로 (실제 모델이 있는 경로로 변경하세요)
    model_paths = [
        "bert-base-uncased",  # HuggingFace Hub 모델
        "gpt2",               # HuggingFace Hub 모델
        # "/path/to/local/model"  # 로컬 모델 경로
    ]
    
    for model_path in model_paths:
        print(f"\n📊 분석 중: {model_path}")
        try:
            analysis = analyzer.analyze_model_directory(model_path)
            
            print(f"  모델 타입: {analysis['model_summary'].get('model_type', 'unknown')}")
            print(f"  파라미터 수: {analysis['model_summary'].get('total_parameters', 0):,}")
            print(f"  지원 태스크: {', '.join(analysis['model_summary'].get('supported_tasks', []))}")
            print(f"  발견된 파일: {len(analysis['files_found'])}개")
            print(f"  누락된 파일: {len(analysis['files_missing'])}개")
            
            if analysis['recommendations']:
                print(f"  권장사항: {len(analysis['recommendations'])}개")
                for rec in analysis['recommendations'][:2]:
                    print(f"    - {rec}")
                    
        except Exception as e:
            print(f"  ❌ 분석 실패: {e}")
    
    print("\n✅ 모델 분석 데모 완료")

def demo_system_monitor():
    """시스템 모니터링 기능 데모"""
    print("\n🖥️ 시스템 모니터링 기능 데모")
    print("=" * 50)
    
    monitor = SystemMonitor(update_interval=2.0)
    
    # 모니터링 데이터 수집 콜백
    def print_system_info(data):
        print(f"⏰ {data['timestamp'].strftime('%H:%M:%S')} - "
              f"CPU: {data['cpu']['percent']:.1f}%, "
              f"Memory: {data['memory']['percent']:.1f}%, "
              f"GPU: {len(data['gpu'])} device(s)")
    
    monitor.add_callback(print_system_info)
    
    print("🚀 모니터링 시작 (5초간 실행)")
    monitor.start_monitoring()
    
    # 5초 동안 모니터링
    time.sleep(5)
    
    print("⏹️ 모니터링 중지")
    monitor.stop_monitoring()
    
    # 히스토리 데이터 확인
    history = monitor.get_history()
    print(f"📊 수집된 데이터 포인트: CPU {len(history['cpu'])}개, Memory {len(history['memory'])}개")
    
    # 시스템 정보 출력
    system_info = monitor.get_system_info()
    print(f"💻 시스템 정보:")
    print(f"  - CPU 코어: {system_info['cpu_count']}개")
    print(f"  - 메모리: {system_info['memory_total'] / (1024**3):.1f} GB")
    print(f"  - GPU: {system_info['gpu_count']}개")
    
    print("✅ 시스템 모니터링 데모 완료")

def demo_model_manager():
    """모델 관리 기능 데모"""
    print("\n🤖 모델 관리 기능 데모")
    print("=" * 50)
    
    manager = MultiModelManager()
    
    # 모델 상태 변경 콜백
    def model_callback(model_name, event_type, data):
        print(f"📢 {model_name}: {event_type}")
        if event_type == "loading_success":
            print(f"   메모리 사용량: {data['memory_usage']:.1f} MB")
    
    manager.add_callback(model_callback)
    
    # 간단한 모델 분석
    print("🔍 모델 분석 테스트")
    try:
        analysis = manager.analyze_model("gpt2")
        print(f"✅ GPT-2 분석 완료: {analysis['model_summary'].get('model_type', 'unknown')}")
    except Exception as e:
        print(f"❌ 분석 실패: {e}")
    
    # 시스템 요약 출력
    summary = manager.get_system_summary()
    print(f"\n📊 시스템 요약:")
    print(f"  - 로드된 모델: {summary['loaded_models_count']}개")
    print(f"  - 총 모델: {summary['total_models_count']}개")
    print(f"  - 메모리 사용량: {summary['total_memory_usage_mb']:.1f} MB")
    
    print("✅ 모델 관리 데모 완료")

def demo_fastapi_server():
    """FastAPI 서버 기능 데모"""
    print("\n🚀 FastAPI 서버 기능 데모")
    print("=" * 50)
    
    manager = MultiModelManager()
    server = FastAPIServer(manager)
    
    # 서버 정보 출력
    server_info = server.get_server_info()
    print(f"🌐 서버 정보:")
    print(f"  - URL: {server_info['url']}")
    print(f"  - 문서: {server_info['docs_url']}")
    print(f"  - 상태: {'실행 중' if server_info['running'] else '중지됨'}")
    
    # 엔드포인트 목록
    endpoints = server.get_available_endpoints()
    print(f"\n📡 사용 가능한 엔드포인트 ({len(endpoints)}개):")
    for ep in endpoints[:5]:  # 처음 5개만 표시
        print(f"  - {ep['methods'][0]} {ep['path']}")
    
    print("💡 실제 서버 시작은 GUI에서 수행하세요.")
    print("✅ FastAPI 서버 데모 완료")

def main():
    """메인 데모 함수"""
    print("🎉 Hugging Face GUI 데모 시작")
    print("=" * 60)
    
    try:
        # 각 기능 데모 실행
        demo_model_analyzer()
        demo_system_monitor()
        demo_model_manager()
        demo_fastapi_server()
        
        print("\n🎊 모든 데모 완료!")
        print("\n📖 사용 방법:")
        print("  - Streamlit 버전: streamlit run app_enhanced.py")
        print("  - CustomTkinter 버전: python run_enhanced.py")
        print("  - 기존 버전도 여전히 사용 가능합니다.")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ 데모가 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n❌ 데모 중 오류 발생: {e}")

if __name__ == "__main__":
    main()