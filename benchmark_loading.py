#!/usr/bin/env python3
"""
Ultra-Fast 모델 로딩 성능 벤치마크 스크립트
"""
import time
import psutil
import torch
from model_manager import MultiModelManager
from model_cache import model_cache
from fast_tensor_loader import fast_loader
from parallel_model_loader import parallel_loader
from cpu_optimizer import cpu_optimizer

def benchmark_ultra_fast_loading():
    """Ultra-Fast 모델 로딩 성능 벤치마크"""
    print("=" * 70)
    print("🚀 ULTRA-FAST 모델 로딩 성능 벤치마크")
    print("=" * 70)
    
    # 시스템 정보
    print("📊 시스템 정보:")
    print(f"   CPU 코어 수: {torch.get_num_threads()}")
    print(f"   CUDA 사용 가능: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    memory = psutil.virtual_memory()
    print(f"   시스템 메모리: {memory.total / (1024**3):.1f}GB")
    print(f"   사용 가능한 메모리: {memory.available / (1024**3):.1f}GB")
    
    # 최적화 정보
    opt_info = cpu_optimizer.get_optimization_info()
    print(f"   Intel Extension: {opt_info['intel_extension_available']}")
    print(f"   torch.compile: {opt_info['torch_compile_available']}")
    print(f"   PyTorch 버전: {opt_info['torch_version']}")
    print()
    
    # 테스트할 모델
    model_path = "tabularisai/multilingual-sentiment-analysis"
    model_name = "ultra-fast-benchmark"
    
    # 모델 매니저 초기화
    manager = MultiModelManager()
    
    # 캐시 정리
    model_cache.clear_cache()
    
    print("🔥 1단계: Ultra-Fast 첫 번째 로딩 (캐시 없음)")
    print("-" * 50)
    
    # 첫 번째 로딩 시간 측정
    start_time = time.time()
    memory_before = psutil.virtual_memory().used
    
    def ultra_fast_callback(name, success, message):
        if success:
            elapsed = time.time() - start_time
            memory_after = psutil.virtual_memory().used
            memory_used = (memory_after - memory_before) / (1024**2)  # MB
            
            print(f"✅ Ultra-Fast 첫 번째 로딩 완료:")
            print(f"   ⚡ 시간: {elapsed:.1f}초")
            print(f"   💾 메모리 사용량: {memory_used:.1f}MB")
            
            # 성능 등급 판정
            if elapsed < 30:
                print("   🏆 등급: ULTRA-FAST (30초 미만)")
            elif elapsed < 60:
                print("   🥇 등급: FAST (60초 미만)")
            elif elapsed < 120:
                print("   🥈 등급: GOOD (120초 미만)")
            else:
                print("   🥉 등급: NEEDS IMPROVEMENT")
            print()
            
            # 직접 텐서 로딩 테스트
            print("🧪 2단계: 직접 텐서 로딩 테스트")
            print("-" * 50)
            
            test_direct_loading(model_path)
            
            # 병렬 로딩 테스트
            print("🔄 3단계: 병렬 로딩 테스트")
            print("-" * 50)
            
            test_parallel_loading(model_path)
            
            # 캐시 테스트
            print("💾 4단계: 캐시 성능 테스트")
            print("-" * 50)
            
            test_cache_performance(manager, name, model_path, elapsed)
            
        else:
            print(f"❌ Ultra-Fast 로딩 실패: {message}")
    
    # Ultra-Fast 로딩 시작
    manager.load_model_async(model_name, model_path, ultra_fast_callback)

def test_direct_loading(model_path):
    """직접 텐서 로딩 성능 테스트"""
    try:
        start_time = time.time()
        
        # 직접 텐서 로딩 테스트
        model, model_time = fast_loader.load_model_ultra_fast(model_path, "cpu")
        tokenizer, tokenizer_time = fast_loader.load_tokenizer_fast(model_path)
        
        total_time = time.time() - start_time
        
        if model and tokenizer:
            print(f"   ✅ 직접 로딩 성공:")
            print(f"      모델: {model_time:.1f}초")
            print(f"      토크나이저: {tokenizer_time:.1f}초")
            print(f"      총 시간: {total_time:.1f}초")
            
            # 모델 검증
            valid = fast_loader.validate_model(model, tokenizer)
            print(f"      검증: {'✅ 성공' if valid else '❌ 실패'}")
        else:
            print("   ❌ 직접 로딩 실패")
            
    except Exception as e:
        print(f"   ❌ 직접 로딩 오류: {e}")
    
    print()

def test_parallel_loading(model_path):
    """병렬 로딩 성능 테스트"""
    try:
        start_time = time.time()
        
        # 병렬 로딩 테스트
        model, tokenizer, load_time = parallel_loader.load_model_and_tokenizer_parallel(model_path, "cpu")
        
        if model and tokenizer:
            print(f"   ✅ 병렬 로딩 성공:")
            print(f"      로딩 시간: {load_time:.1f}초")
            
            # CPU 최적화 테스트
            opt_start = time.time()
            optimized_model = cpu_optimizer.optimize_model_for_cpu(model, optimize_level=3)
            opt_time = time.time() - opt_start
            
            print(f"      최적화: {opt_time:.1f}초")
            
            # 성능 벤치마크
            benchmark_results = cpu_optimizer.benchmark_model_performance(optimized_model, num_runs=5)
            if "average_time" in benchmark_results:
                print(f"      추론 속도: {benchmark_results['average_time']:.4f}초")
                print(f"      처리량: {benchmark_results['throughput']:.1f} 추론/초")
        else:
            print("   ❌ 병렬 로딩 실패")
            
    except Exception as e:
        print(f"   ❌ 병렬 로딩 오류: {e}")
    
    print()

def test_cache_performance(manager, model_name, model_path, first_load_time):
    """캐시 성능 테스트"""
    try:
        # 모델 언로드
        manager.unload_model(model_name)
        
        # 캐시에서 두 번째 로딩
        second_start = time.time()
        
        def cache_callback(name, success, message):
            if success:
                second_elapsed = time.time() - second_start
                speedup = first_load_time / second_elapsed if second_elapsed > 0 else float('inf')
                
                print(f"   ✅ 캐시 로딩 성공:")
                print(f"      시간: {second_elapsed:.1f}초")
                print(f"      가속: {speedup:.1f}배")
                
                # 캐시 통계
                cache_stats = model_cache.get_cache_stats()
                print(f"      캐시 크기: {cache_stats['total_cache_size_gb']:.2f}GB")
                print(f"      히트율: {cache_stats['cache_hit_rate']:.1f}%")
                
                # 최종 결과
                print()
                print("🎯 최종 벤치마크 결과")
                print("-" * 50)
                print(f"   첫 번째 로딩: {first_load_time:.1f}초")
                print(f"   캐시 로딩: {second_elapsed:.1f}초")
                print(f"   성능 향상: {speedup:.1f}배")
                
                # 목표 달성도
                target_time = 60  # 목표: 60초 이하
                if first_load_time <= target_time:
                    print(f"   🎯 목표 달성! ({target_time}초 이하)")
                    improvement = (338 - first_load_time) / 338 * 100  # 기존 338초 대비
                    print(f"   📈 개선율: {improvement:.1f}% (기존 338초 대비)")
                else:
                    remaining = first_load_time - target_time
                    print(f"   ⏰ 목표까지: {remaining:.1f}초 추가 개선 필요")
                
                print("=" * 70)
            else:
                print(f"   ❌ 캐시 로딩 실패: {message}")
        
        # 두 번째 로딩 시작
        manager.load_model_async(f"{model_name}-cache", model_path, cache_callback)
        
    except Exception as e:
        print(f"   ❌ 캐시 테스트 오류: {e}")

if __name__ == "__main__":
    benchmark_ultra_fast_loading()
    
    # 벤치마크 완료 대기
    import threading
    
    # 모든 로딩 스레드가 완료될 때까지 대기
    for thread in threading.enumerate():
        if thread.name.startswith("ModelLoad-"):
            thread.join()
    
    print("🏁 Ultra-Fast 벤치마크 완료!")