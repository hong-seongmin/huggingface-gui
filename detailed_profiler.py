"""
상세한 모델 로딩 병목 분석 프로파일러
"""
import time
import psutil
import os
import threading
from typing import Dict, List, Any
import logging

class DetailedProfiler:
    """모델 로딩의 모든 단계를 상세히 프로파일링"""
    
    def __init__(self):
        self.logger = logging.getLogger("Profiler")
        self.logger.setLevel(logging.INFO)
        self.start_time = None
        self.checkpoints = []
        self.memory_snapshots = []
        self.io_operations = []
        
    def start_profiling(self, operation_name: str):
        """프로파일링 시작"""
        self.start_time = time.time()
        self.checkpoints = []
        self.memory_snapshots = []
        self.io_operations = []
        
        self.checkpoint(f"🚀 {operation_name} 시작")
        self.memory_snapshot("시작")
        
    def checkpoint(self, description: str):
        """체크포인트 기록"""
        if self.start_time is None:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.checkpoints:
            delta = current_time - self.checkpoints[-1]['timestamp']
            self.logger.info(f"⏱️  [{elapsed:6.1f}s] (+{delta:5.1f}s) {description}")
        else:
            self.logger.info(f"⏱️  [{elapsed:6.1f}s] {description}")
        
        self.checkpoints.append({
            'time': elapsed,
            'timestamp': current_time,
            'description': description
        })
    
    def memory_snapshot(self, stage: str):
        """메모리 스냅샷"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            snapshot = {
                'stage': stage,
                'process_memory_mb': memory_info.rss / (1024 * 1024),
                'system_memory_percent': system_memory.percent,
                'available_memory_gb': system_memory.available / (1024**3)
            }
            
            self.memory_snapshots.append(snapshot)
            self.logger.info(f"💾 [{stage}] 프로세스: {snapshot['process_memory_mb']:.1f}MB, "
                           f"시스템: {snapshot['system_memory_percent']:.1f}%, "
                           f"사용가능: {snapshot['available_memory_gb']:.1f}GB")
        except Exception as e:
            self.logger.warning(f"메모리 스냅샷 실패: {e}")
    
    def io_operation(self, operation: str, file_path: str = "", size_mb: float = 0):
        """I/O 작업 기록"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        
        io_record = {
            'time': elapsed,
            'operation': operation,
            'file_path': file_path,
            'size_mb': size_mb
        }
        
        self.io_operations.append(io_record)
        
        if size_mb > 0:
            self.logger.info(f"📁 [{elapsed:6.1f}s] {operation}: {file_path} ({size_mb:.1f}MB)")
        else:
            self.logger.info(f"📁 [{elapsed:6.1f}s] {operation}: {file_path}")
    
    def profile_file_operation(self, operation_name: str, file_path: str):
        """파일 작업 프로파일링 데코레이터"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
                    self.io_operation(f"{operation_name} 시작", file_path, file_size)
                    
                    start_time = time.time()
                    result = func(*args, **kwargs)
                    end_time = time.time()
                    
                    self.io_operation(f"{operation_name} 완료", file_path)
                    self.checkpoint(f"{operation_name} 완료 ({end_time - start_time:.1f}초)")
                    
                    return result
                except Exception as e:
                    self.checkpoint(f"{operation_name} 실패: {e}")
                    raise
            return wrapper
        return decorator
    
    def profile_transformers_loading(self):
        """transformers 로딩 과정을 상세히 프로파일링"""
        
        # transformers 내부 함수들을 패치하여 로깅 추가
        try:
            import transformers
            
            # 원본 함수들 저장
            original_from_pretrained = transformers.PreTrainedModel.from_pretrained
            original_load_state_dict = transformers.modeling_utils.load_state_dict
            
            def profiled_from_pretrained(cls, *args, **kwargs):
                self.checkpoint("🤖 transformers.from_pretrained 시작")
                self.memory_snapshot("from_pretrained 시작")
                
                try:
                    result = original_from_pretrained(*args, **kwargs)
                    self.checkpoint("✅ transformers.from_pretrained 완료")
                    self.memory_snapshot("from_pretrained 완료")
                    return result
                except Exception as e:
                    self.checkpoint(f"❌ transformers.from_pretrained 실패: {e}")
                    raise
            
            def profiled_load_state_dict(checkpoint_file, *args, **kwargs):
                if isinstance(checkpoint_file, str):
                    file_size = os.path.getsize(checkpoint_file) / (1024 * 1024) if os.path.exists(checkpoint_file) else 0
                    self.io_operation("가중치 파일 로딩", checkpoint_file, file_size)
                
                self.checkpoint("📥 load_state_dict 시작")
                start_time = time.time()
                
                try:
                    result = original_load_state_dict(checkpoint_file, *args, **kwargs)
                    load_time = time.time() - start_time
                    self.checkpoint(f"✅ load_state_dict 완료 ({load_time:.1f}초)")
                    return result
                except Exception as e:
                    self.checkpoint(f"❌ load_state_dict 실패: {e}")
                    raise
            
            # 패치 적용
            transformers.PreTrainedModel.from_pretrained = classmethod(profiled_from_pretrained)
            transformers.modeling_utils.load_state_dict = profiled_load_state_dict
            
            self.logger.info("🔧 transformers 프로파일링 패치 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"transformers 패치 실패: {e}")
    
    def profile_safetensors_loading(self):
        """safetensors 로딩 프로파일링"""
        try:
            from safetensors import safe_open
            
            original_safe_open = safe_open
            
            def profiled_safe_open(filename, *args, **kwargs):
                file_size = os.path.getsize(filename) / (1024 * 1024) if os.path.exists(filename) else 0
                self.io_operation("safetensors 파일 열기", filename, file_size)
                self.checkpoint("🔐 safetensors 파일 열기 시작")
                
                start_time = time.time()
                try:
                    result = original_safe_open(filename, *args, **kwargs)
                    load_time = time.time() - start_time
                    self.checkpoint(f"✅ safetensors 파일 열기 완료 ({load_time:.1f}초)")
                    return result
                except Exception as e:
                    self.checkpoint(f"❌ safetensors 파일 열기 실패: {e}")
                    raise
            
            # 패치 적용 (모듈 레벨에서)
            import safetensors
            safetensors.safe_open = profiled_safe_open
            
            self.logger.info("🔧 safetensors 프로파일링 패치 적용 완료")
            
        except Exception as e:
            self.logger.warning(f"safetensors 패치 실패: {e}")
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """병목 지점 분석"""
        if not self.checkpoints:
            return {"error": "프로파일링 데이터 없음"}
        
        # 단계별 시간 분석
        step_times = []
        for i in range(1, len(self.checkpoints)):
            prev_time = self.checkpoints[i-1]['time']
            curr_time = self.checkpoints[i]['time']
            delta = curr_time - prev_time
            
            step_times.append({
                'step': self.checkpoints[i]['description'],
                'duration': delta,
                'cumulative': curr_time
            })
        
        # 병목 지점 식별 (5초 이상 걸리는 단계들)
        bottlenecks = [step for step in step_times if step['duration'] > 5.0]
        bottlenecks.sort(key=lambda x: x['duration'], reverse=True)
        
        # 메모리 사용량 분석
        memory_growth = []
        if len(self.memory_snapshots) > 1:
            for i in range(1, len(self.memory_snapshots)):
                prev_mem = self.memory_snapshots[i-1]['process_memory_mb']
                curr_mem = self.memory_snapshots[i]['process_memory_mb']
                growth = curr_mem - prev_mem
                
                memory_growth.append({
                    'stage': self.memory_snapshots[i]['stage'],
                    'growth_mb': growth,
                    'total_mb': curr_mem
                })
        
        analysis = {
            'total_time': self.checkpoints[-1]['time'] if self.checkpoints else 0,
            'step_count': len(self.checkpoints),
            'bottlenecks': bottlenecks,
            'memory_growth': memory_growth,
            'io_operations': len(self.io_operations)
        }
        
        return analysis
    
    def print_detailed_report(self):
        """상세한 분석 리포트 출력"""
        analysis = self.analyze_bottlenecks()
        
        print("\n" + "="*80)
        print("🔍 상세한 모델 로딩 병목 분석 리포트")
        print("="*80)
        
        print(f"📊 전체 요약:")
        print(f"   총 소요시간: {analysis['total_time']:.1f}초")
        print(f"   총 단계 수: {analysis['step_count']}")
        print(f"   I/O 작업 수: {analysis['io_operations']}")
        
        print(f"\n🚨 주요 병목 지점 (5초 이상):")
        if analysis['bottlenecks']:
            for i, bottleneck in enumerate(analysis['bottlenecks'][:5], 1):
                print(f"   {i}. {bottleneck['step']}")
                print(f"      ⏱️  소요시간: {bottleneck['duration']:.1f}초")
                print(f"      📍 누적시간: {bottleneck['cumulative']:.1f}초")
                print()
        else:
            print("   ✅ 5초 이상 걸리는 단계 없음")
        
        print(f"💾 메모리 사용량 변화:")
        if analysis['memory_growth']:
            for growth in analysis['memory_growth']:
                if growth['growth_mb'] > 10:  # 10MB 이상 증가한 경우만
                    print(f"   📈 {growth['stage']}: +{growth['growth_mb']:.1f}MB "
                          f"(총 {growth['total_mb']:.1f}MB)")
        
        print(f"\n💡 최적화 제안:")
        
        # 병목 기반 제안
        if analysis['bottlenecks']:
            max_bottleneck = analysis['bottlenecks'][0]
            if "모델 생성" in max_bottleneck['step']:
                print("   🎯 모델 생성이 가장 큰 병목입니다")
                print("   💊 해결책: from_config 대신 직접 텐서 로딩 사용")
            elif "텐서 할당" in max_bottleneck['step']:
                print("   🎯 텐서 할당이 가장 큰 병목입니다")
                print("   💊 해결책: 병렬 텐서 로딩 또는 메모리 매핑 사용")
            elif "from_pretrained" in max_bottleneck['step']:
                print("   🎯 transformers from_pretrained가 가장 큰 병목입니다")
                print("   💊 해결책: 환경 변수 최적화 및 검증 우회")
        
        print("="*80)

# 전역 프로파일러
profiler = DetailedProfiler()