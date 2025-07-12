"""
고성능 모델 캐싱 시스템
"""
import os
import pickle
import hashlib
import time
import tempfile
from typing import Dict, Optional, Any, Tuple
import torch
import psutil

class HighPerformanceModelCache:
    """고성능 모델 캐싱 시스템"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_gb: float = 2.0):
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), "hf_model_cache")
        self.max_cache_size_gb = max_cache_size_gb
        self.memory_cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict] = {}
        
        # 캐시 디렉토리 생성
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"[CACHE] 모델 캐시 초기화: {self.cache_dir}")
    
    def _get_cache_key(self, model_path: str, model_config: Dict) -> str:
        """모델 경로와 설정으로 캐시 키 생성"""
        cache_data = f"{model_path}_{str(sorted(model_config.items()))}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _get_model_size_mb(self, model) -> float:
        """모델 메모리 사용량 계산 (MB)"""
        try:
            if hasattr(model, 'get_memory_footprint'):
                return model.get_memory_footprint() / (1024 * 1024)
            else:
                # 대략적인 크기 계산
                total_params = sum(p.numel() for p in model.parameters())
                bytes_per_param = 4  # float32 기준
                return (total_params * bytes_per_param) / (1024 * 1024)
        except:
            return 0.0
    
    def _can_cache_in_memory(self, model_size_mb: float) -> bool:
        """메모리 캐시 가능 여부 확인"""
        memory = psutil.virtual_memory()
        available_gb = memory.available / (1024**3)
        
        # 사용 가능한 메모리의 50% 이하만 사용
        max_cache_mb = min(self.max_cache_size_gb * 1024, available_gb * 1024 * 0.5)
        
        current_cache_size = sum(
            self.cache_metadata.get(key, {}).get('size_mb', 0) 
            for key in self.memory_cache.keys()
        )
        
        return (current_cache_size + model_size_mb) <= max_cache_mb
    
    def _evict_lru_models(self, required_space_mb: float):
        """LRU 방식으로 모델 제거"""
        if not self.memory_cache:
            return
        
        # 액세스 시간 기준으로 정렬
        lru_keys = sorted(
            self.memory_cache.keys(),
            key=lambda k: self.cache_metadata.get(k, {}).get('last_access', 0)
        )
        
        freed_space = 0.0
        for key in lru_keys:
            if freed_space >= required_space_mb:
                break
                
            model_size = self.cache_metadata.get(key, {}).get('size_mb', 0)
            
            # 메모리에서 모델 제거
            if key in self.memory_cache:
                del self.memory_cache[key]
                print(f"[CACHE] LRU 제거: {key[:8]}... ({model_size:.1f}MB)")
                
            freed_space += model_size
    
    def cache_model(self, cache_key: str, model, tokenizer, model_path: str, config: Dict):
        """모델을 캐시에 저장"""
        try:
            model_size_mb = self._get_model_size_mb(model)
            
            # 메모리 캐시 가능 여부 확인
            if self._can_cache_in_memory(model_size_mb):
                # 필요시 LRU 제거
                if not self._can_cache_in_memory(model_size_mb):
                    self._evict_lru_models(model_size_mb)
                
                # 메모리 캐시에 저장
                self.memory_cache[cache_key] = {
                    'model': model,
                    'tokenizer': tokenizer,
                    'model_path': model_path,
                    'config': config
                }
                
                self.cache_metadata[cache_key] = {
                    'size_mb': model_size_mb,
                    'cached_at': time.time(),
                    'last_access': time.time(),
                    'access_count': 1,
                    'location': 'memory'
                }
                
                print(f"[CACHE] 메모리 캐시 저장: {cache_key[:8]}... ({model_size_mb:.1f}MB)")
                return True
            else:
                print(f"[CACHE] 모델이 너무 큼, 메모리 캐시 건너뜀: {model_size_mb:.1f}MB")
                return False
                
        except Exception as e:
            print(f"[CACHE] 캐시 저장 실패: {e}")
            return False
    
    def get_cached_model(self, cache_key: str) -> Optional[Tuple[Any, Any]]:
        """캐시에서 모델 조회"""
        try:
            # 메모리 캐시 확인
            if cache_key in self.memory_cache:
                cached_data = self.memory_cache[cache_key]
                
                # 액세스 시간 업데이트
                if cache_key in self.cache_metadata:
                    self.cache_metadata[cache_key]['last_access'] = time.time()
                    self.cache_metadata[cache_key]['access_count'] += 1
                
                print(f"[CACHE] 메모리 캐시 히트: {cache_key[:8]}...")
                return cached_data['model'], cached_data['tokenizer']
            
            return None
            
        except Exception as e:
            print(f"[CACHE] 캐시 조회 실패: {e}")
            return None
    
    def clear_cache(self):
        """전체 캐시 정리"""
        try:
            self.memory_cache.clear()
            self.cache_metadata.clear()
            
            # GPU 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("[CACHE] 캐시 정리 완료")
            
        except Exception as e:
            print(f"[CACHE] 캐시 정리 실패: {e}")
    
    def get_cache_stats(self) -> Dict:
        """캐시 통계 정보"""
        total_size_mb = sum(
            meta.get('size_mb', 0) for meta in self.cache_metadata.values()
        )
        
        memory = psutil.virtual_memory()
        
        return {
            'cached_models': len(self.memory_cache),
            'total_cache_size_mb': total_size_mb,
            'total_cache_size_gb': total_size_mb / 1024,
            'max_cache_size_gb': self.max_cache_size_gb,
            'memory_usage_percent': memory.percent,
            'available_memory_gb': memory.available / (1024**3),
            'cache_hit_rate': self._calculate_hit_rate()
        }
    
    def _calculate_hit_rate(self) -> float:
        """캐시 히트율 계산"""
        if not self.cache_metadata:
            return 0.0
        
        total_accesses = sum(
            meta.get('access_count', 0) for meta in self.cache_metadata.values()
        )
        
        if total_accesses == 0:
            return 0.0
        
        hits = len(self.cache_metadata)
        return (hits / total_accesses) * 100.0

# 전역 캐시 인스턴스
model_cache = HighPerformanceModelCache()