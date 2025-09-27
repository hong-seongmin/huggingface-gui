"""
Status tracking service for expensive operations with rate limiting.
"""
import time
from typing import Dict, Any
from core.logging_config import get_logger

logger = get_logger(__name__)

# Global status tracking
_last_checks: Dict[str, float] = {}
_model_status_cache: Dict[str, Any] = {}


def should_perform_expensive_check(operation_name: str, base_interval: int = 30) -> bool:
    """
    비용이 많이 드는 작업을 수행할지 결정합니다.

    Args:
        operation_name: 작업 이름
        base_interval: 기본 간격 (초)

    Returns:
        bool: 작업을 수행할지 여부
    """
    current_time = time.time()
    last_check = _last_checks.get(operation_name, 0)

    # 마지막 체크로부터 충분한 시간이 지났는지 확인
    if current_time - last_check >= base_interval:
        _last_checks[operation_name] = current_time
        logger.debug(f"Expensive check approved for {operation_name}")
        return True

    remaining_time = base_interval - (current_time - last_check)
    logger.debug(f"Expensive check skipped for {operation_name}. Wait {remaining_time:.1f}s")
    return False


def smart_model_status_check() -> Dict[str, Any]:
    """
    스마트 모델 상태 체크 (캐시된 결과 활용).

    Returns:
        Dict containing model status information
    """
    if should_perform_expensive_check("model_status_check", 15):
        try:
            # 실제 상태 체크 로직은 model_manager에서 가져옴
            from model_manager import ModelManager

            if hasattr(ModelManager, '_instance') and ModelManager._instance:
                manager = ModelManager._instance
                status = {
                    'loaded_models': list(manager.loaded_models.keys()),
                    'loading_models': list(manager.loading_models),
                    'failed_models': dict(manager.failed_models),
                    'total_loaded': len(manager.loaded_models),
                    'memory_usage': getattr(manager, '_get_total_memory_usage', lambda: 0)(),
                    'last_updated': time.time()
                }
                _model_status_cache['status'] = status
                logger.debug("Model status updated")
                return status
        except Exception as e:
            logger.error(f"Failed to check model status: {e}")

    # 캐시된 결과 반환 또는 기본값
    return _model_status_cache.get('status', {
        'loaded_models': [],
        'loading_models': [],
        'failed_models': {},
        'total_loaded': 0,
        'memory_usage': 0,
        'last_updated': 0
    })


def is_any_model_loading() -> bool:
    """
    현재 로딩 중인 모델이 있는지 확인합니다.

    Returns:
        bool: 로딩 중인 모델 존재 여부
    """
    status = smart_model_status_check()
    return len(status.get('loading_models', [])) > 0


def cleanup_status_tracker():
    """상태 트래커 정리."""
    global _last_checks, _model_status_cache

    current_time = time.time()
    # 오래된 체크 기록 정리 (1시간 이상 된 것)
    old_keys = []
    for key, timestamp in _last_checks.items():
        if current_time - timestamp > 3600:  # 1 hour
            old_keys.append(key)

    for key in old_keys:
        del _last_checks[key]

    # 캐시된 상태도 정리 (5분 이상 된 것)
    if 'status' in _model_status_cache:
        status = _model_status_cache['status']
        if current_time - status.get('last_updated', 0) > 300:  # 5 minutes
            _model_status_cache.clear()

    logger.debug(f"Status tracker cleaned up. Removed {len(old_keys)} old entries.")


def get_operation_last_check(operation_name: str) -> float:
    """작업의 마지막 체크 시간을 반환합니다."""
    return _last_checks.get(operation_name, 0)


def force_next_check(operation_name: str):
    """다음 체크를 강제로 실행하도록 설정합니다."""
    if operation_name in _last_checks:
        del _last_checks[operation_name]
    logger.debug(f"Forced next check for {operation_name}")


def get_all_cached_statuses() -> Dict[str, Any]:
    """모든 캐시된 상태를 반환합니다."""
    return {
        'last_checks': _last_checks.copy(),
        'model_status_cache': _model_status_cache.copy(),
        'current_time': time.time()
    }