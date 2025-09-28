import psutil
import threading
import time
from datetime import datetime
from typing import Dict, List, Callable, Optional
import json
import logging

# Import our adaptive GPU detector
try:
    from utils.gpu_detector import gpu_detector
    GPU_DETECTOR_AVAILABLE = True
except ImportError:
    GPU_DETECTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.monitoring = False
        self.monitor_thread = None
        self.callbacks: List[Callable] = []
        self.history = {
            'cpu': [],
            'memory': [],
            'gpu': [],
            'disk': []
        }
        self.max_history_length = 100
        
    def add_callback(self, callback: Callable):
        """모니터링 데이터 업데이트 콜백 등록"""
        self.callbacks.append(callback)
    
    def start_monitoring(self):
        """모니터링 시작"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_monitoring(self):
        """모니터링 중지"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """모니터링 루프"""
        while self.monitoring:
            try:
                data = self._collect_system_data()
                self._update_history(data)
                
                # 콜백 함수들에 데이터 전달
                for callback in self.callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        print(f"Callback error: {e}")
                    
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(1)
    
    def _collect_system_data(self) -> Dict:
        """시스템 데이터 수집"""
        # CPU 정보
        cpu_percent = psutil.cpu_percent(interval=None)
        cpu_freq = psutil.cpu_freq()
        cpu_count = psutil.cpu_count()
        cpu_count_logical = psutil.cpu_count(logical=True)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # GPU 정보 (adaptive detection 사용)
        gpu_info = []
        if GPU_DETECTOR_AVAILABLE:
            try:
                gpu_list = gpu_detector.get_gpu_info()
                for gpu in gpu_list:
                    gpu_info.append({
                        'id': gpu.id,
                        'name': gpu.name,
                        'load': gpu.load,
                        'memory_util': gpu.memory_util,
                        'memory_total': gpu.memory_total,
                        'memory_used': gpu.memory_used,
                        'memory_free': gpu.memory_free,
                        'temperature': gpu.temperature,
                        'uuid': gpu.uuid
                    })
                logger.debug(f"GPU 정보 수집: {len(gpu_info)}개")
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
        else:
            logger.debug("GPU detector 사용 불가")
        
        # 디스크 정보
        disk_usage = psutil.disk_usage('/')
        
        # 네트워크 정보
        net_io = psutil.net_io_counters()
        
        return {
            'timestamp': datetime.now(),
            'cpu': {
                'percent': cpu_percent,
                'frequency': cpu_freq.current if cpu_freq else 0,
                'frequency_max': cpu_freq.max if cpu_freq else 0,
                'cores': cpu_count,
                'logical_cores': cpu_count_logical,
                'per_cpu': psutil.cpu_percent(percpu=True)
            },
            'memory': {
                'total': memory.total,
                'available': memory.available,
                'used': memory.used,
                'percent': memory.percent,
                'swap_total': swap.total,
                'swap_used': swap.used,
                'swap_percent': swap.percent,
                'buffers': memory.buffers,
                'cached': memory.cached
            },
            'gpu': gpu_info,
            'disk': {
                'total': disk_usage.total,
                'used': disk_usage.used,
                'free': disk_usage.free,
                'percent': (disk_usage.used / disk_usage.total) * 100
            },
            'network': {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        }
    
    def _update_history(self, data: Dict):
        """히스토리 업데이트"""
        timestamp = data['timestamp']
        
        # CPU 히스토리
        self.history['cpu'].append({
            'timestamp': timestamp,
            'percent': data['cpu']['percent'],
            'frequency': data['cpu']['frequency']
        })
        
        # 메모리 히스토리
        self.history['memory'].append({
            'timestamp': timestamp,
            'percent': data['memory']['percent'],
            'used_gb': data['memory']['used'] / (1024**3),
            'available_gb': data['memory']['available'] / (1024**3)
        })
        
        # GPU 히스토리
        gpu_snapshot = []
        for gpu in data['gpu']:
            gpu_snapshot.append({
                'timestamp': timestamp,
                'gpu_id': gpu['id'],
                'load': gpu['load'],
                'memory_util': gpu['memory_util'],
                'temperature': gpu['temperature']
            })
        self.history['gpu'].append(gpu_snapshot)
        
        # 디스크 히스토리
        self.history['disk'].append({
            'timestamp': timestamp,
            'percent': data['disk']['percent'],
            'used_gb': data['disk']['used'] / (1024**3),
            'free_gb': data['disk']['free'] / (1024**3)
        })
        
        # 히스토리 길이 제한
        for key in self.history:
            if len(self.history[key]) > self.max_history_length:
                self.history[key] = self.history[key][-self.max_history_length:]
    
    def get_current_data(self) -> Dict:
        """현재 시스템 데이터 반환"""
        return self._collect_system_data()
    
    def get_history(self) -> Dict:
        """히스토리 데이터 반환"""
        return self.history
    
    def get_system_info(self) -> Dict:
        """시스템 기본 정보"""
        import platform

        # GPU 정보 수집
        gpu_count = 0
        gpu_info = []
        gpu_status_message = "GPU 정보 없음"

        if GPU_DETECTOR_AVAILABLE:
            try:
                status = gpu_detector.get_status()
                gpu_count = len(status.gpus)
                gpu_info = [{'name': gpu.name, 'memory': gpu.memory_total} for gpu in status.gpus]
                gpu_status_message = gpu_detector.get_status_message()
            except Exception as e:
                logger.warning(f"GPU 정보 수집 실패: {e}")

        return {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'architecture': platform.architecture(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total': psutil.virtual_memory().total,
            'gpu_count': gpu_count,
            'gpu_info': gpu_info,
            'gpu_status': gpu_status_message
        }
    
    def get_alerts(self) -> List[Dict]:
        """시스템 알림 생성"""
        alerts = []
        current_data = self.get_current_data()
        
        # CPU 사용률 높음
        if current_data['cpu']['percent'] > 80:
            alerts.append({
                'type': 'warning',
                'message': f"High CPU usage: {current_data['cpu']['percent']:.1f}%",
                'timestamp': current_data['timestamp']
            })
        
        # 메모리 사용률 높음
        if current_data['memory']['percent'] > 80:
            alerts.append({
                'type': 'warning',
                'message': f"High memory usage: {current_data['memory']['percent']:.1f}%",
                'timestamp': current_data['timestamp']
            })
        
        # GPU 온도 높음
        for gpu in current_data['gpu']:
            if gpu['temperature'] > 80:
                alerts.append({
                    'type': 'warning',
                    'message': f"High GPU temperature: {gpu['temperature']}°C on {gpu['name']}",
                    'timestamp': current_data['timestamp']
                })
        
        # 디스크 사용률 높음
        if current_data['disk']['percent'] > 90:
            alerts.append({
                'type': 'error',
                'message': f"High disk usage: {current_data['disk']['percent']:.1f}%",
                'timestamp': current_data['timestamp']
            })
        
        return alerts
    
    def export_data(self, filename: Optional[str] = None) -> str:
        """모니터링 데이터 내보내기"""
        if not filename:
            filename = f"system_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            'system_info': self.get_system_info(),
            'current_data': self.get_current_data(),
            'history': self.history,
            'alerts': self.get_alerts(),
            'export_timestamp': datetime.now().isoformat()
        }
        
        # JSON 직렬화를 위한 datetime 처리
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, default=serialize_datetime, indent=2)
        
        return filename
    
    def clear_history(self):
        """히스토리 초기화"""
        self.history = {
            'cpu': [],
            'memory': [],
            'gpu': [],
            'disk': []
        }
    
    def get_resource_usage_summary(self) -> Dict:
        """리소스 사용량 요약"""
        current = self.get_current_data()
        
        return {
            'cpu': {
                'current': current['cpu']['percent'],
                'status': self._get_usage_status(current['cpu']['percent'])
            },
            'memory': {
                'current': current['memory']['percent'],
                'used_gb': current['memory']['used'] / (1024**3),
                'total_gb': current['memory']['total'] / (1024**3),
                'status': self._get_usage_status(current['memory']['percent'])
            },
            'gpu': [
                {
                    'name': gpu['name'],
                    'load': gpu['load'],
                    'memory_util': gpu['memory_util'],
                    'temperature': gpu['temperature'],
                    'status': self._get_usage_status(gpu['load'])
                }
                for gpu in current['gpu']
            ],
            'disk': {
                'current': current['disk']['percent'],
                'used_gb': current['disk']['used'] / (1024**3),
                'total_gb': current['disk']['total'] / (1024**3),
                'status': self._get_usage_status(current['disk']['percent'])
            }
        }
    
    def _get_usage_status(self, percent: float) -> str:
        """사용률에 따른 상태 반환"""
        if percent < 50:
            return 'normal'
        elif percent < 80:
            return 'warning'
        else:
            return 'critical'