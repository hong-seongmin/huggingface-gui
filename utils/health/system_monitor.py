"""
System Monitor for health checking.

This module monitors system resources including CPU, memory, and disk usage.
It provides real-time monitoring and threshold-based alerting.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime


class SystemMonitor:
    """Monitors system resource usage and performance metrics."""

    def __init__(self, thresholds: Optional[Dict[str, Dict[str, float]]] = None):
        """Initialize system monitor with configurable thresholds."""
        self.logger = logging.getLogger("SystemMonitor")

        # Default thresholds
        self.thresholds = thresholds or {
            'cpu': {'warning': 70.0, 'critical': 90.0},
            'memory': {'warning': 75.0, 'critical': 90.0},
            'disk': {'warning': 85.0, 'critical': 95.0}
        }

        self.monitoring_data = []

    def check_system_resources(self) -> Dict[str, Any]:
        """Check current system resource usage."""
        try:
            self.logger.info("[MONITOR] Checking system resources")

            resource_data = {
                'timestamp': datetime.now().isoformat(),
                'cpu': self._check_cpu_usage(),
                'memory': self._check_memory_usage(),
                'disk': self._check_disk_usage(),
                'processes': self._check_process_count(),
                'alerts': []
            }

            # Check thresholds and generate alerts
            resource_data['alerts'] = self._check_thresholds(resource_data)

            # Store monitoring data
            self.monitoring_data.append(resource_data)

            # Limit stored data to last 100 checks
            if len(self.monitoring_data) > 100:
                self.monitoring_data = self.monitoring_data[-100:]

            return resource_data

        except Exception as e:
            self.logger.error(f"[MONITOR] System resource check failed: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'alerts': [{'level': 'error', 'message': 'System monitoring failed'}]
            }

    def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage statistics."""
        try:
            # Try to import psutil
            import psutil

            # Get CPU usage with a short interval
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_count_logical = psutil.cpu_count(logical=True)

            # Get per-CPU usage
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)

            # Get load average (Unix-like systems)
            load_avg = None
            try:
                load_avg = psutil.getloadavg()
            except (AttributeError, OSError):
                # Windows doesn't have load average
                pass

            cpu_data = {
                'usage_percent': round(cpu_percent, 1),
                'count_physical': cpu_count,
                'count_logical': cpu_count_logical,
                'per_core_usage': [round(usage, 1) for usage in cpu_per_core],
                'load_average': load_avg
            }

            # Determine status
            if cpu_percent >= self.thresholds['cpu']['critical']:
                cpu_data['status'] = 'critical'
            elif cpu_percent >= self.thresholds['cpu']['warning']:
                cpu_data['status'] = 'warning'
            else:
                cpu_data['status'] = 'ok'

            return cpu_data

        except ImportError:
            return {
                'usage_percent': 'unknown',
                'status': 'unknown',
                'error': 'psutil not available'
            }
        except Exception as e:
            return {
                'usage_percent': 'unknown',
                'status': 'error',
                'error': str(e)
            }

    def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage statistics."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()

            memory_data = {
                'total_gb': round(memory.total / 1024 / 1024 / 1024, 2),
                'available_gb': round(memory.available / 1024 / 1024 / 1024, 2),
                'used_gb': round(memory.used / 1024 / 1024 / 1024, 2),
                'usage_percent': round(memory.percent, 1),
                'swap_total_gb': round(swap.total / 1024 / 1024 / 1024, 2),
                'swap_used_gb': round(swap.used / 1024 / 1024 / 1024, 2),
                'swap_percent': round(swap.percent, 1)
            }

            # Determine status
            if memory.percent >= self.thresholds['memory']['critical']:
                memory_data['status'] = 'critical'
            elif memory.percent >= self.thresholds['memory']['warning']:
                memory_data['status'] = 'warning'
            else:
                memory_data['status'] = 'ok'

            return memory_data

        except ImportError:
            return {
                'usage_percent': 'unknown',
                'status': 'unknown',
                'error': 'psutil not available'
            }
        except Exception as e:
            return {
                'usage_percent': 'unknown',
                'status': 'error',
                'error': str(e)
            }

    def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage statistics."""
        try:
            import psutil

            # Check main disk (current directory)
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100

            disk_data = {
                'total_gb': round(disk.total / 1024 / 1024 / 1024, 2),
                'used_gb': round(disk.used / 1024 / 1024 / 1024, 2),
                'free_gb': round(disk.free / 1024 / 1024 / 1024, 2),
                'usage_percent': round(disk_percent, 1)
            }

            # Get disk I/O stats if available
            try:
                disk_io = psutil.disk_io_counters()
                if disk_io:
                    disk_data['read_mb'] = round(disk_io.read_bytes / 1024 / 1024, 2)
                    disk_data['write_mb'] = round(disk_io.write_bytes / 1024 / 1024, 2)
            except Exception:
                pass

            # Determine status
            if disk_percent >= self.thresholds['disk']['critical']:
                disk_data['status'] = 'critical'
            elif disk_percent >= self.thresholds['disk']['warning']:
                disk_data['status'] = 'warning'
            else:
                disk_data['status'] = 'ok'

            return disk_data

        except ImportError:
            return {
                'usage_percent': 'unknown',
                'status': 'unknown',
                'error': 'psutil not available'
            }
        except Exception as e:
            return {
                'usage_percent': 'unknown',
                'status': 'error',
                'error': str(e)
            }

    def _check_process_count(self) -> Dict[str, Any]:
        """Check process count and top processes by resource usage."""
        try:
            import psutil

            # Get total process count
            process_count = len(psutil.pids())

            # Get top processes by CPU and memory usage
            processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            # Sort by CPU usage and get top 5
            top_cpu_processes = sorted(
                processes, key=lambda x: x.get('cpu_percent', 0), reverse=True
            )[:5]

            # Sort by memory usage and get top 5
            top_memory_processes = sorted(
                processes, key=lambda x: x.get('memory_percent', 0), reverse=True
            )[:5]

            return {
                'total_count': process_count,
                'top_cpu_processes': top_cpu_processes,
                'top_memory_processes': top_memory_processes
            }

        except ImportError:
            return {
                'total_count': 'unknown',
                'error': 'psutil not available'
            }
        except Exception as e:
            return {
                'total_count': 'unknown',
                'error': str(e)
            }

    def _check_thresholds(self, resource_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check resource usage against thresholds and generate alerts."""
        alerts = []

        # CPU alerts
        cpu_data = resource_data.get('cpu', {})
        cpu_usage = cpu_data.get('usage_percent')
        if isinstance(cpu_usage, (int, float)):
            if cpu_usage >= self.thresholds['cpu']['critical']:
                alerts.append({
                    'level': 'critical',
                    'category': 'cpu',
                    'message': f'Critical CPU usage: {cpu_usage}%'
                })
            elif cpu_usage >= self.thresholds['cpu']['warning']:
                alerts.append({
                    'level': 'warning',
                    'category': 'cpu',
                    'message': f'High CPU usage: {cpu_usage}%'
                })

        # Memory alerts
        memory_data = resource_data.get('memory', {})
        memory_usage = memory_data.get('usage_percent')
        if isinstance(memory_usage, (int, float)):
            if memory_usage >= self.thresholds['memory']['critical']:
                alerts.append({
                    'level': 'critical',
                    'category': 'memory',
                    'message': f'Critical memory usage: {memory_usage}%'
                })
            elif memory_usage >= self.thresholds['memory']['warning']:
                alerts.append({
                    'level': 'warning',
                    'category': 'memory',
                    'message': f'High memory usage: {memory_usage}%'
                })

        # Disk alerts
        disk_data = resource_data.get('disk', {})
        disk_usage = disk_data.get('usage_percent')
        if isinstance(disk_usage, (int, float)):
            if disk_usage >= self.thresholds['disk']['critical']:
                alerts.append({
                    'level': 'critical',
                    'category': 'disk',
                    'message': f'Critical disk usage: {disk_usage}%'
                })
            elif disk_usage >= self.thresholds['disk']['warning']:
                alerts.append({
                    'level': 'warning',
                    'category': 'disk',
                    'message': f'High disk usage: {disk_usage}%'
                })

        return alerts

    def get_monitoring_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent monitoring data."""
        return self.monitoring_data[-limit:] if self.monitoring_data else []

    def get_system_summary(self) -> Dict[str, Any]:
        """Get a summarized view of system status."""
        latest_data = self.check_system_resources()

        summary = {
            'timestamp': latest_data['timestamp'],
            'overall_status': 'ok',
            'cpu_status': latest_data.get('cpu', {}).get('status', 'unknown'),
            'memory_status': latest_data.get('memory', {}).get('status', 'unknown'),
            'disk_status': latest_data.get('disk', {}).get('status', 'unknown'),
            'alert_count': len(latest_data.get('alerts', [])),
            'critical_alerts': len([
                alert for alert in latest_data.get('alerts', [])
                if alert.get('level') == 'critical'
            ])
        }

        # Determine overall status
        statuses = [summary['cpu_status'], summary['memory_status'], summary['disk_status']]
        if 'critical' in statuses or summary['critical_alerts'] > 0:
            summary['overall_status'] = 'critical'
        elif 'warning' in statuses:
            summary['overall_status'] = 'warning'
        elif 'error' in statuses:
            summary['overall_status'] = 'error'

        return summary

    def set_thresholds(self, thresholds: Dict[str, Dict[str, float]]) -> None:
        """Update monitoring thresholds."""
        self.thresholds.update(thresholds)
        self.logger.info(f"[MONITOR] Updated monitoring thresholds: {thresholds}")

    def clear_history(self) -> None:
        """Clear monitoring history."""
        self.monitoring_data.clear()
        self.logger.info("[MONITOR] Cleared monitoring history")

    def is_psutil_available(self) -> bool:
        """Check if psutil is available for monitoring."""
        try:
            import psutil
            return True
        except ImportError:
            return False