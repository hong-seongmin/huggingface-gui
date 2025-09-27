"""
System monitoring UI components for Streamlit.
"""
import streamlit as st
import time
from typing import Dict, Any, Optional
from core.state_manager import state_manager
from core.logging_config import get_logger
from utils.helpers import should_perform_expensive_check

logger = get_logger(__name__)


def render_monitoring_controls():
    """Render monitoring control buttons."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("🚀 모니터링 시작"):
            start_monitoring()

    with col2:
        if st.button("⏹️ 모니터링 중지"):
            stop_monitoring()

    with col3:
        if st.button("🔄 새로고침"):
            st.rerun()

    with col4:
        render_refresh_interval_selector()


def start_monitoring():
    """Start system monitoring."""
    try:
        if 'system_monitor' in st.session_state:
            st.session_state['system_monitor'].start_monitoring()

        st.session_state['monitoring_active'] = True
        st.session_state['refresh_count'] = 0

        # Save state
        state_manager.save_state(st.session_state)

        logger.info("시스템 모니터링 시작")
        st.success("모니터링이 시작되었습니다.")
        st.rerun()

    except Exception as e:
        error_msg = f"모니터링 시작 실패: {e}"
        st.error(error_msg)
        logger.error(error_msg)


def stop_monitoring():
    """Stop system monitoring."""
    try:
        if 'system_monitor' in st.session_state:
            st.session_state['system_monitor'].stop_monitoring()

        st.session_state['auto_refresh_interval'] = 0
        st.session_state['monitoring_active'] = False

        logger.info("시스템 모니터링 중지")
        st.info("모니터링이 중지되었습니다.")

    except Exception as e:
        error_msg = f"모니터링 중지 실패: {e}"
        st.error(error_msg)
        logger.error(error_msg)


def render_refresh_interval_selector():
    """Render auto refresh interval selector."""
    refresh_options = {
        "자동 갱신 끄기": 0,
        "1초마다": 1,
        "3초마다": 3,
        "10초마다": 10
    }

    # Get current interval
    current_interval = st.session_state.get('auto_refresh_interval', 0)
    current_key = next((k for k, v in refresh_options.items() if v == current_interval), "자동 갱신 끄기")

    selected_refresh = st.selectbox(
        "자동 갱신",
        options=list(refresh_options.keys()),
        index=list(refresh_options.keys()).index(current_key)
    )

    # Update interval if changed
    new_interval = refresh_options[selected_refresh]
    if new_interval != st.session_state.get('auto_refresh_interval', 0):
        st.session_state['auto_refresh_interval'] = new_interval
        state_manager.save_state(st.session_state)
        logger.info(f"자동 갱신 간격 변경: {new_interval}초")
    else:
        st.session_state['auto_refresh_interval'] = new_interval


def render_monitoring_status():
    """Render monitoring status banner."""
    if st.session_state.get('monitoring_active', False):
        refresh_status = f"자동 갱신: {st.session_state.get('auto_refresh_interval', 0)}초" if st.session_state.get('auto_refresh_interval', 0) > 0 else "수동 갱신"
        st.success(f"🟢 **모니터링 상태**: 활성화됨 ({refresh_status})")
    else:
        st.warning("🟡 **모니터링 상태**: 비활성화됨 - 아래 버튼으로 시작하세요")


def render_realtime_system_charts():
    """Render real-time system monitoring charts."""
    auto_refresh_interval = st.session_state.get('auto_refresh_interval', 0)
    monitoring_active = st.session_state.get('monitoring_active', False)

    if not monitoring_active:
        st.info("모니터링을 시작하면 실시간 차트가 표시됩니다.")
        return

    # Check SystemMonitor availability
    system_monitor = st.session_state.get('system_monitor')
    if system_monitor is None:
        st.error("시스템 모니터가 초기화되지 않았습니다. 실시간 차트를 표시할 수 없습니다.")
        return

    # Generate unique chart container ID
    chart_container_id = f"realtime_chart_{int(time.time())}"

    # Create real-time chart HTML with JavaScript
    realtime_chart_html = generate_realtime_chart_html(chart_container_id, auto_refresh_interval)

    # Render the chart
    st.components.v1.html(realtime_chart_html, height=650, scrolling=False)

    # Show auto refresh status
    if monitoring_active and auto_refresh_interval > 0:
        if 'refresh_count' not in st.session_state:
            st.session_state['refresh_count'] = 0

        st.success(f"🔄 **실시간 차트 자동 갱신 활성화** ({auto_refresh_interval}초 간격)")


def generate_realtime_chart_html(container_id: str, refresh_interval: int) -> str:
    """
    Generate HTML for real-time monitoring chart.

    Args:
        container_id: Unique container ID
        refresh_interval: Refresh interval in seconds

    Returns:
        HTML string for the chart
    """
    return f"""
    <div id="{container_id}" style="width:100%; height:600px; min-width:800px;"></div>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script>
    // 실시간 데이터 저장소
    let chartData = {{
        cpu: {{x: [], y: []}},
        memory: {{x: [], y: []}},
        gpu: {{x: [], y: []}},
        disk: {{x: [], y: []}}
    }};

    // 차트 레이아웃 설정
    let layout = {{
        title: '🔄 실시간 시스템 모니터링',
        grid: {{rows: 2, columns: 2, pattern: 'independent'}},
        width: null,
        height: 600,
        autosize: true,
        showlegend: true,
        annotations: [
            {{text: 'CPU 사용률', x: 0.2, y: 0.9, xref: 'paper', yref: 'paper', showarrow: false}},
            {{text: '메모리 사용률', x: 0.8, y: 0.9, xref: 'paper', yref: 'paper', showarrow: false}},
            {{text: 'GPU 사용률', x: 0.2, y: 0.4, xref: 'paper', yref: 'paper', showarrow: false}},
            {{text: '디스크 사용률', x: 0.8, y: 0.4, xref: 'paper', yref: 'paper', showarrow: false}}
        ]
    }};

    // 초기 차트 트레이스
    let traces = [
        {{x: [], y: [], name: 'CPU %', line: {{color: 'blue'}}, xaxis: 'x1', yaxis: 'y1'}},
        {{x: [], y: [], name: 'Memory %', line: {{color: 'red'}}, xaxis: 'x2', yaxis: 'y2'}},
        {{x: [], y: [], name: 'GPU %', line: {{color: 'green'}}, xaxis: 'x3', yaxis: 'y3'}},
        {{x: [], y: [], name: 'Disk %', line: {{color: 'purple'}}, xaxis: 'x4', yaxis: 'y4'}}
    ];

    // 차트 생성
    Plotly.newPlot('{container_id}', traces, layout, {{responsive: true}});

    // 초기 리사이즈
    setTimeout(function() {{
        Plotly.Plots.resize('{container_id}');
    }}, 100);

    // 창 크기 변경 시 자동 리사이즈
    window.addEventListener('resize', function() {{
        Plotly.Plots.resize('{container_id}');
    }});

    // 실시간 데이터 업데이트 함수
    function updateChartData() {{
        let now = new Date();

        // 실제 시스템 메트릭 데이터 가져오기
        fetch('/system/metrics')
            .then(response => response.json())
            .then(data => {{
                if (data.status === 'success') {{
                    let cpuUsage = data.cpu;
                    let memoryUsage = data.memory;
                    let gpuUsage = data.gpu;
                    let diskUsage = data.disk;

                    updateChartWithData(now, cpuUsage, memoryUsage, gpuUsage, diskUsage);
                }} else {{
                    console.error('Failed to fetch system metrics:', data);
                    // 오류 시 0으로 설정
                    updateChartWithData(now, 0, 0, 0, 0);
                }}
            }})
            .catch(error => {{
                console.error('Error fetching system metrics:', error);
                // 네트워크 오류 시 0으로 설정
                updateChartWithData(now, 0, 0, 0, 0);
            }});
    }}

    // 차트 데이터 업데이트 헬퍼 함수
    function updateChartWithData(now, cpuUsage, memoryUsage, gpuUsage, diskUsage) {{

        // 데이터 추가
        chartData.cpu.x.push(now);
        chartData.cpu.y.push(cpuUsage);
        chartData.memory.x.push(now);
        chartData.memory.y.push(memoryUsage);
        chartData.gpu.x.push(now);
        chartData.gpu.y.push(gpuUsage);
        chartData.disk.x.push(now);
        chartData.disk.y.push(diskUsage);

        // 데이터 길이 제한 (최근 50개 포인트만 유지)
        const maxPoints = 50;
        Object.values(chartData).forEach(series => {{
            if (series.x.length > maxPoints) {{
                series.x.shift();
                series.y.shift();
            }}
        }});

        // 차트 업데이트
        let update = {{
            x: [chartData.cpu.x, chartData.memory.x, chartData.gpu.x, chartData.disk.x],
            y: [chartData.cpu.y, chartData.memory.y, chartData.gpu.y, chartData.disk.y]
        }};

        Plotly.redraw('{container_id}', update);
    }}

    // 자동 갱신 설정
    {f"setInterval(updateChartData, {refresh_interval * 1000});" if refresh_interval > 0 else ""}

    // 초기 데이터 로드
    updateChartData();
    </script>
    """


def render_system_stats():
    """Render system statistics cards."""
    if not st.session_state.get('monitoring_active', False):
        st.info("모니터링을 시작하면 시스템 통계가 표시됩니다.")
        return

    # Check SystemMonitor availability
    system_monitor = st.session_state.get('system_monitor')
    if system_monitor is None:
        st.error("시스템 모니터가 초기화되지 않았습니다.")
        return

    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)

    try:
        # Get real system data from SystemMonitor
        system_data = get_current_system_data()

        with col1:
            st.metric("CPU 사용률", f"{system_data.get('cpu', 0):.1f}%")

        with col2:
            st.metric("메모리 사용률", f"{system_data.get('memory', 0):.1f}%")

        with col3:
            st.metric("GPU 사용률", f"{system_data.get('gpu', 0):.1f}%")

        with col4:
            st.metric("디스크 사용률", f"{system_data.get('disk', 0):.1f}%")

    except RuntimeError as e:
        st.error(f"시스템 데이터 조회 실패: {e}")
    except Exception as e:
        st.error(f"예상치 못한 오류: {e}")
        logger.error(f"System stats error: {e}")


def get_current_system_data() -> Dict[str, float]:
    """
    Get current system data from SystemMonitor instance.

    Returns:
        Dictionary with system metrics
    """
    import streamlit as st

    system_monitor = st.session_state.get('system_monitor')
    if system_monitor is None:
        raise RuntimeError("SystemMonitor not initialized")

    try:
        # Get real data from SystemMonitor
        data = system_monitor.get_current_data()

        # Extract GPU average if available
        gpu_percent = 0.0
        if data.get('gpu') and len(data['gpu']) > 0:
            gpu_percent = sum(gpu['load'] for gpu in data['gpu']) / len(data['gpu'])

        return {
            'cpu': data['cpu']['percent'],
            'memory': data['memory']['percent'],
            'disk': data['disk']['percent'],
            'gpu': gpu_percent
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise RuntimeError(f"SystemMonitor error: {e}")


def render_system_monitoring():
    """Render complete system monitoring UI."""
    st.subheader("🖥️ 시스템 리소스 모니터링")

    # Initialize session state
    init_monitoring_state()

    # Status banner
    render_monitoring_status()

    # Control buttons
    render_monitoring_controls()

    st.divider()

    # System statistics
    render_system_stats()

    st.divider()

    # Real-time charts
    render_realtime_system_charts()

    # Monitoring tips
    with st.expander("💡 모니터링 팁"):
        st.markdown("""
        - **CPU**: 높은 사용률이 지속되면 처리 성능이 저하될 수 있습니다
        - **메모리**: 90% 이상 사용 시 시스템이 느려질 수 있습니다
        - **GPU**: 모델 로딩/추론 시 GPU 사용률이 높아집니다
        - **디스크**: 95% 이상 사용 시 시스템 불안정성이 증가합니다
        - **자동 갱신**: 짧은 간격일수록 시스템 부하가 증가합니다
        """)


def init_monitoring_state():
    """Initialize monitoring-related session state."""
    if 'auto_refresh_interval' not in st.session_state:
        st.session_state['auto_refresh_interval'] = 0

    if 'last_refresh_time' not in st.session_state:
        st.session_state['last_refresh_time'] = 0

    if 'monitoring_active' not in st.session_state:
        st.session_state['monitoring_active'] = False

    if 'refresh_count' not in st.session_state:
        st.session_state['refresh_count'] = 0


def should_auto_refresh() -> bool:
    """
    Check if auto refresh should occur.

    Returns:
        True if auto refresh should happen
    """
    auto_refresh_interval = st.session_state.get('auto_refresh_interval', 0)
    monitoring_active = st.session_state.get('monitoring_active', False)

    if not monitoring_active or auto_refresh_interval <= 0:
        return False

    current_time = time.time()
    last_refresh = st.session_state.get('last_refresh_time', 0)

    if current_time - last_refresh >= auto_refresh_interval:
        st.session_state['last_refresh_time'] = current_time
        return True

    return False