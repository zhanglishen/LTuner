"""
实时监控页面
"""
import streamlit as st
import sys
sys.path.insert(0, '/root/GPTuner/src')
import time
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def initialize_monitor():
    """初始化监控器"""
    if 'monitor_initialized' not in st.session_state:
        try:
            from configparser import ConfigParser
            from dbms.postgres import PgDBMS
            from monitoring.postgres_monitor import PostgreSQLMonitor
            from monitoring.workload_analyzer import WorkloadAnalyzer
            
            config_manager = st.session_state.config_manager
            db_config = config_manager.get_database_config()
            
            # 创建临时配置
            temp_config = ConfigParser()
            temp_config['postgresql'] = {
                'host': db_config['host'],
                'port': str(db_config['port']),
                'user': db_config['user'],
                'passwd': db_config['password'],
                'dbname': db_config['database']
            }
            
            # 初始化数据库和监控器
            dbms = PgDBMS.from_file(temp_config)
            monitor = PostgreSQLMonitor(dbms)
            analyzer = WorkloadAnalyzer()
            
            st.session_state.dbms = dbms
            st.session_state.monitor = monitor
            st.session_state.analyzer = analyzer
            st.session_state.monitor_initialized = True
            st.session_state.monitor_history = []
            
            return True, "监控器初始化成功"
            
        except Exception as e:
            st.session_state.monitor_initialized = False
            return False, f"初始化失败: {str(e)}"
    
    return True, "监控器已初始化"


def collect_metrics():
    """采集性能指标"""
    try:
        monitor = st.session_state.monitor
        analyzer = st.session_state.analyzer
        config_manager = st.session_state.config_manager
        tuning_config = config_manager.get_tuning_config()
        window_minutes = tuning_config['window_minutes']
        
        # 采集所有指标
        metrics = monitor.collect_comprehensive_metrics(window_minutes)
        
        # 分析工作负载
        analysis = analyzer.analyze_comprehensive(metrics)
        
        # 记录到历史
        record = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'analysis': analysis
        }
        
        # 保留最近50条记录
        if 'monitor_history' not in st.session_state:
            st.session_state.monitor_history = []
        
        st.session_state.monitor_history.append(record)
        if len(st.session_state.monitor_history) > 50:
            st.session_state.monitor_history.pop(0)
        
        return metrics, analysis
        
    except Exception as e:
        st.error(f"采集指标失败: {e}")
        return None, None


def show():
    """显示监控页面"""
    st.title("📊 实时监控仪表板")
    st.markdown("---")
    
    config_manager = st.session_state.config_manager
    monitoring_config = config_manager.get_monitoring_config()
    
    # 控制面板
    col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
    
    with col1:
        if not monitoring_config['enabled']:
            if st.button("▶️ 启动监控", type="primary", use_container_width=True):
                success, message = initialize_monitor()
                if success:
                    config_manager.set_monitoring_enabled(True)
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        else:
            if st.button("⏸️ 停止监控", use_container_width=True):
                config_manager.set_monitoring_enabled(False)
                st.info("监控已停止")
                st.rerun()
    
    with col2:
        refresh_interval = st.selectbox(
            "刷新间隔",
            [5, 10, 30, 60],
            index=1,
            format_func=lambda x: f"{x} 秒"
        )
    
    with col3:
        auto_refresh = st.checkbox("自动刷新", value=False)
    
    with col4:
        if st.button("🔄", help="手动刷新"):
            st.rerun()
    
    st.markdown("---")
    
    # 如果监控未启动
    if not monitoring_config['enabled']:
        st.info("""### 📊 监控未启动
        
点击「▶️ 启动监控」按钮开始实时监控数据库性能。

监控功能包括：
- 实时性能指标（TPS、QPS、连接数、缓存命中率）
- 工作负载类型识别（OLTP/OLAP/HYBRID）
- 资源压力检测
- 历史趋势图表
        """)
        return
    
    # 初始化监控器
    if not st.session_state.get('monitor_initialized', False):
        with st.spinner("正在初始化监控器..."):
            success, message = initialize_monitor()
            if not success:
                st.error(message)
                return
    
    # 采集指标
    with st.spinner("正在采集性能指标..."):
        metrics, analysis = collect_metrics()
    
    if metrics is None or analysis is None:
        st.error("无法采集性能指标，请检查数据库连接")
        return
    
    # ========== 核心指标展示 ==========
    st.markdown("### 📈 核心性能指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        tps = metrics.get('tps', {}).get('total_tps', 0)
        st.metric(
            label="TPS（事务/秒）",
            value=f"{tps:.2f}",
            help="每秒事务数"
        )
    
    with col2:
        qps = metrics.get('qps', {}).get('qps', 0)
        st.metric(
            label="QPS（查询/秒）",
            value=f"{qps:.2f}",
            help="每秒查询数"
        )
    
    with col3:
        cache_hit = metrics.get('cache', {}).get('buffer_cache_hit_ratio', 0)
        delta_color = "normal" if cache_hit >= 90 else "inverse"
        st.metric(
            label="缓存命中率",
            value=f"{cache_hit:.1f}%",
            delta=f"{'✓' if cache_hit >= 90 else '⚠'}",
            delta_color=delta_color,
            help="Buffer Pool 缓存命中率"
        )
    
    with col4:
        connections = metrics.get('connections', {}).get('active', 0)
        max_conn = metrics.get('connections', {}).get('max_connections', 100)
        conn_pct = (connections / max_conn * 100) if max_conn > 0 else 0
        st.metric(
            label="活跃连接数",
            value=f"{connections}/{max_conn}",
            delta=f"{conn_pct:.0f}%",
            delta_color="inverse" if conn_pct > 80 else "normal",
            help="当前活跃连接数 / 最大连接数"
        )
    
    st.markdown("---")
    
    # ========== 工作负载分析 ==========
    st.markdown("### 🎯 工作负载分析")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        workload = analysis.get('workload', {})
        workload_type = workload.get('workload_type', 'UNKNOWN')
        confidence = workload.get('confidence', 0)
        
        # 工作负载类型徽章
        type_colors = {
            'OLTP': '🟢',
            'OLAP': '🔵',
            'HYBRID': '🟡',
            'UNKNOWN': '⚪'
        }
        
        type_descriptions = {
            'OLTP': '在线事务处理 - 高并发短事务',
            'OLAP': '在线分析处理 - 批量长查询',
            'HYBRID': '混合负载 - OLTP + OLAP',
            'UNKNOWN': '未知负载类型'
        }
        
        st.markdown(f"""#### {type_colors.get(workload_type, '⚪')} {workload_type}
        
**置信度:** {confidence:.1%}

**描述:** {type_descriptions.get(workload_type, '未知')}
        """)
    
    with col2:
        # 资源压力指示器
        pressure = analysis.get('pressure', {})
        
        st.markdown("#### ⚠️ 资源压力")
        
        pressure_items = [
            ("CPU 压力", pressure.get('cpu_pressure', False)),
            ("内存压力", pressure.get('memory_pressure', False)),
            ("磁盘 I/O 压力", pressure.get('io_pressure', False)),
            ("连接池压力", pressure.get('connection_pressure', False))
        ]
        
        for name, has_pressure in pressure_items:
            if has_pressure:
                st.error(f"🔴 {name}")
            else:
                st.success(f"🟢 {name}")
    
    st.markdown("---")
    
    # ========== 详细指标 ==========
    tab1, tab2, tab3, tab4 = st.tabs(["💾 缓存与内存", "💿 磁盘 I/O", "🔗 连接与锁", "📊 历史趋势"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Buffer Pool")
            cache_metrics = metrics.get('cache', {})
            
            df_cache = pd.DataFrame([
                {"指标": "命中率", "值": f"{cache_metrics.get('buffer_cache_hit_ratio', 0):.2f}%"},
                {"指标": "读取次数", "值": f"{cache_metrics.get('blocks_read', 0):,}"},
                {"指标": "命中次数", "值": f"{cache_metrics.get('blocks_hit', 0):,}"}
            ])
            st.dataframe(df_cache, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### Shared Buffers")
            io_metrics = metrics.get('io', {})
            
            shared_read = io_metrics.get('shared_blocks_read', 0)
            shared_written = io_metrics.get('shared_blocks_written', 0)
            
            df_buffers = pd.DataFrame([
                {"指标": "读取块数", "值": f"{shared_read:,}"},
                {"指标": "写入块数", "值": f"{shared_written:,}"},
                {"指标": "读写比", "值": f"{shared_read/(shared_written+1):.2f}"}
            ])
            st.dataframe(df_buffers, hide_index=True, use_container_width=True)
    
    with tab2:
        st.markdown("#### 💿 磁盘 I/O 统计")
        
        io_metrics = metrics.get('io', {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "总读取块数",
                f"{io_metrics.get('total_blocks_read', 0):,}"
            )
        
        with col2:
            st.metric(
                "总写入块数",
                f"{io_metrics.get('total_blocks_written', 0):,}"
            )
        
        with col3:
            st.metric(
                "临时文件",
                f"{io_metrics.get('temp_files', 0):,}"
            )
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔗 连接统计")
            conn_metrics = metrics.get('connections', {})
            
            df_conn = pd.DataFrame([
                {"状态": "活跃", "数量": conn_metrics.get('active', 0)},
                {"状态": "空闲", "数量": conn_metrics.get('idle', 0)},
                {"状态": "等待", "数量": conn_metrics.get('waiting', 0)},
                {"状态": "最大", "数量": conn_metrics.get('max_connections', 0)}
            ])
            st.dataframe(df_conn, hide_index=True, use_container_width=True)
        
        with col2:
            st.markdown("#### 🔒 锁统计")
            lock_metrics = metrics.get('locks', {})
            
            df_locks = pd.DataFrame([
                {"类型": "排他锁", "数量": lock_metrics.get('exclusive_locks', 0)},
                {"类型": "共享锁", "数量": lock_metrics.get('share_locks', 0)},
                {"类型": "等待锁", "数量": lock_metrics.get('waiting_locks', 0)}
            ])
            st.dataframe(df_locks, hide_index=True, use_container_width=True)
    
    with tab4:
        st.markdown("#### 📊 性能趋势（最近采集记录）")
        
        if 'monitor_history' in st.session_state and len(st.session_state.monitor_history) > 0:
            history = st.session_state.monitor_history
            
            # 提取历史数据
            timestamps = [r['timestamp'] for r in history]
            tps_values = [r['metrics'].get('tps', {}).get('total_tps', 0) for r in history]
            qps_values = [r['metrics'].get('qps', {}).get('qps', 0) for r in history]
            cache_values = [r['metrics'].get('cache', {}).get('buffer_cache_hit_ratio', 0) for r in history]
            conn_values = [r['metrics'].get('connections', {}).get('active', 0) for r in history]
            
            # 创建图表
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("TPS 趋势", "QPS 趋势", "缓存命中率", "活跃连接数")
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=tps_values, name="TPS", line=dict(color='#1f77b4')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=qps_values, name="QPS", line=dict(color='#ff7f0e')),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=cache_values, name="缓存命中率", line=dict(color='#2ca02c')),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=timestamps, y=conn_values, name="连接数", line=dict(color='#d62728')),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("暂无历史数据，请等待数据采集...")
    
    # 自动刷新
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()
