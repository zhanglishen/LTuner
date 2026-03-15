"""
调优推荐页面
- Tab 1: 原有的 GPTuner 式调优推荐与审批
- Tab 2: LTuner 自省式独立调优
"""
import streamlit as st
import sys
import os
import subprocess
import signal
import time
import glob
sys.path.insert(0, '/root/GPTuner/src')
import json
from datetime import datetime
import pandas as pd

LTUNER_OUTPUT_DIR = '/root/GPTuner/optimization_results/postgres/ltuner'
LTUNER_REPORT_PATH = os.path.join(LTUNER_OUTPUT_DIR, 'ltuner_workflow_report.json')


def initialize_tuning_system():
    """初始化调优系统"""
    if 'tuning_initialized' not in st.session_state:
        try:
            import os
            # 禁止 HuggingFace 联网检查，直接使用本地缓存
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['HF_HUB_OFFLINE'] = '1'

            from configparser import ConfigParser
            from dbms.postgres import PgDBMS
            from recommendation.tuning_orchestrator import TuningOrchestrator

            # 直接读取 postgres.ini 配置
            pg_config = ConfigParser()
            pg_config.read('/root/GPTuner/configs/postgres.ini')
            sec = pg_config['DATABASE']
            dbms = PgDBMS(
                db=sec['db'],
                user=sec['user'],
                password=sec['password'],
                restart_cmd=sec['restart_cmd'],
                recover_script=sec['recover_script'],
                knob_info_path=sec['knob_info_path'],
            )
            dbms._connect(sec['db'])

            config_manager = st.session_state.config_manager
            tuning_config = config_manager.get_tuning_config()

            orchestrator = TuningOrchestrator(
                dbms,
                use_rag=tuning_config['enable_rag'],
                enable_safety_check=tuning_config['enable_safety_check']
            )
            
            st.session_state.tuning_dbms = dbms
            st.session_state.orchestrator = orchestrator
            st.session_state.tuning_initialized = True
            
            return True, "调优系统初始化成功"
            
        except Exception as e:
            st.session_state.tuning_initialized = False
            return False, f"初始化失败: {str(e)}"
    
    return True, "调优系统已初始化"


def trigger_analysis():
    """触发调优分析"""
    try:
        orchestrator = st.session_state.orchestrator
        config_manager = st.session_state.config_manager
        tuning_config = config_manager.get_tuning_config()
        
        window_minutes = tuning_config['window_minutes']
        enable_safety_check = tuning_config['enable_safety_check']
        
        # 执行分析和推荐
        with st.spinner("🔍 步骤 1/4: 分析工作负载..."):
            analysis_report = orchestrator.analyze_and_recommend(window_minutes)
        
        # 预览变更
        with st.spinner("🔍 步骤 2/4: 预览配置变更..."):
            preview_report = orchestrator.preview_changes(analysis_report['recommendations'])
        
        # 安全检查
        safety_report = None
        if enable_safety_check:
            with st.spinner("🔍 步骤 3/4: 安全检查..."):
                safety_report = orchestrator.safety_check(analysis_report)
        
        # 保存到 session_state
        st.session_state.current_analysis = analysis_report
        st.session_state.current_preview = preview_report
        st.session_state.current_safety = safety_report
        st.session_state.tuning_step = 'review'  # 切换到审查阶段
        
        return True, "分析完成"
        
    except Exception as e:
        return False, f"分析失败: {str(e)}"


def apply_recommendations(backup_description=""):
    """应用推荐配置"""
    try:
        orchestrator = st.session_state.orchestrator
        config_manager = st.session_state.config_manager
        recommendations = st.session_state.current_analysis['recommendations']
        
        # 应用配置
        with st.spinner("🔍 正在备份并应用配置..."):
            apply_report = orchestrator.apply_with_backup(recommendations)
        
        # 更新统计数据
        success = apply_report.get('apply_result', {}).get('successful', [])
        config_manager.update_tuning_stats(success=len(success) > 0)
        
        # 保存应用结果
        st.session_state.current_apply = apply_report
        st.session_state.tuning_step = 'applied'
        
        return True, apply_report
        
    except Exception as e:
        config_manager.update_tuning_stats(success=False)
        return False, f"应用失败: {str(e)}"


def show():
    """显示调优页面 (双 Tab: GPTuner推荐 + LTuner自省调优)"""
    st.title("🎯 调优中心")
    st.markdown("---")

    tab1, tab2 = st.tabs(["📋 GPTuner 推荐调优", "🧠 LTuner 自省调优"])

    with tab1:
        _show_gptuner_tab()

    with tab2:
        _show_ltuner_tab()


def _show_gptuner_tab():
    """原有的 GPTuner 式调优推荐与审批"""
    config_manager = st.session_state.config_manager
    tuning_config = config_manager.get_tuning_config()

    # 初始化调优系统
    if not st.session_state.get('tuning_initialized', False):
        with st.spinner("正在初始化调优系统..."):
            success, message = initialize_tuning_system()
            if not success:
                st.error(message)
                st.info("### 请先配置数据库连接\n"
                        "前往 ⚙️ 配置管理 页面配置数据库连接并测试连接。")
                return

    current_step = st.session_state.get('tuning_step', 'idle')

    # 步骤指示器
    steps = ["👁️ 待触发", "🔍 分析中", "📄 审查中", "✅ 已应用"]
    step_index = {'idle': 0, 'analyzing': 1, 'review': 2, 'applied': 3}.get(current_step, 0)

    cols = st.columns(4)
    for i, (col, step) in enumerate(zip(cols, steps)):
        with col:
            if i < step_index:
                st.success(step)
            elif i == step_index:
                st.info(step)
            else:
                st.text(step)

    st.markdown("---")

    if current_step == 'idle':
        show_trigger_panel(tuning_config)
    elif current_step == 'review':
        show_review_panel(tuning_config)
    elif current_step == 'applied':
        show_applied_panel()


# ================================================================
# LTuner 自省调优 Tab
# ================================================================
def _show_ltuner_tab():
    """LTuner 自省式反馈独立调优面板"""
    st.markdown("### 🧠 LTuner 自省式反馈调优")
    st.markdown("基于大模型的自省反馈循环，自动发现并优化数据库参数。")
    st.markdown("---")

    # 参数配置
    col1, col2 = st.columns(2)
    with col1:
        lt_test = st.selectbox("Benchmark", ["tpch", "tpcc"], key="lt_test",
                               help="TPC-H (OLAP) / TPC-C (OLTP)")
        lt_max_iter = st.slider("最大迭代轮次", 5, 30, 15, key="lt_max_iter")
        lt_top_k = st.slider("Top-K 参数数", 5, 30, 15, key="lt_top_k")
    with col2:
        lt_threshold = st.number_input("收敛阈值 (%)", 0.001, 0.1, 0.02,
                                       format="%.3f", key="lt_threshold")
        lt_scenario = st.selectbox("场景", ["auto", "OLTP", "OLAP", "HYBRID"],
                                   key="lt_scenario")
        lt_timeout = st.slider("每轮压测超时 (秒)", 60, 300, 180, step=30,
                               key="lt_timeout")

    st.markdown("---")

    # 执行控制
    lt_running = st.session_state.get('ltuner_running', False)
    lt_pid = st.session_state.get('ltuner_pid', None)

    col_start, col_stop, col_status = st.columns([1, 1, 2])
    with col_start:
        if st.button("🚀 开始 LTuner 调优", type="primary",
                     use_container_width=True, disabled=lt_running):
            _start_ltuner(lt_test, lt_max_iter, lt_top_k, lt_threshold,
                          lt_scenario, lt_timeout)
            st.rerun()

    with col_stop:
        if st.button("🛑 停止调优", use_container_width=True,
                     disabled=not lt_running, key="lt_stop"):
            _stop_ltuner()
            st.rerun()

    with col_status:
        if lt_running:
            if lt_pid and _is_alive(lt_pid):
                st.warning(f"🔄 **LTuner 运行中** (PID: {lt_pid})")
            else:
                st.session_state['ltuner_running'] = False
                st.session_state['ltuner_pid'] = None
                st.success("✅ LTuner 调优已完成")
        else:
            st.info("⏳ 等待开始")

    # 实时日志
    if lt_running or os.path.exists(LTUNER_OUTPUT_DIR):
        _show_ltuner_log()

    st.markdown("---")

    # 调优结果展示
    _show_ltuner_results()


def _start_ltuner(test, max_iter, top_k, threshold, scenario, timeout):
    """后台启动 LTuner 调优"""
    os.makedirs(LTUNER_OUTPUT_DIR, exist_ok=True)

    cmd = [
        sys.executable, '/root/GPTuner/src/run_ltuner.py',
        'postgres', test, str(timeout),
        '-max_iter', str(max_iter),
        '-top_k', str(top_k),
        '-threshold', str(threshold),
    ]
    if scenario != 'auto':
        cmd += ['-scenario', scenario]

    log_path = os.path.join(LTUNER_OUTPUT_DIR, 'ltuner_stdout.log')
    with open(log_path, 'w') as f:
        proc = subprocess.Popen(
            cmd, stdout=f, stderr=subprocess.STDOUT,
            cwd='/root/GPTuner', start_new_session=True)

    st.session_state['ltuner_running'] = True
    st.session_state['ltuner_pid'] = proc.pid


def _stop_ltuner():
    pid = st.session_state.get('ltuner_pid')
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    st.session_state['ltuner_running'] = False
    st.session_state['ltuner_pid'] = None


def _is_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _show_ltuner_log():
    log_path = os.path.join(LTUNER_OUTPUT_DIR, 'ltuner_stdout.log')
    with st.expander("📋 LTuner 运行日志", expanded=False):
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                tail = lines[-80:] if len(lines) > 80 else lines
                st.code(''.join(tail), language='text')
            except IOError:
                st.warning("无法读取日志")
        else:
            st.info("暂无日志")
        if st.session_state.get('ltuner_running'):
            st.button("🔄 刷新", key="lt_refresh", on_click=lambda: None)


def _show_ltuner_results():
    """展示 LTuner 调优结果"""
    st.markdown("### 📊 LTuner 调优结果")

    if not os.path.exists(LTUNER_REPORT_PATH):
        st.info("暂无调优结果。运行 LTuner 调优后结果将在此展示。")
        return

    try:
        with open(LTUNER_REPORT_PATH, 'r') as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError):
        st.warning("无法解析调优报告")
        return

    status = report.get('status', 'unknown')
    final = report.get('final_result', {})

    if status == 'completed':
        st.success("✅ 调优已完成")
    elif status == 'failed':
        st.error(f"❌ 调优失败: {report.get('error', '未知错误')}")
    else:
        st.warning(f"状态: {status}")

    # Metric cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("性能提升", f"{final.get('improvement_percent', 0):.1f}%")
    with col2:
        st.metric("迭代轮次", final.get('total_iterations', 0))
    with col3:
        st.metric("配置失败次数", final.get('config_failures', 0))
    with col4:
        total_s = report.get('total_time_seconds', 0)
        st.metric("总耗时", f"{total_s/60:.0f} min")

    # Optimization history chart (if available)
    opt = report.get('steps', {}).get('step4_optimize', {}).get('result', {})
    history = opt.get('history', [])
    if history:
        st.markdown("#### 收敛曲线")
        test = report.get('test', 'tpch')
        is_lat = test in ['tpch']
        if is_lat:
            values = [h.get('latency', 0) for h in history]
            metric_name = "延迟 (μs)"
        else:
            values = [h.get('throughput', 0) for h in history]
            metric_name = "TPS"

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=values, mode='lines+markers',
                                 name=metric_name, line=dict(color='#FF5722')))
        fig.update_layout(xaxis_title='迭代', yaxis_title=metric_name,
                          height=350, margin=dict(l=40, r=20, t=30, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # Best config
    best_config = final.get('best_config', {})
    if best_config:
        with st.expander(f"🔧 最佳配置 ({len(best_config)} 个参数)"):
            df = pd.DataFrame([
                {'参数': k, '值': str(v)} for k, v in best_config.items()
            ])
            st.dataframe(df, hide_index=True, use_container_width=True)

    # Full report
    with st.expander("📄 完整报告 JSON"):
        st.json(report)


def show_trigger_panel(tuning_config):
    """显示触发面板"""
    st.markdown("### 🚀 触发调优分析")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        #### 操作说明
        
        点击下方按钮将：
        1. 采集过去 **{} 分钟**的性能指标
        2. 识别工作负载类型（OLTP/OLAP/HYBRID）
        3. 生成针对性的参数调优推荐
        4. {}
        
        **预计耗时：** 30-60 秒
        """.format(
            tuning_config['window_minutes'],
            "进行安全检查验证" if tuning_config['enable_safety_check'] else "跳过安全检查"
        ))
    
    with col2:
        st.markdown("#### 当前配置")
        st.info(f"""
        **监控窗口:** {tuning_config['window_minutes']} 分钟  
        **RAG 增强:** {'✅' if tuning_config['enable_rag'] else '❌'}  
        **安全检查:** {'✅' if tuning_config['enable_safety_check'] else '❌'}
        """)
    
    st.markdown("---")
    
    # 触发按钮
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🚀 开始分析", type="primary", use_container_width=True):
            st.session_state.tuning_step = 'analyzing'
            st.rerun()
    
    with col2:
        if st.button("🛠️ 修改配置", use_container_width=True):
            st.switch_page("pages/config.py")
    
    # 分析状态
    if st.session_state.get('tuning_step') == 'analyzing':
        with st.spinner("🔍 正在分析，请稍候..."):
            success, message = trigger_analysis()
            
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
                st.session_state.tuning_step = 'idle'


def show_review_panel(tuning_config):
    """显示审查面板"""
    st.markdown("### 📄 审查推荐方案")
    
    analysis = st.session_state.current_analysis
    preview = st.session_state.current_preview
    safety = st.session_state.current_safety
    
    recommendations = analysis['recommendations']
    summary = analysis['summary']
    
    # 总体概要
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("场景类型", summary['scenario'])
    
    with col2:
        st.metric("推荐参数数", summary['total_recommendations'])
    
    with col3:
        st.metric("高优先级", summary['high_priority_count'])
    
    with col4:
        has_pressure = summary.get('has_pressure', False)
        st.metric(
            "资源压力",
            "是" if has_pressure else "否",
            delta="⚠️" if has_pressure else "✅"
        )
    
    st.markdown("---")
    
    # 安全检查结果
    if safety and not safety.get('skipped', False):
        risk_assessment = safety.get('risk_assessment', {})
        risk_level = risk_assessment.get('risk_level', 'unknown')
        safe_to_apply = safety.get('summary', {}).get('safe_to_apply', True)
        
        risk_colors = {
            'none': 'success',
            'low': 'info',
            'medium': 'warning',
            'high': 'error'
        }
        
        risk_icons = {
            'none': '🟢',
            'low': '🟡',
            'medium': '🟠',
            'high': '🔴'
        }
        
        color = risk_colors.get(risk_level, 'info')
        icon = risk_icons.get(risk_level, '⚪')
        
        with st.expander(f"🛡️ 安全检查报告 - {icon} 风险等级: {risk_level.upper()}", expanded=(not safe_to_apply)):
            st.markdown(f"**描述:** {risk_assessment.get('risk_description', '未知')}")
            st.markdown(f"**风险得分:** {risk_assessment.get('risk_score', 0)}")
            
            if risk_assessment.get('risk_factors'):
                st.markdown("**风险因素:**")
                for factor in risk_assessment['risk_factors']:
                    st.markdown(f"- {factor}")
            
            if risk_assessment.get('recommendations'):
                st.markdown("**安全建议:**")
                for rec in risk_assessment['recommendations']:
                    st.markdown(rec)
            
            if not safe_to_apply:
                st.error("⚠️ **警告：配置未通过安全检查，不建议应用！**")
    
    st.markdown("---")
    
    # 推荐参数列表
    st.markdown("### 📊 推荐参数详情")
    
    # 按优先级分组
    high_priority = []
    medium_priority = []
    low_priority = []
    
    for knob, rec in recommendations['recommendations'].items():
        item = {
            '参数名': knob,
            '当前值': recommendations['current_config'].get(knob, 'unknown'),
            '推荐值': rec['recommended_value'],
            '优先级': rec['priority'],
            '原因': rec['reason'][:100] + '...' if len(rec['reason']) > 100 else rec['reason']
        }
        
        if rec['priority'] == 'high':
            high_priority.append(item)
        elif rec['priority'] == 'medium':
            medium_priority.append(item)
        else:
            low_priority.append(item)
    
    # 显示高优先级
    if high_priority:
        st.markdown("#### 🔴 高优先级（建议优先应用）")
        df_high = pd.DataFrame(high_priority)
        st.dataframe(
            df_high[['参数名', '当前值', '推荐值', '原因']],
            hide_index=True,
            use_container_width=True
        )
    
    # 显示中优先级
    if medium_priority:
        with st.expander(f"🟡 中优先级 ({len(medium_priority)} 个参数)"):
            df_medium = pd.DataFrame(medium_priority)
            st.dataframe(
                df_medium[['参数名', '当前值', '推荐值', '原因']],
                hide_index=True,
                use_container_width=True
            )
    
    # 显示低优先级
    if low_priority:
        with st.expander(f"⚪ 低优先级 ({len(low_priority)} 个参数)"):
            df_low = pd.DataFrame(low_priority)
            st.dataframe(
                df_low[['参数名', '当前值', '推荐值', '原因']],
                hide_index=True,
                use_container_width=True
            )
    
    st.markdown("---")
    
    # 审批操作
    st.markdown("### ✅ 审批操作")
    
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    
    safe_to_apply = True
    if safety and not safety.get('skipped', False):
        safe_to_apply = safety.get('summary', {}).get('safe_to_apply', True)
    
    with col1:
        if st.button("✅ 批准并应用", type="primary", use_container_width=True, disabled=(not safe_to_apply)):
            with st.spinner("正在应用配置..."):
                success, result = apply_recommendations()
                
                if success:
                    st.success("✅ 配置应用成功！")
                    st.rerun()
                else:
                    st.error(f"❌ {result}")
    
    with col2:
        if st.button("❌ 拒绝", use_container_width=True):
            st.session_state.tuning_step = 'idle'
            st.info("已拒绝此次推荐")
            st.rerun()
    
    with col3:
        if st.button("💾 导出报告", use_container_width=True):
            # 生成 JSON 报告
            report = {
                'timestamp': datetime.now().isoformat(),
                'analysis': analysis,
                'safety_check': safety
            }
            
            st.download_button(
                label="💾 下载 JSON",
                data=json.dumps(report, indent=2, ensure_ascii=False),
                file_name=f"tuning_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col4:
        if st.button("🔄 重新分析", use_container_width=True):
            st.session_state.tuning_step = 'idle'
            st.rerun()
    
    if not safe_to_apply:
        st.error("""
        ### ⚠️ 安全检查未通过
        
        当前配置存在高风险，不建议应用。请：
        1. 查看上方安全检查报告
        2. 修复所有高风险问题
        3. 重新触发分析
        
        或者前往 [⚠️ 配置管理](#) 禁用安全检查（不推荐）
        """)


def show_applied_panel():
    """显示应用结果面板"""
    st.markdown("### ✅ 配置应用成功")
    
    apply_report = st.session_state.current_apply
    backup_info = apply_report.get('backup_info', {})
    apply_result = apply_report.get('apply_result', {})
    
    # 成功消息
    st.success("""
    ### 🎉 调优配置已成功应用！
    
    系统已经：
    - ✅ 备份了当前配置
    - ✅ 应用了新的参数
    - ✅ 重启了数据库（如需）
    """)
    
    # 备份信息
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💾 备份信息")
        st.info(f"""
        **备份 ID:** {backup_info.get('backup_id', 'unknown')}  
        **备份时间:** {backup_info.get('timestamp', 'unknown')}  
        **备份参数数:** {len(backup_info.get('config', {}))} 个
        
        如需回滚，请前往 [📜 历史记录](#) 页面
        """)
    
    with col2:
        st.markdown("#### ✅ 应用结果")
        successful = apply_result.get('successful', [])
        failed = apply_result.get('failed', [])
        
        st.metric("成功应用", len(successful))
        if failed:
            st.metric("应用失败", len(failed), delta="⚠️")
    
    # 应用详情
    with st.expander("📄 查看应用详情"):
        if successful:
            st.markdown("**成功应用的参数：**")
            for param in successful:
                st.markdown(f"- ✅ {param}")
        
        if failed:
            st.markdown("**应用失败的参数：**")
            for param in failed:
                st.markdown(f"- ❌ {param}")
    
    st.markdown("---")
    
    # 后续操作
    st.markdown("### 👉 后续操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 查看监控", type="primary", use_container_width=True):
            st.switch_page("pages/monitoring.py")
    
    with col2:
        if st.button("📜 查看历史", use_container_width=True):
            st.switch_page("pages/history.py")
    
    with col3:
        if st.button("🔄 再次调优", use_container_width=True):
            st.session_state.tuning_step = 'idle'
            st.rerun()
