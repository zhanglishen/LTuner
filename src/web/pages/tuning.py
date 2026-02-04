"""
调优推荐页面
"""
import streamlit as st
import sys
sys.path.insert(0, '/root/GPTuner/src')
import json
from datetime import datetime
import pandas as pd


def initialize_tuning_system():
    """初始化调优系统"""
    if 'tuning_initialized' not in st.session_state:
        try:
            from configparser import ConfigParser
            from dbms.postgres import PgDBMS
            from recommendation.tuning_orchestrator import TuningOrchestrator
            
            config_manager = st.session_state.config_manager
            db_config = config_manager.get_database_config()
            tuning_config = config_manager.get_tuning_config()
            
            # 创建临时配置
            temp_config = ConfigParser()
            temp_config['postgresql'] = {
                'host': db_config['host'],
                'port': str(db_config['port']),
                'user': db_config['user'],
                'passwd': db_config['password'],
                'dbname': db_config['database']
            }
            
            # 初始化数据库和调优编排器
            dbms = PgDBMS.from_file(temp_config)
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
    """显示调优推荐页面"""
    st.title("🎯 调优推荐与审批")
    st.markdown("---")
    
    config_manager = st.session_state.config_manager
    tuning_config = config_manager.get_tuning_config()
    
    # 初始化调优系统
    if not st.session_state.get('tuning_initialized', False):
        with st.spinner("正在初始化调优系统..."):
            success, message = initialize_tuning_system()
            if not success:
                st.error(message)
                st.info("""
                ### 请先配置数据库连接
                
                前往 [⚠️ 配置管理](#) 页面配置数据库连接信息并测试连接。
                """)
                return
    
    # 获取当前步骤
    current_step = st.session_state.get('tuning_step', 'idle')
    
    # ========== 步骤指示器 ==========
    steps = ["👁️ 待触发", "🔍 分析中", "📄 审查中", "✅ 已应用"]
    step_index = {
        'idle': 0,
        'analyzing': 1,
        'review': 2,
        'applied': 3
    }.get(current_step, 0)
    
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
    
    # ========== 页面内容 ==========
    
    if current_step == 'idle':
        # 待触发状态
        show_trigger_panel(tuning_config)
    
    elif current_step == 'review':
        # 审查推荐
        show_review_panel(tuning_config)
    
    elif current_step == 'applied':
        # 应用结果
        show_applied_panel()


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
