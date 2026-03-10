"""
首页 - 系统概览
"""
import streamlit as st
from datetime import datetime


def show():
    """显示首页"""
    st.title("🏠 LTuner 智能数据库调优系统")
    st.markdown("---")
    
    # 欢迎信息
    st.markdown("""
    ### 👋 欢迎使用 LTuner！
    
    这是一个基于**大模型 + RAG** 的智能数据库调优系统，专为 DBA 设计。
    
    #### ✨ 核心功能
    
    - **🔍 实时监控**：采集数据库性能指标，识别工作负载类型
    - **🎯 智能推荐**：基于场景和 RAG 知识库生成调优建议
    - **🛡️ 安全保障**：规则引擎验证参数，避免系统崩溃
    - **📜 历史管理**：记录所有调优操作，支持一键回滚
    """)
    
    st.markdown("---")
    
    # 快速开始
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚀 快速开始")
        st.markdown("""
        1. **配置数据库连接** → 前往 [⚙️ 配置管理](#)
        2. **测试连接** → 确保连接成功
        3. **启动监控** → 在 [📊 实时监控](#) 页面启动
        4. **触发调优** → 在 [🎯 调优推荐](#) 页面生成推荐
        5. **审批应用** → 查看推荐并批准应用
        """)
    
    with col2:
        st.markdown("### 📊 系统状态")
        
        config_manager = st.session_state.config_manager
        db_config = config_manager.get_database_config()
        status = config_manager.get_system_status()
        
        # 数据库连接信息
        st.info(f"""
        **数据库:** {db_config['dbms_type'].upper()}  
        **地址:** {db_config['host']}:{db_config['port']}  
        **数据库:** {db_config['database']}
        """)
        
        # 统计信息
        if status['total_tunings'] > 0:
            success_rate = config_manager.get_success_rate()
            st.success(f"""
            **总调优次数:** {status['total_tunings']}  
            **成功次数:** {status['successful_tunings']}  
            **成功率:** {success_rate:.1f}%
            """)
        else:
            st.warning("尚未执行过调优操作")
    
    st.markdown("---")
    
    # 操作指南
    st.markdown("### 📖 使用模式")
    
    tab1, tab2 = st.tabs(["👨‍💼 手动触发模式（推荐）", "🤖 自动定时模式"])
    
    with tab1:
        st.markdown("""
        #### 适用场景
        - DBA 需要完全控制调优时机
        - 希望在低峰时段手动触发
        - 需要逐个审批每次调优
        
        #### 工作流程
        1. DBA 在需要时手动触发分析
        2. 系统生成推荐方案
        3. DBA 审查推荐（包含安全检查）
        4. DBA 批准后应用配置
        5. 系统自动备份，可随时回滚
        
        #### 优势
        - ✅ 人工把关，安全性最高
        - ✅ 灵活控制调优时机
        - ✅ 适合生产环境
        """)
    
    with tab2:
        st.markdown("""
        #### 适用场景
        - 测试环境或开发环境
        - 工作负载变化频繁
        - 需要持续自动优化
        
        #### 工作流程
        1. 系统每隔 N 分钟自动分析
        2. 自动生成推荐方案
        3. 可选择自动应用或等待审批
        4. 自动记录历史，支持回滚
        
        #### 风险提示
        - ⚠️ 自动应用可能在高峰期触发
        - ⚠️ 建议先在测试环境验证
        - ⚠️ 务必启用安全检查
        """)
    
    st.markdown("---")
    
    # 底部信息
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>LTuner v1.0 | 基于大模型自省式反馈的智能调优系统</p>
        <p>Built with ❤️ for DBAs</p>
    </div>
    """, unsafe_allow_html=True)
