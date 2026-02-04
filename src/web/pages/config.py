"""
配置管理页面
"""
import streamlit as st


def show():
    """显示配置管理页面"""
    st.title("⚙️ 配置管理")
    st.markdown("---")
    
    config_manager = st.session_state.config_manager
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["🗄️ 数据库配置", "🎯 调优策略", "🔔 通知设置"])
    
    # ========== 数据库配置 ==========
    with tab1:
        st.markdown("### 数据库连接配置")
        st.markdown("配置目标数据库的连接信息")
        
        db_config = config_manager.get_database_config()
        
        col1, col2 = st.columns(2)
        
        with col1:
            dbms_type = st.selectbox(
                "数据库类型",
                ["postgres", "mysql"],
                index=0 if db_config['dbms_type'] == 'postgres' else 1,
                help="当前仅支持 PostgreSQL"
            )
            
            host = st.text_input(
                "主机地址",
                value=db_config['host'],
                help="数据库服务器的 IP 或域名"
            )
            
            port = st.number_input(
                "端口",
                min_value=1,
                max_value=65535,
                value=db_config['port'],
                help="数据库服务端口"
            )
        
        with col2:
            database = st.text_input(
                "数据库名",
                value=db_config['database'],
                help="要调优的目标数据库"
            )
            
            user = st.text_input(
                "用户名",
                value=db_config['user'],
                help="数据库连接用户"
            )
            
            password = st.text_input(
                "密码",
                value=db_config['password'],
                type="password",
                help="数据库连接密码"
            )
        
        st.markdown("---")
        
        # SSH 隧道配置（可选）
        with st.expander("🔐 SSH 隧道配置（可选）"):
            ssh_enabled = st.checkbox(
                "启用 SSH 隧道",
                value=db_config['ssh_enabled'],
                help="如果数据库需要通过 SSH 跳板机访问"
            )
            
            if ssh_enabled:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    ssh_host = st.text_input(
                        "SSH 主机",
                        value=db_config['ssh_host']
                    )
                
                with col2:
                    ssh_port = st.number_input(
                        "SSH 端口",
                        min_value=1,
                        max_value=65535,
                        value=db_config['ssh_port']
                    )
                
                with col3:
                    ssh_user = st.text_input(
                        "SSH 用户",
                        value=db_config['ssh_user']
                    )
        
        st.markdown("---")
        
        # 操作按钮
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("💾 保存配置", type="primary", use_container_width=True):
                config_manager.update_database_config(
                    dbms_type=dbms_type,
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    ssh_enabled=ssh_enabled,
                    ssh_host=ssh_host if ssh_enabled else '',
                    ssh_port=ssh_port if ssh_enabled else 22,
                    ssh_user=ssh_user if ssh_enabled else ''
                )
                st.success("✅ 配置已保存！")
                st.rerun()
        
        with col2:
            if st.button("🔍 测试连接", use_container_width=True):
                with st.spinner("正在测试连接..."):
                    success, message = config_manager.test_database_connection()
                    
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
    
    # ========== 调优策略配置 ==========
    with tab2:
        st.markdown("### 调优策略配置")
        st.markdown("配置调优模式和参数")
        
        tuning_config = config_manager.get_tuning_config()
        
        # 调优模式选择
        st.markdown("#### 1. 调优模式")
        mode = st.radio(
            "选择调优模式",
            ["manual", "auto"],
            format_func=lambda x: "👨‍💼 手动触发模式（推荐）" if x == "manual" else "🤖 自动定时模式",
            index=0 if tuning_config['mode'] == 'manual' else 1,
            help="手动模式：DBA 手动触发调优\n自动模式：系统定时自动调优"
        )
        
        if mode == "manual":
            st.info("""
            **手动触发模式**
            - ✅ 完全由 DBA 控制调优时机
            - ✅ 适合生产环境
            - ✅ 每次调优需要人工审批
            """)
        else:
            st.warning("""
            **自动定时模式**
            - ⚠️ 系统会定时自动触发调优
            - ⚠️ 建议在测试环境使用
            - ⚠️ 务必启用安全检查
            """)
        
        st.markdown("---")
        
        # 监控和调优参数
        st.markdown("#### 2. 监控参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            window_minutes = st.slider(
                "监控时间窗口（分钟）",
                min_value=30,
                max_value=360,
                value=tuning_config['window_minutes'],
                step=30,
                help="系统分析过去 N 分钟的工作负载"
            )
        
        with col2:
            auto_interval_minutes = st.slider(
                "自动调优间隔（分钟）",
                min_value=15,
                max_value=240,
                value=tuning_config['auto_interval_minutes'],
                step=15,
                help="自动模式下，每隔 N 分钟触发一次调优",
                disabled=(mode == "manual")
            )
        
        st.markdown("---")
        
        # 高级选项
        st.markdown("#### 3. 高级选项")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            enable_safety_check = st.checkbox(
                "启用安全检查",
                value=tuning_config['enable_safety_check'],
                help="规则引擎会验证参数安全性，强烈推荐启用"
            )
        
        with col2:
            enable_rag = st.checkbox(
                "启用 RAG 增强",
                value=tuning_config['enable_rag'],
                help="使用 RAG 从知识库检索调优建议"
            )
        
        with col3:
            auto_apply = st.checkbox(
                "自动应用推荐",
                value=tuning_config['auto_apply'],
                help="⚠️ 不推荐！会跳过人工审批直接应用"
            )
        
        if auto_apply:
            st.error("""
            ⚠️ **危险：自动应用模式**
            
            启用此选项后，系统会自动应用调优推荐，跳过人工审批。
            这可能导致：
            - 在高峰期触发配置变更
            - 未经审查的参数被应用
            - 潜在的性能问题
            
            **仅建议在测试环境使用！**
            """)
        
        st.markdown("---")
        
        # 保存按钮
        if st.button("💾 保存调优策略", type="primary"):
            config_manager.update_tuning_config(
                mode=mode,
                window_minutes=window_minutes,
                auto_interval_minutes=auto_interval_minutes,
                enable_safety_check=enable_safety_check,
                enable_rag=enable_rag,
                auto_apply=auto_apply
            )
            st.success("✅ 调优策略已保存！")
            st.rerun()
    
    # ========== 通知设置 ==========
    with tab3:
        st.markdown("### 通知设置")
        st.markdown("配置调优完成后的通知方式（即将支持）")
        
        st.info("🚧 通知功能即将上线，敬请期待！")
        
        notification_config = config_manager.config['notification']
        
        enable_notification = st.checkbox(
            "启用通知",
            value=notification_config['enabled'],
            disabled=True
        )
        
        if enable_notification:
            col1, col2 = st.columns(2)
            
            with col1:
                st.text_input(
                    "邮箱地址",
                    value=notification_config['email'],
                    disabled=True,
                    help="接收调优报告的邮箱"
                )
            
            with col2:
                st.text_input(
                    "Webhook URL",
                    value=notification_config['webhook_url'],
                    disabled=True,
                    help="企业微信、钉钉等 Webhook"
                )
