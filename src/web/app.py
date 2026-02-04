#!/usr/bin/env python3
"""
GPTuner Web 界面主应用
基于 Streamlit 的 DBA 友好界面
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

import streamlit as st
from datetime import datetime

# 配置页面
st.set_page_config(
    page_title="GPTuner - 智能数据库调优系统",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 导入配置管理器
from web.config_manager import get_config_manager

# 初始化会话状态
if 'config_manager' not in st.session_state:
    st.session_state.config_manager = get_config_manager()

# 侧边栏导航
st.sidebar.title("🚀 GPTuner")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "导航",
    ["🏠 首页", "⚙️ 配置管理", "📊 实时监控", "🎯 调优推荐", "📜 历史记录"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 系统状态")

config_manager = st.session_state.config_manager
status = config_manager.get_system_status()

# 显示系统状态
monitoring_status = "🟢 运行中" if config_manager.is_monitoring_enabled() else "⚪ 已停止"
st.sidebar.markdown(f"**监控状态:** {monitoring_status}")

tuning_mode = "🤖 自动模式" if config_manager.is_auto_mode() else "👨‍💼 手动模式"
st.sidebar.markdown(f"**调优模式:** {tuning_mode}")

st.sidebar.markdown(f"**总调优次数:** {status['total_tunings']}")

if status['total_tunings'] > 0:
    success_rate = config_manager.get_success_rate()
    st.sidebar.markdown(f"**成功率:** {success_rate:.1f}%")

if status['last_tuning_time']:
    last_time = datetime.fromisoformat(status['last_tuning_time'])
    st.sidebar.markdown(f"**上次调优:** {last_time.strftime('%Y-%m-%d %H:%M')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 关于")
st.sidebar.info(
    "**GPTuner** 是基于大模型和 RAG 的\n"
    "智能数据库调优系统\n\n"
    "🔹 实时性能监控\n"
    "🔹 场景感知推荐\n"
    "🔹 安全规则引擎\n"
    "🔹 一键回滚机制"
)

# ========== 页面路由 ==========

if page == "🏠 首页":
    from web.pages import home
    home.show()

elif page == "⚙️ 配置管理":
    from web.pages import config
    config.show()

elif page == "📊 实时监控":
    from web.pages import monitoring
    monitoring.show()

elif page == "🎯 调优推荐":
    from web.pages import tuning
    tuning.show()

elif page == "📜 历史记录":
    from web.pages import history
    history.show()
