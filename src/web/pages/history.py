"""
历史记录页面
"""
import streamlit as st
import sys
sys.path.insert(0, '/root/GPTuner/src')
import os
import json
from datetime import datetime
import pandas as pd
import glob


def load_backup_list():
    """加载备份列表"""
    try:
        # 查找所有备份文件
        backup_files = glob.glob('/root/GPTuner/backup_*.json')
        
        backups = []
        for file_path in backup_files:
            try:
                with open(file_path, 'r') as f:
                    backup = json.load(f)
                    backup['file_path'] = file_path
                    backups.append(backup)
            except Exception as e:
                print(f"读取备份文件失败 {file_path}: {e}")
        
        # 按时间排序（最新的在前）
        backups.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return backups
        
    except Exception as e:
        st.error(f"加载备份列表失败: {e}")
        return []


def rollback_to_backup(backup_id: str):
    """回滚到指定备份"""
    try:
        from configparser import ConfigParser
        from dbms.postgres import PgDBMS
        from recommendation.config_manager import ConfigManager
        
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
        
        # 初始化数据库
        dbms = PgDBMS.from_file(temp_config)
        config_mgr = ConfigManager(dbms)
        
        # 执行回滚
        success = config_mgr.restore_from_backup(backup_id)
        
        if success:
            return True, "回滚成功！数据库需要重启才能生效。"
        else:
            return False, "回滚失败，请检查备份文件是否存在。"
            
    except Exception as e:
        return False, f"回滚失败: {str(e)}"


def delete_backup(file_path: str):
    """删除备份文件"""
    try:
        os.remove(file_path)
        return True, "备份已删除"
    except Exception as e:
        return False, f"删除失败: {str(e)}"


def show():
    """显示历史记录页面"""
    st.title("📜 调优历史记录")
    st.markdown("---")
    
    # 加载备份列表
    backups = load_backup_list()
    
    if not backups:
        st.info("""
        ### 💭 暂无调优历史
        
        当您执行调优并应用配置时，系统会自动创建备份。
        
        备份包括：
        - 调优前的所有参数配置
        - 备份时间和描述
        - 支持一键回滚
        
        前往 [🎯 调优推荐](#) 页面开始第一次调优。
        """)
        return
    
    # 统计信息
    st.markdown("### 📊 统计概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总备份数", len(backups))
    
    with col2:
        if backups:
            latest = datetime.fromisoformat(backups[0]['timestamp'])
            st.metric("最新备份", latest.strftime('%m-%d %H:%M'))
    
    with col3:
        total_size = sum(os.path.getsize(b['file_path']) for b in backups if os.path.exists(b['file_path']))
        st.metric("总占用空间", f"{total_size / 1024:.1f} KB")
    
    with col4:
        config_manager = st.session_state.config_manager
        status = config_manager.get_system_status()
        st.metric("调优次数", status['total_tunings'])
    
    st.markdown("---")
    
    # 备份列表
    st.markdown("### 💾 备份列表")
    
    # 搜索和筛选
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_keyword = st.text_input("🔍 搜索备份", placeholder="输入备份 ID 或描述")
    
    with col2:
        show_count = st.selectbox("显示数量", [10, 20, 50, 100], index=0)
    
    # 筛选备份
    filtered_backups = backups
    if search_keyword:
        filtered_backups = [
            b for b in backups
            if search_keyword.lower() in b['backup_id'].lower()
            or search_keyword.lower() in b.get('description', '').lower()
        ]
    
    # 限制显示数量
    displayed_backups = filtered_backups[:show_count]
    
    st.markdown(f"显示 **{len(displayed_backups)}** / {len(filtered_backups)} 个备份")
    
    # 显示备份卡片
    for i, backup in enumerate(displayed_backups):
        backup_id = backup['backup_id']
        timestamp = datetime.fromisoformat(backup['timestamp'])
        description = backup.get('description', '无描述')
        config = backup.get('config', {})
        
        with st.expander(
            f"💾 {backup_id} - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            expanded=(i == 0)  # 默认展开最新的
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**描述:** {description}")
                st.markdown(f"**备份时间:** {timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                st.markdown(f"**参数数量:** {len(config)} 个")
                
                # 显示备份的参数
                if config:
                    with st.expander("📄 查看备份参数"):
                        df_config = pd.DataFrame([
                            {'参数名': k, '值': v}
                            for k, v in config.items()
                        ])
                        st.dataframe(df_config, hide_index=True, use_container_width=True)
            
            with col2:
                st.markdown("#### 操作")
                
                # 回滚按钮
                if st.button(
                    "↩️ 回滚",
                    key=f"rollback_{backup_id}",
                    type="primary",
                    use_container_width=True
                ):
                    # 二次确认
                    if 'rollback_confirm' not in st.session_state:
                        st.session_state.rollback_confirm = backup_id
                        st.warning("⚠️ 再次点击确认回滚")
                        st.rerun()
                    elif st.session_state.rollback_confirm == backup_id:
                        with st.spinner("正在回滚..."):
                            success, message = rollback_to_backup(backup_id)
                            
                            if success:
                                st.success(message)
                                st.session_state.pop('rollback_confirm', None)
                            else:
                                st.error(message)
                
                # 导出按钮
                st.download_button(
                    label="💾 导出",
                    data=json.dumps(backup, indent=2, ensure_ascii=False),
                    file_name=f"backup_{backup_id}.json",
                    mime="application/json",
                    key=f"export_{backup_id}",
                    use_container_width=True
                )
                
                # 删除按钮
                if st.button(
                    "🗑️ 删除",
                    key=f"delete_{backup_id}",
                    use_container_width=True
                ):
                    # 二次确认
                    if 'delete_confirm' not in st.session_state:
                        st.session_state.delete_confirm = backup_id
                        st.warning("⚠️ 再次点击确认删除")
                        st.rerun()
                    elif st.session_state.delete_confirm == backup_id:
                        success, message = delete_backup(backup['file_path'])
                        
                        if success:
                            st.success(message)
                            st.session_state.pop('delete_confirm', None)
                            st.rerun()
                        else:
                            st.error(message)
    
    st.markdown("---")
    
    # 底部操作
    st.markdown("### 🛠️ 批量操作")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 查看监控", use_container_width=True):
            st.switch_page("pages/monitoring.py")
    
    with col2:
        if st.button("🎯 开始调优", use_container_width=True):
            st.switch_page("pages/tuning.py")
    
    with col3:
        # 清理过期备份
        if len(backups) > 20:
            if st.button("🗑️ 清理过期备份", use_container_width=True):
                st.warning("🚧 此功能开发中...")
