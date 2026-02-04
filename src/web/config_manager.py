#!/usr/bin/env python3
"""
Web 界面配置管理器
用于存储和管理用户在界面中的配置
"""
import json
import os
from typing import Dict, Any
from datetime import datetime


class WebConfigManager:
    """Web 界面配置管理器"""
    
    def __init__(self, config_file: str = '/root/GPTuner/web_config.json'):
        """
        初始化配置管理器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            # 数据库连接配置
            'database': {
                'dbms_type': 'postgres',
                'host': 'localhost',
                'port': 5432,
                'user': 'admin',
                'password': 'password',
                'database': 'benchbase',
                'ssh_enabled': False,
                'ssh_host': '',
                'ssh_port': 22,
                'ssh_user': ''
            },
            
            # 调优策略配置
            'tuning': {
                'mode': 'manual',  # manual / auto
                'window_minutes': 120,  # 监控时间窗口
                'auto_interval_minutes': 30,  # 自动调优间隔
                'enable_safety_check': True,  # 启用安全检查
                'enable_rag': True,  # 启用 RAG 增强
                'auto_apply': False  # 自动应用（不推荐）
            },
            
            # 监控配置
            'monitoring': {
                'enabled': False,  # 监控是否运行
                'refresh_seconds': 10,  # 仪表板刷新间隔
                'history_hours': 24  # 保留历史数据时长
            },
            
            # 通知配置
            'notification': {
                'enabled': False,
                'email': '',
                'webhook_url': ''
            },
            
            # 系统状态
            'system': {
                'last_tuning_time': None,
                'next_tuning_time': None,
                'total_tunings': 0,
                'successful_tunings': 0
            }
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # 合并默认配置（处理新增字段）
                    default = self._get_default_config()
                    return self._merge_config(default, config)
            except Exception as e:
                print(f"警告：加载配置失败 ({e})，使用默认配置")
                return self._get_default_config()
        else:
            return self._get_default_config()
    
    def _merge_config(self, default: Dict, custom: Dict) -> Dict:
        """合并配置（保留自定义值，补充默认值）"""
        result = default.copy()
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_config(result[key], value)
            else:
                result[key] = value
        return result
    
    def save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"保存配置失败: {e}")
            return False
    
    # ========== 数据库配置 ==========
    
    def get_database_config(self) -> Dict:
        """获取数据库配置"""
        return self.config['database'].copy()
    
    def update_database_config(self, **kwargs):
        """更新数据库配置"""
        for key, value in kwargs.items():
            if key in self.config['database']:
                self.config['database'][key] = value
        self.save_config()
    
    def test_database_connection(self) -> tuple[bool, str]:
        """测试数据库连接"""
        try:
            import sys
            sys.path.insert(0, '/root/GPTuner/src')
            from dbms.postgres import PgDBMS
            from configparser import ConfigParser
            
            # 创建临时配置
            temp_config = ConfigParser()
            temp_config['postgresql'] = {
                'host': self.config['database']['host'],
                'port': str(self.config['database']['port']),
                'user': self.config['database']['user'],
                'passwd': self.config['database']['password'],
                'dbname': self.config['database']['database']
            }
            
            # 尝试连接
            dbms = PgDBMS.from_file(temp_config)
            result = dbms.fetch_results("SELECT version();")
            
            if result and result[0]:
                version = result[0][0]
                return True, f"连接成功！\n{version}"
            else:
                return False, "连接失败：无法获取数据库版本"
                
        except Exception as e:
            return False, f"连接失败：{str(e)}"
    
    # ========== 调优策略配置 ==========
    
    def get_tuning_config(self) -> Dict:
        """获取调优配置"""
        return self.config['tuning'].copy()
    
    def update_tuning_config(self, **kwargs):
        """更新调优配置"""
        for key, value in kwargs.items():
            if key in self.config['tuning']:
                self.config['tuning'][key] = value
        self.save_config()
    
    def is_auto_mode(self) -> bool:
        """是否为自动调优模式"""
        return self.config['tuning']['mode'] == 'auto'
    
    # ========== 监控配置 ==========
    
    def get_monitoring_config(self) -> Dict:
        """获取监控配置"""
        return self.config['monitoring'].copy()
    
    def update_monitoring_config(self, **kwargs):
        """更新监控配置"""
        for key, value in kwargs.items():
            if key in self.config['monitoring']:
                self.config['monitoring'][key] = value
        self.save_config()
    
    def is_monitoring_enabled(self) -> bool:
        """监控是否启用"""
        return self.config['monitoring']['enabled']
    
    def set_monitoring_enabled(self, enabled: bool):
        """设置监控状态"""
        self.config['monitoring']['enabled'] = enabled
        self.save_config()
    
    # ========== 系统状态 ==========
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        return self.config['system'].copy()
    
    def update_tuning_stats(self, success: bool = True):
        """更新调优统计"""
        self.config['system']['total_tunings'] += 1
        if success:
            self.config['system']['successful_tunings'] += 1
        self.config['system']['last_tuning_time'] = datetime.now().isoformat()
        self.save_config()
    
    def get_success_rate(self) -> float:
        """获取调优成功率"""
        total = self.config['system']['total_tunings']
        if total == 0:
            return 0.0
        return self.config['system']['successful_tunings'] / total * 100
    
    # ========== 完整配置 ==========
    
    def get_all_config(self) -> Dict:
        """获取所有配置"""
        return self.config.copy()
    
    def reset_to_default(self):
        """重置为默认配置"""
        self.config = self._get_default_config()
        self.save_config()


# 全局配置实例
_config_manager = None

def get_config_manager() -> WebConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = WebConfigManager()
    return _config_manager
