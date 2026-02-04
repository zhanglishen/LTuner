#!/usr/bin/env python3
"""
配置变更管理器
负责数据库参数配置的备份、应用和回滚
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from typing import Dict, List, Optional
from datetime import datetime
import json
import os


class ConfigManager:
    """配置变更管理器"""
    
    def __init__(self, dbms, backup_dir='/root/GPTuner/config_backups'):
        """
        初始化配置管理器
        
        Args:
            dbms: PgDBMS 实例
            backup_dir: 备份目录路径
        """
        self.dbms = dbms
        self.backup_dir = backup_dir
        
        # 确保备份目录存在
        os.makedirs(backup_dir, exist_ok=True)
    
    def backup_current_config(self, knob_names: List[str], 
                              description: str = '') -> Dict:
        """
        备份当前配置
        
        Args:
            knob_names: 要备份的参数列表
            description: 备份描述
            
        Returns:
            备份信息字典
        """
        print(f"\n备份当前配置...")
        
        # 收集当前配置
        current_config = {}
        for knob in knob_names:
            try:
                result = self.dbms.get_knob_value(knob)
                if result and result[0]:
                    current_config[knob] = result[0][0]
                    print(f"  ✓ {knob}: {result[0][0]}")
            except Exception as e:
                print(f"  ✗ {knob}: 无法获取 ({e})")
                current_config[knob] = None
        
        # 生成备份ID
        timestamp = datetime.now()
        backup_id = timestamp.strftime('%Y%m%d_%H%M%S')
        
        # 备份信息
        backup_info = {
            'backup_id': backup_id,
            'timestamp': timestamp.isoformat(),
            'description': description,
            'knob_count': len(knob_names),
            'config': current_config
        }
        
        # 保存备份文件
        backup_file = os.path.join(self.backup_dir, f'backup_{backup_id}.json')
        with open(backup_file, 'w') as f:
            json.dump(backup_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 配置已备份")
        print(f"   备份ID: {backup_id}")
        print(f"   文件: {backup_file}")
        print(f"   参数数: {len(current_config)}\n")
        
        return backup_info
    
    def apply_config_changes(self, 
                            recommendations: Dict[str, Dict],
                            dry_run: bool = False) -> Dict:
        """
        应用配置变更
        
        Args:
            recommendations: 推荐配置 {knob: {recommended_value, ...}}
            dry_run: 是否仅模拟（不实际应用）
            
        Returns:
            应用结果
        """
        print(f"\n{'='*60}")
        print(f"应用配置变更 (模拟模式: {dry_run})")
        print(f"{'='*60}\n")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dry_run': dry_run,
            'total_changes': len(recommendations),
            'successful': [],
            'failed': [],
            'skipped': []
        }
        
        for knob, rec in recommendations.items():
            recommended_value = rec['recommended_value']
            
            # 跳过无推荐值的参数
            if recommended_value == 'see_knowledge_base':
                results['skipped'].append({
                    'knob': knob,
                    'reason': '需要参考知识库手动设置'
                })
                continue
            
            try:
                if dry_run:
                    print(f"  [模拟] {knob} = {recommended_value}")
                    results['successful'].append({
                        'knob': knob,
                        'value': recommended_value,
                        'simulated': True
                    })
                else:
                    # 实际应用配置
                    success = self.dbms.set_knob(knob, recommended_value)
                    
                    if success:
                        print(f"  ✓ {knob} = {recommended_value}")
                        results['successful'].append({
                            'knob': knob,
                            'value': recommended_value,
                            'simulated': False
                        })
                    else:
                        print(f"  ✗ {knob} 设置失败")
                        results['failed'].append({
                            'knob': knob,
                            'value': recommended_value,
                            'error': 'set_knob returned False'
                        })
                        
            except Exception as e:
                print(f"  ✗ {knob} 异常: {e}")
                results['failed'].append({
                    'knob': knob,
                    'value': recommended_value,
                    'error': str(e)
                })
        
        # 统计
        print(f"\n应用结果:")
        print(f"  成功: {len(results['successful'])} 个")
        print(f"  失败: {len(results['failed'])} 个")
        print(f"  跳过: {len(results['skipped'])} 个\n")
        
        return results
    
    def restart_database(self, timeout: int = 30) -> bool:
        """
        重启数据库使配置生效
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            是否成功
        """
        print(f"\n{'='*60}")
        print("重启数据库使配置生效")
        print(f"{'='*60}\n")
        
        try:
            print("正在重启...")
            success = self.dbms.reconfigure()
            
            if success:
                print("✅ 数据库重启成功\n")
                return True
            else:
                print("❌ 数据库重启失败\n")
                return False
                
        except Exception as e:
            print(f"❌ 重启异常: {e}\n")
            return False
    
    def restore_from_backup(self, backup_id: str) -> bool:
        """
        从备份恢复配置
        
        Args:
            backup_id: 备份ID
            
        Returns:
            是否成功
        """
        print(f"\n{'='*60}")
        print(f"从备份恢复配置: {backup_id}")
        print(f"{'='*60}\n")
        
        # 加载备份文件
        backup_file = os.path.join(self.backup_dir, f'backup_{backup_id}.json')
        
        if not os.path.exists(backup_file):
            print(f"❌ 备份文件不存在: {backup_file}\n")
            return False
        
        try:
            with open(backup_file, 'r') as f:
                backup_info = json.load(f)
            
            config = backup_info['config']
            
            print(f"正在恢复 {len(config)} 个参数...")
            
            success_count = 0
            failed_count = 0
            
            for knob, value in config.items():
                if value is None:
                    print(f"  - {knob}: 跳过（备份时无法获取）")
                    continue
                
                try:
                    success = self.dbms.set_knob(knob, value)
                    if success:
                        print(f"  ✓ {knob} = {value}")
                        success_count += 1
                    else:
                        print(f"  ✗ {knob} 恢复失败")
                        failed_count += 1
                except Exception as e:
                    print(f"  ✗ {knob} 异常: {e}")
                    failed_count += 1
            
            print(f"\n恢复完成:")
            print(f"  成功: {success_count} 个")
            print(f"  失败: {failed_count} 个\n")
            
            # 重启数据库
            if success_count > 0:
                self.restart_database()
            
            return failed_count == 0
            
        except Exception as e:
            print(f"❌ 恢复失败: {e}\n")
            return False
    
    def list_backups(self) -> List[Dict]:
        """列出所有备份"""
        backups = []
        
        for filename in sorted(os.listdir(self.backup_dir), reverse=True):
            if filename.startswith('backup_') and filename.endswith('.json'):
                backup_file = os.path.join(self.backup_dir, filename)
                try:
                    with open(backup_file, 'r') as f:
                        backup_info = json.load(f)
                    
                    backups.append({
                        'backup_id': backup_info['backup_id'],
                        'timestamp': backup_info['timestamp'],
                        'description': backup_info.get('description', ''),
                        'knob_count': backup_info.get('knob_count', 0),
                        'file': backup_file
                    })
                except Exception as e:
                    print(f"警告：无法读取备份文件 {filename}: {e}")
        
        return backups
    
    def print_backups(self):
        """打印备份列表"""
        backups = self.list_backups()
        
        print(f"\n{'='*60}")
        print(f"配置备份列表 (共 {len(backups)} 个)")
        print(f"{'='*60}\n")
        
        if not backups:
            print("  暂无备份\n")
            return
        
        for i, backup in enumerate(backups, 1):
            print(f"{i}. 备份ID: {backup['backup_id']}")
            print(f"   时间: {backup['timestamp']}")
            print(f"   描述: {backup['description'] or '无'}")
            print(f"   参数数: {backup['knob_count']}")
            print()


# 测试代码
if __name__ == '__main__':
    from configparser import ConfigParser
    from dbms.postgres import PgDBMS
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 创建配置管理器
    config_manager = ConfigManager(dbms)
    
    # 测试备份
    test_knobs = ['shared_buffers', 'work_mem', 'max_connections']
    backup_info = config_manager.backup_current_config(
        test_knobs,
        description='测试备份'
    )
    
    # 列出备份
    config_manager.print_backups()
    
    # 测试模拟应用
    test_recommendations = {
        'shared_buffers': {
            'recommended_value': '256MB',
            'reason': '测试'
        },
        'work_mem': {
            'recommended_value': '16MB',
            'reason': '测试'
        }
    }
    
    apply_result = config_manager.apply_config_changes(
        test_recommendations,
        dry_run=True
    )
    
    print(f"✅ 配置管理器测试完成")
