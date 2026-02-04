#!/usr/bin/env python3
"""
参数验证器
验证推荐参数的合法性、资源约束和系统兼容性
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from typing import Dict, List, Tuple, Optional
import re
import psutil


class ParameterValidator:
    """参数验证器 - 确保参数安全性"""
    
    def __init__(self, dbms):
        """
        初始化验证器
        
        Args:
            dbms: PgDBMS 实例
        """
        self.dbms = dbms
        
        # 获取系统资源
        self.system_memory_mb = self._get_system_memory()
        self.cpu_count = psutil.cpu_count()
        
        print(f"✅ 参数验证器已初始化")
        print(f"   系统内存: {self.system_memory_mb} MB")
        print(f"   CPU 核心数: {self.cpu_count}\n")
    
    def _get_system_memory(self) -> int:
        """获取系统总内存（MB）"""
        try:
            mem = psutil.virtual_memory()
            return int(mem.total / (1024 * 1024))
        except Exception as e:
            print(f"警告：无法获取系统内存，使用默认值 8192 MB: {e}")
            return 8192
    
    def _parse_memory_value(self, value: str) -> Optional[int]:
        """
        解析内存值为 MB
        
        Args:
            value: 如 "256MB", "2GB", "8192kB"
            
        Returns:
            MB 数值，如果解析失败返回 None
        """
        if not value:
            return None
        
        # 移除空格
        value = value.strip().upper()
        
        # 匹配数字和单位
        match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?B)?', value)
        if not match:
            return None
        
        number = float(match.group(1))
        unit = match.group(2) or 'MB'
        
        # 转换为 MB
        if unit == 'KB':
            return int(number / 1024)
        elif unit == 'MB':
            return int(number)
        elif unit == 'GB':
            return int(number * 1024)
        elif unit == 'TB':
            return int(number * 1024 * 1024)
        else:
            return int(number)
    
    def validate_memory_constraint(self, knob: str, value: str) -> Tuple[bool, str]:
        """
        验证内存相关参数是否超出系统资源
        
        Args:
            knob: 参数名
            value: 参数值
            
        Returns:
            (是否通过, 错误信息)
        """
        memory_knobs = {
            'shared_buffers': {'max_ratio': 0.4, 'recommended_ratio': 0.25},
            'effective_cache_size': {'max_ratio': 0.8, 'recommended_ratio': 0.5},
            'work_mem': {'max_mb': 2048},
            'maintenance_work_mem': {'max_mb': 4096}
        }
        
        if knob not in memory_knobs:
            return True, ''
        
        value_mb = self._parse_memory_value(value)
        if value_mb is None:
            return False, f"无法解析内存值: {value}"
        
        constraint = memory_knobs[knob]
        
        # 检查比例约束
        if 'max_ratio' in constraint:
            max_mb = int(self.system_memory_mb * constraint['max_ratio'])
            recommended_mb = int(self.system_memory_mb * constraint['recommended_ratio'])
            
            if value_mb > max_mb:
                return False, (
                    f"{knob} = {value} 超出系统内存限制 "
                    f"(最大: {max_mb}MB, 系统内存: {self.system_memory_mb}MB)"
                )
            
            if value_mb > recommended_mb * 1.5:
                return True, (
                    f"警告: {knob} = {value} 超过推荐值 {recommended_mb}MB 的 50%，"
                    f"可能影响其他进程"
                )
        
        # 检查绝对值约束
        if 'max_mb' in constraint:
            if value_mb > constraint['max_mb']:
                return False, (
                    f"{knob} = {value} 超出建议最大值 {constraint['max_mb']}MB"
                )
        
        return True, ''
    
    def validate_connection_constraint(self, knob: str, value: str) -> Tuple[bool, str]:
        """
        验证连接数相关参数
        
        Args:
            knob: 参数名
            value: 参数值
            
        Returns:
            (是否通过, 错误信息)
        """
        if knob != 'max_connections':
            return True, ''
        
        try:
            conn_count = int(value)
        except ValueError:
            return False, f"max_connections 值无效: {value}"
        
        # 基于系统内存估算合理的最大连接数
        # 每个连接大约需要 5-10 MB（work_mem + 连接开销）
        estimated_memory_per_conn = 10  # MB
        max_safe_connections = int(self.system_memory_mb * 0.6 / estimated_memory_per_conn)
        
        if conn_count > max_safe_connections:
            return False, (
                f"max_connections = {conn_count} 过高，"
                f"可能导致内存不足（建议最大: {max_safe_connections}）"
            )
        
        if conn_count < 10:
            return False, f"max_connections = {conn_count} 过低，可能影响并发性能"
        
        return True, ''
    
    def validate_parallel_workers(self, knob: str, value: str) -> Tuple[bool, str]:
        """
        验证并行工作进程参数
        
        Args:
            knob: 参数名
            value: 参数值
            
        Returns:
            (是否通过, 错误信息)
        """
        parallel_knobs = [
            'max_parallel_workers_per_gather',
            'max_parallel_workers',
            'max_worker_processes'
        ]
        
        if knob not in parallel_knobs:
            return True, ''
        
        try:
            worker_count = int(value)
        except ValueError:
            return False, f"{knob} 值无效: {value}"
        
        # 不应超过 CPU 核心数的 2 倍
        max_workers = self.cpu_count * 2
        
        if worker_count > max_workers:
            return False, (
                f"{knob} = {worker_count} 超过 CPU 核心数 ({self.cpu_count}) 的 2 倍，"
                f"可能导致上下文切换开销过大"
            )
        
        if knob == 'max_parallel_workers_per_gather' and worker_count > self.cpu_count:
            return True, (
                f"警告: {knob} = {worker_count} 超过 CPU 核心数 {self.cpu_count}，"
                f"可能在高并发时导致性能下降"
            )
        
        return True, ''
    
    def validate_checkpoint_settings(self, knob: str, value: str) -> Tuple[bool, str]:
        """
        验证 checkpoint 相关参数
        
        Args:
            knob: 参数名
            value: 参数值
            
        Returns:
            (是否通过, 错误信息)
        """
        if knob == 'checkpoint_completion_target':
            try:
                target = float(value)
                if target < 0.1 or target > 0.9:
                    return False, (
                        f"checkpoint_completion_target = {value} 超出合理范围 [0.1, 0.9]"
                    )
            except ValueError:
                return False, f"checkpoint_completion_target 值无效: {value}"
        
        elif knob == 'max_wal_size':
            wal_mb = self._parse_memory_value(value)
            if wal_mb and wal_mb > 10240:  # 10 GB
                return True, (
                    f"警告: max_wal_size = {value} 较大，可能导致恢复时间过长"
                )
        
        return True, ''
    
    def validate_single_parameter(self, knob: str, value: str) -> Dict:
        """
        验证单个参数
        
        Args:
            knob: 参数名
            value: 参数值
            
        Returns:
            验证结果字典
        """
        result = {
            'knob': knob,
            'value': value,
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 验证内存约束
        valid, msg = self.validate_memory_constraint(knob, value)
        if not valid:
            result['valid'] = False
            result['errors'].append(msg)
        elif msg:
            result['warnings'].append(msg)
        
        # 验证连接数约束
        valid, msg = self.validate_connection_constraint(knob, value)
        if not valid:
            result['valid'] = False
            result['errors'].append(msg)
        elif msg:
            result['warnings'].append(msg)
        
        # 验证并行工作进程
        valid, msg = self.validate_parallel_workers(knob, value)
        if not valid:
            result['valid'] = False
            result['errors'].append(msg)
        elif msg:
            result['warnings'].append(msg)
        
        # 验证 checkpoint 设置
        valid, msg = self.validate_checkpoint_settings(knob, value)
        if not valid:
            result['valid'] = False
            result['errors'].append(msg)
        elif msg:
            result['warnings'].append(msg)
        
        return result
    
    def validate_recommendations(self, recommendations: Dict[str, Dict]) -> Dict:
        """
        验证所有推荐参数
        
        Args:
            recommendations: 推荐配置字典
            
        Returns:
            验证报告
        """
        print(f"\n{'='*60}")
        print("参数安全性验证")
        print(f"{'='*60}\n")
        
        validation_report = {
            'total_params': len(recommendations),
            'valid_params': [],
            'invalid_params': [],
            'warnings': []
        }
        
        for knob, rec in recommendations.items():
            value = rec.get('recommended_value')
            
            # 跳过无具体值的推荐
            if value == 'see_knowledge_base' or not value:
                continue
            
            result = self.validate_single_parameter(knob, value)
            
            if result['valid']:
                validation_report['valid_params'].append(result)
                
                # 打印警告
                if result['warnings']:
                    for warning in result['warnings']:
                        print(f"⚠️  {warning}")
                        validation_report['warnings'].append({
                            'knob': knob,
                            'warning': warning
                        })
                else:
                    print(f"✓ {knob} = {value}")
            else:
                validation_report['invalid_params'].append(result)
                
                # 打印错误
                for error in result['errors']:
                    print(f"✗ {error}")
        
        # 打印总结
        print(f"\n验证结果:")
        print(f"  有效参数: {len(validation_report['valid_params'])} 个")
        print(f"  无效参数: {len(validation_report['invalid_params'])} 个")
        print(f"  警告: {len(validation_report['warnings'])} 个\n")
        
        return validation_report


# 测试代码
if __name__ == '__main__':
    from configparser import ConfigParser
    from dbms.postgres import PgDBMS
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 创建验证器
    validator = ParameterValidator(dbms)
    
    # 测试推荐配置
    test_recommendations = {
        'shared_buffers': {
            'recommended_value': '2048MB',
            'reason': '测试'
        },
        'work_mem': {
            'recommended_value': '256MB',
            'reason': '测试'
        },
        'max_connections': {
            'recommended_value': '200',
            'reason': '测试'
        },
        'max_parallel_workers_per_gather': {
            'recommended_value': '4',
            'reason': '测试'
        },
        # 测试无效配置
        'shared_buffers_invalid': {
            'recommended_value': '10GB',  # 超出系统内存
            'reason': '测试无效'
        }
    }
    
    # 执行验证
    validation_report = validator.validate_recommendations(test_recommendations)
    
    print(f"✅ 参数验证器测试完成")
