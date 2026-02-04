#!/usr/bin/env python3
"""
参数冲突检测器
检测参数之间的冲突和依赖关系，确保配置一致性
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from typing import Dict, List, Tuple


class ConflictDetector:
    """参数冲突和依赖关系检测器"""
    
    def __init__(self):
        """初始化冲突检测器"""
        # 定义参数依赖规则
        self.dependency_rules = self._load_dependency_rules()
        
        # 定义冲突规则
        self.conflict_rules = self._load_conflict_rules()
        
        print("✅ 冲突检测器已初始化")
        print(f"   依赖规则: {len(self.dependency_rules)} 条")
        print(f"   冲突规则: {len(self.conflict_rules)} 条\n")
    
    def _load_dependency_rules(self) -> Dict:
        """
        加载参数依赖规则
        
        Returns:
            依赖规则字典
        """
        return {
            'shared_buffers': {
                'suggests': ['effective_cache_size'],
                'rule': 'effective_cache_size 应该 >= shared_buffers * 2',
                'check': lambda sb, ec: self._parse_mb(ec) >= self._parse_mb(sb) * 2
            },
            'max_connections': {
                'suggests': ['work_mem', 'max_worker_processes'],
                'rule': 'work_mem * max_connections 不应超过系统内存的 60%',
                'check': lambda mc, wm, sys_mem: (
                    int(mc) * self._parse_mb(wm) < sys_mem * 0.6
                )
            },
            'max_parallel_workers_per_gather': {
                'suggests': ['max_parallel_workers', 'max_worker_processes'],
                'rule': 'max_parallel_workers >= max_parallel_workers_per_gather',
                'check': lambda mpwpg, mpw: int(mpw) >= int(mpwpg)
            },
            'wal_buffers': {
                'suggests': ['shared_buffers'],
                'rule': 'wal_buffers 通常设置为 shared_buffers 的 3%，最大 16MB',
                'check': lambda wb, sb: self._parse_mb(wb) <= 16
            }
        }
    
    def _load_conflict_rules(self) -> List[Dict]:
        """
        加载冲突规则
        
        Returns:
            冲突规则列表
        """
        return [
            {
                'name': '内存过度分配检测',
                'params': ['shared_buffers', 'work_mem', 'maintenance_work_mem', 'max_connections'],
                'check': self._check_memory_overallocation,
                'severity': 'high'
            },
            {
                'name': '并行工作进程层级检查',
                'params': ['max_parallel_workers_per_gather', 'max_parallel_workers', 'max_worker_processes'],
                'check': self._check_parallel_hierarchy,
                'severity': 'high'
            },
            {
                'name': 'checkpoint 配置一致性',
                'params': ['checkpoint_completion_target', 'max_wal_size', 'min_wal_size'],
                'check': self._check_checkpoint_consistency,
                'severity': 'medium'
            }
        ]
    
    def _parse_mb(self, value: str) -> int:
        """解析内存值为 MB"""
        if not value:
            return 0
        
        value = value.strip().upper()
        
        # 处理纯数字（假设为 MB）
        if value.isdigit():
            return int(value)
        
        # 解析带单位的值
        import re
        match = re.match(r'(\d+(?:\.\d+)?)\s*([KMGT]?B)?', value)
        if not match:
            return 0
        
        number = float(match.group(1))
        unit = match.group(2) or 'MB'
        
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
    
    def _check_memory_overallocation(self, params: Dict, system_memory_mb: int) -> Tuple[bool, str]:
        """
        检查内存是否过度分配
        
        Args:
            params: 参数字典
            system_memory_mb: 系统总内存（MB）
            
        Returns:
            (是否通过, 信息)
        """
        shared_buffers = self._parse_mb(params.get('shared_buffers', '128MB'))
        work_mem = self._parse_mb(params.get('work_mem', '4MB'))
        maintenance_work_mem = self._parse_mb(params.get('maintenance_work_mem', '64MB'))
        max_connections = int(params.get('max_connections', 100))
        
        # 估算总内存使用
        # shared_buffers + (work_mem * max_connections) + maintenance_work_mem
        estimated_memory = shared_buffers + (work_mem * max_connections) + maintenance_work_mem
        
        # 不应超过系统内存的 80%
        threshold = system_memory_mb * 0.8
        
        if estimated_memory > threshold:
            return False, (
                f"内存过度分配警告: "
                f"估算使用 {estimated_memory}MB > 系统内存的 80% ({int(threshold)}MB)\n"
                f"  shared_buffers: {shared_buffers}MB\n"
                f"  work_mem × max_connections: {work_mem}MB × {max_connections} = {work_mem * max_connections}MB\n"
                f"  maintenance_work_mem: {maintenance_work_mem}MB"
            )
        
        return True, f"内存分配合理: {estimated_memory}MB / {system_memory_mb}MB ({estimated_memory/system_memory_mb*100:.1f}%)"
    
    def _check_parallel_hierarchy(self, params: Dict, system_memory_mb: int) -> Tuple[bool, str]:
        """
        检查并行工作进程的层级关系
        
        Args:
            params: 参数字典
            system_memory_mb: 系统总内存
            
        Returns:
            (是否通过, 信息)
        """
        mpwpg = int(params.get('max_parallel_workers_per_gather', 2))
        mpw = int(params.get('max_parallel_workers', 8))
        mwp = int(params.get('max_worker_processes', 8))
        
        # 检查层级关系: max_worker_processes >= max_parallel_workers >= max_parallel_workers_per_gather
        if not (mwp >= mpw >= mpwpg):
            return False, (
                f"并行工作进程层级错误:\n"
                f"  应满足: max_worker_processes >= max_parallel_workers >= max_parallel_workers_per_gather\n"
                f"  当前: {mwp} >= {mpw} >= {mpwpg} (不满足)"
            )
        
        return True, f"并行工作进程层级正确: {mwp} >= {mpw} >= {mpwpg}"
    
    def _check_checkpoint_consistency(self, params: Dict, system_memory_mb: int) -> Tuple[bool, str]:
        """
        检查 checkpoint 配置的一致性
        
        Args:
            params: 参数字典
            system_memory_mb: 系统总内存
            
        Returns:
            (是否通过, 信息)
        """
        cct = float(params.get('checkpoint_completion_target', 0.9))
        max_wal = self._parse_mb(params.get('max_wal_size', '1GB'))
        min_wal = self._parse_mb(params.get('min_wal_size', '80MB'))
        
        # min_wal_size 应该 < max_wal_size
        if min_wal >= max_wal:
            return False, (
                f"Checkpoint 配置错误: min_wal_size ({min_wal}MB) >= max_wal_size ({max_wal}MB)"
            )
        
        # checkpoint_completion_target 应该在 0.1-0.9 之间
        if not (0.1 <= cct <= 0.9):
            return False, (
                f"checkpoint_completion_target = {cct} 超出合理范围 [0.1, 0.9]"
            )
        
        return True, "Checkpoint 配置一致"
    
    def check_dependencies(self, recommendations: Dict[str, Dict], 
                          system_memory_mb: int) -> Dict:
        """
        检查参数依赖关系
        
        Args:
            recommendations: 推荐配置
            system_memory_mb: 系统内存
            
        Returns:
            依赖检查报告
        """
        print(f"\n{'='*60}")
        print("参数依赖关系检查")
        print(f"{'='*60}\n")
        
        report = {
            'dependencies_checked': [],
            'missing_dependencies': [],
            'dependency_violations': []
        }
        
        for knob, rule in self.dependency_rules.items():
            if knob not in recommendations:
                continue
            
            # 检查建议的相关参数是否存在
            missing = [
                dep for dep in rule['suggests']
                if dep not in recommendations
            ]
            
            if missing:
                report['missing_dependencies'].append({
                    'knob': knob,
                    'missing': missing,
                    'suggestion': f"调整 {knob} 时建议同时考虑: {', '.join(missing)}"
                })
                print(f"⚠️  {knob} 建议同时调整: {', '.join(missing)}")
            else:
                report['dependencies_checked'].append(knob)
                print(f"✓ {knob} 相关依赖已包含")
        
        print()
        return report
    
    def detect_conflicts(self, recommendations: Dict[str, Dict],
                        system_memory_mb: int) -> Dict:
        """
        检测参数冲突
        
        Args:
            recommendations: 推荐配置
            system_memory_mb: 系统内存
            
        Returns:
            冲突检测报告
        """
        print(f"\n{'='*60}")
        print("参数冲突检测")
        print(f"{'='*60}\n")
        
        report = {
            'conflicts': [],
            'warnings': [],
            'passed': []
        }
        
        # 构建参数值字典
        param_values = {
            knob: rec.get('recommended_value')
            for knob, rec in recommendations.items()
        }
        
        # 执行冲突规则检查
        for rule in self.conflict_rules:
            # 检查是否有足够的参数来运行检查
            available_params = [
                p for p in rule['params']
                if p in param_values and param_values[p] != 'see_knowledge_base'
            ]
            
            if len(available_params) < 2:
                continue
            
            # 运行冲突检查
            try:
                passed, message = rule['check'](param_values, system_memory_mb)
                
                if not passed:
                    conflict = {
                        'rule_name': rule['name'],
                        'severity': rule['severity'],
                        'message': message,
                        'params': rule['params']
                    }
                    
                    if rule['severity'] == 'high':
                        report['conflicts'].append(conflict)
                        print(f"❌ [高] {rule['name']}")
                        print(f"    {message}\n")
                    else:
                        report['warnings'].append(conflict)
                        print(f"⚠️  [中] {rule['name']}")
                        print(f"    {message}\n")
                else:
                    report['passed'].append({
                        'rule_name': rule['name'],
                        'message': message
                    })
                    print(f"✓ {rule['name']}: {message}")
                    
            except Exception as e:
                print(f"⚠️  规则 '{rule['name']}' 检查失败: {e}")
        
        # 打印总结
        print(f"\n冲突检测结果:")
        print(f"  冲突: {len(report['conflicts'])} 个")
        print(f"  警告: {len(report['warnings'])} 个")
        print(f"  通过: {len(report['passed'])} 个\n")
        
        return report
    
    def run_full_check(self, recommendations: Dict[str, Dict],
                      system_memory_mb: int) -> Dict:
        """
        运行完整的冲突和依赖检查
        
        Args:
            recommendations: 推荐配置
            system_memory_mb: 系统内存
            
        Returns:
            完整检查报告
        """
        print(f"\n{'#'*60}")
        print("# 规则引擎 - 完整性检查")
        print(f"{'#'*60}\n")
        
        # 依赖关系检查
        dependency_report = self.check_dependencies(recommendations, system_memory_mb)
        
        # 冲突检测
        conflict_report = self.detect_conflicts(recommendations, system_memory_mb)
        
        # 整合报告
        full_report = {
            'dependency_check': dependency_report,
            'conflict_detection': conflict_report,
            'summary': {
                'has_conflicts': len(conflict_report['conflicts']) > 0,
                'has_warnings': len(conflict_report['warnings']) > 0,
                'missing_dependencies': len(dependency_report['missing_dependencies']),
                'safe_to_apply': len(conflict_report['conflicts']) == 0
            }
        }
        
        # 打印总结
        print(f"{'#'*60}")
        print("# 总结")
        print(f"{'#'*60}\n")
        
        if full_report['summary']['safe_to_apply']:
            print("✅ 配置通过所有检查，可以安全应用\n")
        else:
            print(f"❌ 发现 {len(conflict_report['conflicts'])} 个冲突，需要修复后才能应用\n")
        
        return full_report


# 测试代码
if __name__ == '__main__':
    detector = ConflictDetector()
    
    # 测试配置
    test_recommendations = {
        'shared_buffers': {
            'recommended_value': '2048MB'
        },
        'effective_cache_size': {
            'recommended_value': '4096MB'
        },
        'work_mem': {
            'recommended_value': '256MB'
        },
        'max_connections': {
            'recommended_value': '50'
        },
        'max_parallel_workers_per_gather': {
            'recommended_value': '4'
        },
        'max_parallel_workers': {
            'recommended_value': '8'
        },
        'max_worker_processes': {
            'recommended_value': '8'
        }
    }
    
    # 执行检查
    report = detector.run_full_check(test_recommendations, system_memory_mb=8192)
    
    print(f"✅ 冲突检测器测试完成")
