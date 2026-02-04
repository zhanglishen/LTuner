#!/usr/bin/env python3
"""
安全引擎
整合参数验证和冲突检测，提供综合风险评估
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from typing import Dict, List
from rule_engine.parameter_validator import ParameterValidator
from rule_engine.conflict_detector import ConflictDetector


class SafetyEngine:
    """安全引擎 - 综合安全检查和风险评估"""
    
    def __init__(self, dbms):
        """
        初始化安全引擎
        
        Args:
            dbms: PgDBMS 实例
        """
        self.dbms = dbms
        
        # 初始化子模块
        self.validator = ParameterValidator(dbms)
        self.conflict_detector = ConflictDetector()
        
        print("✅ 安全引擎已初始化\n")
    
    def assess_risk(self, validation_report: Dict, conflict_report: Dict) -> Dict:
        """
        评估配置变更的总体风险
        
        Args:
            validation_report: 验证报告
            conflict_report: 冲突检测报告
            
        Returns:
            风险评估报告
        """
        risk_score = 0
        risk_factors = []
        
        # 1. 无效参数（高风险）
        invalid_count = len(validation_report.get('invalid_params', []))
        if invalid_count > 0:
            risk_score += invalid_count * 10
            risk_factors.append(f"{invalid_count} 个参数验证失败")
        
        # 2. 严重冲突（高风险）
        conflicts = conflict_report.get('conflicts', [])
        if conflicts:
            risk_score += len(conflicts) * 10
            risk_factors.append(f"{len(conflicts)} 个严重冲突")
        
        # 3. 警告（中风险）
        warnings_count = len(validation_report.get('warnings', []))
        warnings_count += len(conflict_report.get('warnings', []))
        if warnings_count > 0:
            risk_score += warnings_count * 3
            risk_factors.append(f"{warnings_count} 个警告")
        
        # 4. 缺少依赖（低风险）
        missing_deps = len(conflict_report.get('missing_dependencies', []))
        if missing_deps > 0:
            risk_score += missing_deps * 2
            risk_factors.append(f"{missing_deps} 个缺失依赖")
        
        # 确定风险等级
        if risk_score >= 20:
            risk_level = 'high'
            risk_desc = '高风险：不建议应用，可能导致系统不稳定'
            safe_to_apply = False
        elif risk_score >= 10:
            risk_level = 'medium'
            risk_desc = '中风险：建议修复警告后应用'
            safe_to_apply = False
        elif risk_score > 0:
            risk_level = 'low'
            risk_desc = '低风险：可以谨慎应用，建议先测试'
            safe_to_apply = True
        else:
            risk_level = 'none'
            risk_desc = '无风险：配置通过所有检查，可以安全应用'
            safe_to_apply = True
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_description': risk_desc,
            'risk_factors': risk_factors,
            'safe_to_apply': safe_to_apply,
            'recommendations': self._generate_safety_recommendations(
                validation_report, conflict_report, risk_level
            )
        }
    
    def _generate_safety_recommendations(self, validation_report: Dict,
                                        conflict_report: Dict,
                                        risk_level: str) -> List[str]:
        """生成安全建议"""
        recommendations = []
        
        if risk_level == 'high':
            recommendations.append("❌ 禁止应用：请先修复所有无效参数和冲突")
            
            # 列出需要修复的问题
            for invalid in validation_report.get('invalid_params', []):
                recommendations.append(f"  - 修复参数: {invalid['knob']}")
            
            for conflict in conflict_report.get('conflicts', []):
                recommendations.append(f"  - 解决冲突: {conflict['rule_name']}")
        
        elif risk_level == 'medium':
            recommendations.append("⚠️  建议暂缓：修复警告后可提高安全性")
            recommendations.append("  - 可选择忽略部分警告并继续")
            recommendations.append("  - 建议在测试环境先验证")
        
        elif risk_level == 'low':
            recommendations.append("✓ 可以应用：建议采取以下措施")
            recommendations.append("  - 在低峰时段应用")
            recommendations.append("  - 监控应用后的性能变化")
            recommendations.append("  - 准备好回滚方案")
        
        else:
            recommendations.append("✅ 安全应用：配置已通过所有检查")
            recommendations.append("  - 可以直接应用")
            recommendations.append("  - 建议应用后监控 15 分钟")
        
        return recommendations
    
    def run_safety_check(self, recommendations: Dict[str, Dict]) -> Dict:
        """
        运行完整的安全检查
        
        Args:
            recommendations: 推荐配置
            
        Returns:
            完整安全检查报告
        """
        print(f"\n{'='*60}")
        print("安全引擎 - 完整性检查")
        print(f"{'='*60}\n")
        
        system_memory_mb = self.validator.system_memory_mb
        
        # 1. 参数验证
        print("步骤 1/3: 参数验证")
        validation_report = self.validator.validate_recommendations(recommendations)
        
        # 2. 冲突检测
        print("\n步骤 2/3: 冲突检测")
        conflict_report = self.conflict_detector.run_full_check(
            recommendations,
            system_memory_mb
        )
        
        # 3. 风险评估
        print("\n步骤 3/3: 风险评估")
        risk_assessment = self.assess_risk(
            validation_report,
            conflict_report['conflict_detection']
        )
        
        # 打印风险评估
        self.print_risk_assessment(risk_assessment)
        
        # 整合报告
        full_report = {
            'validation': validation_report,
            'conflicts': conflict_report,
            'risk_assessment': risk_assessment,
            'summary': {
                'safe_to_apply': risk_assessment['safe_to_apply'],
                'risk_level': risk_assessment['risk_level'],
                'risk_score': risk_assessment['risk_score'],
                'total_params': len(recommendations),
                'valid_params': len(validation_report.get('valid_params', [])),
                'invalid_params': len(validation_report.get('invalid_params', [])),
                'conflicts': len(conflict_report['conflict_detection'].get('conflicts', [])),
                'warnings': len(validation_report.get('warnings', []))
            }
        }
        
        return full_report
    
    def print_risk_assessment(self, risk_assessment: Dict):
        """打印风险评估结果"""
        print(f"\n{'='*60}")
        print("风险评估结果")
        print(f"{'='*60}\n")
        
        # 风险等级
        level = risk_assessment['risk_level'].upper()
        score = risk_assessment['risk_score']
        
        level_icons = {
            'NONE': '🟢',
            'LOW': '🟡',
            'MEDIUM': '🟠',
            'HIGH': '🔴'
        }
        
        icon = level_icons.get(level, '⚪')
        print(f"{icon} 风险等级: {level} (得分: {score})")
        print(f"   {risk_assessment['risk_description']}\n")
        
        # 风险因素
        if risk_assessment['risk_factors']:
            print("风险因素:")
            for factor in risk_assessment['risk_factors']:
                print(f"  • {factor}")
            print()
        
        # 安全建议
        print("安全建议:")
        for rec in risk_assessment['recommendations']:
            print(rec)
        
        print(f"\n{'='*60}\n")


# 测试代码
if __name__ == '__main__':
    from configparser import ConfigParser
    from dbms.postgres import PgDBMS
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 创建安全引擎
    safety_engine = SafetyEngine(dbms)
    
    # 测试配置（安全的）
    safe_recommendations = {
        'shared_buffers': {'recommended_value': '2048MB'},
        'effective_cache_size': {'recommended_value': '4096MB'},
        'work_mem': {'recommended_value': '64MB'},
        'max_connections': {'recommended_value': '100'},
        'max_parallel_workers_per_gather': {'recommended_value': '2'},
        'max_parallel_workers': {'recommended_value': '4'},
        'max_worker_processes': {'recommended_value': '8'}
    }
    
    print("="*60)
    print("测试 1: 安全配置")
    print("="*60)
    
    report1 = safety_engine.run_safety_check(safe_recommendations)
    
    # 测试配置（有风险的）
    risky_recommendations = {
        'shared_buffers': {'recommended_value': '6GB'},  # 过大
        'work_mem': {'recommended_value': '512MB'},      # 过大
        'max_connections': {'recommended_value': '500'}, # 过多
        'max_parallel_workers_per_gather': {'recommended_value': '8'},
        'max_parallel_workers': {'recommended_value': '4'},  # 层级错误
        'max_worker_processes': {'recommended_value': '2'}   # 层级错误
    }
    
    print("\n" + "="*60)
    print("测试 2: 高风险配置")
    print("="*60)
    
    report2 = safety_engine.run_safety_check(risky_recommendations)
    
    print(f"\n✅ 安全引擎测试完成")
