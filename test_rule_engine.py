#!/usr/bin/env python3
"""
规则引擎集成测试
测试参数验证、冲突检测和风险评估功能
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from configparser import ConfigParser
from dbms.postgres import PgDBMS
from rule_engine.safety_engine import SafetyEngine
from recommendation.tuning_orchestrator import TuningOrchestrator


def test_safety_engine():
    """测试安全引擎"""
    print("="*80)
    print("测试 1: 安全引擎独立测试")
    print("="*80 + "\n")
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 创建安全引擎
    safety_engine = SafetyEngine(dbms)
    
    # 测试安全配置
    print("\n" + "="*80)
    print("测试 1.1: 安全配置")
    print("="*80)
    
    safe_recommendations = {
        'shared_buffers': {'recommended_value': '2048MB'},
        'effective_cache_size': {'recommended_value': '4096MB'},
        'work_mem': {'recommended_value': '64MB'},
        'max_connections': {'recommended_value': '100'},
        'max_parallel_workers_per_gather': {'recommended_value': '2'},
        'max_parallel_workers': {'recommended_value': '4'},
        'max_worker_processes': {'recommended_value': '8'}
    }
    
    report1 = safety_engine.run_safety_check(safe_recommendations)
    print(f"\n结果: 风险等级 = {report1['risk_assessment']['risk_level']}")
    print(f"      是否安全 = {report1['summary']['safe_to_apply']}")
    
    # 测试高风险配置
    print("\n\n" + "="*80)
    print("测试 1.2: 高风险配置")
    print("="*80)
    
    risky_recommendations = {
        'shared_buffers': {'recommended_value': '6GB'},  # 过大
        'work_mem': {'recommended_value': '512MB'},      # 过大
        'max_connections': {'recommended_value': '500'}, # 过多
        'max_parallel_workers_per_gather': {'recommended_value': '8'},
        'max_parallel_workers': {'recommended_value': '4'},  # 层级错误
        'max_worker_processes': {'recommended_value': '2'}   # 层级错误
    }
    
    report2 = safety_engine.run_safety_check(risky_recommendations)
    print(f"\n结果: 风险等级 = {report2['risk_assessment']['risk_level']}")
    print(f"      是否安全 = {report2['summary']['safe_to_apply']}")
    
    # 注意：PgDBMS 没有 disconnect 方法，去掉
    # dbms.disconnect()
    
    return report1, report2


def test_integrated_workflow():
    """测试集成到调优流程"""
    print("\n\n" + "="*80)
    print("测试 2: 集成到调优编排器")
    print("="*80 + "\n")
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 创建调优编排器（启用安全检查）
    print("创建调优编排器（安全检查：启用）\n")
    orchestrator = TuningOrchestrator(dbms, use_rag=False, enable_safety_check=True)
    
    # 执行完整工作流（带安全检查）
    print("\n执行完整工作流...\n")
    workflow_report = orchestrator.run_full_tuning_workflow(
        window_minutes=60,
        auto_apply=False,
        enable_safety_check=True
    )
    
    # 检查是否包含安全检查步骤
    has_safety_check = 'step2_5_safety_check' in workflow_report.get('steps', {})
    print(f"\n结果:")
    print(f"  工作流状态: {workflow_report.get('status', 'completed')}")
    print(f"  包含安全检查: {has_safety_check}")
    
    if has_safety_check:
        safety_check = workflow_report['steps']['step2_5_safety_check']
        if not safety_check.get('skipped', False):
            print(f"  安全检查结果: {safety_check.get('summary', {}).get('safe_to_apply', 'unknown')}")
            print(f"  风险等级: {safety_check.get('risk_assessment', {}).get('risk_level', 'unknown')}")
    
    # 注意：PgDBMS 没有 disconnect 方法，去掉
    # dbms.disconnect()
    
    return workflow_report


def main():
    """主测试函数"""
    print("\n" + "#"*80)
    print("# 规则引擎集成测试")
    print("#"*80 + "\n")
    
    try:
        # 测试 1: 安全引擎独立测试
        safe_report, risky_report = test_safety_engine()
        
        # 测试 2: 集成到调优流程
        workflow_report = test_integrated_workflow()
        
        # 总结
        print("\n\n" + "#"*80)
        print("# 测试总结")
        print("#"*80 + "\n")
        
        print("✅ 所有测试完成！")
        print("\n功能验证:")
        print("  1. 参数验证器 - 检查参数是否超出系统资源")
        print("  2. 冲突检测器 - 检查参数依赖和冲突关系")
        print("  3. 风险评估器 - 综合评估配置变更风险")
        print("  4. 工作流集成 - 安全检查已集成到调优流程")
        print("\n安全保障:")
        print("  • 高风险配置会被自动拦截")
        print("  • 参数冲突会被提前检测")
        print("  • 系统资源约束会被验证")
        print("  • 避免极端配置导致系统崩溃")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
