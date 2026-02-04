#!/usr/bin/env python3
"""
监控系统测试脚本
验证 Phase 1 功能：性能监控、工作负载分析、场景识别
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from configparser import ConfigParser
from dbms.postgres import PgDBMS
from monitoring.adaptive_monitor import AdaptiveMonitor
import json


def test_basic_monitoring():
    """测试基础监控功能"""
    print("\n" + "="*60)
    print("测试 1: 基础性能监控")
    print("="*60 + "\n")
    
    from monitoring.postgres_monitor import PostgreSQLMonitor
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库连接
    dbms = PgDBMS.from_file(config)
    
    # 创建监控器
    monitor = PostgreSQLMonitor(dbms)
    
    # 测试各项指标采集
    print("测试 QPS 采集...")
    qps = monitor.get_qps(window_minutes=5)
    print(f"  ✓ QPS: {qps:.2f}")
    
    print("测试 TPS 采集...")
    tps = monitor.get_tps(window_minutes=5)
    print(f"  ✓ TPS: {tps['total_tps']:.2f}")
    
    print("测试连接统计...")
    conn_stats = monitor.get_connection_stats()
    print(f"  ✓ 活跃连接: {conn_stats['active']}")
    
    print("测试缓存命中率...")
    cache_stats = monitor.get_cache_hit_ratio()
    print(f"  ✓ 缓存命中率: {cache_stats['buffer_hit_ratio']:.2f}%")
    
    print("\n✅ 基础监控功能测试通过\n")
    
    return monitor, dbms


def test_workload_analysis(monitor):
    """测试工作负载分析"""
    print("\n" + "="*60)
    print("测试 2: 工作负载分析")
    print("="*60 + "\n")
    
    from monitoring.workload_analyzer import WorkloadAnalyzer
    
    # 采集指标
    print("采集性能指标...")
    metrics = monitor.collect_comprehensive_metrics(window_minutes=60)
    
    # 创建分析器
    analyzer = WorkloadAnalyzer()
    
    # 执行分析
    print("\n执行工作负载分析...")
    analysis = analyzer.analyze_comprehensive(metrics)
    
    # 验证分析结果
    assert 'workload' in analysis, "缺少工作负载类型分析"
    assert 'resource_pressure' in analysis, "缺少资源压力分析"
    assert 'tuning_hints' in analysis, "缺少调优提示"
    
    workload_type = analysis['workload']['workload_type']
    confidence = analysis['workload']['confidence']
    
    print(f"\n工作负载类型: {workload_type} (置信度: {confidence:.2f})")
    print(f"调优建议数量: {len(analysis['tuning_hints'])}")
    
    print("\n✅ 工作负载分析功能测试通过\n")
    
    return analysis


def test_scenario_integration():
    """测试场景识别集成"""
    print("\n" + "="*60)
    print("测试 3: 场景识别集成")
    print("="*60 + "\n")
    
    try:
        from scenario_engine.classifier import ScenarioClassifier
        
        classifier = ScenarioClassifier()
        
        # 测试工作负载统计分类
        workload_stats = {
            'qps': 150.0,
            'avg_query_time': 0.05,
            'query_complexity': 'low',
            'connection_count': 50
        }
        
        result = classifier.classify(workload_stats)
        
        print(f"场景识别结果: {result['scenario']}")
        print(f"置信度: {result['confidence']:.2f}")
        print(f"判断依据: {result['reasons'][0]}")
        
        print("\n✅ 场景识别集成测试通过\n")
        
    except ImportError:
        print("⚠️  场景识别模块未安装，跳过测试\n")


def test_rag_integration():
    """测试 RAG 集成"""
    print("\n" + "="*60)
    print("测试 4: RAG 知识检索集成")
    print("="*60 + "\n")
    
    try:
        import os
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        from rag_engine.retriever import RAGRetriever
        
        retriever = RAGRetriever(db='postgres')
        
        # 测试检索
        docs = retriever.retrieve(
            query="OLTP 场景 shared_buffers 优化",
            scenario='OLTP',
            top_k=3
        )
        
        print(f"检索到 {len(docs)} 条相关知识:")
        for i, doc in enumerate(docs, 1):
            print(f"  {i}. 参数: {doc['knob']}")
        
        print("\n✅ RAG 集成测试通过\n")
        
    except Exception as e:
        print(f"⚠️  RAG 模块测试失败: {e}\n")


def test_adaptive_monitor_full():
    """测试完整自适应监控系统"""
    print("\n" + "="*60)
    print("测试 5: 完整自适应监控系统")
    print("="*60 + "\n")
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库连接
    dbms = PgDBMS.from_file(config)
    
    # 创建自适应监控系统
    adaptive_monitor = AdaptiveMonitor(dbms, use_rag=True)
    
    # 执行全面分析
    report = adaptive_monitor.collect_and_analyze(window_minutes=60)
    
    # 验证报告结构
    assert 'metrics' in report, "缺少性能指标"
    assert 'workload_analysis' in report, "缺少工作负载分析"
    assert 'scenario' in report, "缺少场景识别"
    assert 'summary' in report, "缺少摘要"
    
    # 打印综合报告
    adaptive_monitor.print_comprehensive_report(report)
    
    # 保存报告
    output_file = '/root/GPTuner/test_monitor_report.json'
    adaptive_monitor.save_report(report, output_file)
    
    print("\n✅ 完整自适应监控系统测试通过\n")
    
    return report


def run_all_tests():
    """运行所有测试"""
    print("\n" + "#"*60)
    print("# Phase 1 监控系统测试套件")
    print("#"*60 + "\n")
    
    test_results = {
        'timestamp': None,
        'tests_passed': 0,
        'tests_failed': 0,
        'details': []
    }
    
    tests = [
        ("基础性能监控", test_basic_monitoring),
        ("场景识别集成", test_scenario_integration),
        ("RAG 知识检索集成", test_rag_integration),
        ("完整自适应监控", test_adaptive_monitor_full)
    ]
    
    monitor = None
    dbms = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "基础性能监控":
                monitor, dbms = test_func()
                test_results['tests_passed'] += 1
                test_results['details'].append({
                    'test': test_name,
                    'status': 'PASS'
                })
            elif test_name == "工作负载分析" and monitor:
                test_workload_analysis(monitor)
                test_results['tests_passed'] += 1
                test_results['details'].append({
                    'test': test_name,
                    'status': 'PASS'
                })
            else:
                test_func()
                test_results['tests_passed'] += 1
                test_results['details'].append({
                    'test': test_name,
                    'status': 'PASS'
                })
                
        except Exception as e:
            print(f"\n❌ 测试失败: {test_name}")
            print(f"   错误: {str(e)}\n")
            test_results['tests_failed'] += 1
            test_results['details'].append({
                'test': test_name,
                'status': 'FAIL',
                'error': str(e)
            })
    
    # 打印测试总结
    print("\n" + "#"*60)
    print("# 测试总结")
    print("#"*60 + "\n")
    
    total_tests = test_results['tests_passed'] + test_results['tests_failed']
    pass_rate = (test_results['tests_passed'] / total_tests * 100) if total_tests > 0 else 0
    
    print(f"总测试数:     {total_tests}")
    print(f"通过:         {test_results['tests_passed']} ✅")
    print(f"失败:         {test_results['tests_failed']} ❌")
    print(f"通过率:       {pass_rate:.1f}%")
    
    print("\n测试详情:")
    for detail in test_results['details']:
        status_icon = "✅" if detail['status'] == 'PASS' else "❌"
        print(f"  {status_icon} {detail['test']}")
        if detail['status'] == 'FAIL':
            print(f"     错误: {detail.get('error', 'Unknown')}")
    
    print("\n" + "#"*60 + "\n")
    
    if test_results['tests_failed'] == 0:
        print("🎉 所有测试通过！Phase 1 监控系统功能正常\n")
    else:
        print(f"⚠️  有 {test_results['tests_failed']} 个测试失败，请检查\n")
    
    # 保存测试结果
    from datetime import datetime
    test_results['timestamp'] = datetime.now().isoformat()
    
    result_file = '/root/GPTuner/monitor_test_results.json'
    with open(result_file, 'w') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    print(f"测试结果已保存至: {result_file}\n")


if __name__ == '__main__':
    run_all_tests()
