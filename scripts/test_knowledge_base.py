#!/usr/bin/env python3
"""
LADO 知识库系统测试脚本
测试 KnowledgeBase、RAG 检索器、症状驱动检索等功能
"""
import os
import sys

# 添加 src 到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from knowledge_handler.knowledge_base import KnowledgeBase, KnowledgeUnit


def test_knowledge_base():
    """测试 KnowledgeBase 基础功能"""
    print("=" * 60)
    print("测试 1: KnowledgeBase 基础功能")
    print("=" * 60)
    
    kb = KnowledgeBase(db='postgres')
    
    # 统计信息
    stats = kb.get_statistics()
    print(f"\n✓ 总参数数: {stats['total_units']}")
    print(f"✓ 平均症状数: {stats['avg_symptoms_per_unit']:.2f}")
    print(f"✓ 有依赖的参数: {stats['units_with_dependencies']}")
    print(f"✓ 有约束的参数: {stats['units_with_constraints']}")
    print(f"✓ 需要重启的参数: {stats['units_needing_restart']}")
    
    print(f"\n分类分布 (前10):")
    for cat, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {cat:25s}: {count:3d} 个参数")
    
    return kb


def test_symptom_search(kb):
    """测试症状驱动检索"""
    print("\n" + "=" * 60)
    print("测试 2: 症状驱动检索")
    print("=" * 60)
    
    test_cases = [
        ("high IO wait", "高 IO 等待"),
        ("OOM error", "内存溢出"),
        ("slow query", "慢查询"),
        ("bloat accumulation", "膨胀"),
        ("checkpoint burst", "检查点风暴")
    ]
    
    for symptom_en, symptom_cn in test_cases:
        print(f"\n查询: '{symptom_en}' ({symptom_cn})")
        results = kb.search_by_symptom(symptom_en, top_k=3)
        
        if results:
            print(f"找到 {len(results)} 个相关参数:")
            for i, unit in enumerate(results, 1):
                print(f"  {i}. {unit.name} ({unit.category})")
                print(f"     描述: {unit.description[:60]}...")
                if unit.symptoms:
                    print(f"     症状: {unit.symptoms[0]}")
        else:
            print("  未找到匹配的参数")


def test_dependency_analysis(kb):
    """测试依赖链分析"""
    print("\n" + "=" * 60)
    print("测试 3: 依赖链分析")
    print("=" * 60)
    
    test_knobs = ['work_mem', 'shared_buffers', 'autovacuum_vacuum_cost_delay']
    
    for knob_name in test_knobs:
        unit = kb.get_unit(knob_name)
        if not unit:
            print(f"\n参数 {knob_name} 不存在")
            continue
        
        print(f"\n参数: {knob_name}")
        print(f"  类别: {unit.category}")
        print(f"  重启需求: {'是' if unit.needs_restart() else '否'}")
        
        if unit.dependencies:
            print(f"  依赖参数: {', '.join(unit.dependencies)}")
        
        if unit.related_knobs:
            print(f"  相关参数: {', '.join(unit.related_knobs)}")
        
        chain = kb.get_dependency_chain(knob_name)
        if len(chain) > 1:
            print(f"  完整依赖链: {' → '.join(chain)}")
        
        dependent = kb.get_dependent_knobs(knob_name)
        if dependent:
            print(f"  被依赖于: {', '.join([u.name for u in dependent])}")


def test_knowledge_unit_views(kb):
    """测试知识单元的多视角嵌入"""
    print("\n" + "=" * 60)
    print("测试 4: 知识单元多视角嵌入")
    print("=" * 60)
    
    test_knob = 'shared_buffers'
    unit = kb.get_unit(test_knob)
    
    if unit:
        print(f"\n参数: {unit.name}")
        views = unit.get_embedding_texts()
        
        for view_name, text in views.items():
            print(f"\n{view_name.upper()} 视角:")
            print(f"  {text[:150]}...")
    else:
        print(f"参数 {test_knob} 不存在")


def test_category_query(kb):
    """测试分类查询"""
    print("\n" + "=" * 60)
    print("测试 5: 分类查询")
    print("=" * 60)
    
    test_categories = ['Autovacuum', 'WAL', 'Resource', 'Planner']
    
    for category in test_categories:
        units = kb.get_by_category(category)
        print(f"\n类别: {category}")
        print(f"  参数数量: {len(units)}")
        if units:
            print(f"  示例参数: {', '.join([u.name for u in units[:5]])}")


def test_constraint_validation(kb):
    """测试约束验证"""
    print("\n" + "=" * 60)
    print("测试 6: 约束规则检查")
    print("=" * 60)
    
    # 检查有约束的参数
    units_with_constraints = [u for u in kb.get_all_units() if u.constraints]
    
    print(f"\n有约束规则的参数: {len(units_with_constraints)} 个")
    print("\n示例约束:")
    
    for unit in units_with_constraints[:5]:
        print(f"\n参数: {unit.name}")
        if unit.constraints.get('hard_rule'):
            print(f"  硬约束: {unit.constraints['hard_rule']}")
        if unit.constraints.get('min'):
            print(f"  最小值: {unit.constraints['min']}")
        if unit.constraints.get('max'):
            print(f"  最大值: {unit.constraints['max']}")


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("LADO 知识库系统测试")
    print("=" * 60)
    
    try:
        # 1. 测试基础功能
        kb = test_knowledge_base()
        
        # 2. 测试症状检索
        test_symptom_search(kb)
        
        # 3. 测试依赖分析
        test_dependency_analysis(kb)
        
        # 4. 测试多视角嵌入
        test_knowledge_unit_views(kb)
        
        # 5. 测试分类查询
        test_category_query(kb)
        
        # 6. 测试约束验证
        test_constraint_validation(kb)
        
        print("\n" + "=" * 60)
        print("✓ 所有测试完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
