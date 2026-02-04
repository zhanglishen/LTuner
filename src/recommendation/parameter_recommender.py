#!/usr/bin/env python3
"""
参数推荐生成器
基于 RAG 知识库、场景识别和工作负载分析生成数据库参数调优建议
"""
import sys
sys.path.insert(0, '/root/GPTuner/src')

from typing import Dict, List, Optional
from datetime import datetime
import json

# 降级导入
try:
    from rag_engine.retriever import RAGRetriever
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    print("警告：RAG 模块未找到，将使用规则基础推荐")

try:
    from scenario_engine.classifier import ScenarioClassifier
    from scenario_engine.prompt_templates import PromptTemplates
    SCENARIO_AVAILABLE = True
except ImportError:
    SCENARIO_AVAILABLE = False
    print("警告：场景模块未找到")


class ParameterRecommender:
    """参数推荐生成器"""
    
    def __init__(self, dbms, use_rag=True):
        """
        初始化推荐器
        
        Args:
            dbms: PgDBMS 实例
            use_rag: 是否使用 RAG 增强
        """
        self.dbms = dbms
        self.use_rag = use_rag and RAG_AVAILABLE
        
        # 初始化 RAG 检索器
        self.rag_retriever = None
        if self.use_rag:
            try:
                import os
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                self.rag_retriever = RAGRetriever(db='postgres')
                print("✅ RAG 知识库已加载")
            except Exception as e:
                print(f"⚠️  RAG 加载失败: {e}")
                self.use_rag = False
        
        # 参数约束规则
        self.knob_constraints = self._load_knob_constraints()
        
    def _load_knob_constraints(self) -> Dict:
        """加载参数约束规则"""
        return {
            'shared_buffers': {
                'min_mb': 128,
                'max_ratio': 0.25,  # 系统内存的 25%
                'recommended_ratio': 0.25,
                'unit': 'MB'
            },
            'work_mem': {
                'min_mb': 4,
                'max_mb': 2048,
                'recommended_mb': {
                    'OLTP': 16,
                    'OLAP': 256,
                    'HYBRID': 64
                },
                'unit': 'MB'
            },
            'maintenance_work_mem': {
                'min_mb': 64,
                'max_mb': 2048,
                'recommended_mb': {
                    'OLTP': 256,
                    'OLAP': 1024,
                    'HYBRID': 512
                },
                'unit': 'MB'
            },
            'effective_cache_size': {
                'min_mb': 1024,
                'max_ratio': 0.75,  # 系统内存的 75%
                'recommended_ratio': 0.5,
                'unit': 'MB'
            },
            'max_connections': {
                'min': 20,
                'max': 1000,
                'recommended': {
                    'OLTP': 200,
                    'OLAP': 50,
                    'HYBRID': 100
                }
            },
            'checkpoint_completion_target': {
                'min': 0.5,
                'max': 0.9,
                'recommended': 0.9
            },
            'wal_buffers': {
                'min_mb': 16,
                'max_mb': 64,
                'recommended_mb': 16,
                'unit': 'MB'
            },
            'default_statistics_target': {
                'min': 10,
                'max': 10000,
                'recommended': {
                    'OLTP': 100,
                    'OLAP': 500,
                    'HYBRID': 200
                }
            },
            'random_page_cost': {
                'min': 1.0,
                'max': 4.0,
                'recommended': {
                    'OLTP': 1.1,  # SSD
                    'OLAP': 1.1,
                    'HYBRID': 1.1
                }
            },
            'effective_io_concurrency': {
                'min': 1,
                'max': 1000,
                'recommended': {
                    'OLTP': 200,
                    'OLAP': 200,
                    'HYBRID': 200
                }
            },
            'max_parallel_workers_per_gather': {
                'min': 0,
                'max': 8,
                'recommended': {
                    'OLTP': 0,
                    'OLAP': 4,
                    'HYBRID': 2
                }
            }
        }
    
    def get_current_config(self, knob_names: List[str]) -> Dict[str, str]:
        """获取当前参数配置"""
        current_config = {}
        
        for knob in knob_names:
            try:
                result = self.dbms.get_knob_value(knob)
                if result and result[0]:
                    current_config[knob] = result[0][0]
            except Exception as e:
                print(f"警告：无法获取 {knob} 当前值: {e}")
                current_config[knob] = "unknown"
        
        return current_config
    
    def _get_system_memory(self) -> int:
        """获取系统内存（MB）"""
        # 简化实现，实际应该读取系统信息
        # 可以通过 /proc/meminfo 或 psutil 获取
        return 8192  # 默认 8GB
    
    def generate_rule_based_recommendations(self, 
                                           scenario: str, 
                                           metrics: Dict,
                                           workload_analysis: Dict) -> Dict[str, Dict]:
        """
        基于规则生成推荐配置
        
        Args:
            scenario: 场景类型 (OLTP/OLAP/HYBRID)
            metrics: 性能指标
            workload_analysis: 工作负载分析结果
            
        Returns:
            推荐配置字典
        """
        recommendations = {}
        system_memory_mb = self._get_system_memory()
        
        # 1. shared_buffers
        constraint = self.knob_constraints['shared_buffers']
        recommended_mb = int(system_memory_mb * constraint['recommended_ratio'])
        recommendations['shared_buffers'] = {
            'recommended_value': f'{recommended_mb}MB',
            'reason': f'建议设置为系统内存的 {constraint["recommended_ratio"]*100:.0f}%',
            'priority': 'high',
            'impact': 'high'
        }
        
        # 2. work_mem（根据场景调整）
        constraint = self.knob_constraints['work_mem']
        if isinstance(constraint['recommended_mb'], dict):
            recommended_mb = constraint['recommended_mb'].get(scenario, 64)
        else:
            recommended_mb = constraint['recommended_mb']
        
        recommendations['work_mem'] = {
            'recommended_value': f'{recommended_mb}MB',
            'reason': f'{scenario} 场景推荐值，影响排序和哈希操作',
            'priority': 'high',
            'impact': 'medium'
        }
        
        # 3. maintenance_work_mem
        constraint = self.knob_constraints['maintenance_work_mem']
        recommended_mb = constraint['recommended_mb'].get(scenario, 512)
        recommendations['maintenance_work_mem'] = {
            'recommended_value': f'{recommended_mb}MB',
            'reason': f'{scenario} 场景，影响 VACUUM、CREATE INDEX 等维护操作',
            'priority': 'medium',
            'impact': 'medium'
        }
        
        # 4. effective_cache_size
        constraint = self.knob_constraints['effective_cache_size']
        recommended_mb = int(system_memory_mb * constraint['recommended_ratio'])
        recommendations['effective_cache_size'] = {
            'recommended_value': f'{recommended_mb}MB',
            'reason': '查询规划器使用，建议设置为系统内存的 50%',
            'priority': 'high',
            'impact': 'high'
        }
        
        # 5. max_connections（根据场景和当前连接数）
        constraint = self.knob_constraints['max_connections']
        current_connections = metrics.get('connections', {}).get('total', 50)
        recommended = constraint['recommended'].get(scenario, 100)
        
        # 如果当前连接数接近推荐值，适当增加
        if current_connections > recommended * 0.8:
            recommended = int(current_connections * 1.5)
            recommended = min(recommended, constraint['max'])
        
        recommendations['max_connections'] = {
            'recommended_value': str(recommended),
            'reason': f'{scenario} 场景推荐值，当前连接数 {current_connections}',
            'priority': 'medium',
            'impact': 'medium'
        }
        
        # 6. checkpoint_completion_target
        recommendations['checkpoint_completion_target'] = {
            'recommended_value': '0.9',
            'reason': '平滑 checkpoint，减少 I/O 峰值',
            'priority': 'medium',
            'impact': 'low'
        }
        
        # 7. wal_buffers
        recommendations['wal_buffers'] = {
            'recommended_value': '16MB',
            'reason': '适合大多数工作负载',
            'priority': 'low',
            'impact': 'low'
        }
        
        # 8. 根据缓存命中率调整
        cache_hit_ratio = metrics.get('cache_hit_ratio', {}).get('buffer', 100)
        if cache_hit_ratio < 90:
            # 缓存命中率低，增加 shared_buffers
            current_shared_buffers = int(recommended_mb * 1.5)
            recommendations['shared_buffers']['recommended_value'] = f'{current_shared_buffers}MB'
            recommendations['shared_buffers']['reason'] += f' (当前缓存命中率 {cache_hit_ratio:.1f}% 偏低)'
        
        # 9. 并行查询（OLAP 场景）
        if scenario == 'OLAP':
            recommendations['max_parallel_workers_per_gather'] = {
                'recommended_value': '4',
                'reason': 'OLAP 场景启用并行查询',
                'priority': 'high',
                'impact': 'high'
            }
            recommendations['max_worker_processes'] = {
                'recommended_value': '8',
                'reason': 'OLAP 场景增加工作进程',
                'priority': 'medium',
                'impact': 'medium'
            }
        
        return recommendations
    
    def generate_rag_enhanced_recommendations(self,
                                             scenario: str,
                                             metrics: Dict,
                                             workload_analysis: Dict) -> Dict[str, Dict]:
        """
        基于 RAG 知识库生成增强推荐
        
        Args:
            scenario: 场景类型
            metrics: 性能指标
            workload_analysis: 工作负载分析
            
        Returns:
            RAG 增强的推荐配置
        """
        if not self.use_rag or not self.rag_retriever:
            return {}
        
        rag_recommendations = {}
        
        # 从知识库检索相关参数
        docs = self.rag_retriever.retrieve(
            query=f"{scenario} 场景性能优化关键参数",
            scenario=scenario,
            top_k=10
        )
        
        # 提取知识库中的参数建议
        for doc in docs:
            knob = doc['knob']
            if knob not in rag_recommendations:
                rag_recommendations[knob] = {
                    'knowledge_source': doc['source'],
                    'reference': doc['content'][:200] + '...',
                    'priority': 'medium',
                    'impact': 'medium'
                }
        
        return rag_recommendations
    
    def merge_recommendations(self,
                             rule_based: Dict[str, Dict],
                             rag_enhanced: Dict[str, Dict]) -> Dict[str, Dict]:
        """合并规则基础和 RAG 增强的推荐"""
        merged = rule_based.copy()
        
        # RAG 提供的参数作为补充建议
        for knob, rag_info in rag_enhanced.items():
            if knob not in merged:
                merged[knob] = {
                    'recommended_value': 'see_knowledge_base',
                    'reason': f'知识库推荐（来源: {rag_info["knowledge_source"]}）',
                    'priority': 'low',
                    'impact': 'low',
                    'knowledge_reference': rag_info.get('reference', '')
                }
            else:
                # 为已有推荐添加知识库参考
                merged[knob]['knowledge_reference'] = rag_info.get('reference', '')
        
        return merged
    
    def generate_recommendations(self,
                                scenario: str,
                                metrics: Dict,
                                workload_analysis: Dict) -> Dict:
        """
        生成完整的参数推荐
        
        Args:
            scenario: 场景类型
            metrics: 性能指标
            workload_analysis: 工作负载分析
            
        Returns:
            完整推荐报告
        """
        print(f"\n{'='*60}")
        print(f"生成参数推荐 - 场景: {scenario}")
        print(f"{'='*60}\n")
        
        # 1. 规则基础推荐
        print("1/3 生成规则基础推荐...")
        rule_based = self.generate_rule_based_recommendations(
            scenario, metrics, workload_analysis
        )
        
        # 2. RAG 增强推荐
        print("2/3 从知识库检索增强建议...")
        rag_enhanced = self.generate_rag_enhanced_recommendations(
            scenario, metrics, workload_analysis
        )
        
        # 3. 合并推荐
        print("3/3 合并推荐结果...")
        merged_recommendations = self.merge_recommendations(rule_based, rag_enhanced)
        
        # 4. 获取当前配置
        knob_names = list(merged_recommendations.keys())
        current_config = self.get_current_config(knob_names)
        
        # 5. 生成完整报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario,
            'total_recommendations': len(merged_recommendations),
            'current_config': current_config,
            'recommendations': merged_recommendations,
            'summary': {
                'high_priority_count': sum(1 for r in merged_recommendations.values() if r['priority'] == 'high'),
                'medium_priority_count': sum(1 for r in merged_recommendations.values() if r['priority'] == 'medium'),
                'low_priority_count': sum(1 for r in merged_recommendations.values() if r['priority'] == 'low'),
            }
        }
        
        print(f"\n✅ 推荐生成完成")
        print(f"   总计: {report['total_recommendations']} 个参数")
        print(f"   高优先级: {report['summary']['high_priority_count']} 个")
        print(f"   中优先级: {report['summary']['medium_priority_count']} 个\n")
        
        return report
    
    def print_recommendations(self, report: Dict):
        """打印推荐报告"""
        print(f"\n{'='*60}")
        print(f"参数调优推荐报告")
        print(f"场景: {report['scenario']} | 时间: {report['timestamp']}")
        print(f"{'='*60}\n")
        
        # 按优先级分组
        high_priority = []
        medium_priority = []
        low_priority = []
        
        for knob, rec in report['recommendations'].items():
            item = (knob, rec)
            if rec['priority'] == 'high':
                high_priority.append(item)
            elif rec['priority'] == 'medium':
                medium_priority.append(item)
            else:
                low_priority.append(item)
        
        # 打印高优先级推荐
        if high_priority:
            print(f"🔴 高优先级推荐 ({len(high_priority)} 个):\n")
            for i, (knob, rec) in enumerate(high_priority, 1):
                current = report['current_config'].get(knob, 'unknown')
                print(f"  {i}. {knob}")
                print(f"     当前值: {current}")
                print(f"     推荐值: {rec['recommended_value']}")
                print(f"     原因: {rec['reason']}")
                print(f"     影响: {rec['impact']}")
                print()
        
        # 打印中优先级推荐
        if medium_priority:
            print(f"🟡 中优先级推荐 ({len(medium_priority)} 个):\n")
            for i, (knob, rec) in enumerate(medium_priority, 1):
                current = report['current_config'].get(knob, 'unknown')
                print(f"  {i}. {knob}: {current} → {rec['recommended_value']}")
                print(f"     原因: {rec['reason']}")
                print()
        
        # 打印低优先级推荐
        if low_priority:
            print(f"🟢 低优先级推荐 ({len(low_priority)} 个):\n")
            for knob, rec in low_priority:
                print(f"  - {knob}: {rec['recommended_value']}")
        
        print(f"\n{'='*60}\n")


# 测试代码
if __name__ == '__main__':
    from configparser import ConfigParser
    from dbms.postgres import PgDBMS
    from monitoring.adaptive_monitor import AdaptiveMonitor
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库
    dbms = PgDBMS.from_file(config)
    
    # 采集监控数据
    print("采集性能指标...")
    monitor = AdaptiveMonitor(dbms, use_rag=False)
    monitor_report = monitor.collect_and_analyze(window_minutes=60)
    
    # 创建推荐器
    recommender = ParameterRecommender(dbms, use_rag=True)
    
    # 生成推荐
    recommendations = recommender.generate_recommendations(
        scenario=monitor_report['summary']['scenario'],
        metrics=monitor_report['metrics'],
        workload_analysis=monitor_report['workload_analysis']
    )
    
    # 打印报告
    recommender.print_recommendations(recommendations)
    
    # 保存报告
    output_file = '/root/GPTuner/parameter_recommendations.json'
    with open(output_file, 'w') as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)
    print(f"✅ 推荐报告已保存至: {output_file}")
