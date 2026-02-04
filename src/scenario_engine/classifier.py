"""
场景分类器模块
根据负载特征自动识别场景类型（OLTP/OLAP/HYBRID）
"""
import json


class ScenarioClassifier:
    """
    场景分类器
    基于工作负载特征自动识别场景类型
    """
    
    # 场景定义
    SCENARIOS = {
        'OLTP': {
            'name': '高并发OLTP',
            'description': '高并发短事务场景，大量随机读写',
            'characteristics': [
                '高 QPS（每秒查询数）',
                '短查询时间（< 100ms）',
                '简单查询为主',
                '高并发连接数',
                '随机IO为主'
            ]
        },
        'OLAP': {
            'name': '批量OLAP',
            'description': '复杂分析查询场景，大数据扫描和聚合',
            'characteristics': [
                '低 QPS',
                '长查询时间（> 1s）',
                '复杂查询（JOIN、聚合、排序）',
                '顺序扫描为主',
                '大量内存和CPU消耗'
            ]
        },
        'HYBRID': {
            'name': '混合负载',
            'description': 'OLTP和OLAP混合场景',
            'characteristics': [
                '中等 QPS',
                '查询时间混合',
                '查询类型多样',
                '需要平衡两类场景'
            ]
        }
    }
    
    def __init__(self):
        # 分类阈值（可调整）
        self.thresholds = {
            'oltp_qps_min': 100,          # OLTP 最小 QPS
            'oltp_avg_time_max': 0.1,     # OLTP 最大平均查询时间（秒）
            'olap_qps_max': 50,           # OLAP 最大 QPS
            'olap_avg_time_min': 1.0,     # OLAP 最小平均查询时间（秒）
            'complex_query_ratio': 0.3    # 复杂查询比例阈值
        }
        
    def classify(self, workload_stats):
        """
        分类工作负载
        
        Args:
            workload_stats: {
                'qps': 查询每秒数,
                'avg_query_time': 平均查询时间（秒）,
                'read_write_ratio': 读写比例,
                'query_complexity': 查询复杂度 ('simple'/'medium'/'complex'),
                'transaction_rate': 事务率（可选）,
                'connection_count': 连接数（可选）,
                'top_queries': 热点查询列表（可选）
            }
            
        Returns:
            {
                'scenario': 'OLTP'/'OLAP'/'HYBRID',
                'confidence': 0.0-1.0,
                'reasons': [原因列表],
                'recommendations': [建议列表]
            }
        """
        qps = workload_stats.get('qps', 0)
        avg_time = workload_stats.get('avg_query_time', 0)
        query_complexity = workload_stats.get('query_complexity', 'medium')
        
        reasons = []
        oltp_score = 0
        olap_score = 0
        
        # 规则1：基于 QPS 判断
        if qps >= self.thresholds['oltp_qps_min']:
            oltp_score += 2
            reasons.append(f"高QPS({qps}) 倾向于OLTP场景")
        elif qps <= self.thresholds['olap_qps_max']:
            olap_score += 2
            reasons.append(f"低QPS({qps}) 倾向于OLAP场景")
            
        # 规则2：基于查询时间判断
        if avg_time <= self.thresholds['oltp_avg_time_max']:
            oltp_score += 2
            reasons.append(f"短查询时间({avg_time:.3f}s) 倾向于OLTP场景")
        elif avg_time >= self.thresholds['olap_avg_time_min']:
            olap_score += 2
            reasons.append(f"长查询时间({avg_time:.3f}s) 倾向于OLAP场景")
            
        # 规则3：基于查询复杂度判断
        if query_complexity == 'simple':
            oltp_score += 1
            reasons.append("简单查询为主，倾向于OLTP场景")
        elif query_complexity == 'complex':
            olap_score += 1
            reasons.append("复杂查询为主，倾向于OLAP场景")
            
        # 规则4：基于连接数判断
        if 'connection_count' in workload_stats:
            conn_count = workload_stats['connection_count']
            if conn_count > 100:
                oltp_score += 1
                reasons.append(f"高并发连接数({conn_count}) 倾向于OLTP场景")
                
        # 规则5：基于读写比例判断
        if 'read_write_ratio' in workload_stats:
            rw_ratio = workload_stats['read_write_ratio']
            if rw_ratio > 0.8:  # 读多写少
                olap_score += 0.5
                reasons.append(f"读多写少(读写比{rw_ratio:.2f}) 可能是OLAP场景")
                
        # 决策
        total_score = oltp_score + olap_score
        if total_score == 0:
            scenario = 'HYBRID'
            confidence = 0.5
            reasons.append("无法明确判断，归类为混合场景")
        elif oltp_score > olap_score * 1.5:
            scenario = 'OLTP'
            confidence = min(oltp_score / (total_score + 1), 0.95)
        elif olap_score > oltp_score * 1.5:
            scenario = 'OLAP'
            confidence = min(olap_score / (total_score + 1), 0.95)
        else:
            scenario = 'HYBRID'
            confidence = 0.5 + abs(oltp_score - olap_score) / (total_score + 1) * 0.3
            reasons.append("OLTP和OLAP特征混合，归类为混合场景")
            
        # 生成推荐
        recommendations = self._generate_recommendations(scenario, workload_stats)
        
        return {
            'scenario': scenario,
            'confidence': confidence,
            'reasons': reasons,
            'recommendations': recommendations,
            'scores': {
                'oltp_score': oltp_score,
                'olap_score': olap_score
            }
        }
        
    def classify_by_benchmark(self, benchmark_name):
        """
        根据基准测试名称直接分类
        
        Args:
            benchmark_name: 'tpch', 'tpcc', 'sysbench', etc.
            
        Returns:
            场景类型
        """
        benchmark_map = {
            'tpch': 'OLAP',
            'tpcc': 'OLTP',
            'tpce': 'OLTP',
            'sysbench': 'OLTP',
            'ycsb': 'OLTP',
            'job': 'OLAP',
            'ssb': 'OLAP'
        }
        
        scenario = benchmark_map.get(benchmark_name.lower(), 'HYBRID')
        
        return {
            'scenario': scenario,
            'confidence': 0.9,
            'reasons': [f"基于基准测试 {benchmark_name} 的已知特征"],
            'recommendations': self._generate_recommendations(scenario, {})
        }
        
    def _generate_recommendations(self, scenario, workload_stats):
        """根据场景生成优化建议"""
        recommendations = []
        
        if scenario == 'OLTP':
            recommendations.extend([
                "重点关注连接池和缓存参数",
                "优化随机IO性能",
                "调整并发控制参数",
                "关注响应时间指标"
            ])
        elif scenario == 'OLAP':
            recommendations.extend([
                "增加工作内存(work_mem)",
                "优化并行查询设置",
                "调整顺序扫描相关参数",
                "关注吞吐量和查询延迟"
            ])
        else:  # HYBRID
            recommendations.extend([
                "平衡OLTP和OLAP参数",
                "考虑读写分离架构",
                "动态调整策略",
                "监控不同类型查询的性能"
            ])
            
        return recommendations
        
    def get_scenario_info(self, scenario):
        """获取场景详细信息"""
        return self.SCENARIOS.get(scenario, None)
        
    def get_all_scenarios(self):
        """获取所有场景定义"""
        return self.SCENARIOS


if __name__ == '__main__':
    # 测试分类器
    classifier = ScenarioClassifier()
    
    # 测试 OLTP 场景
    print("=== 测试 OLTP 场景 ===")
    oltp_stats = {
        'qps': 500,
        'avg_query_time': 0.05,
        'read_write_ratio': 0.6,
        'query_complexity': 'simple',
        'connection_count': 200
    }
    result = classifier.classify(oltp_stats)
    print(f"场景: {result['scenario']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("判断依据:")
    for reason in result['reasons']:
        print(f"  - {reason}")
        
    # 测试 OLAP 场景
    print("\n=== 测试 OLAP 场景 ===")
    olap_stats = {
        'qps': 10,
        'avg_query_time': 5.2,
        'read_write_ratio': 0.95,
        'query_complexity': 'complex'
    }
    result = classifier.classify(olap_stats)
    print(f"场景: {result['scenario']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("判断依据:")
    for reason in result['reasons']:
        print(f"  - {reason}")
        
    # 测试基准测试分类
    print("\n=== 基于基准测试分类 ===")
    result = classifier.classify_by_benchmark('tpch')
    print(f"TPC-H -> {result['scenario']}")
    result = classifier.classify_by_benchmark('tpcc')
    print(f"TPC-C -> {result['scenario']}")
