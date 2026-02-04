#!/usr/bin/env python3
"""
工作负载特征分析器
基于采集的性能指标分析工作负载特征并提供调优建议
"""
from typing import Dict, List
from datetime import datetime


class WorkloadAnalyzer:
    """工作负载特征分析器"""
    
    def __init__(self):
        """初始化分析器"""
        pass
    
    def analyze_workload_type(self, metrics: Dict) -> Dict:
        """分析工作负载类型"""
        tps = metrics['tps']['total_tps']
        qps = metrics['qps']
        
        # 简单判断逻辑
        if tps > 10 and qps > 50:
            workload_type = 'OLTP'
            confidence = 0.85
        elif tps < 10:
            workload_type = 'OLAP'
            confidence = 0.80
        else:
            workload_type = 'HYBRID'
            confidence = 0.60
        
        return {
            'workload_type': workload_type,
            'confidence': confidence,
            'characteristics': {
                'tps': tps,
                'qps': qps,
                'buffer_hit_ratio': metrics['cache_hit_ratio']['buffer']
            },
            'evidence': [f"TPS: {tps:.2f}, QPS: {qps:.2f}"]
        }
    
    def detect_resource_pressure(self, metrics: Dict) -> Dict:
        """检测资源压力"""
        pressure_points = []
        
        if metrics['cache_hit_ratio']['buffer'] < 80.0:
            pressure_points.append({
                'type': 'cache',
                'message': f'缓存命中率偏低 ({metrics["cache_hit_ratio"]["buffer"]:.1f}%)',
                'severity': 'high'
            })
        
        return {
            'has_pressure': len(pressure_points) > 0,
            'pressure_points': pressure_points,
            'severity': 'high' if pressure_points else 'none'
        }
    
    def generate_tuning_hints(self, metrics: Dict, workload_analysis: Dict, pressure_analysis: Dict) -> List[Dict]:
        """生成调优提示"""
        hints = []
        workload_type = workload_analysis['workload_type']
        
        if workload_type == 'OLTP':
            hints.append({
                'category': 'workload',
                'priority': 'high',
                'message': 'OLTP 场景建议优化连接池和事务并发',
                'related_knobs': ['max_connections', 'shared_buffers']
            })
        
        return hints
    
    def analyze_comprehensive(self, metrics: Dict) -> Dict:
        """全面分析工作负载"""
        workload_analysis = self.analyze_workload_type(metrics)
        pressure_analysis = self.detect_resource_pressure(metrics)
        tuning_hints = self.generate_tuning_hints(metrics, workload_analysis, pressure_analysis)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'workload': workload_analysis,
            'resource_pressure': pressure_analysis,
            'tuning_hints': tuning_hints,
            'summary': {
                'workload_type': workload_analysis['workload_type'],
                'confidence': workload_analysis['confidence'],
                'has_pressure': pressure_analysis['has_pressure'],
                'pressure_severity': pressure_analysis['severity'],
                'hint_count': len(tuning_hints)
            }
        }
