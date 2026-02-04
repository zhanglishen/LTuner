#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/GPTuner/src')
from monitoring.postgres_monitor import PostgreSQLMonitor
from monitoring.workload_analyzer import WorkloadAnalyzer

class AdaptiveMonitor:
    def __init__(self, dbms, use_rag=False):
        self.dbms = dbms
        self.monitor = PostgreSQLMonitor(dbms)
        self.analyzer = WorkloadAnalyzer()
        
    def collect_and_analyze(self, window_minutes=120):
        metrics = self.monitor.collect_comprehensive_metrics(window_minutes)
        analysis = self.analyzer.analyze_comprehensive(metrics)
        # 添加 scenario 字段
        scenario = analysis['workload']['workload_type']
        summary = analysis['summary'].copy()
        summary['scenario'] = scenario
        return {
            'metrics': metrics, 
            'workload_analysis': analysis, 
            'scenario': {'scenario': scenario, 'confidence': analysis['workload']['confidence']},
            'summary': summary
        }
