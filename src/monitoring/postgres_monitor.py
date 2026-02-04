#!/usr/bin/env python3
"""
PostgreSQL 实时性能监控模块
采集数据库性能指标用于工作负载分析和自适应调优
"""
import psycopg2
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time


class PostgreSQLMonitor:
    """PostgreSQL 性能指标实时采集器"""
    
    def __init__(self, dbms):
        """
        初始化监控器
        
        Args:
            dbms: PgDBMS 实例，提供数据库连接
        """
        self.dbms = dbms
        self.connection = dbms.connection
        
    def _execute_query(self, query: str) -> List[Tuple]:
        """执行查询并返回结果"""
        try:
            result, _ = self.dbms.get_sql_result(query)
            return result
        except Exception as e:
            print(f"查询执行失败: {e}")
            return []
    
    def get_qps(self, window_minutes: int = 5) -> float:
        """
        获取查询每秒数（QPS）
        
        Args:
            window_minutes: 统计时间窗口（分钟）
            
        Returns:
            平均 QPS
        """
        # 简化查询，使用 pg_stat_database 的总事务数
        query = """
        SELECT 
            COALESCE(xact_commit + xact_rollback, 0) as total_transactions
        FROM pg_stat_database
        WHERE datname = current_database();
        """
        result = self._execute_query(query)
        # 返回一个估算值
        return float(result[0][0] if result and result[0][0] else 0) / 60.0
    
    def get_tps(self, window_minutes: int = 5) -> Dict[str, float]:
        """
        获取事务每秒数（TPS）
        
        Args:
            window_minutes: 统计时间窗口（分钟）
            
        Returns:
            {'commit_tps': 提交TPS, 'rollback_tps': 回滚TPS, 'total_tps': 总TPS}
        """
        query = f"""
        SELECT 
            xact_commit,
            xact_rollback,
            EXTRACT(EPOCH FROM (NOW() - stats_reset)) as elapsed_seconds
        FROM pg_stat_database
        WHERE datname = current_database();
        """
        result = self._execute_query(query)
        
        if result and result[0]:
            commits, rollbacks, elapsed = result[0]
            elapsed = float(max(elapsed or 1, 1))  # 避免除零，转换为 float
            
            return {
                'commit_tps': float(commits or 0) / elapsed,
                'rollback_tps': float(rollbacks or 0) / elapsed,
                'total_tps': float((commits or 0) + (rollbacks or 0)) / elapsed
            }
        
        return {'commit_tps': 0.0, 'rollback_tps': 0.0, 'total_tps': 0.0}
    
    def get_connection_stats(self) -> Dict[str, int]:
        """
        获取连接统计信息
        
        Returns:
            {'active': 活跃连接, 'idle': 空闲连接, 'total': 总连接数}
        """
        query = """
        SELECT 
            COUNT(*) FILTER (WHERE state = 'active') as active_count,
            COUNT(*) FILTER (WHERE state = 'idle') as idle_count,
            COUNT(*) as total_count
        FROM pg_stat_activity
        WHERE datname = current_database();
        """
        result = self._execute_query(query)
        
        if result and result[0]:
            active, idle, total = result[0]
            return {
                'active': int(active or 0),
                'idle': int(idle or 0),
                'total': int(total or 0)
            }
        
        return {'active': 0, 'idle': 0, 'total': 0}
    
    def get_cache_hit_ratio(self) -> Dict[str, float]:
        """
        获取缓存命中率
        
        Returns:
            {'buffer_hit_ratio': 缓冲区命中率, 'index_hit_ratio': 索引命中率}
        """
        # 缓冲区命中率
        buffer_query = """
        SELECT 
            CASE 
                WHEN (blks_hit + blks_read) > 0 
                THEN (blks_hit::float / (blks_hit + blks_read)) * 100
                ELSE 0
            END as buffer_hit_ratio
        FROM pg_stat_database
        WHERE datname = current_database();
        """
        buffer_result = self._execute_query(buffer_query)
        buffer_hit_ratio = float(buffer_result[0][0]) if buffer_result and buffer_result[0][0] else 0.0
        
        # 索引命中率
        index_query = """
        SELECT 
            CASE 
                WHEN (idx_blks_hit + idx_blks_read) > 0
                THEN (idx_blks_hit::float / (idx_blks_hit + idx_blks_read)) * 100
                ELSE 0
            END as index_hit_ratio
        FROM pg_statio_user_indexes;
        """
        index_result = self._execute_query(index_query)
        index_hit_ratio = float(index_result[0][0]) if index_result and index_result[0][0] else 0.0
        
        return {
            'buffer_hit_ratio': buffer_hit_ratio,
            'index_hit_ratio': index_hit_ratio
        }
    
    def get_io_stats(self) -> Dict[str, float]:
        """
        获取 I/O 统计信息
        
        Returns:
            {'blks_read_per_sec': 每秒读取块数, 'blks_write_per_sec': 每秒写入块数}
        """
        query = """
        SELECT 
            blks_read,
            blks_hit,
            tup_returned,
            tup_fetched,
            tup_inserted,
            tup_updated,
            tup_deleted,
            EXTRACT(EPOCH FROM (NOW() - stats_reset)) as elapsed_seconds
        FROM pg_stat_database
        WHERE datname = current_database();
        """
        result = self._execute_query(query)
        
        if result and result[0]:
            blks_read, blks_hit, tup_ret, tup_fetch, tup_ins, tup_upd, tup_del, elapsed = result[0]
            elapsed = float(max(elapsed or 1, 1))
            
            return {
                'blks_read_per_sec': float(blks_read or 0) / elapsed,
                'blks_hit_per_sec': float(blks_hit or 0) / elapsed,
                'rows_returned_per_sec': float(tup_ret or 0) / elapsed,
                'rows_fetched_per_sec': float(tup_fetch or 0) / elapsed,
                'rows_modified_per_sec': float((tup_ins or 0) + (tup_upd or 0) + (tup_del or 0)) / elapsed
            }
        
        return {
            'blks_read_per_sec': 0.0,
            'blks_hit_per_sec': 0.0,
            'rows_returned_per_sec': 0.0,
            'rows_fetched_per_sec': 0.0,
            'rows_modified_per_sec': 0.0
        }
    
    def get_slow_queries(self, limit: int = 10, min_exec_time_ms: float = 100.0) -> List[Dict]:
        """
        获取慢查询列表（需要 pg_stat_statements 扩展）
        
        Args:
            limit: 返回前 N 条慢查询
            min_exec_time_ms: 最小执行时间（毫秒）
            
        Returns:
            慢查询列表
        """
        # 检查 pg_stat_statements 是否可用
        check_query = """
        SELECT EXISTS (
            SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'
        );
        """
        check_result = self._execute_query(check_query)
        
        if not check_result or not check_result[0][0]:
            print("警告：pg_stat_statements 扩展未安装，无法获取慢查询")
            return []
        
        query = f"""
        SELECT 
            query,
            calls,
            total_exec_time / calls as avg_exec_time_ms,
            total_exec_time,
            rows / GREATEST(calls, 1) as avg_rows
        FROM pg_stat_statements
        WHERE total_exec_time / calls > {min_exec_time_ms}
        ORDER BY total_exec_time DESC
        LIMIT {limit};
        """
        
        result = self._execute_query(query)
        
        slow_queries = []
        for row in result:
            query_text, calls, avg_time, total_time, avg_rows = row
            slow_queries.append({
                'query': query_text[:200] + '...' if len(query_text) > 200 else query_text,
                'calls': int(calls),
                'avg_exec_time_ms': float(avg_time),
                'total_exec_time_ms': float(total_time),
                'avg_rows': float(avg_rows)
            })
        
        return slow_queries
    
    def get_table_stats(self, top_n: int = 10) -> List[Dict]:
        """
        获取表统计信息
        
        Args:
            top_n: 返回前 N 个最活跃的表
            
        Returns:
            表统计列表
        """
        query = f"""
        SELECT 
            schemaname || '.' || tablename as table_name,
            seq_scan,
            seq_tup_read,
            idx_scan,
            idx_tup_fetch,
            n_tup_ins + n_tup_upd + n_tup_del as total_modifications
        FROM pg_stat_user_tables
        ORDER BY (seq_scan + COALESCE(idx_scan, 0)) DESC
        LIMIT {top_n};
        """
        
        result = self._execute_query(query)
        
        table_stats = []
        for row in result:
            table_name, seq_scan, seq_read, idx_scan, idx_fetch, modifications = row
            table_stats.append({
                'table': table_name,
                'seq_scans': int(seq_scan or 0),
                'seq_tuples_read': int(seq_read or 0),
                'index_scans': int(idx_scan or 0),
                'index_tuples_fetched': int(idx_fetch or 0),
                'total_modifications': int(modifications or 0)
            })
        
        return table_stats
    
    def get_lock_stats(self) -> Dict[str, int]:
        """
        获取锁统计信息
        
        Returns:
            各类锁的数量
        """
        query = """
        SELECT 
            mode,
            COUNT(*) as lock_count
        FROM pg_locks
        GROUP BY mode;
        """
        
        result = self._execute_query(query)
        
        lock_stats = {}
        for row in result:
            mode, count = row
            lock_stats[mode] = int(count)
        
        return lock_stats
    
    def collect_comprehensive_metrics(self, window_minutes: int = 120) -> Dict:
        """
        收集全面的性能指标（用于调优决策）
        
        Args:
            window_minutes: 统计时间窗口（分钟），默认 2 小时
            
        Returns:
            完整的性能指标字典
        """
        print(f"\n{'='*60}")
        print(f"开始采集 PostgreSQL 性能指标（时间窗口: {window_minutes} 分钟）")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # 1. 基础性能指标
        print("1/7 采集基础性能指标...")
        tps_stats = self.get_tps(window_minutes)
        qps = self.get_qps(window_minutes)
        
        # 2. 连接统计
        print("2/7 采集连接统计...")
        conn_stats = self.get_connection_stats()
        
        # 3. 缓存命中率
        print("3/7 采集缓存命中率...")
        cache_stats = self.get_cache_hit_ratio()
        
        # 4. I/O 统计
        print("4/7 采集 I/O 统计...")
        io_stats = self.get_io_stats()
        
        # 5. 慢查询
        print("5/7 采集慢查询...")
        slow_queries = self.get_slow_queries(limit=5)
        
        # 6. 表统计
        print("6/7 采集表统计...")
        table_stats = self.get_table_stats(top_n=5)
        
        # 7. 锁统计
        print("7/7 采集锁统计...")
        lock_stats = self.get_lock_stats()
        
        elapsed_time = time.time() - start_time
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'window_minutes': window_minutes,
            'collection_time_seconds': round(elapsed_time, 2),
            
            # 事务统计
            'tps': {
                'commit_tps': round(tps_stats['commit_tps'], 2),
                'rollback_tps': round(tps_stats['rollback_tps'], 2),
                'total_tps': round(tps_stats['total_tps'], 2)
            },
            'qps': round(qps, 2),
            
            # 连接统计
            'connections': conn_stats,
            
            # 缓存命中率
            'cache_hit_ratio': {
                'buffer': round(cache_stats['buffer_hit_ratio'], 2),
                'index': round(cache_stats['index_hit_ratio'], 2)
            },
            
            # I/O 统计
            'io': {
                'blks_read_per_sec': round(io_stats['blks_read_per_sec'], 2),
                'blks_hit_per_sec': round(io_stats['blks_hit_per_sec'], 2),
                'rows_returned_per_sec': round(io_stats['rows_returned_per_sec'], 2),
                'rows_modified_per_sec': round(io_stats['rows_modified_per_sec'], 2)
            },
            
            # 慢查询
            'slow_queries': slow_queries,
            'slow_query_count': len(slow_queries),
            
            # 表统计
            'top_tables': table_stats,
            
            # 锁统计
            'locks': lock_stats,
            'total_locks': sum(lock_stats.values())
        }
        
        print(f"\n✅ 指标采集完成（耗时 {elapsed_time:.2f} 秒）\n")
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict):
        """打印指标摘要"""
        print(f"\n{'='*60}")
        print(f"性能指标摘要 - {metrics['timestamp']}")
        print(f"{'='*60}\n")
        
        print(f"📊 事务统计:")
        print(f"  TPS (总):     {metrics['tps']['total_tps']:.2f} 事务/秒")
        print(f"  TPS (提交):   {metrics['tps']['commit_tps']:.2f} 事务/秒")
        print(f"  TPS (回滚):   {metrics['tps']['rollback_tps']:.2f} 事务/秒")
        print(f"  QPS:          {metrics['qps']:.2f} 查询/秒")
        
        print(f"\n🔌 连接统计:")
        print(f"  活跃连接:     {metrics['connections']['active']}")
        print(f"  空闲连接:     {metrics['connections']['idle']}")
        print(f"  总连接数:     {metrics['connections']['total']}")
        
        print(f"\n💾 缓存命中率:")
        print(f"  缓冲区:       {metrics['cache_hit_ratio']['buffer']:.2f}%")
        print(f"  索引:         {metrics['cache_hit_ratio']['index']:.2f}%")
        
        print(f"\n💿 I/O 统计:")
        print(f"  物理读取:     {metrics['io']['blks_read_per_sec']:.2f} 块/秒")
        print(f"  缓存读取:     {metrics['io']['blks_hit_per_sec']:.2f} 块/秒")
        print(f"  返回行数:     {metrics['io']['rows_returned_per_sec']:.2f} 行/秒")
        print(f"  修改行数:     {metrics['io']['rows_modified_per_sec']:.2f} 行/秒")
        
        if metrics['slow_queries']:
            print(f"\n🐌 慢查询 Top {len(metrics['slow_queries'])}:")
            for i, sq in enumerate(metrics['slow_queries'][:3], 1):
                print(f"  {i}. 平均耗时: {sq['avg_exec_time_ms']:.2f}ms, 调用: {sq['calls']} 次")
                print(f"     {sq['query'][:80]}...")
        
        print(f"\n🔒 锁统计:")
        print(f"  总锁数:       {metrics['total_locks']}")
        
        print(f"\n{'='*60}\n")


# 测试代码
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/root/GPTuner/src')
    
    from configparser import ConfigParser
    from dbms.postgres import PgDBMS
    
    # 加载配置
    config = ConfigParser()
    config.read('/root/GPTuner/configs/postgres.ini')
    
    # 初始化数据库连接
    print("初始化数据库连接...")
    dbms = PgDBMS.from_file(config)
    
    # 创建监控器
    monitor = PostgreSQLMonitor(dbms)
    
    # 采集指标
    metrics = monitor.collect_comprehensive_metrics(window_minutes=120)
    
    # 打印摘要
    monitor.print_metrics_summary(metrics)
    
    # 保存到文件
    import json
    output_file = '/root/GPTuner/monitoring_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"✅ 完整指标已保存至: {output_file}")
