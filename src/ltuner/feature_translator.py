#!/usr/bin/env python3
"""
特征语义翻译器 (Feature Translator)
将数据库原始数值指标转化为 LLM 可理解的自然语言描述，
包含静态特征解析（硬件配置）和动态特征翻译（运行时指标）。
"""
import sys
import os
import psutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class FeatureTranslator:
    """
    特征语义翻译器
    负责将数据库的原始数值指标转化为 LLM 可理解的语义描述，
    使大模型能够基于自然语言进行逻辑推理而非仅处理数字。
    """

    # 缓存命中率阈值定义
    CACHE_THRESHOLDS = {
        'critical': 70.0,   # 严重不足
        'warning': 85.0,    # 偏低
        'good': 95.0,       # 良好
        'excellent': 99.0   # 优秀
    }

    # TPS 分级阈值
    TPS_THRESHOLDS = {
        'idle': 1.0,
        'low': 50.0,
        'moderate': 500.0,
        'high': 2000.0
    }

    def __init__(self, dbms=None):
        """
        初始化翻译器

        Args:
            dbms: PgDBMS 实例（可选，用于获取当前配置）
        """
        self.dbms = dbms
        self.hardware_info = self._collect_hardware_info()

    def _collect_hardware_info(self) -> dict:
        """收集静态硬件信息"""
        info = {
            'cpu_cores': psutil.cpu_count(logical=False) or 1,
            'cpu_threads': psutil.cpu_count(logical=True) or 1,
            'total_memory_mb': int(psutil.virtual_memory().total / (1024 * 1024)),
            'total_memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),
            'storage_type': 'SSD',  # 默认假设 SSD，生产环境可检测
            'pg_version': '14'
        }

        # 尝试获取 PostgreSQL 版本
        if self.dbms:
            try:
                result, _ = self.dbms.get_sql_result("SHOW server_version;")
                if result and result[0]:
                    info['pg_version'] = str(result[0][0])
            except Exception:
                pass

        return info

    def translate_static_features(self) -> str:
        """
        翻译静态硬件与环境特征为自然语言描述

        Returns:
            硬件环境的自然语言描述
        """
        hw = self.hardware_info
        lines = [
            "## 当前实例硬件环境",
            f"- CPU: {hw['cpu_cores']} 物理核心, {hw['cpu_threads']} 逻辑线程",
            f"- 内存: {hw['total_memory_gb']} GB ({hw['total_memory_mb']} MB)",
            f"- 存储类型: {hw['storage_type']}",
            f"- PostgreSQL 版本: {hw['pg_version']}",
            "",
            "### 资源预算参考",
            f"- shared_buffers 建议上限: {int(hw['total_memory_mb'] * 0.25)} MB (RAM 的 25%)",
            f"- effective_cache_size 建议值: {int(hw['total_memory_mb'] * 0.5)} MB (RAM 的 50%)",
            f"- 总内存安全阈值: {int(hw['total_memory_mb'] * 0.8)} MB (RAM 的 80%，需为 OS 预留空间)",
        ]
        return "\n".join(lines)

    def translate_dynamic_metrics(self, metrics: dict) -> str:
        """
        将运行时动态指标翻译为自然语言描述

        Args:
            metrics: monitoring/postgres_monitor.py 采集的完整指标字典

        Returns:
            运行时状态的自然语言描述
        """
        sections = [
            "## 当前数据库运行状态分析",
            "",
            self._translate_transaction_metrics(metrics),
            self._translate_cache_metrics(metrics),
            self._translate_io_metrics(metrics),
            self._translate_connection_metrics(metrics),
            self._translate_lock_metrics(metrics),
            self._translate_slow_queries(metrics),
        ]
        return "\n\n".join(sections)

    def _translate_transaction_metrics(self, metrics: dict) -> str:
        """翻译事务性能指标"""
        tps_data = metrics.get('tps', {})
        total_tps = tps_data.get('total_tps', 0)
        commit_tps = tps_data.get('commit_tps', 0)
        rollback_tps = tps_data.get('rollback_tps', 0)
        qps = metrics.get('qps', 0)

        # 判断 TPS 水平
        if total_tps < self.TPS_THRESHOLDS['idle']:
            level_desc = "数据库当前处于空闲状态，事务活动极低"
        elif total_tps < self.TPS_THRESHOLDS['low']:
            level_desc = "数据库事务负载较轻，处于低负载运行状态"
        elif total_tps < self.TPS_THRESHOLDS['moderate']:
            level_desc = "数据库处于中等负载，事务吞吐量适中"
        elif total_tps < self.TPS_THRESHOLDS['high']:
            level_desc = "数据库处于高负载状态，事务吞吐量较大"
        else:
            level_desc = "数据库处于极高负载状态，事务吞吐量非常大，需关注资源瓶颈"

        # 检查回滚率
        rollback_warning = ""
        if total_tps > 0 and rollback_tps / total_tps > 0.05:
            rollback_warning = (
                f"\n**警告**: 事务回滚率偏高 ({rollback_tps/total_tps*100:.1f}%)，"
                "可能存在锁竞争或死锁问题，建议检查并发事务逻辑。"
            )

        return (
            f"### 事务性能\n"
            f"{level_desc}\n"
            f"- 总 TPS: {total_tps:.2f} 事务/秒 (提交: {commit_tps:.2f}, 回滚: {rollback_tps:.2f})\n"
            f"- QPS: {qps:.2f} 查询/秒"
            f"{rollback_warning}"
        )

    def _translate_cache_metrics(self, metrics: dict) -> str:
        """翻译缓存命中率指标"""
        cache = metrics.get('cache_hit_ratio', {})
        buffer_ratio = cache.get('buffer', 0)
        index_ratio = cache.get('index', 0)

        # 缓冲区命中率诊断
        if buffer_ratio < self.CACHE_THRESHOLDS['critical']:
            buf_desc = (
                f"**严重**: 缓冲区命中率仅为 {buffer_ratio:.1f}%，远低于 90% 的健康基线。"
                " 大量数据读取绕过内存缓存直接访问磁盘，造成严重的 I/O 等待。"
                " 强烈建议增大 shared_buffers 参数。"
            )
        elif buffer_ratio < self.CACHE_THRESHOLDS['warning']:
            buf_desc = (
                f"缓冲区命中率为 {buffer_ratio:.1f}%，略低于推荐值。"
                " 存在一定的缓存缺页现象，可适当增大 shared_buffers。"
            )
        elif buffer_ratio < self.CACHE_THRESHOLDS['good']:
            buf_desc = f"缓冲区命中率 {buffer_ratio:.1f}%，处于良好水平，缓存配置基本合理。"
        else:
            buf_desc = f"缓冲区命中率 {buffer_ratio:.1f}%，表现优秀，缓存配置充足。"

        return f"### 缓存命中率\n{buf_desc}\n- 索引命中率: {index_ratio:.1f}%"

    def _translate_io_metrics(self, metrics: dict) -> str:
        """翻译 I/O 指标"""
        io = metrics.get('io', {})
        blks_read = io.get('blks_read_per_sec', 0)
        blks_hit = io.get('blks_hit_per_sec', 0)
        rows_modified = io.get('rows_modified_per_sec', 0)

        # 判断 IO 模式
        total_io = blks_read + blks_hit
        if total_io > 0:
            physical_ratio = blks_read / total_io * 100
        else:
            physical_ratio = 0

        if physical_ratio > 30:
            io_desc = (
                f"物理磁盘读取占总 I/O 的 {physical_ratio:.1f}%，磁盘 I/O 压力较大。"
                " 当前工作集超出缓冲池容量，建议增大内存缓存或检查查询是否存在全表扫描。"
            )
        elif physical_ratio > 10:
            io_desc = f"物理磁盘读取占比 {physical_ratio:.1f}%，I/O 负载适中。"
        else:
            io_desc = f"物理磁盘读取占比仅 {physical_ratio:.1f}%，绝大部分请求由内存缓存满足，I/O 表现良好。"

        # 写入模式
        write_desc = ""
        if rows_modified > 100:
            write_desc = f"\n数据修改速率 {rows_modified:.1f} 行/秒，写入活动频繁，需关注 WAL 日志和 checkpoint 配置。"

        return f"### I/O 状态\n{io_desc}{write_desc}"

    def _translate_connection_metrics(self, metrics: dict) -> str:
        """翻译连接统计"""
        conn = metrics.get('connections', {})
        active = conn.get('active', 0)
        idle = conn.get('idle', 0)
        total = conn.get('total', 0)

        if total > 100:
            conn_desc = f"当前连接数 {total} (活跃 {active}, 空闲 {idle})，连接数偏多，需确保 max_connections 配置足够。"
        elif active > 50:
            conn_desc = f"活跃连接 {active} 个，并发较高，建议关注连接池效率和 work_mem 的总内存开销。"
        else:
            conn_desc = f"当前连接数 {total} (活跃 {active}, 空闲 {idle})，连接负载正常。"

        return f"### 连接状态\n{conn_desc}"

    def _translate_lock_metrics(self, metrics: dict) -> str:
        """翻译锁统计"""
        total_locks = metrics.get('total_locks', 0)
        lock_stats = metrics.get('locks', {})

        exclusive_locks = sum(
            v for k, v in lock_stats.items()
            if 'Exclusive' in k and 'Share' not in k
        )

        if exclusive_locks > 10:
            lock_desc = (
                f"当前存在 {exclusive_locks} 个排他锁，锁竞争较为激烈。"
                " 可能导致事务等待，建议检查长事务和高并发写入场景。"
            )
        elif total_locks > 50:
            lock_desc = f"总锁数 {total_locks}，锁活动较多但以共享锁为主，暂无严重竞争。"
        else:
            lock_desc = f"总锁数 {total_locks}，锁竞争程度低，并发控制正常。"

        return f"### 锁状态\n{lock_desc}"

    def _translate_slow_queries(self, metrics: dict) -> str:
        """翻译慢查询信息"""
        slow_queries = metrics.get('slow_queries', [])
        count = metrics.get('slow_query_count', 0)

        if count == 0:
            return "### 慢查询\n当前无显著慢查询，查询执行效率良好。"

        lines = [f"### 慢查询\n检测到 {count} 条慢查询:"]
        for i, sq in enumerate(slow_queries[:3], 1):
            avg_ms = sq.get('avg_exec_time_ms', 0)
            calls = sq.get('calls', 0)
            lines.append(
                f"  {i}. 平均耗时 {avg_ms:.1f}ms, 调用 {calls} 次 — "
                f"{'该查询严重影响性能' if avg_ms > 1000 else '建议关注优化'}"
            )

        if count > 0:
            lines.append(
                "\n建议: 对慢查询考虑增大 work_mem（排序/哈希操作）或启用并行查询。"
            )

        return "\n".join(lines)

    def translate_performance_delta(self, old_metrics: dict, new_metrics: dict,
                                     config_changes: dict) -> str:
        """
        翻译性能变化对比，用于自省反馈环节

        Args:
            old_metrics: 变更前的指标
            new_metrics: 变更后的指标
            config_changes: 本轮配置变更 {knob: {old_value, new_value}}

        Returns:
            性能变化的自然语言分析
        """
        lines = ["## 配置变更后性能对比分析", ""]

        # 变更内容
        lines.append("### 本轮配置变更")
        for knob, change in config_changes.items():
            lines.append(f"- {knob}: {change.get('old_value', '?')} -> {change.get('new_value', '?')}")
        lines.append("")

        # TPS 变化
        old_tps = old_metrics.get('tps', {}).get('total_tps', 0)
        new_tps = new_metrics.get('tps', {}).get('total_tps', 0)
        if old_tps > 0:
            tps_delta = (new_tps - old_tps) / old_tps * 100
            tps_dir = "提升" if tps_delta > 0 else "下降"
            lines.append(f"### TPS 变化: {old_tps:.2f} -> {new_tps:.2f} ({tps_dir} {abs(tps_delta):.1f}%)")
        else:
            tps_delta = 0
            lines.append(f"### TPS 变化: {old_tps:.2f} -> {new_tps:.2f}")

        # 缓存命中率变化
        old_cache = old_metrics.get('cache_hit_ratio', {}).get('buffer', 0)
        new_cache = new_metrics.get('cache_hit_ratio', {}).get('buffer', 0)
        cache_delta = new_cache - old_cache
        lines.append(f"### 缓存命中率变化: {old_cache:.1f}% -> {new_cache:.1f}% ({'改善' if cache_delta > 0 else '恶化'} {abs(cache_delta):.1f}%)")

        # IO 变化
        old_io = old_metrics.get('io', {}).get('blks_read_per_sec', 0)
        new_io = new_metrics.get('io', {}).get('blks_read_per_sec', 0)
        if old_io > 0:
            io_delta = (new_io - old_io) / old_io * 100
            io_dir = "增加" if io_delta > 0 else "减少"
            lines.append(f"### 物理 I/O 读取变化: {io_dir} {abs(io_delta):.1f}%")

        # 综合诊断
        lines.append("")
        lines.append("### 综合诊断")

        if tps_delta > 5:
            lines.append("- 正向信号: TPS 显著提升，当前调整方向正确。")
            if tps_delta > 20:
                lines.append("- 建议: 继续保持增长趋势，但减小探测步长以防止越过性能极点。")
            else:
                lines.append("- 建议: 可继续沿当前方向微调。")
        elif tps_delta > -2:
            lines.append("- 中性信号: TPS 变化不显著，可能需要调整其他维度的参数。")
        else:
            lines.append("- 负向信号: TPS 出现下降，本轮调整可能带来了负面效果。")
            lines.append("- 建议: 回退本轮变更，转而关注其他性能瓶颈维度。")

        if cache_delta < -2:
            lines.append("- 缓存命中率下降，当前内存分配策略可能不合理。")
        if new_io > old_io * 1.3 and old_io > 0:
            lines.append("- 物理 I/O 显著增加，磁盘子系统压力加大，建议关注 checkpoint 和 WAL 配置。")

        return "\n".join(lines)

    def generate_full_context(self, metrics: dict) -> str:
        """
        生成供 LLM 决策使用的完整上下文文本

        Args:
            metrics: 完整性能指标

        Returns:
            完整的环境+运行状态上下文
        """
        parts = [
            self.translate_static_features(),
            "",
            self.translate_dynamic_metrics(metrics),
        ]
        return "\n".join(parts)
