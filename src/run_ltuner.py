#!/usr/bin/env python3
"""
LTuner 主入口
基于 LLM 自省式反馈的数据库参数自动化调优系统
仅支持 PostgreSQL

用法:
    python run_ltuner.py postgres tpcc 300
    python run_ltuner.py postgres tpch 600 -max_iter 20 -top_k 25
"""
from configparser import ConfigParser
import argparse
import time
import os
import sys

from dbms.postgres import PgDBMS
from ltuner.ltuner_orchestrator import LTunerOrchestrator


def main():
    parser = argparse.ArgumentParser(
        description="LTuner - 基于 LLM 自省式反馈的数据库参数调优系统"
    )
    parser.add_argument("db", type=str, choices=["postgres"],
                        help="数据库类型（仅支持 postgres）")
    parser.add_argument("test", type=str,
                        help="BenchBase 基准测试名 (tpcc/tpch/twitter 等)")
    parser.add_argument("timeout", type=int,
                        help="单次 benchmark 超时秒数")
    parser.add_argument("-seed", type=int, default=1,
                        help="随机种子（默认 1）")
    parser.add_argument("-max_iter", type=int, default=15,
                        help="最大迭代次数（默认 15）")
    parser.add_argument("-top_k", type=int, default=20,
                        help="MoE 筛选 Top-K 参数数（默认 20）")
    parser.add_argument("-threshold", type=float, default=0.02,
                        help="收敛阈值百分比（默认 0.02，即 2%%）")
    parser.add_argument("-scenario", type=str, default="auto",
                        choices=["auto", "OLTP", "OLAP", "HYBRID"],
                        help="场景类型（默认 auto 自动检测）")

    args = parser.parse_args()
    print(f"\n{'='*60}")
    print(f"LTuner - 基于 LLM 自省式反馈的数据库参数调优系统")
    print(f"{'='*60}")
    print(f"输入参数: {args}")
    time.sleep(1)

    # 加载数据库配置
    if args.db == 'postgres':
        config_path = "./configs/postgres.ini"
        config = ConfigParser()
        config.read(config_path)
        dbms = PgDBMS.from_file(config)
    else:
        raise ValueError("LTuner 仅支持 PostgreSQL!")

    # 连接数据库
    dbms._connect("benchbase")

    # LLM API 配置（阿里云 DashScope）
    api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key = "sk-8695e5513e7d451d9fd1dd8fe155a2da"
    model = "qwen-plus"

    # 结果输出目录
    output_dir = f"../optimization_results/{args.db}/ltuner/"
    os.makedirs(output_dir, exist_ok=True)

    # 场景检测
    scenario = args.scenario if args.scenario != "auto" else "HYBRID"

    # 创建 LTuner 编排器
    orchestrator = LTunerOrchestrator(
        dbms=dbms,
        test=args.test,
        timeout=args.timeout,
        api_base=api_base,
        api_key=api_key,
        model=model,
        max_iterations=args.max_iter,
        convergence_threshold=args.threshold,
        top_k_knobs=args.top_k,
        scenario=scenario,
        output_dir=output_dir
    )

    # 执行调优
    workflow_report = orchestrator.run()

    # 输出最终状态
    status = workflow_report.get('status', 'unknown')
    if status == 'completed':
        final = workflow_report.get('final_result', {})
        print(f"\n{'='*60}")
        print(f"LTuner 调优成功完成!")
        print(f"  性能提升: {final.get('improvement_percent', 0):.1f}%")
        print(f"  调优轮次: {final.get('total_iterations', 0)}")
        print(f"  结果目录: {output_dir}")
        print(f"{'='*60}\n")
    else:
        print(f"\n[WARNING] LTuner 调优未正常完成，状态: {status}")
        if 'error' in workflow_report:
            print(f"  错误: {workflow_report['error']}")


if __name__ == '__main__':
    main()
