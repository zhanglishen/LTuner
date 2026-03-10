#!/usr/bin/env python3
"""
真实对比实验: GPTuner (BO) vs LTuner (Self-Reflective)
使用 TPC-H benchmark 在真实 PostgreSQL 上运行

GPTuner: Coarse 15 轮 + Fine 20 轮 = 35 轮 BO
LTuner: 10 轮自省反馈

用法:
    cd /root/GPTuner
    python src/experiments/run_real_comparison.py
"""
import sys
import os
import json
import time
import traceback
from datetime import datetime

# 确保工作目录正确
os.chdir("/root/GPTuner")
sys.path.insert(0, '/root/GPTuner/src')

from configparser import ConfigParser
from dbms.postgres import PgDBMS
from config_recommender.coarse_stage import CoarseStage
from config_recommender.fine_stage import FineStage
from config_recommender.workload_runner import BenchbaseRunner
from ltuner.ltuner_orchestrator import LTunerOrchestrator
from experiments.visualizer import ExperimentVisualizer

# ============================================================
# 配置
# ============================================================
TEST = "tpch"
TIMEOUT = 180
SEED = 42
BO_COARSE_TRIALS = 30
BO_FINE_TRIALS = 40   # fine 阶段总试验数（含 coarse 导入）
LTUNER_MAX_ITER = 20
TARGET_KNOBS_PATH = "./knowledge_collection/postgres/target_knobs.txt"
OUTPUT_DIR = "./optimization_results/comparison_real"

API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY = "sk-8695e5513e7d451d9fd1dd8fe155a2da"
MODEL = "qwen-plus"

# SMAC3 会自动加 smac3_output/ 前缀到 name 参数
# 用 ../ 抵消该前缀，使最终路径落在 ./optimization_results/...
# SMAC 输出: smac3_output/../optimization_results/... → ./optimization_results/...
# fine_space.py 期望: ./optimization_results/postgres/coarse/{seed}/runhistory.json
SMAC_COARSE_NAME = "../optimization_results/postgres/coarse/"
SMAC_FINE_NAME = "../optimization_results/postgres/fine/"

# 实际结果路径（SMAC 解析后）
SMAC_COARSE_RESULT = "./optimization_results/postgres/coarse/"
SMAC_FINE_RESULT = "./optimization_results/postgres/fine/"


def make_dbms():
    """正确初始化 PgDBMS"""
    config = ConfigParser()
    config.read("./configs/postgres.ini")
    sec = config["DATABASE"]
    dbms = PgDBMS(
        db=sec["db"],
        user=sec["user"],
        password=sec["password"],
        restart_cmd=sec["restart_cmd"],
        recover_script=sec["recover_script"],
        knob_info_path=sec["knob_info_path"],
    )
    dbms._connect("benchbase")
    return dbms


def run_baseline():
    """运行默认配置基准测试（独立的，不依赖 SMAC 初始化）"""
    print(f"\n{'#'*60}")
    print("# Phase 0: 默认配置基准测试")
    print(f"{'#'*60}\n")

    import threading
    dbms = make_dbms()
    dbms.reset_config()
    dbms.reconfigure()

    runner = BenchbaseRunner(dbms=dbms, test=TEST, target_path="./optimization_results/temp_results")
    runner.clear_summary_dir()
    t = threading.Thread(target=runner.run_benchmark)
    t.start()
    t.join(timeout=TIMEOUT)
    if t.is_alive():
        print("Baseline benchmark timeout, terminating...")
        runner.process.terminate()
        time.sleep(2)
        dbms._disconnect()
        return 0.0, float('inf')

    throughput = runner.get_throughput()
    latency = runner.get_latency()
    dbms._disconnect()
    print(f"\n[Baseline] TPS={throughput:.2f}, Latency={latency:.2f}")
    return throughput, latency


def run_gptuner_bo():
    """运行 GPTuner BO（Coarse + Fine）"""
    print(f"\n{'#'*60}")
    print(f"# Phase 1: GPTuner BO (Coarse {BO_COARSE_TRIALS} + Fine {BO_FINE_TRIALS})")
    print(f"{'#'*60}\n")

    # 创建标准目录
    os.makedirs("./optimization_results/postgres/log", exist_ok=True)
    os.makedirs("./optimization_results/postgres/coarse", exist_ok=True)
    os.makedirs("./optimization_results/postgres/fine", exist_ok=True)

    # 清理旧的 SMAC 输出（避免 overwrite=False 导致继续旧实验）
    import shutil
    # 清理实际结果目录
    coarse_seed_dir = f"{SMAC_COARSE_RESULT}{SEED}"
    fine_seed_dir = f"{SMAC_FINE_RESULT}{SEED}"
    # 也清理 smac3_output 目录下的残留
    smac3_coarse = f"smac3_output/{SMAC_COARSE_RESULT}{SEED}" if not SMAC_COARSE_NAME.startswith("..") else None
    for d in [coarse_seed_dir, fine_seed_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"[清理] 删除旧 SMAC 输出: {d}")
    # 清理 smac3_output 中可能残留的旧路径
    old_smac_coarse = f"smac3_output/optimization_results/postgres/coarse/{SEED}"
    old_smac_fine = f"smac3_output/optimization_results/postgres/fine/{SEED}"
    for d in [old_smac_coarse, old_smac_fine]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"[清理] 删除旧 smac3_output 残留: {d}")

    start_time = time.time()
    gptuner_iterations = []

    # ─── Coarse Stage ───
    print("\n--- GPTuner Coarse Stage ---")
    dbms1 = make_dbms()
    try:
        coarse_stage = CoarseStage(
            dbms=dbms1,
            target_knobs_path=TARGET_KNOBS_PATH,
            test=TEST,
            timeout=TIMEOUT,
            seed=SEED,
        )
        coarse_stage.optimize(
            name=SMAC_COARSE_NAME,
            trials_number=BO_COARSE_TRIALS,
            initial_config_number=3
        )
    except Exception as e:
        print(f"[ERROR] Coarse stage failed: {e}")
        traceback.print_exc()
    finally:
        try:
            dbms1._disconnect()
        except:
            pass

    time.sleep(5)

    # ─── Fine Stage ───
    print("\n--- GPTuner Fine Stage ---")
    dbms2 = make_dbms()
    try:
        fine_stage = FineStage(
            dbms=dbms2,
            target_knobs_path=TARGET_KNOBS_PATH,
            test=TEST,
            timeout=TIMEOUT,
            seed=SEED,
        )
        fine_stage.optimize(
            name=SMAC_FINE_NAME,
            trials_number=BO_FINE_TRIALS,
        )
    except Exception as e:
        print(f"[ERROR] Fine stage failed: {e}")
        traceback.print_exc()
    finally:
        try:
            dbms2._disconnect()
        except:
            pass

    total_time = time.time() - start_time

    # 收集 BO 结果
    result = collect_bo_results(total_time)
    return result


def collect_bo_results(total_time):
    """从 SMAC runhistory + log 文件收集 BO 结果"""
    result = {
        'method': 'GPTuner (BO)',
        'test': TEST,
        'total_time_seconds': round(total_time, 1),
        'total_iterations': 0,
        'iteration_latency': [],
        'iteration_tps': [],
        'iteration_times': [],
        'best_latency': float('inf'),
        'best_tps': 0,
    }

    # 解析 SMAC runhistory（coarse + fine）
    for stage, path_prefix in [("coarse", SMAC_COARSE_RESULT), ("fine", SMAC_FINE_RESULT)]:
        rh_path = os.path.join(path_prefix, str(SEED), "runhistory.json")
        if not os.path.exists(rh_path):
            print(f"[WARN] {stage} runhistory 不存在: {rh_path}")
            continue

        with open(rh_path, 'r') as f:
            rh = json.load(f)

        data_entries = rh.get("data", [])
        print(f"[INFO] {stage} runhistory: {len(data_entries)} 条记录")

        for entry in data_entries:
            # SMAC3 format: [config_id, instance_id, seed, budget, cost, time, status, additional_info]
            if len(entry) >= 5:
                cost = entry[4]
                # TPC-H: cost = latency (越小越好)
                # TPC-C: cost = -throughput (SMAC 最小化)
                if TEST in ['tpch']:
                    latency = cost
                    result['iteration_latency'].append(latency)
                    if latency < result['best_latency']:
                        result['best_latency'] = latency
                else:
                    tps = -cost
                    result['iteration_tps'].append(tps)
                    if tps > result['best_tps']:
                        result['best_tps'] = tps

    result['total_iterations'] = max(
        len(result['iteration_latency']),
        len(result['iteration_tps'])
    )

    # 解析 log 文件获取时间信息
    log_file = f"./optimization_results/postgres/log/{SEED}_log.txt"
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        for line in lines[1:]:  # 跳过表头
            parts = line.strip().split('\t')
            if len(parts) >= 4:
                try:
                    elapsed = float(parts[3])
                    result['iteration_times'].append(round(elapsed, 1))
                except ValueError:
                    pass
        print(f"[INFO] 日志记录 {len(result['iteration_times'])} 轮时间数据")

    return result


def run_ltuner_real():
    """运行 LTuner 真实实验"""
    print(f"\n{'#'*60}")
    print(f"# Phase 2: LTuner Self-Reflective ({LTUNER_MAX_ITER} 轮)")
    print(f"{'#'*60}\n")

    ltuner_output = os.path.join(OUTPUT_DIR, "ltuner")
    os.makedirs(ltuner_output, exist_ok=True)

    dbms = make_dbms()

    orchestrator = LTunerOrchestrator(
        dbms=dbms,
        test=TEST,
        timeout=TIMEOUT,
        api_base=API_BASE,
        api_key=API_KEY,
        model=MODEL,
        max_iterations=LTUNER_MAX_ITER,
        convergence_threshold=0.005,
        top_k_knobs=15,
        output_dir=ltuner_output,
    )

    workflow_report = orchestrator.run()

    try:
        dbms._disconnect()
    except:
        pass

    return workflow_report


def build_comparison_data(baseline_tps, baseline_latency, bo_result, ltuner_report):
    """构建对比数据结构（兼容 visualizer）"""

    # ─── GPTuner 数据 ───
    gptuner_data = {
        'method': 'GPTuner (BO)',
        'test': TEST,
        'baseline_tps': baseline_tps,
        'baseline_latency': baseline_latency,
        'total_iterations': bo_result.get('total_iterations', 0),
        'total_time_seconds': bo_result.get('total_time_seconds', 0),
        'config_failures': 0,
        'iteration_tps': bo_result.get('iteration_tps', []),
        'iteration_latency': bo_result.get('iteration_latency', []),
        'best_tps': bo_result.get('best_tps', 0),
        'best_latency': bo_result.get('best_latency', float('inf')),
        'improvement_percent': 0,
    }
    # 计算 GPTuner 提升
    if TEST in ['tpch'] and gptuner_data['iteration_latency']:
        best_lat = min(gptuner_data['iteration_latency'])
        gptuner_data['best_latency'] = best_lat
        if baseline_latency > 0:
            gptuner_data['improvement_percent'] = round(
                (baseline_latency - best_lat) / baseline_latency * 100, 2
            )
    elif gptuner_data['iteration_tps']:
        best_tps = max(gptuner_data['iteration_tps'])
        gptuner_data['best_tps'] = best_tps
        if baseline_tps > 0:
            gptuner_data['improvement_percent'] = round(
                (best_tps - baseline_tps) / baseline_tps * 100, 2
            )

    # ─── LTuner 数据 ───
    ltuner_final = ltuner_report.get('final_result', {})
    ltuner_opt = ltuner_report.get('steps', {}).get('step4_optimize', {}).get('result', {})

    lt_iteration_tps = []
    lt_iteration_latency = []
    for rec in ltuner_opt.get('history', []):
        lt_iteration_tps.append(rec.get('throughput', 0))
        lt_iteration_latency.append(rec.get('latency', 0))

    ltuner_data = {
        'method': 'LTuner (Self-Reflective)',
        'test': TEST,
        'baseline_tps': baseline_tps,
        'baseline_latency': baseline_latency,
        'total_iterations': ltuner_final.get('total_iterations', 0),
        'total_time_seconds': ltuner_report.get('total_time_seconds', 0),
        'config_failures': ltuner_final.get('config_failures', 0),
        'iteration_tps': lt_iteration_tps,
        'iteration_latency': lt_iteration_latency,
        'best_tps': 0,
        'best_latency': float('inf'),
        'improvement_percent': 0,  # 统一从 baseline 计算，不用自报告值
    }
    # 统一计算 LTuner 提升（与 GPTuner 一致）
    if TEST in ['tpch'] and lt_iteration_latency:
        valid_lats = [x for x in lt_iteration_latency if x > 0 and x < float('inf')]
        if valid_lats:
            best_lat = min(valid_lats)
            ltuner_data['best_latency'] = best_lat
            if baseline_latency > 0:
                ltuner_data['improvement_percent'] = round(
                    (baseline_latency - best_lat) / baseline_latency * 100, 2
                )
    elif lt_iteration_tps:
        best_tps = max(lt_iteration_tps)
        ltuner_data['best_tps'] = best_tps
        if baseline_tps > 0:
            ltuner_data['improvement_percent'] = round(
                (best_tps - baseline_tps) / baseline_tps * 100, 2
            )

    # ─── 综合对比 ───
    gp_impr = gptuner_data['improvement_percent']
    lt_impr = ltuner_data['improvement_percent']

    comparison = {
        'experiment_time': datetime.now().isoformat(),
        'test': TEST,
        'mode': 'real',
        'gptuner': gptuner_data,
        'ltuner': ltuner_data,
        'comparison_summary': {
            'gptuner_improvement': gp_impr,
            'ltuner_improvement': lt_impr,
            'improvement_delta': round(lt_impr - gp_impr, 2),
            'gptuner_iterations': gptuner_data['total_iterations'],
            'ltuner_iterations': ltuner_data['total_iterations'],
            'iteration_reduction': round(
                (1 - ltuner_data['total_iterations'] / max(gptuner_data['total_iterations'], 1)) * 100, 1
            ),
            'gptuner_time_seconds': gptuner_data['total_time_seconds'],
            'ltuner_time_seconds': ltuner_data['total_time_seconds'],
            'time_reduction_percent': round(
                (1 - ltuner_data['total_time_seconds'] / max(gptuner_data['total_time_seconds'], 1)) * 100, 1
            ) if gptuner_data['total_time_seconds'] > 0 else 0,
            'gptuner_failures': gptuner_data['config_failures'],
            'ltuner_failures': ltuner_data['config_failures'],
            'winner': 'LTuner' if lt_impr >= gp_impr else 'GPTuner'
        }
    }
    return comparison


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("./optimization_results/temp_results", exist_ok=True)
    os.makedirs("./optimization_results/postgres/log", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"真实对比实验: GPTuner (BO) vs LTuner (Self-Reflective)")
    print(f"{'='*60}")
    print(f"  Benchmark: {TEST}")
    print(f"  GPTuner: Coarse {BO_COARSE_TRIALS} + Fine {BO_FINE_TRIALS} 轮")
    print(f"  LTuner: {LTUNER_MAX_ITER} 轮自省反馈")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # ── Phase 0: Baseline ──
    baseline_tps, baseline_latency = run_baseline()
    print(f"\n[Baseline] TPS={baseline_tps:.2f}, Latency={baseline_latency:.2f}")

    # ── Phase 1: GPTuner BO ──
    bo_result = run_gptuner_bo()

    # ── Phase 2: LTuner ──
    ltuner_report = run_ltuner_real()

    # ── Phase 3: 生成对比图表 ──
    print(f"\n{'#'*60}")
    print("# Phase 3: 生成对比图表")
    print(f"{'#'*60}\n")

    comparison = build_comparison_data(
        baseline_tps, baseline_latency,
        bo_result, ltuner_report
    )

    # 保存原始数据
    result_path = os.path.join(OUTPUT_DIR, 'comparison_real_tpch.json')
    with open(result_path, 'w') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    print(f"[保存] 对比数据: {result_path}")

    # 生成可视化
    try:
        viz = ExperimentVisualizer(output_dir=OUTPUT_DIR)
        charts = viz.visualize_all(comparison, prefix='real_tpch')
        print(f"[图表] 已生成 {len(charts)} 张图表到 {OUTPUT_DIR}")
    except Exception as e:
        print(f"[WARN] 图表生成失败: {e}")

    # 打印总结
    summary = comparison['comparison_summary']
    print(f"\n{'='*60}")
    print(f"  真实对比实验结果")
    print(f"{'='*60}")
    print(f"  基准 TPS: {baseline_tps:.2f}")
    print(f"  基准 Latency: {baseline_latency:.0f} μs")
    print(f"  ────────────────────────────────────")
    print(f"  GPTuner 提升: {summary['gptuner_improvement']:.1f}%")
    print(f"  LTuner 提升:  {summary['ltuner_improvement']:.1f}%")
    print(f"  ────────────────────────────────────")
    print(f"  GPTuner 迭代: {summary['gptuner_iterations']} 轮")
    print(f"  LTuner 迭代:  {summary['ltuner_iterations']} 轮")
    print(f"  ────────────────────────────────────")
    print(f"  GPTuner 耗时: {summary['gptuner_time_seconds']:.0f}s")
    print(f"  LTuner 耗时:  {summary['ltuner_time_seconds']:.0f}s")
    print(f"  ────────────────────────────────────")
    print(f"  优胜方: {summary['winner']}")
    print(f"  图表目录: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
