#!/usr/bin/env python3
"""
真实对比实验: GPTuner (BO) vs SMAC-only vs LTuner (Self-Reflective)
支持 TPC-H / TPC-C, 多 session 重复, 日志流式输出

用法:
    cd /root/GPTuner
    python src/experiments/run_real_comparison.py                      # 默认 tpch
    python src/experiments/run_real_comparison.py --test tpcc          # tpcc
    python src/experiments/run_real_comparison.py --sessions 3         # 3次重复
    python src/experiments/run_real_comparison.py --methods gptuner ltuner smac  # 指定方法
"""
import sys
import os
import json
import time
import shutil
import argparse
import traceback
import logging
from datetime import datetime

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
# 默认配置
# ============================================================
DEFAULT_CONFIG = {
    'test': 'tpch',
    'timeout': 180,
    'seed': 42,
    'bo_coarse_trials': 30,
    'bo_fine_trials': 40,
    'ltuner_max_iter': 20,
    'smac_trials': 70,
    'sessions': 1,
    'methods': ['gptuner', 'ltuner'],
    'output_dir': './optimization_results/comparison_real',
    'target_knobs_path': './knowledge_collection/postgres/target_knobs.txt',
    'api_base': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    'api_key': 'sk-8695e5513e7d451d9fd1dd8fe155a2da',
    'model': 'qwen-plus',
}

SMAC_COARSE_NAME = "../optimization_results/postgres/coarse/"
SMAC_FINE_NAME = "../optimization_results/postgres/fine/"
SMAC_COARSE_RESULT = "./optimization_results/postgres/coarse/"
SMAC_FINE_RESULT = "./optimization_results/postgres/fine/"

# ============================================================
# 日志系统 - 支持 Web 实时读取
# ============================================================
_log_file = None
_progress_file = None

def setup_logging(output_dir):
    global _log_file, _progress_file
    os.makedirs(output_dir, exist_ok=True)
    _log_file = os.path.join(output_dir, 'experiment.log')
    _progress_file = os.path.join(output_dir, 'experiment_progress.json')
    # 清空旧日志
    with open(_log_file, 'w') as f:
        f.write(f"[{datetime.now().isoformat()}] Experiment started\n")
    _write_progress('idle', 0, 0, '')

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f"[{ts}] {msg}"
    print(line)
    if _log_file:
        with open(_log_file, 'a') as f:
            f.write(line + '\n')

def _write_progress(status, current, total, method, extra=None):
    """写入进度文件, Web 页面轮询读取"""
    if not _progress_file:
        return
    data = {
        'status': status,
        'method': method,
        'current_iter': current,
        'total_iter': total,
        'percent': round(current / max(total, 1) * 100, 1),
        'timestamp': datetime.now().isoformat(),
    }
    if extra:
        data.update(extra)
    try:
        with open(_progress_file, 'w') as f:
            json.dump(data, f, ensure_ascii=False)
    except:
        pass


# ============================================================
# 数据库初始化
# ============================================================
def make_dbms():
    config = ConfigParser()
    config.read("./configs/postgres.ini")
    sec = config["DATABASE"]
    dbms = PgDBMS(
        db=sec["db"], user=sec["user"], password=sec["password"],
        restart_cmd=sec["restart_cmd"], recover_script=sec["recover_script"],
        knob_info_path=sec["knob_info_path"],
    )
    dbms._connect("benchbase")
    return dbms


# ============================================================
# Phase 0: Baseline
# ============================================================
def run_baseline(cfg):
    log("Phase 0: Running default config baseline...")
    _write_progress('running', 0, 1, 'baseline')
    import threading
    dbms = make_dbms()
    dbms.reset_config()
    dbms.reconfigure()
    runner = BenchbaseRunner(dbms=dbms, test=cfg['test'],
                             target_path="./optimization_results/temp_results")
    runner.clear_summary_dir()
    t = threading.Thread(target=runner.run_benchmark)
    t.start()
    t.join(timeout=cfg['timeout'])
    if t.is_alive():
        log("Baseline benchmark timeout")
        runner.process.terminate()
        time.sleep(2)
        dbms._disconnect()
        return 0.0, float('inf')
    throughput = runner.get_throughput()
    latency = runner.get_latency()
    dbms._disconnect()
    log(f"Baseline: TPS={throughput:.2f}, Latency={latency:.2f}")
    _write_progress('running', 1, 1, 'baseline',
                    {'tps': throughput, 'latency': latency})
    return throughput, latency


# ============================================================
# Phase 1: GPTuner BO (Coarse + Fine)
# ============================================================
def run_gptuner_bo(cfg):
    seed = cfg['seed']
    coarse_trials = cfg['bo_coarse_trials']
    fine_trials = cfg['bo_fine_trials']
    total = coarse_trials + fine_trials
    log(f"GPTuner BO: Coarse {coarse_trials} + Fine {fine_trials} = {total} rounds")
    _write_progress('running', 0, total, 'GPTuner')

    os.makedirs("./optimization_results/postgres/log", exist_ok=True)
    os.makedirs("./optimization_results/postgres/coarse", exist_ok=True)
    os.makedirs("./optimization_results/postgres/fine", exist_ok=True)

    # Clean old SMAC outputs
    for d in [f"{SMAC_COARSE_RESULT}{seed}", f"{SMAC_FINE_RESULT}{seed}",
              f"smac3_output/optimization_results/postgres/coarse/{seed}",
              f"smac3_output/optimization_results/postgres/fine/{seed}"]:
        if os.path.exists(d):
            shutil.rmtree(d)

    start_time = time.time()

    # Coarse Stage
    log("GPTuner: Starting Coarse Stage...")
    dbms1 = make_dbms()
    try:
        cs = CoarseStage(dbms=dbms1, target_knobs_path=cfg['target_knobs_path'],
                         test=cfg['test'], timeout=cfg['timeout'], seed=seed)
        cs.optimize(name=SMAC_COARSE_NAME, trials_number=coarse_trials,
                    initial_config_number=3)
    except Exception as e:
        log(f"Coarse stage error: {e}")
        traceback.print_exc()
    finally:
        try: dbms1._disconnect()
        except: pass

    _write_progress('running', coarse_trials, total, 'GPTuner')
    time.sleep(3)

    # Fine Stage
    log("GPTuner: Starting Fine Stage...")
    dbms2 = make_dbms()
    try:
        fs = FineStage(dbms=dbms2, target_knobs_path=cfg['target_knobs_path'],
                       test=cfg['test'], timeout=cfg['timeout'], seed=seed)
        fs.optimize(name=SMAC_FINE_NAME, trials_number=fine_trials)
    except Exception as e:
        log(f"Fine stage error: {e}")
        traceback.print_exc()
    finally:
        try: dbms2._disconnect()
        except: pass

    elapsed = time.time() - start_time
    _write_progress('running', total, total, 'GPTuner')
    return collect_bo_results(elapsed, cfg)


def collect_bo_results(total_time, cfg):
    result = {
        'method': 'GPTuner (BO)', 'test': cfg['test'],
        'total_time_seconds': round(total_time, 1),
        'total_iterations': 0,
        'iteration_latency': [], 'iteration_tps': [],
        'best_latency': float('inf'), 'best_tps': 0,
    }
    for stage, prefix in [("coarse", SMAC_COARSE_RESULT), ("fine", SMAC_FINE_RESULT)]:
        rh_path = os.path.join(prefix, str(cfg['seed']), "runhistory.json")
        if not os.path.exists(rh_path):
            continue
        with open(rh_path, 'r') as f:
            rh = json.load(f)
        for entry in rh.get("data", []):
            if len(entry) >= 5:
                cost = entry[4]
                if cfg['test'] in ['tpch']:
                    result['iteration_latency'].append(cost)
                    result['best_latency'] = min(result['best_latency'], cost)
                else:
                    tps = -cost
                    result['iteration_tps'].append(tps)
                    result['best_tps'] = max(result['best_tps'], tps)
    result['total_iterations'] = max(len(result['iteration_latency']),
                                     len(result['iteration_tps']))
    log(f"GPTuner: {result['total_iterations']} iterations, time={total_time:.0f}s")
    return result


# ============================================================
# Phase 1b: SMAC-only (Pure BO without knowledge enhancement)
# ============================================================
def run_smac_only(cfg):
    """Run pure SMAC BO without GPTuner knowledge enhancement"""
    seed = cfg['seed']
    trials = cfg.get('smac_trials', 70)
    log(f"SMAC-only: {trials} rounds (pure BO, no knowledge)")
    _write_progress('running', 0, trials, 'SMAC-only')

    smac_result_dir = "./optimization_results/postgres/smac_only/"
    smac_name = "../optimization_results/postgres/smac_only/"
    os.makedirs(smac_result_dir, exist_ok=True)

    seed_dir = f"{smac_result_dir}{seed}"
    if os.path.exists(seed_dir):
        shutil.rmtree(seed_dir)
    old_smac = f"smac3_output/optimization_results/postgres/smac_only/{seed}"
    if os.path.exists(old_smac):
        shutil.rmtree(old_smac)

    start_time = time.time()
    dbms = make_dbms()
    try:
        # Use CoarseStage (which uses default space without fine-tuning)
        cs = CoarseStage(dbms=dbms, target_knobs_path=cfg['target_knobs_path'],
                         test=cfg['test'], timeout=cfg['timeout'], seed=seed)
        cs.optimize(name=smac_name, trials_number=trials, initial_config_number=5)
    except Exception as e:
        log(f"SMAC-only error: {e}")
        traceback.print_exc()
    finally:
        try: dbms._disconnect()
        except: pass

    elapsed = time.time() - start_time
    _write_progress('running', trials, trials, 'SMAC-only')

    # Collect results
    result = {
        'method': 'SMAC-only (Pure BO)', 'test': cfg['test'],
        'total_time_seconds': round(elapsed, 1), 'total_iterations': 0,
        'iteration_latency': [], 'iteration_tps': [],
        'best_latency': float('inf'), 'best_tps': 0,
    }
    rh_path = os.path.join(smac_result_dir, str(seed), "runhistory.json")
    if os.path.exists(rh_path):
        with open(rh_path, 'r') as f:
            rh = json.load(f)
        for entry in rh.get("data", []):
            if len(entry) >= 5:
                cost = entry[4]
                if cfg['test'] in ['tpch']:
                    result['iteration_latency'].append(cost)
                    result['best_latency'] = min(result['best_latency'], cost)
                else:
                    tps = -cost
                    result['iteration_tps'].append(tps)
                    result['best_tps'] = max(result['best_tps'], tps)
    result['total_iterations'] = max(len(result['iteration_latency']),
                                     len(result['iteration_tps']))
    log(f"SMAC-only: {result['total_iterations']} iterations, time={elapsed:.0f}s")
    return result


# ============================================================
# Phase 2: LTuner Self-Reflective
# ============================================================
def run_ltuner_real(cfg):
    log(f"LTuner: {cfg['ltuner_max_iter']} rounds self-reflective feedback")
    _write_progress('running', 0, cfg['ltuner_max_iter'], 'LTuner')

    ltuner_output = os.path.join(cfg['output_dir'], "ltuner")
    os.makedirs(ltuner_output, exist_ok=True)

    dbms = make_dbms()
    orchestrator = LTunerOrchestrator(
        dbms=dbms, test=cfg['test'], timeout=cfg['timeout'],
        api_base=cfg['api_base'], api_key=cfg['api_key'], model=cfg['model'],
        max_iterations=cfg['ltuner_max_iter'], convergence_threshold=0.005,
        top_k_knobs=15, output_dir=ltuner_output,
    )
    workflow_report = orchestrator.run()
    try: dbms._disconnect()
    except: pass

    _write_progress('running', cfg['ltuner_max_iter'], cfg['ltuner_max_iter'], 'LTuner')
    return workflow_report


# ============================================================
# Build comparison data (supports 2 or 3 methods)
# ============================================================
def _extract_method_data(method_name, cfg, baseline_tps, baseline_lat, raw_result):
    """Extract unified data structure from method result"""
    is_latency = cfg['test'] in ['tpch']

    if method_name == 'ltuner':
        final = raw_result.get('final_result', {})
        opt = raw_result.get('steps', {}).get('step4_optimize', {}).get('result', {})
        it_tps = [r.get('throughput', 0) for r in opt.get('history', [])]
        it_lat = [r.get('latency', 0) for r in opt.get('history', [])]
        data = {
            'method': 'LTuner (Self-Reflective)', 'test': cfg['test'],
            'baseline_tps': baseline_tps, 'baseline_latency': baseline_lat,
            'total_iterations': final.get('total_iterations', len(it_lat)),
            'total_time_seconds': raw_result.get('total_time_seconds', 0),
            'config_failures': final.get('config_failures', 0),
            'iteration_tps': it_tps, 'iteration_latency': it_lat,
            'best_tps': 0, 'best_latency': float('inf'), 'improvement_percent': 0,
        }
    else:
        data = {
            'method': raw_result.get('method', method_name),
            'test': cfg['test'],
            'baseline_tps': baseline_tps, 'baseline_latency': baseline_lat,
            'total_iterations': raw_result.get('total_iterations', 0),
            'total_time_seconds': raw_result.get('total_time_seconds', 0),
            'config_failures': 0,
            'iteration_tps': raw_result.get('iteration_tps', []),
            'iteration_latency': raw_result.get('iteration_latency', []),
            'best_tps': raw_result.get('best_tps', 0),
            'best_latency': raw_result.get('best_latency', float('inf')),
            'improvement_percent': 0,
        }

    # Calculate improvement
    if is_latency and data['iteration_latency']:
        valid = [x for x in data['iteration_latency'] if 0 < x < float('inf')]
        if valid:
            data['best_latency'] = min(valid)
            if baseline_lat > 0:
                data['improvement_percent'] = round(
                    (baseline_lat - data['best_latency']) / baseline_lat * 100, 2)
    elif data['iteration_tps']:
        best = max(data['iteration_tps'])
        data['best_tps'] = best
        if baseline_tps > 0:
            data['improvement_percent'] = round(
                (best - baseline_tps) / baseline_tps * 100, 2)
    return data


def build_comparison_data(cfg, baseline_tps, baseline_lat, results_dict):
    """Build comparison data supporting gptuner/smac/ltuner"""
    methods_data = {}
    for name, raw in results_dict.items():
        methods_data[name] = _extract_method_data(name, cfg, baseline_tps, baseline_lat, raw)

    # Find winner
    best_name, best_impr = '', -999
    for name, d in methods_data.items():
        if d['improvement_percent'] > best_impr:
            best_impr = d['improvement_percent']
            best_name = name

    comparison = {
        'experiment_time': datetime.now().isoformat(),
        'test': cfg['test'], 'mode': 'real',
        'methods': list(methods_data.keys()),
        'baseline_tps': baseline_tps,
        'baseline_latency': baseline_lat,
    }

    # Add each method's data
    for name, d in methods_data.items():
        comparison[name] = d

    # Backward compat: if gptuner+ltuner present, add legacy keys
    if 'gptuner' in methods_data and 'ltuner' in methods_data:
        gp = methods_data['gptuner']
        lt = methods_data['ltuner']
        comparison['comparison_summary'] = {
            'gptuner_improvement': gp['improvement_percent'],
            'ltuner_improvement': lt['improvement_percent'],
            'improvement_delta': round(lt['improvement_percent'] - gp['improvement_percent'], 2),
            'gptuner_iterations': gp['total_iterations'],
            'ltuner_iterations': lt['total_iterations'],
            'iteration_reduction': round(
                (1 - lt['total_iterations'] / max(gp['total_iterations'], 1)) * 100, 1),
            'gptuner_time_seconds': gp['total_time_seconds'],
            'ltuner_time_seconds': lt['total_time_seconds'],
            'time_reduction_percent': round(
                (1 - lt['total_time_seconds'] / max(gp['total_time_seconds'], 1)) * 100, 1
            ) if gp['total_time_seconds'] > 0 else 0,
            'gptuner_failures': gp['config_failures'],
            'ltuner_failures': lt['config_failures'],
            'winner': best_name,
        }

    # General summary for any method set
    comparison['summary'] = {
        'winner': best_name,
        'winner_improvement': best_impr,
        'methods_compared': len(methods_data),
    }
    for name, d in methods_data.items():
        comparison['summary'][f'{name}_improvement'] = d['improvement_percent']
        comparison['summary'][f'{name}_iterations'] = d['total_iterations']
        comparison['summary'][f'{name}_time'] = d['total_time_seconds']

    return comparison


# ============================================================
# Main experiment runner
# ============================================================
def run_experiment(cfg=None):
    """Main entry - can be called from CLI or Web"""
    if cfg is None:
        cfg = dict(DEFAULT_CONFIG)

    output_dir = cfg['output_dir']
    setup_logging(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("./optimization_results/temp_results", exist_ok=True)
    os.makedirs("./optimization_results/postgres/log", exist_ok=True)

    methods = cfg.get('methods', ['gptuner', 'ltuner'])
    sessions = cfg.get('sessions', 1)

    log(f"{'='*60}")
    log(f"Experiment: {' vs '.join(m.upper() for m in methods)}")
    log(f"Benchmark: {cfg['test'].upper()}, Sessions: {sessions}")
    log(f"{'='*60}")

    all_session_results = []

    for session_idx in range(sessions):
        if sessions > 1:
            log(f"\n{'#'*60}")
            log(f"# Session {session_idx + 1}/{sessions}")
            log(f"{'#'*60}")
            cfg['seed'] = 42 + session_idx  # different seed per session

        # Phase 0: Baseline
        baseline_tps, baseline_lat = run_baseline(cfg)

        # Run each method
        results_dict = {}
        for method in methods:
            log(f"\n--- Running {method.upper()} ---")
            _write_progress('running', 0, 1, method)

            if method == 'gptuner':
                results_dict['gptuner'] = run_gptuner_bo(cfg)
            elif method == 'smac':
                results_dict['smac'] = run_smac_only(cfg)
            elif method == 'ltuner':
                results_dict['ltuner'] = run_ltuner_real(cfg)
            else:
                log(f"Unknown method: {method}")

        comparison = build_comparison_data(cfg, baseline_tps, baseline_lat, results_dict)
        comparison['session'] = session_idx + 1
        all_session_results.append(comparison)

    # Save results
    if sessions == 1:
        final_result = all_session_results[0]
    else:
        final_result = _aggregate_sessions(all_session_results, cfg)

    result_path = os.path.join(output_dir,
                               f'comparison_real_{cfg["test"]}.json')
    with open(result_path, 'w') as f:
        json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)
    log(f"Results saved: {result_path}")

    # Generate charts
    try:
        viz = ExperimentVisualizer(output_dir=output_dir)
        charts = viz.visualize_all(final_result, prefix=f'real_{cfg["test"]}')
        log(f"Generated {len(charts)} charts")
    except Exception as e:
        log(f"Chart generation failed: {e}")
        traceback.print_exc()

    # Print summary
    _print_summary(final_result)
    _write_progress('completed', 1, 1, 'done')
    return final_result


def _aggregate_sessions(sessions_list, cfg):
    """Aggregate multiple sessions: compute median and quartiles"""
    import numpy as np
    methods = cfg.get('methods', ['gptuner', 'ltuner'])

    # Use first session as template
    agg = dict(sessions_list[0])
    agg['sessions_count'] = len(sessions_list)
    agg['all_sessions'] = sessions_list

    for method in methods:
        if method not in agg:
            continue
        improvements = []
        for s in sessions_list:
            if method in s:
                improvements.append(s[method].get('improvement_percent', 0))
        if improvements:
            agg[method]['improvement_median'] = round(float(np.median(improvements)), 2)
            agg[method]['improvement_q1'] = round(float(np.percentile(improvements, 25)), 2)
            agg[method]['improvement_q3'] = round(float(np.percentile(improvements, 75)), 2)
            agg[method]['improvement_all'] = improvements

    return agg


def _print_summary(result):
    test = result.get('test', 'unknown')
    log(f"\n{'='*60}")
    log(f"Experiment Results - {test.upper()}")
    log(f"{'='*60}")

    for method_name in result.get('methods', ['gptuner', 'ltuner']):
        d = result.get(method_name, {})
        if not d:
            continue
        impr = d.get('improvement_percent', 0)
        iters = d.get('total_iterations', 0)
        t = d.get('total_time_seconds', 0)
        log(f"  {d.get('method', method_name):30s} | {impr:6.1f}% | {iters:3d} iters | {t:.0f}s")

    winner = result.get('summary', {}).get('winner', '?')
    log(f"  Winner: {winner}")
    log(f"{'='*60}\n")


# ============================================================
# CLI entry
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='GPTuner vs LTuner Comparison Experiment')
    parser.add_argument('--test', default='tpch', choices=['tpch', 'tpcc'])
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--sessions', type=int, default=1)
    parser.add_argument('--methods', nargs='+', default=['gptuner', 'ltuner'],
                        choices=['gptuner', 'ltuner', 'smac'])
    parser.add_argument('--bo-coarse', type=int, default=30)
    parser.add_argument('--bo-fine', type=int, default=40)
    parser.add_argument('--ltuner-iter', type=int, default=20)
    parser.add_argument('--smac-trials', type=int, default=70)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        'test': args.test,
        'timeout': args.timeout,
        'sessions': args.sessions,
        'methods': args.methods,
        'bo_coarse_trials': args.bo_coarse,
        'bo_fine_trials': args.bo_fine,
        'ltuner_max_iter': args.ltuner_iter,
        'smac_trials': args.smac_trials,
        'seed': args.seed,
    })
    run_experiment(cfg)


if __name__ == '__main__':
    main()