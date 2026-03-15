"""
对比实验页面 - 配置、执行、查看实验结果
"""
import streamlit as st
import sys
import os
import json
import subprocess
import signal
import time
import glob
from datetime import datetime

sys.path.insert(0, '/root/GPTuner/src')

RESULTS_DIR = '/root/GPTuner/optimization_results/comparison_real'
LOG_FILE = os.path.join(RESULTS_DIR, 'experiment.log')
PROGRESS_FILE = os.path.join(RESULTS_DIR, 'experiment_progress.json')


def show():
    st.title("🧪 对比实验")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["⚙️ 配置与执行", "📊 实验结果", "📁 历史实验"])

    with tab1:
        _show_config_and_run()
    with tab2:
        _show_results()
    with tab3:
        _show_history()


# ================================================================
# Tab 1: 配置与执行
# ================================================================
def _show_config_and_run():
    st.markdown("### 实验参数配置")

    col1, col2 = st.columns(2)
    with col1:
        test = st.selectbox("Benchmark", ["tpch", "tpcc"],
                            help="TPC-H (OLAP, 延迟) / TPC-C (OLTP, 吞吐量)")
        methods = st.multiselect(
            "对比方法", ["gptuner", "ltuner", "smac"],
            default=["gptuner", "ltuner"],
            help="GPTuner(BO+知识) / LTuner(自省反馈) / SMAC-only(纯BO基线)")
        sessions = st.selectbox("重复实验次数", [1, 3], index=0,
                                help="3次取中位数+四分位范围，论文标准做法")
        timeout = st.slider("每轮压测超时 (秒)", 60, 300, 180, step=30)

    with col2:
        bo_coarse = st.slider("GPTuner Coarse 轮次", 10, 50, 30)
        bo_fine = st.slider("GPTuner Fine 轮次", 20, 60, 40)
        ltuner_iter = st.slider("LTuner 自省迭代轮次", 5, 30, 20)
        smac_trials = st.slider("SMAC-only 总轮次", 30, 100, 70)
        seed = st.number_input("随机种子", value=42, min_value=0, max_value=9999)

    st.markdown("---")

    # 实验预估
    total_per_session = 0
    if 'gptuner' in methods:
        total_per_session += (bo_coarse + bo_fine) * timeout
    if 'smac' in methods:
        total_per_session += smac_trials * timeout
    if 'ltuner' in methods:
        total_per_session += ltuner_iter * timeout
    total_est_min = total_per_session * sessions / 60
    st.info(f"📏 **预估总耗时**: ~{total_est_min:.0f} 分钟 "
            f"({sessions} session × {len(methods)} 方法)")

    st.markdown("---")

    # 执行控制
    col_start, col_stop, col_status = st.columns([1, 1, 2])

    running = st.session_state.get('experiment_running', False)
    pid = st.session_state.get('experiment_pid', None)

    with col_start:
        if st.button("🚀 开始实验", type="primary", use_container_width=True,
                     disabled=running):
            cfg = {
                'test': test, 'timeout': timeout, 'seed': seed,
                'bo_coarse_trials': bo_coarse, 'bo_fine_trials': bo_fine,
                'ltuner_max_iter': ltuner_iter, 'smac_trials': smac_trials,
                'sessions': sessions, 'methods': methods,
            }
            _start_experiment(cfg)
            st.rerun()

    with col_stop:
        if st.button("🛑 停止实验", use_container_width=True, disabled=not running):
            _stop_experiment()
            st.rerun()

    with col_status:
        if running:
            st.warning(f"🔄 **实验运行中** (PID: {pid})")
        else:
            # Check if process actually finished
            if pid and not _is_process_alive(pid):
                st.session_state['experiment_running'] = False
                st.session_state['experiment_pid'] = None
                st.success("✅ 实验已完成")
            else:
                st.info("⏳ 等待开始")

    # 实时进度
    if running or (pid and _is_process_alive(pid)):
        _show_live_progress()

    # 实时日志
    if running or os.path.exists(LOG_FILE):
        _show_live_log()


def _start_experiment(cfg):
    """后台启动实验子进程"""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Write config to temp file for subprocess to read
    cfg_path = os.path.join(RESULTS_DIR, '_run_config.json')
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)

    # Build command
    cmd = [
        sys.executable,
        '/root/GPTuner/src/experiments/run_real_comparison.py',
        '--test', cfg['test'],
        '--timeout', str(cfg['timeout']),
        '--sessions', str(cfg['sessions']),
        '--methods'] + cfg['methods'] + [
        '--bo-coarse', str(cfg['bo_coarse_trials']),
        '--bo-fine', str(cfg['bo_fine_trials']),
        '--ltuner-iter', str(cfg['ltuner_max_iter']),
        '--smac-trials', str(cfg['smac_trials']),
        '--seed', str(cfg['seed']),
    ]

    log_path = os.path.join(RESULTS_DIR, 'experiment_stdout.log')
    with open(log_path, 'w') as log_f:
        proc = subprocess.Popen(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd='/root/GPTuner', start_new_session=True)

    st.session_state['experiment_running'] = True
    st.session_state['experiment_pid'] = proc.pid
    st.session_state['experiment_start_time'] = datetime.now().isoformat()


def _stop_experiment():
    """终止实验进程"""
    pid = st.session_state.get('experiment_pid')
    if pid:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    st.session_state['experiment_running'] = False
    st.session_state['experiment_pid'] = None


def _is_process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _show_live_progress():
    """显示实时进度条"""
    if not os.path.exists(PROGRESS_FILE):
        return
    try:
        with open(PROGRESS_FILE, 'r') as f:
            prog = json.load(f)
    except (json.JSONDecodeError, IOError):
        return

    status = prog.get('status', 'idle')
    method = prog.get('method', '')
    current = prog.get('current_iter', 0)
    total = prog.get('total_iter', 1)
    pct = prog.get('percent', 0)

    if status == 'completed':
        st.success(f"✅ 实验完成!")
        st.session_state['experiment_running'] = False
    elif status == 'running':
        st.progress(min(pct / 100.0, 1.0),
                    text=f"正在运行 {method}: {current}/{total} ({pct:.0f}%)")


def _show_live_log():
    """显示实时日志 (最新100行)"""
    with st.expander("📋 实验日志", expanded=False):
        log_path = LOG_FILE
        if not os.path.exists(log_path):
            log_path = os.path.join(RESULTS_DIR, 'experiment_stdout.log')
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    lines = f.readlines()
                tail = lines[-100:] if len(lines) > 100 else lines
                st.code(''.join(tail), language='text')
            except IOError:
                st.warning("无法读取日志文件")
        else:
            st.info("暂无日志")

        if st.session_state.get('experiment_running'):
            st.button("🔄 刷新日志", key="refresh_log",
                     on_click=lambda: None)


# ================================================================
# Tab 2: 实验结果查看
# ================================================================
def _show_results():
    st.markdown("### 📊 最新实验结果")

    # Find latest result JSON
    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'comparison_real_*.json')),
                        key=os.path.getmtime, reverse=True)
    if not json_files:
        st.info("暂无实验结果。请先在「配置与执行」页运行实验。")
        return

    selected = st.selectbox("选择实验结果", json_files,
                            format_func=lambda x: os.path.basename(x))

    try:
        with open(selected, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"读取结果文件失败: {e}")
        return

    # Key metrics cards
    methods = data.get('methods', ['gptuner', 'ltuner'])
    test = data.get('test', 'tpch')
    is_latency = test in ['tpch']

    st.markdown(f"**Benchmark**: {test.upper()} | **方法**: {', '.join(methods)}")
    st.markdown("---")

    # Metric cards
    cols = st.columns(len(methods) + 1)
    with cols[0]:
        baseline = data.get('baseline_latency', 0) if is_latency else data.get('baseline_tps', 0)
        metric_name = "基线延迟" if is_latency else "基线TPS"
        st.metric(metric_name, f"{baseline:.0f}")

    for i, m in enumerate(methods):
        md = data.get(m, {})
        impr = md.get('improvement_percent', 0)
        iters = md.get('total_iterations', 0)
        t = md.get('total_time_seconds', 0) / 60
        with cols[i + 1]:
            label = m.upper()
            st.metric(f"{label} 提升", f"{impr:.1f}%",
                     delta=f"{iters}轮 / {t:.0f}min")

    st.markdown("---")

    # Display charts
    st.markdown("### 📈 对比图表")
    prefix = os.path.basename(selected).replace('comparison_real_', '').replace('.json', '')
    chart_prefix = f'real_{prefix}'

    chart_names = [
        ('convergence', '收敛曲线'),
        ('performance', '性能对比'),
        ('time', '耗时对比'),
        ('safety', '综合对比表'),
        ('dashboard', '仪表盘'),
        ('multi_session', '多Session误差带'),
    ]

    for suffix, title in chart_names:
        img_path = os.path.join(RESULTS_DIR, f'{chart_prefix}_{suffix}.png')
        if os.path.exists(img_path):
            st.markdown(f"#### {title}")
            st.image(img_path, use_container_width=True)

    # JSON details
    with st.expander("📄 JSON 数据详情"):
        st.json(data)


# ================================================================
# Tab 3: 历史实验
# ================================================================
def _show_history():
    st.markdown("### 📁 历史实验列表")

    json_files = sorted(glob.glob(os.path.join(RESULTS_DIR, 'comparison_real_*.json')),
                        key=os.path.getmtime, reverse=True)

    if not json_files:
        st.info("暂无历史实验记录。")
        return

    for fp in json_files:
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
        except:
            continue

        fname = os.path.basename(fp)
        exp_time = data.get('experiment_time', '未知时间')
        test = data.get('test', '?').upper()
        methods = data.get('methods', [])
        winner = data.get('summary', {}).get('winner', '?')
        winner_impr = data.get('summary', {}).get('winner_improvement', 0)
        sessions = data.get('sessions_count', 1)

        with st.expander(f"📋 {fname} ({exp_time[:19]})"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Benchmark", test)
            with col2:
                st.metric("方法数", len(methods))
            with col3:
                st.metric("Sessions", sessions)
            with col4:
                st.metric("Winner", f"{winner} ({winner_impr:.1f}%)")

            # Per-method details
            for m in methods:
                md = data.get(m, {})
                if md:
                    impr = md.get('improvement_percent', 0)
                    iters = md.get('total_iterations', 0)
                    t = md.get('total_time_seconds', 0) / 60
                    st.write(f"**{m.upper()}**: 提升 {impr:.1f}%, "
                             f"{iters} 轮, {t:.0f} 分钟")
