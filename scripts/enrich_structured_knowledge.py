#!/usr/bin/env python3
"""补充结构化知识中 GPTuner 所需的字段。

使用项目已有的 KGTrans 模块（knowledge_transformation.py），
通过 LLM 从 tuning_lake 文本中提取 suggested_values / min_value / max_value，
并将这些字段合并到现有的 structured_knowledge/normal/*.json 中，
同时保留已有的 LADO 格式字段（description, symptoms, dependencies 等）。

使用方式：
    cd /root/GPTuner
    python scripts/enrich_structured_knowledge.py

环境变量（可选）：
    VOTE_ROUNDS   - 每个 knob 投票轮次（默认 3）
    OVERWRITE     - 是否覆盖已有 suggested_values（默认 0）
    KNOB_FILTER   - 仅处理指定 knob（逗号分隔，如 "shared_buffers,work_mem"）
"""

import os
import sys
import json
import time
from pathlib import Path
from collections import Counter

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from knowledge_handler.knowledge_transformation import KGTrans


# ── 配置 ──────────────────────────────────────────────
API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY  = "sk-8695e5513e7d451d9fd1dd8fe155a2da"
MODEL    = "qwen-plus"

TARGET_KNOBS_PATH = PROJECT_ROOT / "knowledge_collection" / "postgres" / "target_knobs.txt"
NORMAL_DIR = PROJECT_ROOT / "knowledge_collection" / "postgres" / "structured_knowledge" / "normal"
SPECIAL_DIR = PROJECT_ROOT / "knowledge_collection" / "postgres" / "structured_knowledge" / "special"

VOTE_ROUNDS = int(os.environ.get("VOTE_ROUNDS", "3"))
OVERWRITE   = os.environ.get("OVERWRITE", "0") == "1"
KNOB_FILTER = os.environ.get("KNOB_FILTER", "")


def load_target_knobs():
    """加载目标 knob 列表。"""
    with open(TARGET_KNOBS_PATH, "r") as f:
        knobs = [line.strip() for line in f if line.strip()]
    if KNOB_FILTER:
        allowed = set(k.strip() for k in KNOB_FILTER.split(","))
        knobs = [k for k in knobs if k in allowed]
    return knobs


def extract_gptuner_fields(kgtrans: KGTrans, knob: str, rounds: int = 3):
    """对一个 knob 调用 KGTrans.get_skill() 多轮投票，返回 suggested_values/min_value/max_value。"""
    min_list = []
    max_list = []
    suggested_list = []
    last_hw = {}

    for i in range(rounds):
        print(f"  [VOTE {i+1}/{rounds}] {knob}")
        try:
            result = kgtrans.get_skill(knob)
        except Exception as e:
            print(f"  [WARN] round {i+1} failed: {e}")
            continue

        sv = result.get("suggested_values", [])
        mn = result.get("min_value")
        mx = result.get("max_value")
        min_list.append(mn)
        max_list.append(mx)
        suggested_list.extend(sv)
        last_hw = {
            "cpu": result.get("cpu"),
            "ram": result.get("ram"),
            "disk_size": result.get("disk_size"),
            "disk_type": result.get("disk_type"),
        }

    # 投票逻辑（与原始 KGTrans.vote 一致）
    gptuner_fields = {}

    # min_value: 多数投票
    min_counts = Counter(str(v) for v in min_list)
    sorted_min = sorted(min_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_min and sorted_min[0][0] != "None":
        gptuner_fields["min_value"] = _try_parse(sorted_min[0][0])
    else:
        gptuner_fields["min_value"] = None

    # max_value: 多数投票
    max_counts = Counter(str(v) for v in max_list)
    sorted_max = sorted(max_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_max and sorted_max[0][0] != "None":
        gptuner_fields["max_value"] = _try_parse(sorted_max[0][0])
    else:
        gptuner_fields["max_value"] = None

    # suggested_values: 选取出现频率最高的一组
    if suggested_list:
        sv_counts = Counter(str(v) for v in suggested_list)
        sorted_sv = sorted(sv_counts.items(), key=lambda x: x[1], reverse=True)
        top_count = sorted_sv[0][1]
        gptuner_fields["suggested_values"] = [
            _try_parse(item[0]) for item in sorted_sv if item[1] == top_count
        ]
    else:
        gptuner_fields["suggested_values"] = []

    # 硬件信息
    gptuner_fields.update(last_hw)

    return gptuner_fields


def _try_parse(value):
    """尝试将字符串解析为数值，失败则返回原字符串。"""
    if value is None or value == "None" or value == "null":
        return None
    try:
        if "." in str(value):
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def merge_into_json(knob: str, gptuner_fields: dict):
    """将 GPTuner 字段合并到已有的 structured_knowledge JSON 文件中。"""
    json_path = NORMAL_DIR / f"{knob}.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = {"name": knob}

    # 合并 GPTuner 字段
    existing["suggested_values"] = gptuner_fields["suggested_values"]
    existing["min_value"] = gptuner_fields["min_value"]
    existing["max_value"] = gptuner_fields["max_value"]

    # 可选：记录硬件快照
    hw_keys = ["cpu", "ram", "disk_size", "disk_type"]
    for k in hw_keys:
        if gptuner_fields.get(k) is not None:
            existing[k] = gptuner_fields[k]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print(f"  [MERGED] {json_path.relative_to(PROJECT_ROOT)}")


def main():
    print("=" * 60)
    print("GPTuner 结构化知识字段补充工具")
    print(f"API: {API_BASE} | Model: {MODEL}")
    print(f"投票轮次: {VOTE_ROUNDS} | 覆盖已有: {OVERWRITE}")
    print("=" * 60)

    knobs = load_target_knobs()
    print(f"\n待处理 knob 数量: {len(knobs)}")

    # 初始化 KGTrans（项目已有模块）
    kgtrans = KGTrans(api_base=API_BASE, api_key=API_KEY, db="postgres", model=MODEL)

    # 修补 tiktoken 不支持 qwen-plus 的问题
    original_calc_token = kgtrans.calc_token
    def safe_calc_token(in_text, out_text=""):
        try:
            return original_calc_token(in_text, out_text)
        except Exception:
            # qwen-plus 不在 tiktoken 模型列表中，粗略估算
            if isinstance(out_text, dict):
                out_text = json.dumps(out_text)
            return len(str(in_text) + str(out_text)) // 4
    kgtrans.calc_token = safe_calc_token

    success = 0
    skipped = 0
    failed = 0
    start_time = time.time()

    for idx, knob in enumerate(knobs, 1):
        print(f"\n[{idx}/{len(knobs)}] Processing: {knob}")

        # 检查是否已有 suggested_values
        json_path = NORMAL_DIR / f"{knob}.json"
        if json_path.exists() and not OVERWRITE:
            with open(json_path, "r") as f:
                data = json.load(f)
            if "suggested_values" in data:
                print(f"  [SKIP] 已有 suggested_values")
                skipped += 1
                continue

        try:
            # 使用 KGTrans 提取 GPTuner 字段
            gptuner_fields = extract_gptuner_fields(kgtrans, knob, rounds=VOTE_ROUNDS)

            # 合并到现有 JSON
            merge_into_json(knob, gptuner_fields)

            # 同时确保 special knob 分类也存在
            special_path = SPECIAL_DIR / f"{knob}.json"
            if not special_path.exists():
                try:
                    kgtrans.prepare_special_skill(knob)
                    print(f"  [SPECIAL] 已生成 special 分类")
                except Exception as e:
                    print(f"  [WARN] special 分类失败: {e}")

            success += 1

        except Exception as e:
            print(f"  [FAIL] {knob}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"完成！耗时: {elapsed:.1f}s")
    print(f"成功: {success} | 跳过: {skipped} | 失败: {failed}")
    print("=" * 60)


if __name__ == "__main__":
    main()
