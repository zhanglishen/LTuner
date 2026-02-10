#!/usr/bin/env python3
"""构建结构化知识库：

从 knowledge_collection/postgres/tuning_lake 下的文本调优建议中，
通过大模型抽取结构化 JSON（LADO 所需的知识单元）。

使用方式（示例）：

  export GPT_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
  export GPT_API_KEY="<your_api_key>"
  cd /root/GPTuner
  python scripts/build_structured_data.py

注意：
- 已存在的 structured_knowledge/normal/*.json 会被保留（不会覆盖），
  方便你手工精修核心参数。
- 调用大模型可能产生费用，请先在少量参数上试跑验证效果。
"""

import os
import json
import glob
import sys
from pathlib import Path
from typing import Dict, Any

# 将项目根目录加入路径，便于导入 src 内模块
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.knowledge_handler.gpt import GPT


# 路径配置
BASE_DIR = PROJECT_ROOT
TUNING_LAKE_DIR = BASE_DIR / "knowledge_collection" / "postgres" / "tuning_lake"
OUTPUT_DIR = BASE_DIR / "knowledge_collection" / "postgres" / "structured_knowledge" / "normal"
EXISTING_INFO_PATH = BASE_DIR / "knowledge_collection" / "postgres" / "knob_info" / "system_view.json"


# 结构化抽取 Prompt（基于你给出的 schema 进行了精简整合）
EXTRACTION_PROMPT = """
You are a PostgreSQL Database Expert and a Data Engineer. Your task is to extract structured configuration knowledge from unstructured tuning advice and system documentation.

Task
-----
Read the provided text describing a PostgreSQL configuration parameter (knob), together with optional system metadata.
Extract key information and format it into a valid JSON object following the schema below.

Output JSON Schema
------------------
{
  "name": "Parameter name (string)",
  "category": "Category like Resource, Network, WAL, Planner, etc. (string)",
  "context": "One of: 'user', 'sighup', 'postmaster' (infer from text or system metadata; default to 'user')",
  "unit": "Unit like kB, MB, ms, s, count (string or null)",
  "description": "A concise, technical definition of what the knob does (string)",
  "symptoms": [
    "List of 3-5 specific performance issues or errors caused by misconfiguration.",
    "Use short English phrases like 'high IO wait', 'OOM error', 'slow sorting'."
  ],
  "dependencies": [
    "List of other parameters that are functionally related (list of strings)"
  ],
  "constraints": {
    "hard_rule": "A logical expression for safety (e.g., 'value < RAM * 0.5'), or null if unknown"
  },
  "tuning_tips": "A summary of how to tune this value (1-2 sentences)",
  "default_formula": "A rough mathematical formula for a starting value based on hardware (e.g., 'RAM * 0.25'), or null if purely empirical"
}

Rules
-----
- If the text or system metadata mentions 'restart' or context 'postmaster', set context to 'postmaster'.
- If the text or system metadata mentions 'reload' or context 'sighup', set context to 'sighup'.
- Otherwise, default context to 'user'.
- Infer symptoms: for example, if the text says 'increases memory usage', use symptoms like 'High memory usage' or 'OOM error'.
- Use short, precise technical English in description and symptoms.
- Do NOT output markdown code blocks. Output raw JSON only.
"""


def load_system_view() -> Dict[str, Any]:
    """加载 system_view.json 提供的元数据（上下文、单位、min/max 等）。"""
    if not EXISTING_INFO_PATH.exists():
        return {}
    with open(EXISTING_INFO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 直接使用原始结构：{ knob_name: { ... fields ... } }
    return data


def merge_with_system_view(knob_name: str, structured: Dict[str, Any], system_data: Dict[str, Any]) -> Dict[str, Any]:
    """将 system_view 中的信息合并到 LLM 抽取结果中。

    - 补全 category/context/unit
    - 将 min_val/max_val 写入 constraints.min / constraints.max（若存在）
    """
    sys_info = system_data.get(knob_name)
    if not sys_info:
        return structured

    # 确保有 constraints 字段
    constraints = structured.get("constraints") or {}

    # category
    if not structured.get("category") and sys_info.get("category"):
        structured["category"] = sys_info["category"]

    # context：system_view 中的 context 更权威
    if sys_info.get("context"):
        # PostgreSQL 中 context 可能为 'user', 'sighup', 'postmaster', 'superuser' 等
        ctx = sys_info["context"]
        if ctx in ("user", "sighup", "postmaster"):
            structured["context"] = ctx
        else:
            # superuser 等，仍然可以视为 user 级别
            structured.setdefault("context", "user")

    # unit
    if not structured.get("unit") and sys_info.get("unit"):
        structured["unit"] = sys_info["unit"]

    # min/max（如果存在）
    min_val = sys_info.get("min_val")
    max_val = sys_info.get("max_val")
    if min_val is not None:
        constraints["min"] = min_val
    if max_val is not None:
        constraints["max"] = max_val

    structured["constraints"] = constraints
    return structured


def build_structured_files():
    """遍历 tuning_lake，调用 LLM 将调优建议转换为结构化 JSON。"""
    # 1. 读取 API 配置
    api_base = os.environ.get("GPT_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    api_key = os.environ.get("GPT_API_KEY")
    model = os.environ.get("GPT_MODEL_NAME", "qwen-plus")

    if not api_key:
        raise RuntimeError("GPT_API_KEY 未设置，请在环境变量中配置后再运行。")

    # 2. 初始化 LLM（不启用 RAG，这里只做抽取）
    llm = GPT(api_base=api_base, api_key=api_key, model=model, use_rag=False, db="postgres")

    # 3. 加载 system_view 元数据
    system_data = load_system_view()

    # 4. 遍历 tuning_lake 文件（可选按前缀过滤）
    prefix = os.environ.get("KNOB_PREFIX")
    overwrite = os.environ.get("OVERWRITE_EXISTING") == "1"
    pattern = f"{prefix}*.txt" if prefix else "*.txt"
    txt_files = glob.glob(str(TUNING_LAKE_DIR / pattern))
    print(f"Found {len(txt_files)} tuning lake files (pattern={pattern}). Starting extraction...")

    for file_path in txt_files:
        knob_name = Path(file_path).stem
        output_path = OUTPUT_DIR / f"{knob_name}.json"

        # 已存在的 JSON：根据环境变量决定是否覆盖
        if output_path.exists() and not overwrite:
            print(f"[SKIP] {knob_name} (structured JSON already exists)")
            continue

        print(f"[PROCESS] {knob_name} ...")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # 从 system_view 获取元信息（如果有），作为上下文提供给 LLM
        sys_info = system_data.get(knob_name, {})

        # 构造完整 Prompt
        prompt = (
            f"{EXTRACTION_PROMPT}\n\n"
            f"Parameter name: {knob_name}\n\n"
            f"System metadata (if any):\n{json.dumps(sys_info, indent=2)}\n\n"
            f"Tuning advice text:\n{content}"
        )

        try:
            # 使用 JSON 格式输出
            response_obj = llm.get_GPT_response_json(prompt, json_format=True)
            if not isinstance(response_obj, dict):
                # 兼容性处理：如果底层返回的是字符串
                response_obj = json.loads(str(response_obj))

            structured = response_obj

            # 强制名称为当前 knob
            structured["name"] = knob_name

            # 与 system_view 元数据融合
            structured = merge_with_system_view(knob_name, structured, system_data)

            # 保存结果
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(structured, out_f, ensure_ascii=False, indent=2)

            print(f"[OK] Saved {output_path.relative_to(BASE_DIR)}")

        except Exception as e:
            print(f"[FAIL] {knob_name}: {e}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    build_structured_files()
