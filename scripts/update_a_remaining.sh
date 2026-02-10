#!/bin/bash
# 更新剩余的 a 开头参数 JSON

FILES=(
    "archive_mode"
    "autovacuum_analyze_scale_factor"
    "autovacuum_analyze_threshold"
    "autovacuum"
    "autovacuum_max_workers"
    "autovacuum_multixact_freeze_max_age"
    "autovacuum_vacuum_cost_limit"
    "autovacuum_vacuum_insert_scale_factor"
    "autovacuum_vacuum_insert_threshold"
    "autovacuum_vacuum_threshold"
    "autovacuum_work_mem"
)

export GPT_API_KEY="sk-8695e5513e7d451d9fd1dd8fe155a2da"
export GPT_API_BASE="https://dashscope.aliyuncs.com/compatible-mode/v1"
export GPT_MODEL_NAME="qwen-plus"
export OVERWRITE_EXISTING="1"

cd /root/GPTuner

for knob in "${FILES[@]}"; do
    echo "=== Processing: $knob ==="
    rm -f "knowledge_collection/postgres/structured_knowledge/normal/${knob}.json"
    export KNOB_PREFIX="${knob}"
    python scripts/build_structured_data.py 2>&1 | grep -E "(Found|PROCESS|OK|FAIL)"
    sleep 2
done

echo ""
echo "=== 完成！验证结果 ==="
for knob in "${FILES[@]}"; do
    file="knowledge_collection/postgres/structured_knowledge/normal/${knob}.json"
    if [ -f "$file" ]; then
        size=$(stat -c%s "$file")
        if [ $size -gt 500 ]; then
            echo "✓ $knob: $size bytes (完整格式)"
        else
            echo "✗ $knob: $size bytes (格式不完整)"
        fi
    else
        echo "✗ $knob: 文件不存在"
    fi
done
