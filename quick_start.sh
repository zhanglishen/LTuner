#!/bin/bash
#===============================================================================
# LTuner 快速启动脚本
# 用于快速启动 Web 界面
#
# 使用方法:
#   ./quick_start.sh          # 默认启动
#   ./quick_start.sh --help   # 查看帮助
#===============================================================================

set -e

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

# 解析参数
START_MODE="web"  # web, ltuner, gptuner
DB_TYPE="postgres"

while [[ $# -gt 0 ]]; do
    case $1 in
        --ltuner)
            START_MODE="ltuner"
            ;;
        --gptuner)
            START_MODE="gptuner"
            ;;
        --mysql)
            DB_TYPE="mysql"
            ;;
        --help|-h)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --ltuner      启动 LTuner 命令行调优"
            echo "  --gptuner     启动 GPTuner 命令行调优"
            echo "  --mysql       使用 MySQL (默认 PostgreSQL)"
            echo "  --help        显示帮助信息"
            echo ""
            echo "示例:"
            echo "  $0              # 启动 Web 界面"
            echo "  $0 --ltuner     # 命令行运行 LTuner"
            exit 0
            ;;
        *)
            ;;
    esac
    shift
done

cd "$PROJECT_ROOT"

# 检查 Python 虚拟环境
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}[WARN]${NC} 虚拟环境不存在，正在创建..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    echo -e "${GREEN}[OK]${NC} 虚拟环境创建完成"
fi

# 激活虚拟环境
source venv/bin/activate

# 设置离线模式（避免 HuggingFace 联网检查）
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# 加载环境变量（如果存在）
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 检查数据库服务
echo -e "${BLUE}[INFO]${NC} 检查数据库服务..."
if ! service postgresql status &> /dev/null; then
    echo -e "${YELLOW}[WARN]${NC} PostgreSQL 未启动，正在启动..."
    service postgresql start || sudo service postgresql start
fi

# 根据模式启动
case $START_MODE in
    web)
        echo -e "${GREEN}[INFO]${NC} 启动 Web 界面..."
        echo -e "${GREEN}[INFO]${NC} 访问地址: http://localhost:8501"
        streamlit run src/web/app.py --server.port=8501
        ;;
    ltuner)
        echo -e "${GREEN}[INFO]${NC} 启动 LTuner 命令行调优..."
        python src/run_ltuner.py --db $DB_TYPE --benchmark tpch
        ;;
    gptuner)
        echo -e "${GREEN}[INFO]${NC} 启动 GPTuner 命令行调优..."
        python src/run_gptuner.py --db $DB_TYPE --benchmark tpch
        ;;
esac
