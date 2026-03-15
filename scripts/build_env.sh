#!/bin/bash
#===============================================================================
# LTuner 项目完整构建脚本
# 用于在新环境快速搭建 LTuner 开发/运行平台
#
# 使用方法:
#   chmod +x build_env.sh
#   ./build_env.sh [postgres|mysql] [tpch|tpcc|...]
#
# 参数说明:
#   $1: 数据库类型 (默认 postgres)
#   $2: 基准测试类型 (默认 tpch)
#===============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

#-------------------------------------------------------------------------------
# 配置变量（根据实际情况修改）
#-------------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DB_TYPE="${1:-postgres}"
BENCHMARK_TYPE="${2:-tpch}"
PYTHON_VERSION="3.10"

# LLM API 配置（需要根据实际情况修改）
export QWEN_API_KEY="${QWEN_API_KEY:-your-api-key-here}"
export QWEN_BASE_URL="${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"

# PostgreSQL 配置
export PG_HOST="${PG_HOST:-localhost}"
export PG_PORT="${PG_PORT:-5432}"
export PG_USER="${PG_USER:-postgres}"
export DB_PASSWORD="${DB_PASSWORD:-postgres}"

#-------------------------------------------------------------------------------
# 辅助函数
#-------------------------------------------------------------------------------
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "$1 命令未找到，请先安装"
        exit 1
    fi
    log_success "$1 已安装: $(command -v $1)"
}

#-------------------------------------------------------------------------------
# Step 1: 系统依赖安装
#-------------------------------------------------------------------------------
install_system_deps() {
    log_info "========== Step 1: 安装系统依赖 =========="

    # 更新包列表
    sudo apt-get update

    # 安装基础工具
    sudo apt-get install -y \
        build-essential \
        git \
        curl \
        wget \
        vim \
        htop \
        net-tools \
        lsof \
        tree

    # 安装 Java (BenchBase 需要)
    log_info "安装 Java 21 JDK..."
    sudo apt-get install -y openjdk-21-jdk

    # 验证 Java
    java -version
    export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
    log_success "JAVA_HOME=$JAVA_HOME"

    # 安装 Python (如果系统中没有)
    if ! command -v python3 &> /dev/null; then
        log_info "安装 Python $PYTHON_VERSION..."
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:deadsnakes/ppa
        sudo apt-get update
        sudo apt-get install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev
    fi

    log_success "系统依赖安装完成"
}

#-------------------------------------------------------------------------------
# Step 2: Python 环境配置
#-------------------------------------------------------------------------------
setup_python_env() {
    log_info "========== Step 2: 配置 Python 环境 =========="

    # 创建虚拟环境
    cd "$PROJECT_ROOT"
    if [ ! -d "venv" ]; then
        log_info "创建 Python 虚拟环境..."
        python3 -m venv venv
    fi

    # 激活虚拟环境
    source venv/bin/activate

    # 升级 pip
    log_info "升级 pip..."
    pip install --upgrade pip -q

    # 安装项目依赖
    log_info "安装 Python 依赖..."
    pip install -r requirements.txt -q

    # 安装 RAG 模型（首次运行时会下载到 ~/.cache/huggingface/）
    # 设置离线模式，避免联网检查
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_OFFLINE=1

    log_success "Python 环境配置完成"
}

#-------------------------------------------------------------------------------
# Step 3: PostgreSQL 安装与配置
#-------------------------------------------------------------------------------
setup_postgres() {
    if [ "$DB_TYPE" != "postgres" ]; then
        log_warn "跳过 PostgreSQL 安装 (DB_TYPE=$DB_TYPE)"
        return
    fi

    log_info "========== Step 3: 配置 PostgreSQL =========="

    # 检查 PostgreSQL 是否已安装
    if command -v psql &> /dev/null; then
        log_info "PostgreSQL 已安装: $(psql --version)"
    else
        log_info "安装 PostgreSQL 14..."
        sudo apt-get install -y postgresql-14 postgresql-client-14
    fi

    # 启动 PostgreSQL 服务
    log_info "启动 PostgreSQL 服务..."
    sudo service postgresql start || sudo pg_ctlcluster 14 main start

    # 配置 PostgreSQL
    log_info "配置 PostgreSQL..."

    # 设置密码
    sudo -u postgres psql -c "ALTER USER postgres PASSWORD '$DB_PASSWORD';"

    # 允许远程连接（可选）
    PG_HBA_CONF="/etc/postgresql/14/main/pg_hba.conf"
    if [ -f "$PG_HBA_CONF" ]; then
        # 本地 trust，远程 md5
        sudo sed -i 's/local\s\+all\s\+all\s\+peer/local   all             all                                    trust/' $PG_HBA_CONF
    fi

    # 配置 postgresql.conf
    PG_CONF="/etc/postgresql/14/main/postgresql.conf"
    if [ -f "$PG_CONF" ]; then
        # 启用 ALTER SYSTEM
        sudo sed -i "s/#alter_system = 'off'/alter_system = 'always'/" $PG_CONF

        # 建议的参数（可选）
        log_info "建议手动调整以下参数以获得更好的性能:"
        echo "  shared_buffers = 2GB           # 建议 25% 物理内存"
        echo "  work_mem = 128MB"
        echo "  effective_cache_size = 4GB     # 建议 75% 物理内存"
        echo "  max_connections = 200"
    fi

    # 测试连接
    PGPASSWORD=$DB_PASSWORD psql -h localhost -U postgres -c "SELECT version();"
    log_success "PostgreSQL 配置完成"
}

#-------------------------------------------------------------------------------
# Step 4: MySQL 安装与配置 (可选)
#-------------------------------------------------------------------------------
setup_mysql() {
    if [ "$DB_TYPE" != "mysql" ]; then
        return
    fi

    log_info "========== Step 4: 配置 MySQL =========="

    # 安装 MySQL
    sudo apt-get install -y mysql-server mysql-client

    # 启动服务
    sudo service mysql start

    # 设置 root 密码
    sudo mysql -e "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '$DB_PASSWORD';"
    sudo mysql -e "FLUSH PRIVILEGES;"

    log_success "MySQL 配置完成"
}

#-------------------------------------------------------------------------------
# Step 5: 编译 BenchBase
#-------------------------------------------------------------------------------
setup_benchbase() {
    log_info "========== Step 5: 编译 BenchBase =========="

    cd "$PROJECT_ROOT/../benchbase"

    if [ -f "target/benchbase-${DB_TYPE}.tgz" ]; then
        log_info "BenchBase 已编译完成，跳过"
        return
    fi

    # 编译 BenchBase
    log_info "编译 BenchBase for $DB_TYPE (这可能需要 5-10 分钟)..."
    ./mvnw clean package -P $DB_TYPE -DskipTests -q

    # 解压
    if [ -f "target/benchbase-${DB_TYPE}.tgz" ]; then
        cd target
        tar xzf benchbase-${DB_TYPE}.tgz
        log_success "BenchBase 编译完成"
    else
        log_error "BenchBase 编译失败"
        exit 1
    fi
}

#-------------------------------------------------------------------------------
# Step 6: 初始化基准测试数据
#-------------------------------------------------------------------------------
init_benchmark_data() {
    log_info "========== Step 6: 初始化基准测试数据 =========="

    BENCHBASE_DIR="$PROJECT_ROOT/../benchbase/target/benchbase-${DB_TYPE}"

    if [ ! -d "$BENCHBASE_DIR" ]; then
        log_error "BenchBase 目录不存在: $BENCHBASE_DIR"
        return
    fi

    cd "$BENCHBASE_DIR"

    # 创建测试数据
    log_info "创建 $BENCHMARK_TYPE 基准测试数据 (SF=0.01)..."
    java -jar benchbase.jar \
        -b $BENCHMARK_TYPE \
        -c config/$DB_TYPE/sample_${BENCHMARK_TYPE}_config.xml \
        --create=true \
        --load=true \
        --clear=true \
        --execute=false

    log_success "基准测试数据初始化完成"
}

#-------------------------------------------------------------------------------
# Step 7: 创建必要的目录
#-------------------------------------------------------------------------------
setup_directories() {
    log_info "========== Step 7: 创建必要的目录 =========="

    cd "$PROJECT_ROOT"

    mkdir -p optimization_results/postgres/coarse
    mkdir -p optimization_results/postgres/fine
    mkdir -p optimization_results/postgres/ltuner
    mkdir -p optimization_results/postgres/tuner
    mkdir -p optimization_results/temp_results
    mkdir -p optimization_results/comparison
    mkdir -p optimization_results/comparison_real

    log_success "目录创建完成"
}

#-------------------------------------------------------------------------------
# Step 8: 配置环境变量
#-------------------------------------------------------------------------------
setup_env_file() {
    log_info "========== Step 8: 配置环境变量 =========="

    cd "$PROJECT_ROOT"

    # 创建 .env 文件（如果不存在）
    if [ ! -f ".env" ]; then
        cat > .env << 'EOF'
#===============================================================================
# LTuner 环境配置文件
# 请根据实际情况修改以下配置
#===============================================================================

# LLM API 配置 (通义千问)
QWEN_API_KEY=your-api-key-here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1

# 数据库配置
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
DB_PASSWORD=postgres
DB_TYPE=postgres

# 项目路径
PROJECT_ROOT=/root/GPTuner
BENCHBASE_ROOT=/root/benchbase
EOF
        log_info "已创建 .env 文件，请编辑配置"
    else
        log_info ".env 文件已存在"
    fi

    # 导出环境变量
    export $(cat .env | grep -v '^#' | xargs)
    log_success "环境变量配置完成"
}

#-------------------------------------------------------------------------------
# Step 9: 验证安装
#-------------------------------------------------------------------------------
verify_installation() {
    log_info "========== Step 9: 验证安装 =========="

    cd "$PROJECT_ROOT"
    source venv/bin/activate

    # 验证 Python 依赖
    log_info "验证 Python 包..."
    python -c "import psycopg2; import streamlit; import faiss; print('Python 依赖 OK')"

    # 验证数据库连接
    log_info "验证数据库连接..."
    if [ "$DB_TYPE" = "postgres" ]; then
        PGPASSWORD=$DB_PASSWORD psql -h $PG_HOST -U $PG_USER -c "SELECT 1;" -q && log_success "PostgreSQL 连接 OK"
    fi

    # 验证 BenchBase
    log_info "验证 BenchBase..."
    if [ -f "$PROJECT_ROOT/../benchbase/target/benchbase-${DB_TYPE}/benchbase.jar" ]; then
        log_success "BenchBase OK"
    else
        log_warn "BenchBase 未找到"
    fi

    log_success "安装验证完成"
}

#-------------------------------------------------------------------------------
# 主函数
#-------------------------------------------------------------------------------
main() {
    log_info "============================================"
    log_info "   LTuner 项目构建脚本"
    log_info "   数据库: $DB_TYPE"
    log_info "   基准测试: $BENCHMARK_TYPE"
    log_info "============================================"

    # 检查是否为 root 用户
    if [ "$EUID" -ne 0 ]; then
        log_warn "建议使用 root 用户运行，以安装系统依赖"
    fi

    # 1. 安装系统依赖
    install_system_deps

    # 2. 配置 Python 环境
    setup_python_env

    # 3. 配置数据库
    if [ "$DB_TYPE" = "postgres" ]; then
        setup_postgres
    elif [ "$DB_TYPE" = "mysql" ]; then
        setup_mysql
    fi

    # 4. 编译 BenchBase (可选，耗时较长)
    if [ "$BENCHBASE_SKIP" != "true" ]; then
        setup_benchbase
        init_benchmark_data
    else
        log_warn "跳过 BenchBase 编译"
    fi

    # 5. 创建目录
    setup_directories

    # 6. 配置环境变量
    setup_env_file

    # 7. 验证安装
    verify_installation

    log_success "============================================"
    log_success "   构建完成!"
    log_success "============================================"
    log_info ""
    log_info "后续步骤:"
    log_info "  1. 编辑 .env 文件，配置 API Key"
    log_info "  2. 启动 Web 界面: streamlit run src/web/app.py"
    log_info "  3. 访问 http://localhost:8501"
    log_info ""
}

#-------------------------------------------------------------------------------
# 使用说明
#-------------------------------------------------------------------------------
usage() {
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  postgres      使用 PostgreSQL (默认)"
    echo "  mysql         使用 MySQL"
    echo "  tpch          使用 TPC-H 基准测试 (默认)"
    echo "  tpcc          使用 TPC-C 基准测试"
    echo "  --skip-bench  跳过 BenchBase 编译"
    echo "  -h, --help    显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0                    # 默认配置"
    echo "  $0 postgres tpch      # PostgreSQL + TPC-H"
    echo "  $0 mysql tpcc         # MySQL + TPC-C"
    echo "  $0 --skip-bench       # 跳过 BenchBase 编译"
}

#-------------------------------------------------------------------------------
# 解析参数
#-------------------------------------------------------------------------------
BENCHBASE_SKIP="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        postgres|mysql)
            DB_TYPE="$1"
            ;;
        tpch|tpcc|seats|auctionmark)
            BENCHMARK_TYPE="$1"
            ;;
        --skip-bench)
            BENCHBASE_SKIP="true"
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

main
