#!/bin/bash
set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== WhisperX API Server 部署脚本 ===${NC}"

# 1. 检查 Root 权限
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}请使用 root 权限运行此脚本 (sudo ./deploy.sh)${NC}"
  exit 1
fi

# 获取当前脚本所在目录的上一级目录（假设脚本在 deploy_scripts/ 或根目录下）
# 确保我们在项目根目录
PROJECT_ROOT=$(pwd)
if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
    # 尝试上一级
    PROJECT_ROOT=$(dirname $(pwd))
    if [ ! -f "$PROJECT_ROOT/pyproject.toml" ]; then
        echo -e "${RED}错误：找不到 pyproject.toml，请在 WhisperX 项目根目录下运行此脚本。${NC}"
        exit 1
    fi
fi
echo -e "${GREEN}项目根目录: ${PROJECT_ROOT}${NC}"

# 2. 安装系统依赖
echo -e "${YELLOW}正在更新系统并安装依赖 (ffmpeg, supervisor, python3-venv)...${NC}"
apt-get update
apt-get install -y ffmpeg supervisor python3-pip python3-venv git

# 3. 询问配置
echo -e "${YELLOW}=== 配置设置 ===${NC}"

read -p "请输入 Hugging Face Token (用于 Diarization，必填，留空将跳过配置但功能会受限): " HF_TOKEN
read -p "请输入服务运行端口 [默认 8000]: " PORT
PORT=${PORT:-8000}

read -p "请输入默认 ASR 模型大小 (tiny, small, medium, large-v2, large-v3) [默认 large-v2]: " WHISPER_MODEL
WHISPER_MODEL=${WHISPER_MODEL:-large-v2}

read -p "是否使用 CUDA (y/n) [默认 y]: " USE_CUDA
USE_CUDA=${USE_CUDA:-y}

DEVICE="cpu"
COMPUTE_TYPE="int8"
if [[ "$USE_CUDA" == "y" || "$USE_CUDA" == "Y" ]]; then
    DEVICE="cuda"
    COMPUTE_TYPE="float16"
fi

# 4. 设置 Python 环境
echo -e "${YELLOW}正在设置 Python 虚拟环境...${NC}"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

# 激活虚拟环境进行安装
source "$VENV_DIR/bin/activate"

echo -e "${YELLOW}正在安装 Python 依赖 (这可能需要几分钟)...${NC}"
# 升级 pip
pip install --upgrade pip

# 安装依赖
# 优先尝试安装 uv 来加速安装，如果不行则用 pip
if pip install uv; then
    echo "使用 uv 安装依赖..."
    # 使用 uv 安装，遵循 pyproject.toml
    # 注意：在 venv 中使用 uv pip install
    uv pip install .
else
    echo "使用 pip 安装依赖..."
    pip install .
fi

# 验证安装
if ! python3 -c "import whisperx; print('WhisperX imported successfully')"; then
    echo -e "${RED}依赖安装失败，请检查错误日志。${NC}"
    exit 1
fi

# 5. 配置 Supervisor
echo -e "${YELLOW}正在配置 Supervisor...${NC}"

CONF_PATH="/etc/supervisor/conf.d/whisperx_api.conf"
LOG_DIR="/var/log/whisperx"
mkdir -p "$LOG_DIR"

# 构造环境变量字符串
ENV_VARS="PORT=$PORT,WHISPERX_MODEL=$WHISPER_MODEL,WHISPERX_DEVICE=$DEVICE,WHISPERX_COMPUTE_TYPE=$COMPUTE_TYPE"
if [ -n "$HF_TOKEN" ]; then
    ENV_VARS="$ENV_VARS,HF_TOKEN=$HF_TOKEN"
fi
# 添加 CUDA 库路径 (有时 supervisor 环境找不到)
ENV_VARS="$ENV_VARS,LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu"

cat > "$CONF_PATH" <<EOF
[program:whisperx_api]
directory=$PROJECT_ROOT
command=$VENV_DIR/bin/uvicorn whisperx.api_server:app --host 0.0.0.0 --port $PORT
autostart=true
autorestart=true
stderr_logfile=$LOG_DIR/err.log
stdout_logfile=$LOG_DIR/out.log
user=root
environment=$ENV_VARS
stopasgroup=true
killasgroup=true
EOF

echo -e "${GREEN}Supervisor 配置文件已写入: $CONF_PATH${NC}"

# 6. 启动/重载服务
echo -e "${YELLOW}正在启动服务...${NC}"
supervisorctl reread
supervisorctl update
supervisorctl restart whisperx_api

echo -e "${GREEN}=== 部署完成 ===${NC}"
echo -e "服务状态："
supervisorctl status whisperx_api
echo -e "${GREEN}API 服务正在端口 $PORT 上运行${NC}"
echo -e "查看日志: tail -f $LOG_DIR/out.log"
