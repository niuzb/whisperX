#!/bin/bash
set -e

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== WhisperX API Server 升级脚本 ===${NC}"

# 1. 检查 Root 权限
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}请使用 root 权限运行此脚本 (sudo ./upgrade.sh)${NC}"
  exit 1
fi

# 2. 定位项目根目录
# 假设脚本在 deploy_scripts/ 或根目录下运行
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

cd "$PROJECT_ROOT"

# 3. 拉取最新代码
echo -e "${YELLOW}正在拉取最新代码 (git pull)...${NC}"
# 注意：这需要当前目录是干净的，且正确配置了 remote
if git pull; then
    echo -e "${GREEN}代码已更新${NC}"
else
    echo -e "${RED}Git 拉取失败。可能是因为有本地修改冲突或网络问题。${NC}"
    echo -e "${YELLOW}如果您有本地修改，请手动 stash 或 commit 后再运行。${NC}"
    exit 1
fi

# 4. 更新依赖
echo -e "${YELLOW}正在检查并更新依赖...${NC}"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}未找到虚拟环境 ($VENV_DIR)，似乎尚未部署。请先运行 deploy.sh${NC}"
    exit 1
fi

# 激活环境
source "$VENV_DIR/bin/activate"

# 更新依赖
if pip show uv > /dev/null 2>&1; then
    echo "使用 uv 更新依赖..."
    uv pip install .
else
    echo "使用 pip 更新依赖..."
    pip install .
fi

# 5. 重启服务
echo -e "${YELLOW}正在重启 Supervisor 服务...${NC}"
supervisorctl restart whisperx_api

# 6. 检查状态
echo -e "${GREEN}=== 升级完成 ===${NC}"
sleep 2 # 等待几秒看启动状态
supervisorctl status whisperx_api

echo -e "${GREEN}如需查看详细日志: tail -f /var/log/whisperx/out.log${NC}"
