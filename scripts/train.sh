#!/bin/bash
# MTRL 训练启动脚本
# 用法: bash scripts/train.sh [config_file]

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}    MTRL Training Script${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查配置文件
CONFIG_FILE=${1:-"example_config.yaml"}

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件 '$CONFIG_FILE' 不存在${NC}"
    echo -e "${YELLOW}用法: bash scripts/train.sh [config_file]${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 配置文件: $CONFIG_FILE${NC}"

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python 版本: $(python --version)${NC}"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo -e "${GREEN}✓ 检测到 $GPU_COUNT 张 GPU${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n 1
else
    echo -e "${YELLOW}⚠ 警告: 未检测到 CUDA，将使用 CPU 训练（非常慢）${NC}"
fi

# 检查必要的包
echo -e "\n${GREEN}检查依赖...${NC}"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')"
python -c "import ray; print(f'✓ Ray {ray.__version__}')"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')"

# 设置环境变量
export TOKENIZERS_PARALLELISM=true
export NCCL_DEBUG=WARN
export VLLM_LOGGING_LEVEL=WARN
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_DEVICE_MAX_CONNECTIONS=1

echo -e "\n${GREEN}开始训练...${NC}"
echo -e "${YELLOW}日志将保存到当前目录${NC}"
echo -e "${YELLOW}按 Ctrl+C 停止训练${NC}\n"

# 运行训练
python -m mtrl.training.training_main config=$CONFIG_FILE

echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}    训练完成！${NC}"
echo -e "${GREEN}========================================${NC}"

