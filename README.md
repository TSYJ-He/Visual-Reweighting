# Multi-Modal Token-level RL Reweighting （还没想好名儿）

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)


## 📋 目录

- [主要特性](#主要特性)
- [核心算法](#核心算法)
- [环境要求](#环境要求)
- [安装](#安装)
- [快速开始](#快速开始)
- [详细配置](#详细配置)
- [训练流程](#训练流程)
- [高级功能](#高级功能)
- [性能优化](#性能优化)
- [示例](#示例)
- [架构说明](#架构说明)
- [常见问题](#常见问题)

## ✨ 主要特性

- 🚀 **高性能分布式训练**: 基于 Ray 和 FSDP 的分布式训练架构
- 🎯 **Token级优化**: 支持 Token Gradient Filtering (TGF) 和 Trajectory Advantage Shaping (TAS)
- 🖼️ **多模态支持**: 原生支持文本、图像、视频等多模态输入
- ⚡ **动态批处理**: Padding-Free 训练和动态批处理，显著提升训练效率
- 🔧 **灵活配置**: 基于 OmegaConf 的层次化配置系统
- 📊 **完善监控**: 支持 WandB、TensorBoard、MLflow 等多种日志系统
- 💾 **断点续训**: 完整的检查点管理和自动恢复机制

## 🧠 核心算法

MTRL 实现了创新的 **Token Perception Reinforcement Learning (TPRL)** 算法，包含两个核心技术：

### Token Gradient Filtering (TGF) - 微观级优化
- **熵过滤** (`enable_entropy_filtering`): 基于 token 熵值筛选高不确定性的 token
- **感知过滤** (`enable_perception_filtering`): 基于视觉依赖度筛选视觉相关 token

### Trajectory Advantage Shaping (TAS) - 宏观级优化
- **轨迹重加权** (`enable_trajectory_reweighting`): 基于轨迹的视觉感知敏感度动态调整优势函数

## 📦 环境要求

- Python >= 3.9
- CUDA >= 11.8 (推荐 12.1+)
- PyTorch >= 2.0
- 多 GPU 环境 (推荐 8 卡或以上)


## 🔧 安装





### 1. 创建虚拟环境

```bash
conda create -n mtrl python=3.10
conda activate mtrl
```

### 2. 安装依赖

```bash
# 安装 PyTorch (根据 CUDA 版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 Flash Attention
pip install flash-attn --no-build-isolation

# 安装 MTRL
pip install -e .



### 3. 验证安装

```bash
python -c "import mtrl; print('MTRL 安装成功!')"
```



### 准备数据

数据格式应为 JSONL，每行包含以下字段：

```json
{
    "prompt": "请描述这张图片",
    "answer": "这是一张...",
    "images": ["path/to/image1.jpg"],
    "videos": []
}
```

支持的字段：
- `prompt`: 输入提示文本
- `answer`: 参考答案（可选）
- `images`: 图片路径列表
- `videos`: 视频路径列表

### 创建配置文件

创建 `config.yaml`:

```yaml
# ==================== 数据集配置 ====================
dataset:
  train_files: "data/train.jsonl"
  val_files: "data/val.jsonl"
  prompt_key: "prompt"
  answer_key: "answer"
  image_key: "images"
  video_key: "videos"
  image_dir: "data/images"
  max_prompt_length: 512
  max_response_length: 512
  rollout_batch_size: 256
  val_batch_size: 64

# ==================== 训练配置 ====================
training:
  total_epochs: 10
  project_name: "mtrl_training"
  experiment_name: "qwen2_vl_7b"
  logger: ["console", "wandb"]
  nnodes: 1
  n_gpus_per_node: 8
  save_freq: 500
  val_freq: 100

# ==================== 优化配置 ====================
optimization:
  # 基础优化参数
  gamma: 1.0
  lam: 0.95
  advantage_estimator: "gae"
  
  # KL 散度控制
  divergence_coef: 0.01
  divergence_type: "adaptive"
  divergence_target: 6.0
  divergence_horizon: 10000.0
  
  # TPRL 
  enable_entropy_filtering: true       # 启用熵过滤（SAR的遗产，先放这了）
  top_p_entropy_tokens: 0.2           # 选择 top 20% 高熵 token
  
  enable_perception_filtering: true    # 启用感知过滤
  top_p_perception_tokens: 0.2        # 选择 top 20% 视觉相关 token
  
  enable_trajectory_reweighting: true  # 启用轨迹重加权
  trajectory_scaling_min: 0.8         # 最小缩放因子

# ==================== Worker 配置 ====================
worker:
  hybrid_engine: true
  
  # Actor 配置
  actor:
    model:
      model_path: "Qwen/Qwen2.5-VL-7B-Instruct" #qwen3好一些
      enable_gradient_checkpointing: true
      trust_remote_code: true
    
    optim:
      lr: 5e-7
      weight_decay: 0.01
      warmup_steps: 10
    
    fsdp:
      sharding_strategy: "FULL_SHARD"
      mixed_precision_dtype: "bf16"
      cpu_offload: false
    
    # 训练参数
    global_batch_size: 256
    micro_batch_size_per_device_for_update: 2
    ppo_epochs: 1
    clip_ratio_low: 0.2
    clip_ratio_high: 0.3
    max_grad_norm: 1.0
    padding_free: true
    dynamic_batching: true
  
  # Rollout 配置
  rollout:
    name: "vllm"
    gpu_memory_utilization: 0.4
    temperature: 1.0
    top_p: 1.0
    top_k: -1
  
  # Reward 配置
  reward:
    reward_type: "function"
    reward_fn_path: "your_reward_function.py"
```

### 运行训练

```bash
# 基础训练
python -m mtrl.training.training_main config=config.yaml

# 使用命令行覆盖配置
python -m mtrl.training.training_main \
    config=config.yaml \
    training.experiment_name=my_experiment \
    optimization.enable_entropy_filtering=true \
    optimization.enable_perception_filtering=true
```

## ⚙️ 详细配置

### 数据集配置 (DatasetConfig)

```yaml
dataset:
  # 数据路径
  train_files: "data/train.jsonl"      # 训练数据，支持逗号分隔多个文件
  val_files: "data/val.jsonl"          # 验证数据
  
  # 数据字段名
  prompt_key: "prompt"                 # 输入提示的字段名
  answer_key: "answer"                 # 答案字段名
  image_key: "images"                  # 图片路径字段名
  video_key: "videos"                  # 视频路径字段名
  
  # 多模态配置
  image_dir: "data/images"             # 图片目录（如果路径是相对的）
  video_fps: 2.0                       # 视频采样帧率
  min_pixels: 262144                   # 最小像素数 (512x512)
  max_pixels: 4194304                  # 最大像素数 (2048x2048)
  
  # 长度配置
  max_prompt_length: 512               # 最大提示长度
  max_response_length: 512             # 最大生成长度
  
  # 批处理配置
  rollout_batch_size: 256              # 生成批次大小
  val_batch_size: 64                   # 验证批次大小
  
  # 数据处理
  shuffle: true                        # 是否打乱数据
  seed: 42                            # 随机种子
  filter_overlong_prompts: true       # 过滤过长的提示
```

### 优化配置 (OptimizationConfig)

```yaml
optimization:
  # ============ 基础 RL 参数 ============
  gamma: 1.0                          # 折扣因子
  lam: 0.95                          # GAE lambda 参数
  advantage_estimator: "gae"         # 优势估计器: gae, grpo, rloo, remax
  
  # ============ KL 散度控制 ============
  disable_divergence: false          # 是否禁用参考模型
  use_divergence_loss: false         # 使用散度损失而非奖励惩罚
  divergence_penalty: "kl"           # 散度类型: kl, abs, mse, low_var_kl
  divergence_coef: 0.01              # 散度系数
  divergence_type: "adaptive"        # 控制器: fixed, adaptive
  divergence_target: 6.0             # 自适应目标散度
  divergence_horizon: 10000.0        # 自适应时间范围
  
  # ============ 在线过滤 ============
  online_filtering: false            # 启用在线样本过滤
  filter_key: "overall"              # 过滤使用的奖励键
  filter_low: 0.01                   # 过滤低于此分位数的样本
  filter_high: 0.99                  # 过滤高于此分位数的样本
  
  # ============ TPRL 算法 ============
  # Token Gradient Filtering - 熵过滤
  enable_entropy_filtering: true     # 启用基于熵的 token 过滤
  top_p_entropy_tokens: 0.2         # 选择高熵 token 的比例
  
  # Token Gradient Filtering - 感知过滤
  enable_perception_filtering: true  # 启用基于视觉感知的 token 过滤
  top_p_perception_tokens: 0.2      # 选择高感知 token 的比例
  
  # Trajectory Advantage Shaping
  enable_trajectory_reweighting: true # 启用轨迹级重加权
  trajectory_scaling_min: 0.8        # 最小缩放因子 (0-1)
  
  # 熵惩罚
  use_entropy_penalty: false         # 启用熵正则化
  entropy_penalty_coef: 0.06         # 熵惩罚系数
```

### 模型配置 (ModelConfig)

```yaml
worker:
  actor:
    model:
      # 模型路径
      model_path: "Qwen/Qwen2-VL-7B-Instruct"
      
      # 模型设置
      trust_remote_code: true                    # 信任远程代码
      enable_gradient_checkpointing: true        # 梯度检查点（节省显存）
      peft_type: null                           # PEFT 类型: lora, qlora, null
      
    optim:
      # 优化器配置
      lr: 5e-7                                  # 学习率
      weight_decay: 0.01                        # 权重衰减
      warmup_steps: 10                          # 预热步数
      lr_scheduler_type: "constant_with_warmup" # 学习率调度器
      betas: [0.9, 0.95]                       # Adam beta 参数
    
    fsdp:
      # FSDP 配置
      sharding_strategy: "FULL_SHARD"           # 分片策略
      mixed_precision_dtype: "bf16"             # 混合精度: bf16, fp16
      cpu_offload: false                        # CPU offload
    
    # 训练超参数
    global_batch_size: 256                      # 全局批次大小
    micro_batch_size_per_device_for_update: 2   # 每卡更新批次
    micro_batch_size_per_device_for_experience: 8  # 每卡推理批次
    ppo_epochs: 1                               # PPO 更新轮数
    
    # PPO 裁剪
    clip_ratio_low: 0.2                         # 下界裁剪比率
    clip_ratio_high: 0.3                        # 上界裁剪比率
    clip_ratio_dual: 3.0                        # 双重裁剪常数
    
    # 其他
    max_grad_norm: 1.0                          # 梯度裁剪
    padding_free: true                          # Padding-free 训练
    dynamic_batching: true                      # 动态批处理
    ulysses_size: 1                            # Ulysses 序列并行大小
    use_torch_compile: true                     # 使用 torch.compile
```

### Rollout 配置 (RolloutConfig)

```yaml
worker:
  rollout:
    name: "vllm"                        # 推理引擎: vllm
    gpu_memory_utilization: 0.4         # GPU 显存利用率
    
    # 生成参数
    temperature: 1.0                    # 采样温度
    top_p: 1.0                         # nucleus 采样
    top_k: -1                          # top-k 采样 (-1 禁用)
    max_new_tokens: 512                # 最大生成长度
    
    # vLLM 特定配置
    tensor_parallel_size: 1            # 张量并行大小
    enable_prefix_caching: false       # 前缀缓存
```

### Reward 配置 (RewardConfig)

```yaml
worker:
  reward:
    reward_type: "function"             # 奖励类型: function, model
    reward_fn_path: "path/to/reward.py" # 奖励函数路径
    num_cpus: 4                        # CPU 核心数
```

奖励函数示例 (`reward.py`):

```python
def reward_function(data_dict):
    """
    计算奖励函数
    
    Args:
        data_dict: 包含以下字段的字典
            - prompt: 输入提示
            - response: 模型生成的响应
            - ground_truth: 参考答案（如果有）
            - images: 图片数据
    
    Returns:
        reward: float，奖励值
        metrics: dict，额外的指标
    """
    prompt = data_dict['prompt']
    response = data_dict['response']
    ground_truth = data_dict.get('ground_truth', '')
    
    # 示例：基于长度和关键词的简单奖励
    reward = 0.0
    
    # 长度奖励
    if 50 <= len(response) <= 200:
        reward += 0.5
    
    # 关键词匹配
    if ground_truth:
        keywords = set(ground_truth.split())
        response_words = set(response.split())
        overlap = len(keywords & response_words) / len(keywords)
        reward += overlap
    
    metrics = {
        "length": len(response),
        "overlap": overlap if ground_truth else 0
    }
    
    return reward, metrics
```

## 📊 训练流程

### 完整训练流程

```python
# 1. 数据准备
# 2. 配置文件编写
# 3. 启动训练
python -m mtrl.training.training_main config=config.yaml

# 4. 监控训练
# - 查看 WandB 面板
# - 检查日志文件
# - 监控 GPU 使用率

# 5. 评估和部署
# - 加载检查点
# - 运行评估脚本
# - 导出最终模型
```

### 训练监控

训练过程中会输出以下关键指标：

- **训练指标**:
  - `actor/pg_loss`: 策略梯度损失
  - `actor/approx_kl`: 近似 KL 散度
  - `actor/clipfrac`: 裁剪比例
  - `actor/entropy`: 策略熵
  - `actor/grad_norm`: 梯度范数

- **TPRL 特定指标**:
  - `actor/entropy_token_fraction`: 熵过滤选择的 token 比例
  - `actor/perception_token_fraction`: 感知过滤选择的 token 比例
  - `actor/sensitivity_score_mean`: 平均敏感度分数
  - `actor/scaling_factor_mean`: 平均缩放因子

- **性能指标**:
  - `perf/tokens_per_second`: 每秒处理 token 数
  - `perf/samples_per_second`: 每秒样本数
  - `perf/throughput`: 吞吐量

- **奖励指标**:
  - `reward/overall`: 总体奖励
  - `val/reward_score`: 验证集奖励

## 🎯 高级功能

### 1. 使用 TPRL 算法

完整启用 TPRL（Token Perception RL）:

```yaml
optimization:
  # 启用所有 TPRL 组件
  enable_entropy_filtering: true
  top_p_entropy_tokens: 0.2
  
  enable_perception_filtering: true
  top_p_perception_tokens: 0.2
  
  enable_trajectory_reweighting: true
  trajectory_scaling_min: 0.8
```

**工作原理**:

1. **熵过滤**: 识别模型不确定的 token，重点优化这些位置
2. **感知过滤**: 识别依赖视觉信息的 token，强化视觉理解
3. **轨迹重加权**: 根据整体视觉敏感度调整轨迹权重

### 2. 分布式训练

#### 单节点多卡

```bash
python -m mtrl.training.training_main \
    config=config.yaml \
    training.nnodes=1 \
    training.n_gpus_per_node=8
```

#### 多节点训练

```bash
# 在每个节点上运行
# 节点 0 (master)
RAY_ADDRESS='auto' python -m mtrl.training.training_main config=config.yaml

# 节点 1, 2, ... (workers)
RAY_ADDRESS='ip-of-node-0:6379' ray start --address='ip-of-node-0:6379'
```

### 3. 断点续训

```yaml
training:
  load_checkpoint_path: "checkpoints/experiment/step_1000"
  find_last_checkpoint: true  # 自动找到最新检查点
```

或使用命令行:

```bash
python -m mtrl.training.training_main \
    config=config.yaml \
    training.load_checkpoint_path=checkpoints/experiment/step_1000
```

### 4. 仅验证模式

```bash
python -m mtrl.training.training_main \
    config=config.yaml \
    training.val_only=true \
    training.load_checkpoint_path=checkpoints/best_model
```

### 5. 使用 LoRA/QLoRA

```yaml
worker:
  actor:
    model:
      peft_type: "lora"
      lora_rank: 8
      lora_alpha: 16
      lora_dropout: 0.05
      lora_target_modules: ["q_proj", "v_proj"]
```

## ⚡ 性能优化

### 显存优化

```yaml
worker:
  actor:
    model:
      enable_gradient_checkpointing: true  # 梯度检查点
    
    fsdp:
      cpu_offload: true                   # CPU offload（慢但省显存）
      sharding_strategy: "FULL_SHARD"     # 全分片
    
    offload:
      offload_params: true                # 参数 offload
      offload_optimizer: true             # 优化器 offload
    
    # 减小批次大小
    micro_batch_size_per_device_for_update: 1
    
  rollout:
    gpu_memory_utilization: 0.3          # 降低 vLLM 显存占用
```

### 速度优化

```yaml
worker:
  actor:
    # 增大批次大小
    global_batch_size: 512
    micro_batch_size_per_device_for_update: 4
    
    # 性能优化选项
    padding_free: true                   # Padding-free 训练
    dynamic_batching: true               # 动态批处理
    use_torch_compile: true              # Torch compile
    ulysses_size: 2                      # Ulysses 序列并行
    
    fsdp:
      mixed_precision_dtype: "bf16"      # 使用 BF16
      
  rollout:
    gpu_memory_utilization: 0.6          # 提高 vLLM 利用率
    enable_prefix_caching: true          # 启用前缀缓存
```

### 混合精度训练

```yaml
worker:
  actor:
    fsdp:
      mixed_precision_dtype: "bf16"  # 推荐 BF16（A100/H100）
      # mixed_precision_dtype: "fp16"  # V100 使用 FP16
```

## 📝 示例

### 示例 1: 基础文本 RL 训练

```yaml
dataset:
  train_files: "data/train.jsonl"
  max_prompt_length: 512
  max_response_length: 256

optimization:
  advantage_estimator: "gae"
  gamma: 0.99
  lam: 0.95

worker:
  actor:
    model:
      model_path: "meta-llama/Llama-2-7b-hf"
    global_batch_size: 256
```

### 示例 2: 多模态 VQA 训练

```yaml
dataset:
  train_files: "data/vqa_train.jsonl"
  image_dir: "data/images"
  image_key: "image_path"
  max_prompt_length: 512
  max_response_length: 128

optimization:
  enable_entropy_filtering: true
  enable_perception_filtering: true
  enable_trajectory_reweighting: true

worker:
  actor:
    model:
      model_path: "Qwen/Qwen2-VL-7B-Instruct"
```

### 示例 3: 数学推理训练

```yaml
dataset:
  train_files: "data/math_train.jsonl"
  max_response_length: 1024

optimization:
  advantage_estimator: "grpo"
  enable_entropy_filtering: true
  top_p_entropy_tokens: 0.3  # 数学推理需要更多探索

worker:
  actor:
    model:
      model_path: "deepseek-ai/deepseek-math-7b"
    ppo_epochs: 2  # 数学任务可以多轮更新
```

## 🏗️ 架构说明

### 模块结构

```
mtrl/
├── agents/                 # 策略代理
│   ├── base.py            # 基础代理接口
│   ├── parallel_policy_agent.py  # 分布式策略代理
│   └── config.py          # 代理配置
├── optimization/          # 优化算法
│   ├── policy_optimization.py  # 策略优化函数
│   └── config.py          # 优化配置
├── single_controller/     # 分布式控制
│   ├── base/             # 基础控制器
│   └── ray/              # Ray 实现
├── training/             # 训练流程
│   ├── distributed_trainer.py  # 分布式训练器
│   ├── training_main.py   # 训练入口
│   └── training_config.py # 训练配置
├── workers/              # 工作进程
│   ├── actor/            # Actor worker
│   ├── critic/           # Critic worker
│   ├── rollout/          # Rollout worker
│   └── reward/           # Reward worker
├── models/               # 模型适配
├── utils/                # 工具函数
└── protocol.py           # 数据协议
```

### 训练流程图

```
1. 数据加载 → 2. Rollout (生成) → 3. Reward 计算 
                                          ↓
6. 更新模型 ← 5. PPO 更新 ← 4. Advantage 计算
     ↓
7. 重复步骤 2-6
```

### TPRL 算法流程

```
输入: 多模态数据 (文本 + 图像)
  ↓
生成响应
  ↓
计算 log_probs 和 aug_log_probs
  ↓
Token-level 过滤:
  ├─ 熵过滤 → 选择高不确定性 token
  └─ 感知过滤 → 选择视觉相关 token
  ↓
Trajectory-level 重加权:
  └─ 基于敏感度调整 advantage
  ↓
PPO 更新
```



 推荐的起始值：

```yaml
optimization:
  enable_entropy_filtering: true
  top_p_entropy_tokens: 0.2          # 起始值，可调整 0.1-0.3
  
  enable_perception_filtering: true
  top_p_perception_tokens: 0.2       # 起始值，可调整 0.1-0.3
  
  enable_trajectory_reweighting: true
  trajectory_scaling_min: 0.8        # 起始值，可调整 0.6-0.9
```

- 如果模型过度关注某些 token，降低 `top_p` 值
- 如果希望更强的视觉理解，增加 `top_p_perception_tokens`
- 如果训练不稳定，增加 `trajectory_scaling_min`

### 如何加速训练？

**A**: 性能优化建议：

1. **启用所有加速选项**:
```yaml
worker:
  actor:
    padding_free: true
    dynamic_batching: true
    use_torch_compile: true
```

2. **使用混合精度**: `mixed_precision_dtype: "bf16"`

3. **优化 vLLM 配置**:
```yaml
worker:
  rollout:
    gpu_memory_utilization: 0.5
    enable_prefix_caching: true
```

4. **增大批次大小**（如果显存允许）

###  训练中断恢复

**A**: MTRL 支持自动断点续训：

```yaml
training:
  find_last_checkpoint: true  # 自动寻找最新检查点
  load_checkpoint_path: "checkpoints/project/experiment"
```

或手动指定：

```bash
python -m mtrl.training.training_main \
    config=config.yaml \
    training.load_checkpoint_path=checkpoints/experiment/step_5000
```

### Q5: 如何调试奖励函数？

**A**: 使用验证模式快速测试：

```bash
python -m mtrl.training.training_main \
    config=config.yaml \
    training.val_only=true \
    training.val_before_train=true \
    dataset.val_batch_size=10
```

### Q6: 多模态训练注意事项？

**A**: 

1. **确保图片路径正确**:
```yaml
dataset:
  image_dir: "/absolute/path/to/images"  # 使用绝对路径
```

2. **调整图片分辨率**:
```yaml
dataset:
  min_pixels: 262144   # 512x512
  max_pixels: 1048576  # 1024x1024（降低以节省显存）
```

3. **对于视觉密集任务，增强感知过滤**:
```yaml
optimization:
  enable_perception_filtering: true
  top_p_perception_tokens: 0.3  # 增加到 30%
```

### Q7: 如何使用自定义模型？

**A**: 只需指定 HuggingFace 模型路径：

```yaml
worker:
  actor:
    model:
      model_path: "your-org/your-model"
      trust_remote_code: true  # 如果需要
```

对于本地模型：

```yaml
worker:
  actor:
    model:
      model_path: "/path/to/local/model"
```

### Q8: 显存不足怎么办？

**A**: 多种显存优化策略：

1. **启用梯度检查点**:
```yaml
worker:
  actor:
    model:
      enable_gradient_checkpointing: true
```

2. **启用 CPU offload**:
```yaml
worker:
  actor:
    fsdp:
      cpu_offload: true
    offload:
      offload_params: true
      offload_optimizer: true
```

3. **减小批次大小**:
```yaml
worker:
  actor:
    micro_batch_size_per_device_for_update: 1
```

4. **降低 vLLM 显存占用**:
```yaml
worker:
  rollout:
    gpu_memory_utilization: 0.3
```

5. **使用 LoRA**:
```yaml
worker:
  actor:
    model:
      peft_type: "lora"
```

## 🙏 致谢

本项目基于以下优秀开源项目：

- [Transformers](https://github.com/huggingface/transformers)
- [vLLM](https://github.com/vllm-project/vllm)
- [Ray](https://github.com/ray-project/ray)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)


