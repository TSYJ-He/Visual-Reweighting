# MTRL Training Module

## 概述

`mtrl/training/` 模块提供了完整的分布式强化学习训练框架，支持多模态模型的训练。

## 模块结构

```
mtrl/training/
├── __init__.py                    # 模块导出
├── training_config.py             # 训练配置类
├── training_metrics.py           # 训练指标计算
├── dataset_manager.py            # 数据加载器管理
├── distributed_trainer.py        # 分布式训练器（核心）
└── training_main.py              # 训练主入口
```

## 核心组件

### 1. RLTrainingConfig (`training_config.py`)

训练配置类，包含：
- `DatasetConfig`: 数据集配置
- `TrainingConfig`: 训练过程配置
- `RLTrainingConfig`: 完整训练配置（整合dataset、worker、optimization、training）

### 2. DistributedRLTrainer (`distributed_trainer.py`)

分布式训练器，核心功能：
- **资源管理**: `ComputeResourceManager` 管理GPU资源池
- **训练循环**: `train()` 方法实现完整的RL训练流程
- **检查点管理**: 自动保存和加载checkpoint
- **验证**: 支持训练过程中的验证

### 3. Training Metrics (`training_metrics.py`)

指标计算函数：
- `aggregate_metrics`: 聚合指标
- `calculate_sequence_length_metrics`: 序列长度指标
- `calculate_batch_statistics`: 批次统计
- `calculate_timing_statistics`: 时间统计
- `calculate_throughput_statistics`: 吞吐量统计

### 4. Dataset Manager (`dataset_manager.py`)

数据加载器创建：
- `create_training_dataloaders`: 创建训练和验证数据加载器

## 使用示例

```python
from mtrl.training import (
    RLTrainingConfig,
    DistributedRLTrainer,
    ComputeResourceManager,
    WorkerRole,
    create_training_dataloaders,
)
from mtrl.optimization import OptimizationConfig

# 创建配置
config = RLTrainingConfig(
    dataset=DatasetConfig(
        train_files="path/to/train",
        val_files="path/to/val",
    ),
    optimization=OptimizationConfig(
        enable_perception_filtering=True,
        enable_trajectory_reweighting=True,
    ),
    training=TrainingConfig(
        total_epochs=15,
        project_name="my_project",
    ),
)

# 创建数据加载器
train_dataloader, val_dataloader = create_training_dataloaders(
    config.dataset, tokenizer, processor
)

# 创建训练器
trainer = DistributedRLTrainer(
    config=config,
    tokenizer=tokenizer,
    processor=processor,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    role_worker_mapping=role_worker_mapping,
    resource_manager=resource_manager,
)

# 初始化并训练
trainer.initialize_workers()
trainer.train()
```

## 训练流程

1. **初始化阶段**
   - 创建资源池
   - 初始化worker groups
   - 加载checkpoint（如果存在）

2. **训练循环**（每个step）
   - 生成批次数据（rollout）
   - 计算奖励
   - 计算优势值
   - 更新策略（应用TGF和TAS）
   - 更新critic（如果使用）
   - 验证（按频率）
   - 保存checkpoint（按频率）

3. **后处理**
   - 最终验证
   - 保存最终checkpoint

- **分布式训练**: 基于Ray的分布式架构
- **自动checkpoint**: 支持断点续训
- **灵活配置**: 模块化配置系统
- **性能监控**: 详细的训练指标
- **多模态支持**: 支持图像和视频输入

