
"""
Training configuration
"""

import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ..optimization.config import OptimizationConfig

from ..workers.config import WorkerConfig


def recursive_post_init(dataclass_obj):
    if hasattr(dataclass_obj, "post_init"):
        dataclass_obj.post_init()

    for attr in fields(dataclass_obj):
        if is_dataclass(getattr(dataclass_obj, attr.name)):
            recursive_post_init(getattr(dataclass_obj, attr.name))


@dataclass
class DatasetConfig:
    train_files: str = ""
    val_files: str = ""
    prompt_key: str = "prompt"
    answer_key: str = "answer"
    image_key: str = "images"
    video_key: str = "videos"
    image_dir: Optional[str] = None
    video_fps: float = 2.0
    max_prompt_length: int = 512
    max_response_length: int = 512
    rollout_batch_size: int = 512
    mini_rollout_batch_size: Optional[int] = None
    val_batch_size: int = -1
    format_prompt: Optional[str] = None
    override_chat_template: Optional[str] = None
    shuffle: bool = True
    seed: int = 1
    min_pixels: Optional[int] = 262144
    max_pixels: Optional[int] = 4194304
    filter_overlong_prompts: bool = True
    filter_overlong_prompts_workers: int = 16

    def post_init(self):
        if self.image_dir is not None:
            if os.path.exists(self.image_dir):  # ray job uses absolute path
                self.image_dir = os.path.abspath(self.image_dir)
            else:
                print(f"Image directory {self.image_dir} not found.")
                self.image_dir = None

        if self.format_prompt is not None:
            if os.path.exists(self.format_prompt):  # ray job uses absolute path
                self.format_prompt = os.path.abspath(self.format_prompt)
            else:
                print(f"Format prompt file {self.format_prompt} not found.")
                self.format_prompt = None


@dataclass
class TrainingConfig:
    total_epochs: int = 15
    """total epochs for training"""
    max_steps: Optional[int] = None
    """max steps for training, if specified, total_epochs is ignored"""
    project_name: str = "mtrl_training"
    """project name for logger"""
    experiment_name: str = "demo"
    """experiment name for logger"""
    logger: Tuple[str] = ("console", "wandb")
    """logger type, support `console`, `mlflow`, `swanlab`, `tensorboard`, `wandb`"""
    nnodes: int = 1
    """number of nodes for training"""
    n_gpus_per_node: int = 8
    """number of gpus per node for training"""
    max_try_make_batch: int = 20
    """max number of generations for online filtering, -1 means no limit"""
    critic_warmup: int = 0
    """critic warmup steps"""
    val_freq: int = -1
    """validation frequency, -1 means no validation"""
    val_before_train: bool = True
    """validate before training"""
    val_only: bool = False
    """validate only, skip training"""
    val_generations_to_log: int = 0
    """number of generations to log for validation"""
    save_freq: int = -1
    """save frequency, -1 means no saving"""
    save_limit: int = -1
    """max number of checkpoints to save, -1 means no limit"""
    save_model_only: bool = False
    """save model only, no optimizer state dict"""
    save_checkpoint_path: Optional[str] = None
    """save checkpoint path, if not specified, use `checkpoints/project_name/experiment_name`"""
    load_checkpoint_path: Optional[str] = None
    """load checkpoint path"""
    ray_timeline: Optional[str] = None
    """file to save ray timeline"""
    find_last_checkpoint: bool = True
    """automatically find the last checkpoint in the save checkpoint path to resume training"""

    def post_init(self):
        if self.save_checkpoint_path is None:
            self.save_checkpoint_path = os.path.join("checkpoints", self.project_name, self.experiment_name)

        self.save_checkpoint_path = os.path.abspath(self.save_checkpoint_path)  # ray job uses absolute path
        if self.load_checkpoint_path is not None:
            if os.path.exists(self.load_checkpoint_path):  # ray job uses absolute path
                self.load_checkpoint_path = os.path.abspath(self.load_checkpoint_path)
            else:
                print(f"Model checkpoint {self.load_checkpoint_path} not found.")
                self.load_checkpoint_path = None


@dataclass
class RLTrainingConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    optimization: Optional["OptimizationConfig"] = field(default=None)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def post_init(self):
        self.worker.rollout.prompt_length = self.dataset.max_prompt_length
        self.worker.rollout.response_length = self.dataset.max_response_length
        self.worker.rollout.trust_remote_code = self.worker.actor.model.trust_remote_code
        
        # Sync optimization config to worker config
        if self.optimization is not None:
            self.worker.actor.disable_divergence = self.optimization.disable_divergence
            self.worker.actor.use_divergence_loss = self.optimization.use_divergence_loss
            self.worker.actor.divergence_penalty = self.optimization.divergence_penalty
            self.worker.actor.divergence_coef = self.optimization.divergence_coef

            # Sync token filtering configs
            self.worker.actor.enable_entropy_filtering = self.optimization.enable_entropy_filtering
            self.worker.actor.top_p_entropy_tokens = self.optimization.top_p_entropy_tokens

            self.worker.actor.enable_perception_filtering = self.optimization.enable_perception_filtering
            self.worker.actor.top_p_perception_tokens = self.optimization.top_p_perception_tokens

            # Sync entropy penalty config
            self.worker.actor.use_entropy_penalty = self.optimization.use_entropy_penalty
            self.worker.actor.entropy_penalty_coef = self.optimization.entropy_penalty_coef

            # Sync trajectory reweighting config
            self.worker.actor.enable_trajectory_reweighting = self.optimization.enable_trajectory_reweighting
            self.worker.actor.trajectory_scaling_min = self.optimization.trajectory_scaling_min
        else:
            # Set default values if optimization config is None
            from ..optimization.config import OptimizationConfig
            default_optimization = OptimizationConfig()
            self.optimization = default_optimization
            # Sync default values
            self.worker.actor.disable_divergence = default_optimization.disable_divergence
            self.worker.actor.use_divergence_loss = default_optimization.use_divergence_loss
            self.worker.actor.divergence_penalty = default_optimization.divergence_penalty
            self.worker.actor.divergence_coef = default_optimization.divergence_coef
            self.worker.actor.enable_entropy_filtering = default_optimization.enable_entropy_filtering
            self.worker.actor.top_p_entropy_tokens = default_optimization.top_p_entropy_tokens
            self.worker.actor.enable_perception_filtering = default_optimization.enable_perception_filtering
            self.worker.actor.top_p_perception_tokens = default_optimization.top_p_perception_tokens
            self.worker.actor.use_entropy_penalty = default_optimization.use_entropy_penalty
            self.worker.actor.entropy_penalty_coef = default_optimization.entropy_penalty_coef
            self.worker.actor.enable_trajectory_reweighting = default_optimization.enable_trajectory_reweighting
            self.worker.actor.trajectory_scaling_min = default_optimization.trajectory_scaling_min

    def deep_post_init(self):
        recursive_post_init(self)

    def to_dict(self):
        return asdict(self)

