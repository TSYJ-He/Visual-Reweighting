
import json

import ray
from omegaconf import OmegaConf

from ..single_controller.ray import RayDistributedWorkerGroup
from ..utils.tokenizer import get_processor, get_tokenizer
from ..workers.fsdp_workers import FSDPWorker
from ..workers.reward import BatchFunctionRewardManager, SequentialFunctionRewardManager
from .training_config import RLTrainingConfig
from .dataset_manager import create_training_dataloaders
from .distributed_trainer import DistributedRLTrainer, ComputeResourceManager, WorkerRole


# please make sure main_task is not scheduled on head
@ray.remote(num_cpus=1)
class TrainingRunner:
    """A runner for RL training."""

    def run(self, config: RLTrainingConfig):
        # print config
        print(json.dumps(config.to_dict(), indent=2))

        # instantiate tokenizer
        tokenizer = get_tokenizer(
            config.worker.actor.model.model_path,
            override_chat_template=config.dataset.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )
        processor = get_processor(
            config.worker.actor.model.model_path,
            override_chat_template=config.dataset.override_chat_template,
            trust_remote_code=config.worker.actor.model.trust_remote_code,
            use_fast=True,
        )

        # define worker classes
        ray_worker_group_cls = RayDistributedWorkerGroup
        role_worker_mapping = {
            WorkerRole.PolicyAgentRolloutRef: ray.remote(FSDPWorker),
            WorkerRole.ValueEstimator: ray.remote(FSDPWorker),
        }
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.training.n_gpus_per_node] * config.training.nnodes,
        }
        mapping = {
            WorkerRole.PolicyAgentRolloutRef: global_pool_id,
            WorkerRole.ValueEstimator: global_pool_id,
        }
        resource_manager = ComputeResourceManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        if config.worker.reward.reward_type == "sequential":
            RewardManager = SequentialFunctionRewardManager
        elif config.worker.reward.reward_type == "batch":
            RewardManager = BatchFunctionRewardManager
        else:
            raise NotImplementedError(f"Unknown reward type {config.worker.reward.reward_type}.")

        RemoteRewardManager = ray.remote(RewardManager).options(num_cpus=config.worker.reward.num_cpus)
        reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)
        val_reward_fn = RemoteRewardManager.remote(config.worker.reward, tokenizer)

        train_dataloader, val_dataloader = create_training_dataloaders(config.dataset, tokenizer, processor)

        trainer = DistributedRLTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            role_worker_mapping=role_worker_mapping,
            resource_manager=resource_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
        )
        trainer.initialize_workers()
        trainer.train()


def main():
    cli_args = OmegaConf.from_cli()
    default_config = OmegaConf.structured(RLTrainingConfig())

    if hasattr(cli_args, "config"):
        config_path = cli_args.pop("config", None)
        file_config = OmegaConf.load(config_path)
        default_config = OmegaConf.merge(default_config, file_config)

    training_config = OmegaConf.merge(default_config, cli_args)
    training_config: RLTrainingConfig = OmegaConf.to_object(training_config)
    training_config.deep_post_init()

    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "TOKENIZERS_PARALLELISM": "true",
                "NCCL_DEBUG": "WARN",
                "VLLM_LOGGING_LEVEL": "WARN",
                "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            }
        }
        ray.init(runtime_env=runtime_env)

    runner = TrainingRunner.remote()
    ray.get(runner.run.remote(training_config))

    if training_config.training.ray_timeline is not None:
        # use `export RAY_PROFILING=1` to record the ray timeline
        ray.timeline(filename=training_config.training.ray_timeline)


if __name__ == "__main__":
    main()

