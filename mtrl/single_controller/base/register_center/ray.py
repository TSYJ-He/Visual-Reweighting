
import ray


@ray.remote
class DistributedWorkerGroupRegistry:
    def __init__(self, rank_zero_information):
        self.rank_zero_information = rank_zero_information

    def get_rank_zero_information(self):
        return self.rank_zero_information


def create_distributed_worker_group_registry(name, info):
    return DistributedWorkerGroupRegistry.options(name=name).remote(info)
