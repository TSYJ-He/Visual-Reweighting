
import os
import random
import re
import string
import time
from typing import Any, Optional
from unittest.mock import patch

import ray
from ray.actor import ActorHandle
from ray.experimental.state.api import get_actor
from ray.util import list_named_actors
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy

from ..base import ComputeResourcePool, DistributedWorkerGroup, InitializationArguments, Worker
from ..base.decorator import _DISPATCH_DECORATOR_ATTR


__all__ = ["Worker"]


def generate_random_string(length: int) -> str:
    letters_digits = string.ascii_letters + string.digits
    return "".join(random.choice(letters_digits) for _ in range(length))


def create_function_generator(self, method_name, dispatch_fn, collect_fn, execute_fn, blocking):
    def func(*args, **kwargs):
        args, kwargs = dispatch_fn(self, *args, **kwargs)
        output = execute_fn(method_name, *args, **kwargs)
        if blocking:
            output = ray.get(output)
        output = collect_fn(self, output)
        return output

    return func


def sort_placement_groups_by_node_ip(placement_groups: list[PlacementGroup]) -> list[PlacementGroup]:
    """
    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in placement_groups:
        specs = ray._private.state.state.placement_group_table(pg.id)
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]

    return sorted(placement_groups, key=lambda pg: pg_ip[pg.id])


class RayComputeResourcePool(ComputeResourcePool):
    def __init__(
        self,
        processes_per_node: list[int] = None,
        use_gpu: bool = True,
        name_prefix_string: str = "",
        max_colocation_count: int = 5,
        detached: bool = False,
    ) -> None:
        super().__init__(processes_per_node, max_colocation_count)
        self.use_gpu = use_gpu
        self.name_prefix_string = name_prefix_string
        self.pgs = None
        self.detached = detached

    def get_placement_group_configurations(self, strategy: str = "STRICT_PACK", name: Optional[str] = None) -> list[PlacementGroup]:
        if self.pgs is not None:
            return self.pgs

        pg_name_prefix = (
            name if name else f"{self.name_prefix_string}mtrl_group_{'_'.join([str(count) for count in self._store])}:"
        )
        pg_scheme = [
            [
                {"CPU": self.max_colocation_count, "GPU": 1} if self.use_gpu else {"CPU": self.max_colocation_count}
                for _ in range(process_count)
            ]
            for process_count in self._store
        ]

        lifetime = "detached" if self.detached else None

        pgs = [
            placement_group(bundles=bundles, strategy=strategy, name=pg_name_prefix + str(idx), lifetime=lifetime)
            for idx, bundles in enumerate(pg_scheme)
        ]

        ray.get([pg.ready() for pg in pgs])

        self.pgs = pgs
        return pgs


def extract_placement_groups_from_existing(
    resource_pools: dict[str, RayComputeResourcePool], src_role_names: list[str], resource_pool: RayComputeResourcePool
) -> list[PlacementGroup]:
    src_pgs = [
        pg
        for role_name, resource_pool in resource_pools.items()
        for pg in resource_pool.get_placement_group_configurations()
        if role_name in src_role_names
    ]

    sorted_src_pgs = sorted(src_pgs, key=lambda pg: pg.bundle_count, reverse=True)
    sorted_process_on_nodes = sorted([(val, idx) for idx, val in enumerate(resource_pool.store)], reverse=True)

    unsorted_pgs: list[tuple[int, PlacementGroup]] = []
    searching_idx = 0
    for request_process, original_idx in sorted_process_on_nodes:
        assert searching_idx < len(sorted_src_pgs), f"no enough nodes for request: searching {searching_idx} th node"
        assert request_process <= sorted_src_pgs[searching_idx].bundle_count, (
            f"requesting {request_process} processes, bundle count cannot satisfy"
        )
        unsorted_pgs.append((original_idx, sorted_src_pgs[searching_idx]))
        searching_idx += 1

    return [pg for _, pg in sorted(unsorted_pgs)]


def merge_compute_resource_pools(rp1: RayComputeResourcePool, rp2: RayComputeResourcePool) -> RayComputeResourcePool:
    assert rp1.use_gpu == rp2.use_gpu, "Both RayComputeResourcePool must either use_gpu or not"
    assert rp1.max_colocation_count == rp2.max_colocation_count, (
        "Both RayComputeResourcePool must has the same max_colocation_count"
    )
    assert rp1.n_gpus_per_node == rp2.n_gpus_per_node, "Both RayComputeResourcePool must has the same n_gpus_per_node"
    assert rp1.detached == rp2.detached, "Detached ResourcePool cannot be merged with non-detached ResourcePool"

    new_store = rp1.store + rp2.store

    merged = RayComputeResourcePool(new_store, rp1.use_gpu, f"{rp1.name_prefix_string}_{rp2.name_prefix_string}")
    merged.pgs = rp1.get_placement_group_configurations() + rp2.get_placement_group_configurations()

    return merged


class RayInitializationArguments(InitializationArguments):
    def __init__(self, cls, *args, **kwargs) -> None:
        super().__init__(cls, *args, **kwargs)
        self._options = {}
        self._additional_resource = {}

    def set_additional_resources(self, additional_resource):
        self._additional_resource = additional_resource

    def update_runtime_options(self, options: dict):
        self._options.update(options)

    def __call__(
        self,
        placement_group: PlacementGroup,
        placement_group_bundle_index: int,
        use_gpu: bool = True,
        num_gpus: int = 1,
        shared_with_worker: Worker = None,
    ) -> Any:
        if shared_with_worker is not None:
            try:
                target_node_id = ray.get(shared_with_worker.get_node_id.remote())
            except AttributeError:
                target_node_id = None
            cuda_visible_devices = ray.get(shared_with_worker.get_cuda_device_visibility.remote())
            if target_node_id is not None:
                options = {"scheduling_strategy": NodeAffinitySchedulingStrategy(node_id=target_node_id, soft=False)}
                return self.cls.options(**options).remote(
                    *self.args, cuda_visible_devices=cuda_visible_devices, **self.kwargs
                )
            else:
                return self.cls.options(**{}).remote(
                    *self.args, cuda_visible_devices=cuda_visible_devices, **self.kwargs
                )

        options = {
            "scheduling_strategy": PlacementGroupSchedulingStrategy(
                placement_group=placement_group, placement_group_bundle_index=placement_group_bundle_index
            )
        }
        options.update(self._options)

        if use_gpu:
            options["num_gpus"] = num_gpus

        if len(self._additional_resource) > 1:
            for k, v in self._additional_resource.items():
                options[k] = v

        return self.cls.options(**options).remote(*self.args, **self.kwargs)


class RayDistributedWorkerGroup(DistributedWorkerGroup):
    def __init__(
        self,
        resource_pool: RayComputeResourcePool = None,
        ray_class_with_initialization: RayInitializationArguments = None,
        enable_bin_packing: bool = True,
        name_prefix_string: str = None,
        detached: bool = False,
        worker_name_list: list[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(resource_pool=resource_pool, **kwargs)
        self.ray_class_with_initialization = ray_class_with_initialization
        self.name_prefix_string = generate_random_string(length=6) if name_prefix_string is None else name_prefix_string

        if worker_name_list is not None:
            assert self._is_init_with_detached_workers
            self._worker_names = worker_name_list

        if self._is_init_with_detached_workers:
            self._initialize_with_detached_workers(worker_names=worker_name_list)
        else:
            self._initialize_with_resource_pool(
                resource_pool=resource_pool, ray_class_with_initialization=ray_class_with_initialization, enable_bin_packing=enable_bin_packing, detached=detached
            )

        if ray_class_with_initialization is not None:
            self._bind_worker_methods(self.ray_class_with_initialization.cls, create_function_generator)

    def _check_worker_aliveness(self, worker: ActorHandle) -> bool:
        worker_state_dict = get_actor(worker._actor_id.hex())
        return worker_state_dict.get("state", "undefined") == "ALIVE" if worker_state_dict is not None else False

    def _initialize_with_detached_workers(self, worker_names: list[str]) -> None:
        workers = [ray.get_actor(name=name) for name in worker_names]
        self._workers = workers
        self._world_size = len(worker_names)

    def _initialize_with_resource_pool(
        self, resource_pool: RayComputeResourcePool, ray_class_with_initialization: RayInitializationArguments, enable_bin_packing: bool, detached: bool
    ):
        use_gpu = resource_pool.use_gpu

        strategy = "PACK"
        if enable_bin_packing:
            strategy = "STRICT_PACK"

        pgs = resource_pool.get_placement_group_configurations(strategy=strategy)
        world_size = resource_pool.world_size
        self._world_size = world_size
        num_gpus = 1 / resource_pool.max_colocation_count

        rank = -1
        local_world_size = resource_pool.store[0]
        for pg_idx, pg in enumerate(sort_placement_groups_by_node_ip(pgs)):
            assert local_world_size <= pg.bundle_count, f"when generating for {self.name_prefix_string}, for the "
            for local_rank in range(local_world_size):
                rank += 1

                env_vars = {
                    "WORLD_SIZE": str(world_size),
                    "RANK": str(rank),
                    "WG_PREFIX": self.name_prefix_string,
                    "WG_BACKEND": "ray",
                    "RAY_LOCAL_WORLD_SIZE": str(local_world_size),
                    "RAY_LOCAL_RANK": str(local_rank),
                }
                if rank != 0:
                    env_vars["MASTER_ADDR"] = self._master_addr
                    env_vars["MASTER_PORT"] = self._master_port

                cia_name = type(ray_class_with_initialization.cls).__name__
                match = re.search(r"ActorClass\(([^)]+)\)", cia_name)
                cia_name = match.group(1) if match else cia_name
                name = f"{self.name_prefix_string}{cia_name}_{pg_idx}:{local_rank}"

                ray_class_with_initialization.update_runtime_options({"runtime_env": {"env_vars": env_vars}, "name": name})

                if detached:
                    ray_class_with_initialization.update_runtime_options({"lifetime": "detached"})

                worker = ray_class_with_initialization(
                    placement_group=pg, placement_group_bundle_index=local_rank, use_gpu=use_gpu, num_gpus=num_gpus
                )
                self._workers.append(worker)
                self._worker_names.append(name)

                if rank == 0:
                    register_center_actor = None
                    for _ in range(120):
                        if f"{self.name_prefix_string}_registry" not in list_named_actors():
                            time.sleep(1)
                        else:
                            register_center_actor = ray.get_actor(f"{self.name_prefix_string}_registry")
                            break
                    assert register_center_actor is not None, (
                        f"failed to get register_center_actor: {self.name_prefix_string}_registry in {list_named_actors(all_namespaces=True)}"
                    )
                    rank_zero_information = ray.get(register_center_actor.get_rank_zero_information.remote())
                    self._master_addr, self._master_port = rank_zero_information["MASTER_ADDR"], rank_zero_information["MASTER_PORT"]

    @property
    def worker_names(self):
        return self._worker_names

    @classmethod
    def create_from_detached_workers(cls, worker_names=None, ray_class_with_initialization=None):
        worker_group = cls(
            resource_pool=None, ray_class_with_initialization=ray_class_with_initialization, name_prefix_string=None, worker_name_list=worker_names
        )
        return worker_group

    def spawn_worker_groups(self, prefix_set):
        """
        spawn to a dictionary of worker groups, each with a subset of method with prefix.
        """

        def _rebind_actor_method_implementations(worker_group, actor_name):
            """
            bind the method with actor_prefix to its original name
            """
            prefix: str = actor_name + "_"
            for method_name in dir(worker_group):
                if method_name.startswith(prefix):
                    original_method_name = method_name.removeprefix(prefix)
                    method = getattr(worker_group, method_name)
                    setattr(worker_group, original_method_name, method)

        new_worker_group_dict = {}
        for prefix in prefix_set:
            new_worker_group = self.create_from_detached_workers(
                worker_names=self._worker_names, ray_class_with_initialization=self.ray_class_with_initialization
            )

            _rebind_actor_method_implementations(new_worker_group, prefix)
            new_worker_group_dict[prefix] = new_worker_group
        return new_worker_group_dict

    def execute_on_rank_zero_synchronously(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_on_rank_zero_asynchronously(method_name, *args, **kwargs))

    def execute_on_rank_zero_asynchronously(self, method_name: str, *args, **kwargs):
        remote_call = getattr(self._workers[0], method_name)
        return remote_call.remote(*args, **kwargs)

    def execute_rank_zero(self, method_name: str, *args, **kwargs):
        return self.execute_on_rank_zero_asynchronously(method_name, *args, **kwargs)

    def execute_all(self, method_name: str, *args, **kwargs):
        return self.execute_on_all_workers_asynchronously(method_name, *args, **kwargs)

    def execute_on_all_workers_synchronously(self, method_name: str, *args, **kwargs):
        return ray.get(self.execute_on_all_workers_asynchronously(method_name, *args, **kwargs))

    def execute_on_all_workers_asynchronously(self, method_name: str, *args, **kwargs):
        length = len(self._workers)
        if all(isinstance(arg, list) for arg in args) and all(isinstance(kwarg, list) for kwarg in kwargs.values()):
            if all(len(arg) == length for arg in args) and all(len(kwarg) == length for kwarg in kwargs.values()):
                result = []
                for i in range(length):
                    sliced_args = tuple(arg[i] for arg in args)
                    sliced_kwargs = {k: v[i] for k, v in kwargs.items()}
                    remote_call = getattr(self._workers[i], method_name)
                    result.append(remote_call.remote(*sliced_args, **sliced_kwargs))
                return result

        return [getattr(worker, method_name).remote(*args, **kwargs) for worker in self._workers]

    @property
    def master_address(self):
        return self._master_addr

    @property
    def master_port(self):
        return self._master_port

    @property
    def workers(self):
        return self._workers

    @property
    def world_size(self):
        return self._world_size


def _bind_worker_methods_to_parent_class(cls, key, user_defined_cls):
    """
    Binds the methods of each worker to the WorkerDict.
    Note that we only bind public methods that are decorated by register
    """
    for method_name in dir(user_defined_cls):
        try:
            method = getattr(user_defined_cls, method_name)
            assert callable(method), f"{method_name} in {user_defined_cls} is not callable"
        except Exception:
            continue

        if hasattr(method, _DISPATCH_DECORATOR_ATTR):

            def generate_function(name):
                def func(self, *args, **kwargs):
                    return getattr(self.worker_dict[key], name)(*args, **kwargs)

                return func

            func = generate_function(method_name)
            setattr(func, _DISPATCH_DECORATOR_ATTR, getattr(method, _DISPATCH_DECORATOR_ATTR))
            try:
                method_name_with_prefix = key + "_" + method_name
                setattr(cls, method_name_with_prefix, func)
            except Exception:
                raise ValueError(f"Fail to set method_name {method_name}")


def _unwrap_ray_remote_decorator(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


def create_colocated_worker_class(class_dictionary: dict[str, RayInitializationArguments]):
    """
    This function should return a class instance that delegates the calls to every
    cls in cls_dict
    """
    cls_dict = {}
    initialization_arguments_dictionary = {}
    worker_cls = None
    for key, cls in class_dictionary.items():
        if worker_cls is None:
            worker_cls = cls.cls.__ray_actor_class__.__base__
        else:
            assert worker_cls == cls.cls.__ray_actor_class__.__base__, (
                "the worker class should be the same when share the same process"
            )
        cls_dict[key] = cls.cls
        initialization_arguments_dictionary[key] = {"args": cls.args, "kwargs": cls.kwargs}

    assert cls_dict.keys() == initialization_arguments_dictionary.keys()

    class WorkerDict(worker_cls):
        def __init__(self):
            super().__init__()
            self.worker_dict = {}
            for key, user_defined_cls in cls_dict.items():
                user_defined_cls = _unwrap_ray_remote_decorator(user_defined_cls)
                with patch.dict(os.environ, {"DISABLE_WORKER_INIT": "1"}):
                    self.worker_dict[key] = user_defined_cls(
                        *initialization_arguments_dictionary[key].get("args", ()), **initialization_arguments_dictionary[key].get("kwargs", {})
                    )

    for key, user_defined_cls in cls_dict.items():
        user_defined_cls = _unwrap_ray_remote_decorator(user_defined_cls)
        _bind_worker_methods_to_parent_class(WorkerDict, key, user_defined_cls)

    remote_cls = ray.remote(WorkerDict)
    remote_cls = RayInitializationArguments(cls=remote_cls)
    return remote_cls
