
from enum import Enum, auto
from functools import wraps
from types import FunctionType
from typing import TYPE_CHECKING, Literal, Union

import ray

from ...protocol import DataProto, DataProtoFuture


if TYPE_CHECKING:
    from .worker_group import DistributedWorkerGroup


_DISPATCH_DECORATOR_ATTR = "__mtrl_dispatch_decorator_attr__"


class Dispatch(Enum):
    RANK_ZERO = auto()
    ONE_TO_ALL = auto()
    ALL_TO_ALL = auto()
    DP_COMPUTE = auto()
    DP_COMPUTE_PROTO = auto()
    DP_COMPUTE_PROTO_WITH_FUNC = auto()
    DP_COMPUTE_METRIC = auto()


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1


def _split_data_proto_arguments(chunks: int, *args, **kwargs):
    splitted_args = []
    for arg in args:
        assert isinstance(arg, (DataProto, DataProtoFuture))
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, value in kwargs.items():
        assert isinstance(value, (DataProto, DataProtoFuture))
        splitted_kwargs[key] = value.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all_workers(worker_group: "DistributedWorkerGroup", *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dispatch_all_to_all_workers(worker_group: "DistributedWorkerGroup", *args, **kwargs):
    return args, kwargs


def collect_all_to_all_workers(worker_group: "DistributedWorkerGroup", output):
    return output


def _concatenate_data_proto_or_future(outputs: list[DataProto]) -> DataProto:
    for output in outputs:
        assert type(output) is type(outputs[0])

    output = outputs[0]

    if isinstance(output, DataProto):
        return DataProto.concat(outputs)
    elif isinstance(output, ray.ObjectRef):
        return DataProtoFuture.concat(outputs)
    else:
        raise NotImplementedError


def dispatch_data_parallel_compute(worker_group: "DistributedWorkerGroup", *args, **kwargs):
    for arg in args:
        assert isinstance(arg, (tuple, list)) and len(arg) == worker_group.world_size

    for value in kwargs.values():
        assert isinstance(value, (tuple, list)) and len(value) == worker_group.world_size

    return args, kwargs


def collect_data_parallel_compute(worker_group: "DistributedWorkerGroup", outputs: list[DataProto]) -> list[DataProto]:
    assert len(outputs) == worker_group.world_size
    return outputs


def dispatch_data_parallel_compute_data_proto(worker_group: "DistributedWorkerGroup", *args, **kwargs):
    splitted_args, splitted_kwargs = _split_data_proto_arguments(worker_group.world_size, *args, **kwargs)
    return splitted_args, splitted_kwargs


def dispatch_data_parallel_compute_data_proto_with_function(worker_group: "DistributedWorkerGroup", *args, **kwargs):
    assert type(args[0]) is FunctionType
    splitted_args, splitted_kwargs = _split_data_proto_arguments(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_data_parallel_compute_data_proto(worker_group: "DistributedWorkerGroup", outputs: list[DataProto]) -> DataProto:
    for output in outputs:
        assert isinstance(output, (DataProto, ray.ObjectRef)), f"Expect a DataProto, but got {type(output)}"

    outputs = collect_data_parallel_compute(worker_group, outputs)
    return _concatenate_data_proto_or_future(outputs)


def get_predefined_dispatch_function(dispatch_mode: Dispatch):
    predefined_dispatch_mode_fn = {
        Dispatch.ONE_TO_ALL: {
            "dispatch_fn": dispatch_one_to_all_workers,
            "collect_fn": collect_all_to_all_workers,
        },
        Dispatch.ALL_TO_ALL: {
            "dispatch_fn": dispatch_all_to_all_workers,
            "collect_fn": collect_all_to_all_workers,
        },
        Dispatch.DP_COMPUTE: {
            "dispatch_fn": dispatch_data_parallel_compute,
            "collect_fn": collect_data_parallel_compute,
        },
        Dispatch.DP_COMPUTE_PROTO: {
            "dispatch_fn": dispatch_data_parallel_compute_data_proto,
            "collect_fn": collect_data_parallel_compute_data_proto,
        },
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            "dispatch_fn": dispatch_data_parallel_compute_data_proto_with_function,
            "collect_fn": collect_data_parallel_compute_data_proto,
        },
        Dispatch.DP_COMPUTE_METRIC: {
            "dispatch_fn": dispatch_data_parallel_compute_data_proto,
            "collect_fn": collect_data_parallel_compute,
        },
    }
    return predefined_dispatch_mode_fn[dispatch_mode]


def get_predefined_execute_function(execute_mode: Execute):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {"execute_fn_name": "execute_all"},
        Execute.RANK_ZERO: {"execute_fn_name": "execute_rank_zero"},
    }
    return predefined_execute_mode_fn[execute_mode]


def _check_dispatch_mode(dispatch_mode: Union[Dispatch, dict[Literal["dispatch_fn", "collect_fn"], FunctionType]]):
    assert isinstance(dispatch_mode, (Dispatch, dict)), (
        f"dispatch_mode must be a Dispatch or a Dict. Got {dispatch_mode}"
    )
    if isinstance(dispatch_mode, dict):
        necessary_keys = ["dispatch_fn", "collect_fn"]
        for key in necessary_keys:
            assert key in dispatch_mode, f"key {key} should be in dispatch_mode if it is a dictionary"


def _check_execute_mode(execute_mode: Execute):
    assert isinstance(execute_mode, Execute), f"execute_mode must be a Execute. Got {execute_mode}"


def _materialize_future_objects(*args, **kwargs):
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        new_args.append(arg)

    for key, value in kwargs.items():
        if isinstance(value, DataProtoFuture):
            kwargs[key] = value.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):
        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_future_objects(*args, **kwargs)
            return func(*args, **kwargs)

        attrs = {"dispatch_mode": dispatch_mode, "execute_mode": execute_mode, "blocking": blocking}
        setattr(inner, _DISPATCH_DECORATOR_ATTR, attrs)
        return inner

    return decorator
