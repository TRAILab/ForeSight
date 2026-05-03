"""Ray-distributed worker with per-task GPU allocation.

Like `worker_ray_no_torch.RayDistributedNoTorch`, but registers GPUs with
ray (`num_gpus_per_node`) and stamps every dispatched task with
`num_gpus_per_task` so each ray worker process gets pinned to its own
fraction of a GPU. Lets PDM-Score eval distribute the model forward across
all visible CUDA devices instead of crowding a single one.

Pairs with the yaml at:
  navsim/planning/script/config/common/worker/ray_distributed_torch.yaml
"""

import logging
import os
import dataclasses
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Iterable, List, Optional, Union

import ray
from psutil import cpu_count

from nuplan.planning.utils.multithreading.ray_execution import ray_map
from nuplan.planning.utils.multithreading.worker_pool import Task, WorkerPool, WorkerResources

logger = logging.getLogger(__name__)
logging.getLogger("botocore").setLevel(logging.WARNING)


def _detect_num_gpus() -> int:
    """Honor CUDA_VISIBLE_DEVICES first; fall back to torch.cuda."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        return len([x for x in cvd.split(",") if x.strip() != ""])
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except Exception:
        pass
    return 0


class RayDistributedTorch(WorkerPool):
    """Ray pool that allocates GPU fractions to each task."""

    def __init__(
        self,
        master_node_ip: Optional[str] = None,
        threads_per_node: Optional[int] = None,
        num_gpus_per_node: Optional[int] = None,
        num_gpus_per_task: float = 0.0,
        debug_mode: bool = False,
        log_to_driver: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        logs_subdir: Optional[str] = "logs",
        use_distributed: bool = False,
    ):
        """
        :param threads_per_node: Total ray task slots per node (CPU side).
        :param num_gpus_per_node: GPUs to register with ray. None = autodetect.
        :param num_gpus_per_task: GPU fraction allocated per task (e.g. 0.25
            packs 4 tasks per GPU; 1.0 = exclusive). 0 means CPU-only.
        :param master_node_ip / debug_mode / log_to_driver / use_distributed:
            same as `RayDistributedNoTorch`.
        """
        self._master_node_ip = master_node_ip
        self._threads_per_node = threads_per_node
        self._num_gpus_per_node = num_gpus_per_node
        self._num_gpus_per_task = float(num_gpus_per_task)
        self._local_mode = debug_mode
        self._log_to_driver = log_to_driver
        self._log_dir: Optional[Path] = (
            Path(output_dir) / (logs_subdir or "") if output_dir is not None else None
        )
        self._use_distributed = use_distributed
        super().__init__(self.initialize())

    def initialize(self) -> WorkerResources:
        if ray.is_initialized():
            logger.warning("Ray is already running; shutting down before re-init.")
            ray.shutdown()

        n_cpus = self._threads_per_node or cpu_count(logical=True)
        n_gpus = (
            self._num_gpus_per_node
            if self._num_gpus_per_node is not None
            else _detect_num_gpus()
        )

        if self._master_node_ip and self._use_distributed:
            ray.init(
                address=f"ray://{self._master_node_ip}:10001",
                local_mode=self._local_mode,
                log_to_driver=self._log_to_driver,
            )
            n_nodes = 1
        else:
            ray.init(
                num_cpus=n_cpus,
                num_gpus=n_gpus,
                dashboard_host="0.0.0.0",
                local_mode=self._local_mode,
                log_to_driver=self._log_to_driver,
            )
            n_nodes = 1

        logger.info(
            f"Ray started with num_cpus={n_cpus}, num_gpus={n_gpus}, "
            f"per-task num_gpus={self._num_gpus_per_task}"
        )
        return WorkerResources(
            number_of_nodes=n_nodes,
            number_of_cpus_per_node=n_cpus,
            number_of_gpus_per_node=n_gpus,
        )

    def shutdown(self) -> None:
        ray.shutdown()

    def _stamp_gpus(self, task: Task) -> Task:
        """Set num_gpus on a task if not already set; preserves explicit
        per-call overrides from the caller."""
        if self._num_gpus_per_task > 0 and (task.num_gpus is None or task.num_gpus == 0):
            return dataclasses.replace(task, num_gpus=self._num_gpus_per_task)
        return task

    def _map(self, task: Task, *item_lists: Iterable[List[Any]], verbose: bool = False) -> List[Any]:
        del verbose
        return ray_map(self._stamp_gpus(task), *item_lists, log_dir=self._log_dir)  # type: ignore

    def submit(self, task: Task, *args: Any, **kwargs: Any) -> Future[Any]:
        task = self._stamp_gpus(task)
        remote_fn = ray.remote(task.fn).options(
            num_gpus=task.num_gpus, num_cpus=task.num_cpus,
        )
        object_ids: ray._raylet.ObjectRef = remote_fn.remote(*args, **kwargs)
        return object_ids.future()  # type: ignore
