"""Schedulers.

The scheduler package contains a set of scheduler classes which submit
compute jobs either through a job-scheduling software or through a
system call.
"""

from queens.schedulers.cluster_scheduler import ClusterScheduler
from queens.schedulers.local_cluster_scheduler import LocalClusterScheduler
from queens.schedulers.local_cluster_scheduler_charon import CharonLocalCluster
from queens.schedulers.local_scheduler import LocalScheduler
from queens.schedulers.pool_scheduler import PoolScheduler
from queens.schedulers.scheduler import Scheduler

VALID_TYPES = {
    "local": LocalScheduler,
    "cluster": ClusterScheduler,
    "local_cluster": LocalClusterScheduler,
    "local_cluster_charon": CharonLocalCluster,
    "pool": PoolScheduler,
}
