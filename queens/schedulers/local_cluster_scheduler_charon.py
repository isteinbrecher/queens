"""Cluster scheduler for QUEENS runs."""

import logging
import time
from datetime import timedelta

from dask.distributed import Client

from queens.schedulers.cluster_scheduler import VALID_WORKLOAD_MANAGERS, timedelta_to_str
from queens.schedulers.dask_scheduler import DaskScheduler
from queens.utils.config_directories import experiment_directory  # Do not change this import!
from queens.utils.logger_settings import log_init_args
from queens.utils.remote_operations import get_port
from queens.utils.rsync import rsync
from queens.utils.valid_options_utils import get_option
from dask_jobqueue import SLURMCluster

_logger = logging.getLogger(__name__)


class CharonLocalCluster(DaskScheduler):
    """Local Cluster scheduler for QUEENS.

    Can be used to schedule jobs to a cluster scheduler with local
    access i.e. without a network connection.
    """

    @log_init_args
    def __init__(
        self,
        experiment_name,*,
        env={},
        n_processors=None,
        n_nodes=None,
        n_processors_per_node=None,
        n_dask_workers=None,
        walltime=None,
        num_jobs=1,
        min_jobs=0,
        num_procs=1,
        num_nodes=1,
        queue=None,
        cluster_internal_address=None,
        restart_workers=False,
        allowed_failures=5,
        **kwargs
    ):
        """Init method for the cluster scheduler.

        The total number of cores per job is given by num_procs*num_nodes.

        Args:
            experiment_name (str): name of the current experiment
            workload_manager (str): Workload manager ("pbs" or "slurm")
            walltime (str): Walltime for each worker job. Format (hh:mm:ss)
            num_jobs (int, opt): Maximum number of parallel jobs
            min_jobs (int, opt): Minimum number of active workers for the cluster
            num_procs (int, opt): Number of processors per job per node
            num_nodes (int, opt): Number of cluster nodes per job
            queue (str, opt): Destination queue for each worker job
            cluster_internal_address (str, opt): Internal address of cluster
            restart_workers (bool): If true, restart workers after each finished job. For larger
                                    jobs (>1min) this should be set to true in most cases.
            allowed_failures (int): Number of allowed failures for a task before an error is raised
        """

        n_dask_workers = num_jobs
        n_processors = num_procs

        experiment_dir = experiment_directory(experiment_name=experiment_name)
        _logger.debug(
            "experiment directory: %s",
            experiment_dir,
        )

        # Check processor count input
        if n_processors is not None and (
            n_nodes is not None or n_processors_per_node is not None
        ):
            raise ValueError(
                "The arguments n_processors and (n_nodes, n_processors_per_node) are mutually exclusive"
            )
        elif n_processors is None and (
            n_nodes is None or n_processors_per_node is None
        ):
            raise ValueError(
                "you have to specify both arguments (n_nodes, n_processors_per_node)"
            )

        # Define modules needed for the SLURM worker
        job_script_prologue = [
            "source /etc/profile.d/modules.sh",
            "source /home/opt/cluster_tools/core/load_baci_environment.sh",
        ]

        # Export all variables in env, so the worker will have access to them
        for key, item in env.items():
            job_script_prologue.append(f'export {key}="{item}"')

        # Add SLURM options
        job_extra_directives = []
        if n_processors is not None:
            job_extra_directives.append(f"--ntasks={n_processors}")
        else:
            job_extra_directives.append(f"--nodes={n_nodes}")
            job_extra_directives.append(f"--ntasks-per-node={n_processors_per_node}")

        # Setup the cluster
        cluster = SLURMCluster(
            cores=1,  # Set cores and process to one, because we only want one python process per job
            processes=1,
            memory="100GB",  # Temp value, is required but will be overwritten and not used
            job_directives_skip=[
                "--cpus-per-task",  # Skip the line where the number of cores is equal to 1
                "-n",  # We specify the number of cores manually
                "--mem",  # Memory allocation is not supported by SLURM on charon
            ],
            job_extra_directives=job_extra_directives,
            job_script_prologue=job_script_prologue,
            job_name=experiment_name + "_worker",
            interface="ib0",  # Use the infiniband interface
            worker_extra_args=(
                ["--memory-limit", "auto"]
            ),  # Since we use an arbirtrary memory above, we redefine the worker memory to auto here
            log_directory= str(experiment_dir),
            **kwargs,
        )
        
        try:
         

            dask_jobscript = experiment_dir / "dask_jobscript.sh"
            _logger.info("Writing dask jobscript to:")
            _logger.info(dask_jobscript)
            dask_jobscript.write_text(str(cluster.job_script()))
        except Exception as e:
            raise RuntimeError() from e

        for i in range(20, 0, -1):  # 20 tries to connect
            _logger.debug("Trying to connect to Dask Cluster: try #%d", i)
            try:
                # client = Client(address=f"localhost:{local_port}", timeout=10)
                cluster.scale(jobs=n_dask_workers)
                client = Client(cluster)
                break
            except OSError as exc:
                if i == 1:
                    raise OSError() from exc
                time.sleep(1)

        _logger.debug("Submitting dummy job to check basic functionality of client.")
        client.submit(lambda: "Dummy job").result(timeout=180)
        _logger.debug("Dummy job was successful.")
        _logger.info(
            "To view the Dask dashboard open this link in your browser: "
            "http://localhost:%i/status",
            "local_port_dashboard",
        )

        super().__init__(
            experiment_name=experiment_name,
            experiment_dir=experiment_dir,
            num_jobs=num_jobs,
            num_procs=num_procs,
            client=client,
            restart_workers=restart_workers,
        )

    def restart_worker(self, worker):
        """Restart a worker.

        This method retires a dask worker. The Client.adapt method of dask takes cares of submitting
        new workers subsequently.

        Args:
            worker (str, tuple): Worker to restart. This can be a worker address, name, or a both.
        """
        self.client.retire_workers(workers=list(worker))

    def copy_files_to_experiment_dir(self, paths):
        """Copy file to experiment directory.

        Args:
            paths (Path, list): paths to files or directories that should be copied to experiment
                                directory
        """
        destination = f"{self.experiment_dir}/"
        rsync(paths, destination)
