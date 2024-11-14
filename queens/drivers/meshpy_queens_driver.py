# -*- coding: utf-8 -*-
"""Driver for Queens to run a general MeshPy example"""

import os
import logging

from queens.drivers import MpiDriver
from queens.utils.metadata import SimulationMetadata
from queens.utils.logger_settings import log_init_args


class MeshPyDriver(MpiDriver):
    """Driver to run a generic MeshPy/CubitPy run"""

    @log_init_args
    def __init__(self, meshpy_model, *, parameters=None, **kwargs):
        """Initialize MeshPyDriver object and store the function to be evaluated.

        Args
        ----
        meshpy_model: function
            MeshPy based function to be evaluated
        four_c_executable: str
            Path to the executable of 4C
        mpi_cmd: str
            Command to be used to call mpi on the system
        """

        super().__init__(parameters, input_template="", **kwargs)

        # Store members and set env variables for MeshPy run_four_c function
        self.meshpy_model = meshpy_model
        os.environ["MESHPY_FOUR_C_EXE"] = self.jobscript_options["executable"]
        os.environ["MESHPY_MPI_COMMAND"] = self.jobscript_options["mpi_cmd"]

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Evaluate the given model"""

        # Setup the simulation directory
        job_dir, output_dir_from_queens, _, _, _, _ = self._manage_paths(
            job_id, experiment_dir, experiment_name
        )
        output_dir_from_queens.rmdir()

        # Sample with dictionary
        if self.parameters is None:
            sample_dict = {"sample": sample}
        else:
            sample_dict = self.parameters.sample_as_dict(sample)

        # Setup the queens metadata
        metadata = SimulationMetadata(
            job_id=job_id, inputs=sample_dict, job_dir=job_dir
        )

        # Set environment variables required to run 4C
        os.environ["MESHPY_MPI_NUM_PROC"] = str(num_procs)
        os.environ["QUEENS_JOB_ID"] = str(job_id)

        # Setup the logger for MeshPy
        # TODO: Would be nice if we dont have to pass this but could access it directly in the calling function
        meshpy_logger = logging.getLogger("MeshPyDriverLogger")
        meshpy_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(
            os.path.join(job_dir, "meshpy_driver_logger.log")
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        meshpy_logger.addHandler(file_handler)

        # Run the forward model
        with metadata.time_code("run_forward_model"):
            result = self.meshpy_model(meshpy_logger, job_dir, sample)
            metadata.outputs = result

        # Finish the logger
        for handler in meshpy_logger.handlers:
            handler.flush()
            handler.close()
        meshpy_logger.handlers.clear()

        return result, None
