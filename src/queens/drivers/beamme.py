"""Driver for Queens to run a general BeamMe example."""

import logging
import os

from queens.drivers import Mpi
from queens.utils.logger_settings import log_init_args
from queens.utils.metadata import SimulationMetadata


class BeamMeDriver(Mpi):
    """Driver to run a generic BeamMe/CubitPy run."""

    @log_init_args
    def __init__(self, beamme_model, parameters=None, **kwargs):
        """Initialize MeshPyDriver object and store the function to be evaluated.

        Args
        ----
        beamme_model: function
            BeamMe based function to be evaluated
        four_c_executable: str
            Path to the executable of 4C
        mpi_cmd: str
            Command to be used to call mpi on the system
        """

        super().__init__(parameters, input_templates="", **kwargs)

        # Store members and set env variables for MeshPy run_four_c function
        self.beamme_model = beamme_model
        if "BEAMME_FOUR_C_EXE" not in os.environ:
            os.environ["BEAMME_FOUR_C_EXE"] = self.jobscript_options["executable"]
        if "BEAMME_MPI_COMMAND" not in os.environ:
            os.environ["BEAMME_MPI_COMMAND"] = self.jobscript_options["mpi_cmd"]

    def run(self, sample, job_id, num_procs, experiment_dir, experiment_name):
        """Evaluate the given model"""

        # Setup the simulation directory
        job_dir, output_dir_from_queens, _, _, _ = self._manage_paths(
            job_id, experiment_dir
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
        os.environ["BEAMME_MPI_NUM_PROC"] = str(num_procs)
        os.environ["QUEENS_JOB_ID"] = str(job_id)

        # Setup the logger for MeshPy
        # TODO: Would be nice if we dont have to pass this but could access it directly in the calling function
        beamme_logger = logging.getLogger("BeamMeDriverLogger")
        beamme_logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(
            os.path.join(job_dir, "beamme_driver_logger.log")
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        beamme_logger.addHandler(file_handler)

        # Run the forward model
        with metadata.time_code("run_forward_model"):
            keys = list(sample_dict.keys())
            keys.sort()
            q = [sample_dict[key] for key in keys]
            result = self.beamme_model(beamme_logger, job_dir, q)
            metadata.outputs = result

        # Finish the logger
        for handler in beamme_logger.handlers:
            handler.flush()
            handler.close()
        beamme_logger.handlers.clear()

        return result, None
