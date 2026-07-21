#
# SPDX-License-Identifier: LGPL-3.0-or-later
# Copyright (c) 2024-2025, QUEENS contributors.
#
# This file is part of QUEENS.
#
# QUEENS is free software: you can redistribute it and/or modify it under the terms of the GNU
# Lesser General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version. QUEENS is distributed in the hope that it will
# be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details. You
# should have received a copy of the GNU Lesser General Public License along with QUEENS. If not,
# see <https://www.gnu.org/licenses/>.
#
"""Function Driver which passes the number of processors to the function."""

from pathlib import Path
import numpy as np
from queens.drivers.function import Function

class FunctionMpi(Function):
    """Driver to run an python function and pass the number of processors to the function."""

    def run(
        self,
        sample: np.ndarray,
        job_id: int,
        num_procs: int,
        experiment_dir: Path,
        experiment_name: str,
    ) -> dict:
        """This is a simply copy of the original method, only with the addition of the num_procs argument to the function call."""
        sample_dict = self.parameters.sample_as_dict(sample)
        if self.function_requires_job_id:
            sample_dict["job_id"] = job_id
        sample_dict["experiment_dir"] = experiment_dir
        sample_dict["experiment_name"] = experiment_name
        sample_dict["num_procs"] = num_procs
        results = self.function(sample_dict)
        return results
