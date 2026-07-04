"""Shared numpy array type aliases used across the package."""

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.complex128]
RealArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
