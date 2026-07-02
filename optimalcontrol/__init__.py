import numpy

__version__ = "0.3.0"

from optimalcontrol.bloch import propagate_bloch_ensemble
from optimalcontrol.crop import CROPPulse, crop_eta, crop_waveform
from optimalcontrol.operators import Ix, Iy, Iz, comm, liouvillian_comm
from optimalcontrol.optimizers import run_grape
from optimalcontrol.rope import rope_g, rope_waveform
from optimalcontrol.spin_system import SpinSystem, two_spin_system
from optimalcontrol.states import fidelity_real, state_from_label


def set_random_seed(seed: int) -> None:
    """Seed NumPy's global random state for reproducible waveform guesses."""
    numpy.random.seed(seed)


__all__ = [
    "__version__",
    "set_random_seed",
    "Ix",
    "Iy",
    "Iz",
    "comm",
    "liouvillian_comm",
    "SpinSystem",
    "two_spin_system",
    "state_from_label",
    "fidelity_real",
    "rope_g",
    "rope_waveform",
    "crop_eta",
    "crop_waveform",
    "CROPPulse",
    "run_grape",
    "propagate_bloch_ensemble",
]
