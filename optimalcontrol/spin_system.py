"""Spin-system dataclasses for two-spin NMR optimal-control models."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optimalcontrol.operators import Ix, Iy, Iz, place_operator

GAMMA_1H_RAD_S_T = 267.52218744e6
"""Approximate 1H gyromagnetic ratio in rad/s/T."""

GAMMA_13C_RAD_S_T = 67.2828402e6
"""Approximate 13C gyromagnetic ratio in rad/s/T."""

TWO_PI = 2.0 * np.pi


@dataclass
class Spin:
    """Single spin description.

    Attributes:
        isotope: Isotope or abstract spin label.
        gamma: Gyromagnetic ratio in rad/s/T.
        channel: RF/control channel label.
    """

    isotope: str
    gamma: float
    channel: str


@dataclass
class Coupling:
    """Scalar coupling between two spins.

    Attributes:
        spin_i: Index of the first coupled spin.
        spin_j: Index of the second coupled spin.
        J_hz: Scalar coupling in Hz.
    """

    spin_i: int
    spin_j: int
    J_hz: float


@dataclass
class RelaxationRates:
    """Relaxation and cross-correlation rates, all stored in rad/s."""

    kDD: float
    kCSA_I: float
    kCSA_S: float
    ka: float
    kc: float
    ka_prime: float
    kc_prime: float

    @property
    def kI(self) -> float:
        """Total transverse relaxation rate of the I/source spin in rad/s."""
        return self.kDD + self.kCSA_I

    @property
    def kS(self) -> float:
        """Total transverse relaxation rate of the S/target spin in rad/s."""
        return self.kDD + self.kCSA_S


@dataclass
class SpinSystem:
    """Container for spin, coupling, shift, relaxation, and basis metadata."""

    spins: list[Spin]
    couplings: list[Coupling]
    shifts_hz: dict[int, float]
    relaxation: RelaxationRates
    basis: str = "dense"


def two_spin_system(
    J_hz: float,
    kDD: float,
    kCSA_I: float,
    kCSA_S: float,
    ka: float,
    kc: float,
    ka_prime: float,
    kc_prime: float,
) -> SpinSystem:
    """Return the on-resonance heteronuclear I-S system used in paper examples.

    Args:
        J_hz: Scalar coupling between spins 0 (I/source) and 1 (S/target), in Hz.
        kDD: Dipole-dipole relaxation rate in rad/s.
        kCSA_I: CSA relaxation contribution for the I/source spin, in rad/s.
        kCSA_S: CSA relaxation contribution for the S/target spin, in rad/s.
        ka: DD-CSA cross-correlation rate for I longitudinal terms, in rad/s.
        kc: DD-CSA cross-correlation rate for I transverse terms, in rad/s.
        ka_prime: DD-CSA cross-correlation rate for S longitudinal terms, in rad/s.
        kc_prime: DD-CSA cross-correlation rate for S transverse terms, in rad/s.
    """
    return SpinSystem(
        spins=[
            Spin(isotope="1H", gamma=GAMMA_1H_RAD_S_T, channel="I"),
            Spin(isotope="13C", gamma=GAMMA_13C_RAD_S_T, channel="S"),
        ],
        couplings=[Coupling(spin_i=0, spin_j=1, J_hz=J_hz)],
        shifts_hz={0: 0.0, 1: 0.0},
        relaxation=RelaxationRates(
            kDD=kDD,
            kCSA_I=kCSA_I,
            kCSA_S=kCSA_S,
            ka=ka,
            kc=kc,
            ka_prime=ka_prime,
            kc_prime=kc_prime,
        ),
    )


def _validate_spin_index(spin_index: int, n_spins: int) -> None:
    """Raise ValueError if spin_index is outside the spin system."""
    if spin_index < 0 or spin_index >= n_spins:
        raise ValueError(f"spin_index {spin_index} is outside system with {n_spins} spins")


def _zero_hamiltonian(n_spins: int) -> npt.NDArray[np.complex128]:
    """Return a zero Hilbert-space operator for an n-spin-1/2 system."""
    dim = 2**n_spins
    return np.zeros((dim, dim), dtype=np.complex128)


def drift_hamiltonian(sys: SpinSystem) -> npt.NDArray[np.complex128]:
    """Build the scalar J-coupling drift Hamiltonian in rad/s.

    Uses H_J = sum 2*pi*J_hz*Iz_i*Iz_j, where public J values are supplied in Hz.
    """
    n_spins = len(sys.spins)
    H = _zero_hamiltonian(n_spins)
    for coupling in sys.couplings:
        _validate_spin_index(coupling.spin_i, n_spins)
        _validate_spin_index(coupling.spin_j, n_spins)
        if coupling.spin_i == coupling.spin_j:
            raise ValueError("coupling spin indices must be distinct")
        Iz_i = place_operator(Iz(), coupling.spin_i, n_spins)
        Iz_j = place_operator(Iz(), coupling.spin_j, n_spins)
        H += (TWO_PI * coupling.J_hz) * (Iz_i @ Iz_j)
    return H


def shift_hamiltonian(sys: SpinSystem) -> npt.NDArray[np.complex128]:
    """Build the chemical-shift Hamiltonian in rad/s.

    Uses H_CS = 2*pi*sum(delta_hz_i*Iz_i), where public shifts are supplied in Hz.
    """
    n_spins = len(sys.spins)
    H = _zero_hamiltonian(n_spins)
    for spin_index, delta_hz in sys.shifts_hz.items():
        _validate_spin_index(spin_index, n_spins)
        H += (TWO_PI * delta_hz) * place_operator(Iz(), spin_index, n_spins)
    return H


def control_operators(sys: SpinSystem) -> dict[str, npt.NDArray[np.complex128]]:
    """Return two-spin RF control operators for I and S channels."""
    n_spins = len(sys.spins)
    if n_spins != 2:
        raise ValueError("control_operators currently supports exactly two spins")
    return {
        "Ix": place_operator(Ix(), 0, n_spins),
        "Iy": place_operator(Iy(), 0, n_spins),
        "Sx": place_operator(Ix(), 1, n_spins),
        "Sy": place_operator(Iy(), 1, n_spins),
    }
