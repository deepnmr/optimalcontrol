"""Spin-system dataclasses and generator builders for two-spin NMR models."""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from optimalcontrol.operators import (
    Ix,
    Iy,
    Iz,
    lindblad_dissipator,
    liouvillian_comm,
    place_operator,
)

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


def _zero_liouvillian(n_spins: int) -> npt.NDArray[np.complex128]:
    """Return a zero Liouville-space operator for an n-spin-1/2 system."""
    dim = 2**n_spins
    dim2 = dim * dim
    return np.zeros((dim2, dim2), dtype=np.complex128)


def _two_spin_z_operators(
    sys: SpinSystem,
) -> tuple[
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
    npt.NDArray[np.complex128],
]:
    """Return (Iz, Sz, 2IzSz) Hilbert-space operators for the two-spin model."""
    n_spins = len(sys.spins)
    if n_spins != 2:
        raise ValueError("relaxation builders currently support exactly two spins")
    Iz_i = place_operator(Iz(), 0, n_spins)
    Sz_s = place_operator(Iz(), 1, n_spins)
    two_IzSz = np.complex128(2.0) * (Iz_i @ Sz_s)
    return Iz_i, Sz_s, two_IzSz


def _has_crosscorr_rates(relaxation: RelaxationRates) -> bool:
    """Return True if any CROP-style effective/cross-correlated rates are set."""
    return any(
        rate != 0.0
        for rate in (
            relaxation.ka,
            relaxation.kc,
            relaxation.ka_prime,
            relaxation.kc_prime,
        )
    )


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


def relaxation_liouvillian(sys: SpinSystem) -> npt.NDArray[np.complex128]:
    """Build the secular relaxation Liouvillian without DD-CSA cross-correlation.

    The non-cross-correlated paper model uses effective transverse relaxation rates
    kI = kDD + kCSA_I and kS = kDD + kCSA_S. These rates are represented as
    pure-dephasing dissipators generated by Iz and Sz, so I-spin transverse terms
    decay with kI and S-spin transverse terms decay with kS.
    """
    if len(sys.spins) == 0:
        return _zero_liouvillian(0)
    Iz_i, Sz_s, _ = _two_spin_z_operators(sys)
    rates = np.array(
        [
            [np.complex128(2.0 * sys.relaxation.kI), np.complex128(0.0)],
            [np.complex128(0.0), np.complex128(2.0 * sys.relaxation.kS)],
        ],
        dtype=np.complex128,
    )
    return lindblad_dissipator([Iz_i, Sz_s], rates)


def relaxation_liouvillian_crosscorr(sys: SpinSystem) -> npt.NDArray[np.complex128]:
    """Build the two-spin relaxation Liouvillian with DD-CSA cross-correlation.

    The CROP notation uses effective autocorrelated rates (ka, ka_prime) and
    cross-correlated rates (kc, kc_prime). If the effective autocorrelated rates
    are left at zero, the non-cross-correlated kI/kS rates are used as the
    baseline. The off-diagonal coefficient terms couple the 2IzSz dephasing axis
    to Iz and Sz, giving single-transition relaxation rates ka +/- kc and
    ka_prime +/- kc_prime.
    """
    if len(sys.spins) == 0:
        return _zero_liouvillian(0)
    Iz_i, Sz_s, two_IzSz = _two_spin_z_operators(sys)
    relaxation = sys.relaxation
    rate_i = relaxation.ka if relaxation.ka != 0.0 else relaxation.kI
    rate_s = relaxation.ka_prime if relaxation.ka_prime != 0.0 else relaxation.kS
    coeffs = np.array(
        [
            [
                np.complex128(2.0 * rate_i),
                np.complex128(0.0),
                np.complex128(relaxation.kc),
            ],
            [
                np.complex128(0.0),
                np.complex128(2.0 * rate_s),
                np.complex128(relaxation.kc_prime),
            ],
            [
                np.complex128(relaxation.kc),
                np.complex128(relaxation.kc_prime),
                np.complex128(0.0),
            ],
        ],
        dtype=np.complex128,
    )
    return lindblad_dissipator([Iz_i, Sz_s, two_IzSz], coeffs)


def total_generator(sys: SpinSystem, controls: dict[str, float]) -> npt.NDArray[np.complex128]:
    """Assemble the Liouville-space generator for drift, relaxation, and controls.

    Control amplitudes are assumed to be in rad/s and multiply the dimensionless
    control spin operators returned by control_operators().
    """
    H_drift = drift_hamiltonian(sys) + shift_hamiltonian(sys)
    generator = liouvillian_comm(H_drift)
    if _has_crosscorr_rates(sys.relaxation):
        generator += relaxation_liouvillian_crosscorr(sys)
    else:
        generator += relaxation_liouvillian(sys)

    available_controls = control_operators(sys)
    for name, amplitude in controls.items():
        if name not in available_controls:
            available = ", ".join(sorted(available_controls))
            raise ValueError(f"Unknown control {name!r}; expected one of: {available}")
        generator += amplitude * liouvillian_comm(available_controls[name])
    return generator


def validate_onresonance(sys: SpinSystem) -> None:
    """Raise ValueError if any chemical shift is non-zero for an on-resonance model."""
    for spin_index, shift_hz in sys.shifts_hz.items():
        _validate_spin_index(spin_index, len(sys.spins))
        if shift_hz != 0.0:
            raise ValueError(
                f"spin {spin_index} has non-zero chemical shift {shift_hz} Hz; "
                "on-resonance model requires all shifts to be zero"
            )
