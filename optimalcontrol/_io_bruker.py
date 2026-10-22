"""Bruker shape-file export helpers."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from optimalcontrol._io_common import Waveform, _format_float, _output_path, xy_channel_indices
from optimalcontrol._types import RealArray


def _bruker_amplitude_phase_deg(wfm: Waveform) -> tuple[RealArray, RealArray]:
    """Return amplitude and degree phase arrays for minimal Bruker export."""
    if len(wfm.channels) == 1:
        amplitude = np.asarray(wfm.data[0, :], dtype=np.float64)
        phase_deg = np.zeros_like(amplitude, dtype=np.float64)
        return amplitude, phase_deg

    if len(wfm.channels) != 2:
        raise ValueError(
            f"Bruker amplitude/phase export supports 1 or 2 (x/y) channels, got {len(wfm.channels)}"
        )
    x_index, y_index = xy_channel_indices(wfm)
    x_values = np.asarray(wfm.data[x_index, :], dtype=np.float64)
    y_values = np.asarray(wfm.data[y_index, :], dtype=np.float64)
    amplitude = np.asarray(np.hypot(x_values, y_values), dtype=np.float64)
    phase_deg = np.asarray(np.degrees(np.arctan2(y_values, x_values)), dtype=np.float64)
    return amplitude, phase_deg


def export_bruker(wfm: Waveform, path: str | Path) -> None:
    """Write a minimal Bruker-style shape text file.

    This is an interoperability stub, not a production-ready Bruker exporter.
    It writes an amplitude/phase ``##XYPOINTS`` table plus a few metadata tags,
    but it does not perform spectrometer calibration, power normalisation,
    vendor-specific shape parameter validation, or safety checks required before
    loading a pulse on hardware.
    """
    amplitude, phase_deg = _bruker_amplitude_phase_deg(wfm)
    output = _output_path(path)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("##TITLE= optimalcontrol minimal Bruker shape\n")
        handle.write("##JCAMP-DX= 5.00\n")
        handle.write("##DATA TYPE= Shape Data\n")
        handle.write("##ORIGIN= optimalcontrol\n")
        handle.write("##OWNER= optimalcontrol\n")
        handle.write("##UNITS= " + wfm.units + "\n")
        handle.write("##NPOINTS= " + str(int(wfm.times.shape[0])) + "\n")
        handle.write("##CHANNELS= amplitude,phase_deg\n")
        handle.write("##$OPTIMALCONTROL_PROBLEM_HASH= " + wfm.problem_hash + "\n")
        handle.write(
            "##$OPTIMALCONTROL_LIMITATIONS= "
            "Minimal stub only; not production-ready for spectrometer use.\n"
        )
        handle.write(
            "##$OPTIMALCONTROL_TIMES= "
            + " ".join(_format_float(float(value)) for value in wfm.times)
            + "\n"
        )
        handle.write("##XYPOINTS= (XY..XY)\n")
        for step_index in range(wfm.times.shape[0]):
            handle.write(
                _format_float(float(amplitude[step_index]))
                + ", "
                + _format_float(float(phase_deg[step_index]))
                + "\n"
            )
        handle.write("##END=\n")


def export_bruker_shape(
    path: str | Path,
    title: str,
    amplitude_percent: RealArray,
    phase_deg: RealArray,
    *,
    minx: float = 0.0,
    maxy: float = 360.0,
    totrot: float = 180.0,
    bwfac: float | None = 0.0,
    integfac: float | None = None,
    shape_mode: int = 1,
    extra_tags: Sequence[str] = (),
) -> Path:
    """Write a Bruker shape-library JCAMP file from amplitude/phase arrays.

    This is the shared writer behind the example scripts; :func:`export_bruker`
    remains the minimal :class:`Waveform` round-trip stub. Amplitude is in
    percent of the calibrated RF maximum and phase in degrees. ``extra_tags``
    are raw ``##...`` lines inserted verbatim before ``##NPOINTS``; passing
    ``bwfac=None`` omits the ``##$SHAPE_BWFAC`` tag, and ``integfac`` defaults
    to the mean amplitude fraction.
    """
    amplitude_percent = np.asarray(amplitude_percent, dtype=np.float64)
    phase_deg = np.asarray(phase_deg, dtype=np.float64)
    if integfac is None:
        integfac = float(np.mean(amplitude_percent)) / 100.0
    lines = [
        f"##TITLE= {title}",
        "##JCAMP-DX= 5.00 Bruker JCAMP library",
        "##DATA TYPE= Shape Data",
        "##ORIGIN= optimalcontrol",
        "##OWNER= optimalcontrol",
        f"##MINX= {minx:.6e}",
        "##MAXX= 1.000000e+02",
        "##MINY= 0.000000e+00",
        f"##MAXY= {maxy:.6e}",
        "##$SHAPE_EXMODE= None",
        f"##$SHAPE_TOTROT= {totrot:.6e}",
    ]
    if bwfac is not None:
        lines.append(f"##$SHAPE_BWFAC= {bwfac:.6e}")
    lines.append(f"##$SHAPE_INTEGFAC= {integfac:.9e}")
    lines.append(f"##$SHAPE_MODE= {shape_mode}")
    lines.extend(extra_tags)
    lines.append(f"##NPOINTS= {amplitude_percent.size}")
    lines.append("##XYPOINTS= (XY..XY)")
    for amplitude, phase in zip(amplitude_percent, phase_deg, strict=True):
        lines.append(f"{float(amplitude):.9e}, {float(phase):.9e}")
    lines.append("##END=")
    output = _output_path(path)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")
    return output
