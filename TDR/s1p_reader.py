"""
S1P Touchstone file reader with DC extrapolation and uniform grid interpolation.

Steps 1-3 of TDR pipeline:
1. Load S11 (MA/RI/DB → complex)
2. DC: linear extrapolation of Re(S11) and Im(S11) from first two measured points at f=0
3. Uniform grid f=0 … f_max with step df = median step; linear interpolation Re/Im
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np


def _parse_option_line(line: str) -> Tuple[str, str, str, float]:
    """Parse Touchstone option line: '# GHZ S MA R 50'"""
    line = line.strip()
    if not line.startswith("#"):
        raise ValueError(f"Invalid option line: {line!r}")
    parts = line[1:].split()
    if len(parts) < 5:
        raise ValueError(f"Incomplete option line: {line!r}")
    freq_unit = parts[0].upper()
    param = parts[1].upper()
    fmt = parts[2].upper()
    if parts[3].upper() != "R" or len(parts) < 5:
        raise ValueError("Expected format '# <unit> S <MA|RI|DB> R <Z0>'")
    z0 = float(parts[4])
    return freq_unit, param, fmt, z0


def _freq_to_hz(value: float, unit: str) -> float:
    """Convert frequency to Hz from various units."""
    u = unit.upper()
    mult = {
        "HZ": 1.0,
        "KHZ": 1e3,
        "MHZ": 1e6,
        "GHZ": 1e9,
        "THZ": 1e12,
    }
    if u not in mult:
        raise ValueError(f"Unsupported frequency unit: {unit}")
    return value * mult[u]


def _s_from_ma(mag: float, ang_deg: float) -> complex:
    """Convert Magnitude/Angle to complex number."""
    rad = math.radians(ang_deg)
    return complex(mag * math.cos(rad), mag * math.sin(rad))


def _s_from_db(db: float, ang_deg: float) -> complex:
    """Convert dB/Angle to complex number."""
    mag = 10 ** (db / 20.0)
    return _s_from_ma(mag, ang_deg)


def read_touchstone_1port(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Read single-port Touchstone file (.s1p).
    
    Returns:
        Tuple of:
        - f_hz: frequencies in Hz (ascending order)
        - s11: complex S11 values
        - z0: characteristic impedance [Ω]
    """
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    freq_unit = "GHZ"
    fmt = "MA"
    z0 = 50.0
    data_start = 0

    for i, raw in enumerate(text):
        s = raw.strip()
        if not s or s.startswith("!"):
            continue
        if s.startswith("#"):
            fu, param, ffmt, z0 = _parse_option_line(s)
            if param != "S":
                raise ValueError(f"Expected parameter S, found {param}")
            freq_unit = fu
            fmt = ffmt
            if fmt not in ("MA", "RI", "DB"):
                raise ValueError(f"Unsupported S format: {fmt}")
            data_start = i + 1
            break

    freqs: list[float] = []
    s11: list[complex] = []

    for raw in text[data_start:]:
        s = raw.strip()
        if not s or s.startswith("!"):
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        f_hz = _freq_to_hz(float(parts[0]), freq_unit)
        if fmt == "RI":
            s11.append(complex(float(parts[1]), float(parts[2])))
        elif fmt == "MA":
            s11.append(_s_from_ma(float(parts[1]), float(parts[2])))
        else:
            s11.append(_s_from_db(float(parts[1]), float(parts[2])))
        freqs.append(f_hz)

    if len(freqs) < 2:
        raise ValueError("Too few frequency points in file.")

    f_arr = np.asarray(freqs, dtype=float)
    s_arr = np.asarray(s11, dtype=np.complex128)
    order = np.argsort(f_arr)
    return f_arr[order], s_arr[order], z0


def _s11_dc_extrapolate_re_im(f_hz: np.ndarray, s11: np.ndarray) -> complex:
    """
    Extrapolate S11 at f=0 Hz using linear extrapolation of Real and Imaginary parts.
    
    Uses first two measured points to extrapolate backward to f=0.
    """
    f0, f1 = float(f_hz[0]), float(f_hz[1])
    re0, re1 = float(s11[0].real), float(s11[1].real)
    im0, im1 = float(s11[0].imag), float(s11[1].imag)
    if abs(f1 - f0) < 1e-18:
        raise ValueError("First two frequencies coincident, cannot extrapolate DC.")
    re_dc = re0 + (re1 - re0) * (0.0 - f0) / (f1 - f0)
    im_dc = im0 + (im1 - im0) * (0.0 - f0) / (f1 - f0)
    return complex(re_dc, im_dc)


class S1pData(NamedTuple):
    """Data from S1P file after steps 1-3."""
    f_uniform_hz: np.ndarray  # Uniform frequency grid [Hz]
    s_uniform: np.ndarray      # Interpolated S11 on uniform grid
    z0: float                  # Characteristic impedance [Ω]
    df_hz: float              # Frequency step [Hz]
    n_uni: int                # Number of uniform grid points


def read_and_process_s1p(
    path: Path,
) -> S1pData:
    """
    Steps 1-3: Read .s1p file, extrapolate DC, interpolate to uniform grid.
    
    Args:
        path: Path to .s1p file
        
    Returns:
        S1pData with uniform grid frequency, interpolated S11, impedance, etc.
    """
    # Step 1: Load S11 (MA/RI/DB → complex)
    f_hz, s11, z0 = read_touchstone_1port(path)
    
    f_max = float(f_hz[-1])
    if f_max <= 0:
        raise ValueError("Invalid f_max.")
    
    df_meas = float(np.median(np.diff(f_hz)))
    if df_meas <= 0:
        raise ValueError("Invalid frequency step df.")
    
    # Step 3: Uniform grid f=0 … f_max with step df
    f_uniform = np.arange(0.0, f_max + 0.5 * df_meas, df_meas, dtype=np.float64)
    f_uniform = f_uniform[f_uniform <= f_max + 1e-9]
    n_uni = len(f_uniform)
    if n_uni < 2:
        raise ValueError("Uniform grid too short.")
    
    # Step 2: DC extrapolation and interpolation
    if f_hz[0] <= 1e-12:
        s_dc = complex(s11[0])
    else:
        s_dc = _s11_dc_extrapolate_re_im(f_hz, s11)
    
    s_uniform = np.empty(n_uni, dtype=np.complex128)
    for i, ff in enumerate(f_uniform):
        if ff <= 1e-18:
            s_uniform[i] = s_dc
        else:
            s_uniform[i] = complex(
                float(np.interp(ff, f_hz, s11.real)),
                float(np.interp(ff, f_hz, s11.imag)),
            )
    
    return S1pData(
        f_uniform_hz=f_uniform,
        s_uniform=s_uniform,
        z0=z0,
        df_hz=df_meas,
        n_uni=n_uni,
    )
