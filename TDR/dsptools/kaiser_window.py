"""
Kaiser window and windowing functions for DSP operations.

Provides Kaiser window generation and application for spectral analysis
with configurable shape parameter (beta) for controlling sidelobe levels.
"""

from __future__ import annotations

import numpy as np
from scipy.special import i0


def kaiser_window(beta: float, n: int) -> np.ndarray:
    """
    Generate a Kaiser window.

    The Kaiser window is a tapered window that allows trading off main lobe width
    for side lobe level. It is defined in terms of the zeroth-order modified Bessel
    function of the first kind, I_0.

    Args:
        beta: Shape parameter. Controls the side lobe attenuation:
            - beta ≈ 0: approximately rectangular window
            - beta = 5: ~40 dB side lobe attenuation
            - beta = 6: ~50 dB side lobe attenuation
            - beta = 8.6: ~60 dB side lobe attenuation
        n: Length of the window (number of points)

    Returns:
        Kaiser window coefficients, shape (n,)

    Examples:
        >>> w = kaiser_window(beta=6.0, n=512)
        >>> assert w.shape == (512,)
        >>> assert np.allclose(w[0], w[-1])  # symmetric
    """
    if n < 1:
        return np.array([])
    if n == 1:
        return np.ones(1)

    m = np.arange(0, n)
    alpha = (n - 1) / 2.0
    arg = beta * np.sqrt(1 - (2 * m / (n - 1) - 1) ** 2)
    w = i0(arg) / i0(beta)
    return w


def apply_kaiser_window(
    spectrum: np.ndarray, kaiser_beta: float
) -> np.ndarray:
    """
    Apply a Kaiser window to a spectrum.

    Args:
        spectrum: Input spectrum (1D array of complex values)
        kaiser_beta: Kaiser window beta parameter (default ~6 for moderate sidelobe control)

    Returns:
        Windowed spectrum (same shape as input)
    """
    n = len(spectrum)
    w = kaiser_window(kaiser_beta, n)
    return spectrum * w


def apply_kaiser_window_or_ones(
    spectrum: np.ndarray, kaiser_beta: float, disable: bool = False
) -> np.ndarray:
    """
    Apply Kaiser window to spectrum, or return ones if disabled.

    Convenience function for optional windowing (e.g., debug mode).

    Args:
        spectrum: Input spectrum
        kaiser_beta: Kaiser window beta parameter
        disable: If True, return spectrum unchanged (no windowing)

    Returns:
        Windowed spectrum, or original spectrum if disable=True
    """
    if disable:
        return spectrum * np.ones(len(spectrum), dtype=spectrum.dtype)
    return apply_kaiser_window(spectrum, kaiser_beta)
