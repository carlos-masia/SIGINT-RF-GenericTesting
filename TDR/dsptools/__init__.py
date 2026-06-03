"""
Digital Signal Processing (DSP) utilities for TDR analysis.

Provides windowing, filtering, and spectral analysis functions.
"""

from dsptools.kaiser_window import (
    kaiser_window,
    apply_kaiser_window,
    apply_kaiser_window_or_ones,
)

__all__ = [
    "kaiser_window",
    "apply_kaiser_window",
    "apply_kaiser_window_or_ones",
]
