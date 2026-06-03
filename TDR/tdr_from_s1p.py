#!/usr/bin/env python3
"""
TDR analysis from Touchstone 1-port (.s1p) files.

Pipeline:
- Steps 1-3: Read S1P file, extrapolate DC, uniform grid interpolation (s1p_reader.py)
- Steps 4-10: Kaiser window, FFT, impulse response, impedance calculation (here)
- Visualization: Interactive plotting with markers and zoom (plot_graph.py)

Interactive mode: drag markers, zoom with right-click, double-click to reset.
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np

from s1p_reader import read_and_process_s1p
from plot_graph import TdrResult, plot_tdr
from dsptools import apply_kaiser_window_or_ones
import plot_graph  # For backend configuration


def tdr_pipeline_irfft_steps_4_to_10(
    s1p_data,
    z0: float,
    kaiser_beta: float = 6.0,
    no_kaiser: bool = False,
) -> TdrResult:
    """
    Steps 4-10 of TDR pipeline: Kaiser window, FFT, impulse response, impedance.
    
    Steps 1-3 (file reading, DC extrapolation, uniform grid) handled by s1p_reader.
    
    Args:
        s1p_data: S1pData from read_and_process_s1p()
        z0: Characteristic impedance [Ω]
        kaiser_beta: Kaiser window beta parameter (default 6)
        no_kaiser: Disable Kaiser window (debug mode)
        
    Returns:
        TdrResult with time-domain analysis
    """
    f_uniform = s1p_data.f_uniform_hz
    s_uni = s1p_data.s_uniform
    df_meas = s1p_data.df_hz
    n_uni = s1p_data.n_uni

    # Step 4: Kaiser window
    s_win = apply_kaiser_window_or_ones(s_uni, kaiser_beta, disable=no_kaiser)

    # Step 5: Zero-padding
    nfft = 4 * n_uni
    n_out = 2 * nfft
    n_spec = nfft + 1  # irfft(a, n=n_out) richiede len(a) == n_out//2 + 1

    spec = np.zeros(n_spec, dtype=np.complex128)
    spec[:n_uni] = s_win

    # Step 6: Inverse Real FFT
    h = np.fft.irfft(spec, n=n_out)

    # Step 7: Time axis
    t_window = 1.0 / df_meas
    dt = t_window / (2.0 * nfft)
    t_s = np.arange(n_out, dtype=np.float64) * dt

    # Step 9-10: Reflection coefficient and impedance
    step = np.cumsum(h)
    gamma = np.clip(step, -0.9999, 0.9999)
    z_ohm = z0 * (1.0 + gamma) / (1.0 - gamma)

    return TdrResult(
        t_s=t_s,
        h=h,
        step=step,
        z_ohm=z_ohm,
        df_hz=df_meas,
        n_uni=n_uni,
        nfft=nfft,
        f_uniform_hz=f_uniform,
    )



def ask_touchstone_path() -> Path | None:
    """Open file dialog to select a Touchstone (.s1p) file."""
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    path_str = filedialog.askopenfilename(
        title="Select Touchstone file (.s1p)",
        filetypes=[
            ("Touchstone 1-port", "*.s1p"),
            ("Touchstone", "*.s*p"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path_str:
        return None
    return Path(path_str)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="TDR from .s1p: irfft, time axis = 1/(df·2NFFT), Kaiser, cumsum, impedance."
    )
    p.add_argument(
        "s1p",
        type=Path,
        nargs="?",
        default=None,
        help="Path to .s1p file (optional, shows dialog if omitted).",
    )
    p.add_argument(
        "--kaiser-beta",
        type=float,
        default=6.0,
        help="Kaiser window β parameter on unilateral spectrum (default 6).",
    )
    p.add_argument(
        "--no-kaiser",
        action="store_true",
        help="Disable Kaiser window (debug mode; increases ringing).",
    )
    p.add_argument(
        "--er",
        type=float,
        default=2.23,
        help="Effective εr for d=t·v/2 (default 2.23 ≈ RO4350B microstrip).",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=["matplotlib", "plotly"],
        default=None,
        help="Plotting backend: if omitted, use PLOT_BACKEND in plot_graph module.",
    )
    p.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Save output (PNG for matplotlib, HTML for plotly).",
    )
    p.add_argument("--no-show", action="store_true", help="Do not show plot window.")
    args = p.parse_args(argv)

    path = args.s1p
    if path is None:
        path = ask_touchstone_path()
        if path is None:
            print("No file selected.", file=sys.stderr)
            return 1
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    # Determine backend: CLI overrides module default when provided
    chosen_backend = args.backend if args.backend is not None else plot_graph.PLOT_BACKEND
    plot_graph.PLOT_BACKEND = chosen_backend

    # Auto-save to outputs/ for plotly if no output path specified
    output_path = args.output
    if output_path is None and chosen_backend == "plotly":
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        output_path = outputs_dir / "tdr_plot.html"

    # Steps 1-3: Read file, extrapolate DC, uniform grid
    s1p_data = read_and_process_s1p(path)
    
    # Steps 4-10: Kaiser, FFT, impulse response, impedance
    res = tdr_pipeline_irfft_steps_4_to_10(
        s1p_data,
        s1p_data.z0,
        kaiser_beta=args.kaiser_beta,
        no_kaiser=args.no_kaiser,
    )

    plot_tdr(res, s1p_data.z0, args.er, args.kaiser_beta, output_path, show=not args.no_show)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
