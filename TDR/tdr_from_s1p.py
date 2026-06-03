#!/usr/bin/env python3
"""
TDR da file Touchstone 1-porta (.s1p), pipeline come da specifica:

1. Carica S11 (MA/RI/DB → complesso).
2. DC: estrapolazione lineare di Re(S11) e Im(S11) dai primi due punti misurati a f=0
   (se f_min>0); se f=0 è già misurato si usa quel punto.
3. Griglia uniforme f=0 … f_max con passo df = passo medio della misura; interp lineare Re/Im.
4. Finestra Kaiser (β configurabile, default 6) sullo spettro unilaterale.
5. Zero-pad: NFFT = 4 * N_uni; spettro complesso di lunghezza NFFT+1 per irfft.
6. h = np.fft.irfft(S11_padded, n=2*NFFT)  (uscita reale, lunghezza 2*NFFT).
7. Asse tempo: T_window = 1/df, dt = T_window / (2*NFFT), t = arange(2*NFFT)*dt
   (non usare dt = 1/(2*f_max)).
8. Distanza one-way: d = t * v_phase / 2, v_phase = c / sqrt(er_eff).
9. step = cumsum(h).
10. gamma = clip(step, -0.9999, 0.9999); Z = Z0 * (1+gamma) / (1-gamma).

Avvio senza argomenti: dialog selezione file .s1p.
"""

from __future__ import annotations

import argparse
import math
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from typing import NamedTuple, Tuple


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector


def _parse_option_line(line: str) -> Tuple[str, str, str, float]:
    line = line.strip()
    if not line.startswith("#"):
        raise ValueError(f"Riga opzione non valida: {line!r}")
    parts = line[1:].split()
    if len(parts) < 5:
        raise ValueError(f"Riga opzione incompleta: {line!r}")
    freq_unit = parts[0].upper()
    param = parts[1].upper()
    fmt = parts[2].upper()
    if parts[3].upper() != "R" or len(parts) < 5:
        raise ValueError("Atteso formato '# <unit> S <MA|RI|DB> R <Z0>'")
    z0 = float(parts[4])
    return freq_unit, param, fmt, z0


def _freq_to_hz(value: float, unit: str) -> float:
    u = unit.upper()
    mult = {
        "HZ": 1.0,
        "KHZ": 1e3,
        "MHZ": 1e6,
        "GHZ": 1e9,
        "THZ": 1e12,
    }
    if u not in mult:
        raise ValueError(f"Unità frequenza non supportata: {unit}")
    return value * mult[u]


def _s_from_ma(mag: float, ang_deg: float) -> complex:
    rad = math.radians(ang_deg)
    return complex(mag * math.cos(rad), mag * math.sin(rad))


def _s_from_db(db: float, ang_deg: float) -> complex:
    mag = 10 ** (db / 20.0)
    return _s_from_ma(mag, ang_deg)


def read_touchstone_1port(path: Path) -> Tuple[np.ndarray, np.ndarray, float]:
    """Frequenze [Hz] crescenti, S11 complesso, Z0."""
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
                raise ValueError(f"Atteso parametro S, trovato {param}")
            freq_unit = fu
            fmt = ffmt
            if fmt not in ("MA", "RI", "DB"):
                raise ValueError(f"Formato S non supportato: {fmt}")
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
        raise ValueError("Troppi pochi punti frequenza nel file.")

    f_arr = np.asarray(freqs, dtype=float)
    s_arr = np.asarray(s11, dtype=np.complex128)
    order = np.argsort(f_arr)
    return f_arr[order], s_arr[order], z0


def _s11_dc_extrapolate_re_im(f_hz: np.ndarray, s11: np.ndarray) -> complex:
    """Estrapola Re e Im separatamente dai primi due punti misurati a f=0."""
    f0, f1 = float(f_hz[0]), float(f_hz[1])
    re0, re1 = float(s11[0].real), float(s11[1].real)
    im0, im1 = float(s11[0].imag), float(s11[1].imag)
    if abs(f1 - f0) < 1e-18:
        raise ValueError("Prime due frequenze coincidenti, impossibile estrapolare DC.")
    re_dc = re0 + (re1 - re0) * (0.0 - f0) / (f1 - f0)
    im_dc = im0 + (im1 - im0) * (0.0 - f0) / (f1 - f0)
    return complex(re_dc, im_dc)


class TdrResult(NamedTuple):
    t_s: np.ndarray
    h: np.ndarray
    step: np.ndarray
    z_ohm: np.ndarray
    df_hz: float
    n_uni: int
    nfft: int
    f_uniform_hz: np.ndarray


def tdr_pipeline_irfft(
    f_hz: np.ndarray,
    s11: np.ndarray,
    z0: float,
    kaiser_beta: float = 6.0,
    no_kaiser: bool = False,
) -> TdrResult:
    """
    Pipeline TDR da S11: griglia uniforme, Kaiser, irfft, asse t corretto, step, Z.
    """
    f_max = float(f_hz[-1])
    if f_max <= 0:
        raise ValueError("f_max non valido.")

    df_meas = float(np.median(np.diff(f_hz)))
    if df_meas <= 0:
        raise ValueError("Passo frequenza df non valido.")

    # Griglia uniforme 0 … f_max con passo df
    f_uniform = np.arange(0.0, f_max + 0.5 * df_meas, df_meas, dtype=np.float64)
    f_uniform = f_uniform[f_uniform <= f_max + 1e-9]
    n_uni = len(f_uniform)
    if n_uni < 2:
        raise ValueError("Griglia uniforme troppo corta.")

    # DC sullo 0 Hz della griglia
    if f_hz[0] <= 1e-12:
        s_dc = complex(s11[0])
    else:
        s_dc = _s11_dc_extrapolate_re_im(f_hz, s11)

    s_uni = np.empty(n_uni, dtype=np.complex128)
    for i, ff in enumerate(f_uniform):
        if ff <= 1e-18:
            s_uni[i] = s_dc
        else:
            s_uni[i] = complex(
                float(np.interp(ff, f_hz, s11.real)),
                float(np.interp(ff, f_hz, s11.imag)),
            )

    if no_kaiser:
        w = np.ones(n_uni, dtype=np.float64)
    else:
        w = np.kaiser(n_uni, kaiser_beta)
    s_win = s_uni * w

    nfft = 4 * n_uni
    n_out = 2 * nfft
    n_spec = nfft + 1  # irfft(a, n=n_out) richiede len(a) == n_out//2 + 1

    spec = np.zeros(n_spec, dtype=np.complex128)
    spec[:n_uni] = s_win

    h = np.fft.irfft(spec, n=n_out)

    t_window = 1.0 / df_meas
    dt = t_window / (2.0 * nfft)
    t_s = np.arange(n_out, dtype=np.float64) * dt

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


def _sync_twiny_from_tlim(ax0, ax_top, t_ps: np.ndarray, dist_mm: np.ndarray) -> None:
    lo, hi = ax0.get_xlim()
    if lo > hi:
        lo, hi = hi, lo
    d_lo = float(np.interp(lo, t_ps, dist_mm))
    d_hi = float(np.interp(hi, t_ps, dist_mm))
    if d_lo > d_hi:
        d_lo, d_hi = d_hi, d_lo
    ax_top.set_xlim(d_lo, d_hi)


def _data_dx_to_display_px(ax, dt: float) -> float:
    lo, hi = ax.get_xlim()
    if hi == lo:
        return 0.0
    p0, _ = ax.transData.transform((lo, 0.0))
    p1, _ = ax.transData.transform((hi, 0.0))
    scale = abs(p1 - p0) / abs(hi - lo)
    return abs(dt) * scale


def _attach_tdr_interactions(
    fig,
    axes: list,
    ax_top,
    t_ps: np.ndarray,
    h: np.ndarray,
    step: np.ndarray,
    z_ohm: np.ndarray,
    z0: float,
    dist_mm: np.ndarray,
    t_full_lo: float,
    t_full_hi: float,
) -> None:
    time_axes = axes
    n = len(t_ps)
    t_min, t_max = float(t_ps[0]), float(t_ps[n - 1])

    mx = [
        float(t_min + 0.22 * (t_max - t_min)),
        float(t_min + 0.78 * (t_max - t_min)),
    ]
    vlines: list[list] = []
    for j in range(2):
        color = ("#c200c2", "#0066cc")[j]
        lines = []
        for ax in time_axes:
            ln = ax.axvline(mx[j], color=color, ls="--", lw=1.4, alpha=0.95, zorder=9)
            lines.append(ln)
        vlines.append(lines)

    info = fig.text(
        0.02,
        0.02,
        "",
        transform=fig.transFigure,
        fontsize=9,
        verticalalignment="bottom",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.85),
    )

    def update_info_text() -> None:
        def iv(xq: float) -> tuple[float, float, float]:
            return (
                float(np.interp(xq, t_ps, h)),
                float(np.interp(xq, t_ps, step)),
                float(np.interp(xq, t_ps, z_ohm)),
            )

        h0, s0, z0v = iv(mx[0])
        h1, s1, z1v = iv(mx[1])
        d0 = float(np.interp(mx[0], t_ps, dist_mm))
        d1 = float(np.interp(mx[1], t_ps, dist_mm))
        dt = abs(mx[1] - mx[0])
        dd = abs(d1 - d0)
        info.set_text(
            f"M1: t={mx[0]:.3f} ps  d={d0:.4f} mm  h={h0:.4g}  step={s0:.4g}  Z={z0v:.2f} Ω\n"
            f"M2: t={mx[1]:.3f} ps  d={d1:.4f} mm  h={h1:.4g}  step={s1:.4g}  Z={z1v:.2f} Ω\n"
            f"Δt = {dt:.3f} ps   Δd = {dd:.4f} mm   ΔZ = {abs(z1v - z0v):.2f} Ω   Z0={z0:g} Ω"
        )

    def set_marker_x(idx: int, x_new: float) -> None:
        x_new = float(np.clip(x_new, t_min, t_max))
        mx[idx] = x_new
        for ln in vlines[idx]:
            ln.set_xdata([x_new, x_new])
        update_info_text()
        fig.canvas.draw_idle()

    drag: dict = {"idx": None, "cid_move": None, "cid_up": None}

    def on_button_press(ev) -> None:
        if getattr(ev, "dblclick", False) and ev.inaxes in time_axes:
            for ax in time_axes:
                ax.set_xlim(t_full_lo, t_full_hi)
            _sync_twiny_from_tlim(axes[0], ax_top, t_ps, dist_mm)
            fig.canvas.draw_idle()
            return
        if ev.button != 1 or ev.xdata is None:
            return
        ax_ev = ev.inaxes
        if ax_ev not in time_axes:
            return
        best_i = None
        best_px = 1e9
        for i in range(2):
            px = _data_dx_to_display_px(ax_ev, ev.xdata - mx[i])
            if px < 12 and px < best_px:
                best_px = px
                best_i = i
        if best_i is None:
            return
        drag["idx"] = best_i

        def on_move(e) -> None:
            if e.inaxes in time_axes and e.xdata is not None:
                set_marker_x(drag["idx"], e.xdata)

        def on_up(_e) -> None:
            if drag["cid_move"] is not None:
                fig.canvas.mpl_disconnect(drag["cid_move"])
            if drag["cid_up"] is not None:
                fig.canvas.mpl_disconnect(drag["cid_up"])
            drag["cid_move"] = drag["cid_up"] = None
            drag["idx"] = None

        drag["cid_move"] = fig.canvas.mpl_connect("motion_notify_event", on_move)
        drag["cid_up"] = fig.canvas.mpl_connect("button_release_event", on_up)

    def on_xlim_changed(ax) -> None:
        if ax in time_axes:
            _sync_twiny_from_tlim(axes[0], ax_top, t_ps, dist_mm)

    axes[0].callbacks.connect("xlim_changed", on_xlim_changed)

    def on_rect_select(eclick, erelease) -> None:
        if eclick.xdata is None or erelease.xdata is None:
            return
        if eclick.inaxes is not axes[2] and erelease.inaxes is not axes[2]:
            return
        xmin = min(eclick.xdata, erelease.xdata)
        xmax = max(eclick.xdata, erelease.xdata)
        if xmax - xmin < 1e-9:
            return
        for ax in time_axes:
            ax.set_xlim(xmin, xmax)
        _sync_twiny_from_tlim(axes[0], ax_top, t_ps, dist_mm)
        fig.canvas.draw_idle()

    rs = RectangleSelector(
        axes[2],
        on_rect_select,
        useblit=False,
        button=[3],
        minspanx=3,
        spancoords="data",
        interactive=False,
    )
    rs.set_active(True)
    fig._tdr_rect_selector = rs

    fig.canvas.mpl_connect("button_press_event", on_button_press)
    update_info_text()

    fig.subplots_adjust(bottom=0.12)
    fig.text(
        0.5,
        0.005,
        "Marker: sinistro su tutti i pannelli  |  Zoom tempo: tasto destro sul grafico Z  |  Doppio click: reset  |  irfft + dt=1/(df·2NFFT)",
        ha="center",
        fontsize=8.5,
        transform=fig.transFigure,
        color="0.25",
    )


def plot_tdr(
    res: TdrResult,
    z0: float,
    er_eff: float,
    kaiser_beta: float,
    out_path: Path | None,
    show: bool,
) -> None:
    t_ps = res.t_s * 1e12
    c0 = 299792458.0
    v_phase = c0 / math.sqrt(er_eff)
    dist_m = res.t_s * v_phase / 2.0
    dist_mm = dist_m * 1e3

    fig = plt.figure(figsize=(9, 8))
    ax0 = fig.add_subplot(311)
    ax1 = fig.add_subplot(312, sharex=ax0)
    ax2 = fig.add_subplot(313, sharex=ax0)
    axes = [ax0, ax1, ax2]

    ax0.plot(t_ps, res.h, color="C0", lw=0.9)
    ax0.set_ylabel("Impulso h\nirfft(S11)")
    ax0.set_title(
        f"TDR (N_uni={res.n_uni}, NFFT={res.nfft}, df={res.df_hz / 1e6:.4g} MHz, Kaiser β={kaiser_beta:g})"
    )
    ax0.grid(True, alpha=0.3)

    ax1.plot(t_ps, res.step, color="C1", lw=0.9)
    ax1.set_ylabel("step = cumsum(h)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_ps, res.z_ohm, color="C2", lw=0.9)
    ax2.axhline(z0, color="k", ls="--", lw=0.7, alpha=0.5)
    ax2.set_ylabel("Z = Z0(1+γ)/(1−γ)\nγ = clip(step)")
    ax2.set_xlabel("Tempo t (ps)")
    ax2.grid(True, alpha=0.3)

    ax_top = ax0.twiny()
    ax_top.set_xlim(dist_mm[0], dist_mm[len(dist_mm) - 1])
    ax_top.set_xlabel(f"Distanza one-way d=t·v/2 (mm), εr,eff={er_eff:g}, v=c/√εr")

    fig.tight_layout()
    t_lo, t_hi = float(t_ps[0]), float(t_ps[len(t_ps) - 1])

    if out_path:
        fig.savefig(out_path, dpi=150)

    if show:
        _attach_tdr_interactions(
            fig,
            axes,
            ax_top,
            t_ps,
            res.h,
            res.step,
            res.z_ohm,
            z0,
            dist_mm,
            t_lo,
            t_hi,
        )
        plt.show()
    else:
        plt.close(fig)


def ask_touchstone_path() -> Path | None:
    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except tk.TclError:
        pass
    path_str = filedialog.askopenfilename(
        title="Seleziona file Touchstone (.s1p)",
        filetypes=[
            ("Touchstone 1 porta", "*.s1p"),
            ("Touchstone", "*.s*p"),
            ("Tutti i file", "*.*"),
        ],
    )
    root.destroy()
    if not path_str:
        return None
    return Path(path_str)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="TDR da .s1p: irfft, asse t = 1/(df·2NFFT), Kaiser, cumsum, Z."
    )
    p.add_argument(
        "s1p",
        type=Path,
        nargs="?",
        default=None,
        help="File .s1p (se omesso, dialog di selezione).",
    )
    p.add_argument(
        "--kaiser-beta",
        type=float,
        default=6.0,
        help="Parametro β finestra Kaiser sullo spettro unilaterale (default 6).",
    )
    p.add_argument(
        "--no-kaiser",
        action="store_true",
        help="Disattiva la finestra Kaiser (solo debug; aumenta il ringing).",
    )
    p.add_argument(
        "--er",
        type=float,
        default=2.23,
        help="εr,eff per d=t·v/2 (default 2.23 ≈ RO4350B microstrip indicativo).",
    )
    p.add_argument("-o", "--output", type=Path, default=None, help="Salva PNG.")
    p.add_argument("--no-show", action="store_true", help="Non mostrare la finestra.")
    args = p.parse_args(argv)

    path = args.s1p
    if path is None:
        path = ask_touchstone_path()
        if path is None:
            print("Nessun file selezionato.", file=sys.stderr)
            return 1
    if not path.is_file():
        print(f"File non trovato: {path}", file=sys.stderr)
        return 1

    f_hz, s11, z0 = read_touchstone_1port(path)
    res = tdr_pipeline_irfft(
        f_hz,
        s11,
        z0,
        kaiser_beta=args.kaiser_beta,
        no_kaiser=args.no_kaiser,
    )

    plot_tdr(res, z0, args.er, args.kaiser_beta, args.output, show=not args.no_show)
    if args.output:
        print(f"Figura salvata: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
