"""Spettro di potenza in dBm (Blackman, Δf≈0.1 Hz, SNR 20 dB) + modalità interattiva stile Data Cursor."""
from __future__ import annotations

import argparse

_parser = argparse.ArgumentParser(description="Spettro / campioni con finestra interattiva (matplotlib + mplcursors).")
_parser.add_argument(
    "-i",
    "--interactive",
    action="store_true",
    help="Apre una finestra grafica: marker sui campioni/spettro, trascinabile lungo la curva (tipo Data Cursor).",
)
_args = _parser.parse_args()

import matplotlib

matplotlib.use("TkAgg" if _args.interactive else "Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# Risoluzione frequenziale df = fs / N  =>  N = fs / df
df_hz = 0.1
fs_hz = 1000.0
N = int(round(fs_hz / df_hz))
T_s = N / fs_hz
t = np.arange(N, dtype=np.float64) / fs_hz

f0_hz = 1.0
R_ohm = 50.0
p_ref_w = 1e-3

A = float(np.sqrt(2.0 * R_ohm * p_ref_w))
snr_db = 20.0
snr_lin = 10.0 ** (snr_db / 10.0)
sigma_v = A / np.sqrt(2.0 * snr_lin)

rng = np.random.default_rng(42)
x_v = A * np.sin(2.0 * np.pi * f0_hz * t) + rng.normal(0.0, sigma_v, N)

win = signal.windows.blackman(N)
freqs_hz, psd_v2 = signal.periodogram(
    x_v,
    fs=fs_hz,
    window=win,
    detrend=False,
    scaling="spectrum",
    return_onesided=True,
)
p_w = np.maximum(psd_v2 / R_ohm, 1e-30)
p_dbm = 10.0 * np.log10(p_w / p_ref_w)

print(
    f"fs={fs_hz} Hz, durata={T_s:.1f} s, N={N}, df={fs_hz / N:.3f} Hz, "
    f"SNR={snr_db} dB, finestra=blackman"
)

if _args.interactive:
    import mplcursors

    fig, (ax_t, ax_f) = plt.subplots(2, 1, figsize=(10, 7), sharex=False)
    fig.suptitle(
        "Clic sulla curva per il marker; tieni premuto il tasto sinistro e trascina lungo la curva per spostarlo. "
        "Tasto destro: rimuovi. Shift+frecce: campione precedente/successivo. Un marker per grafico."
    )

    (line_t,) = ax_t.plot(t, x_v, "-", lw=0.7, color="C0")
    ax_t.set_xlim(0.0, min(2.0, T_s))
    ax_t.set_xlabel("t (s)")
    ax_t.set_ylabel("Tensione (V)")
    ax_t.set_title("Campioni nel tempo (zoom/pan dalla barra strumenti)")
    ax_t.grid(True, alpha=0.3)

    (line_f,) = ax_f.plot(freqs_hz, p_dbm, "-", lw=0.9, color="C1")
    ax_f.set_xlim(0.0, min(25.0, fs_hz / 2.0))
    ax_f.set_xlabel("Frequenza (Hz)")
    ax_f.set_ylabel("Potenza (dBm)")
    ax_f.set_title("Spettro di potenza")
    ax_f.grid(True, alpha=0.3)

    fig.tight_layout()

    # multiple=False abilita il trascinamento con tasto sinistro premuto (motion_notify).
    crs_t = mplcursors.cursor(line_t, multiple=False)
    crs_f = mplcursors.cursor(line_f, multiple=False)

    @crs_t.connect("add")
    def _on_add_time(sel: mplcursors.Selection) -> None:
        t0, v0 = float(sel.target[0]), float(sel.target[1])
        n = int(np.clip(np.round(t0 * fs_hz), 0, N - 1))
        v_sample = float(x_v[n])
        sel.annotation.set(
            text=f"n = {n}\nt = {t0:.6f} s\nV (interp.) = {v0:.6f}\nV[n] = {v_sample:.6f}",
            fontsize=9,
        )

    @crs_f.connect("add")
    def _on_add_freq(sel: mplcursors.Selection) -> None:
        f0, db0 = float(sel.target[0]), float(sel.target[1])
        sel.annotation.set(text=f"f = {f0:.4f} Hz\n{db0:.2f} dBm", fontsize=9)

    print(
        "Finestra interattiva: clic sulla curva, poi trascina con tasto sinistro tenuto premuto; "
        "destro sul marker per rimuoverlo; Shift+←/→ tra i campioni; un marker per grafico."
    )
    plt.show()
else:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(freqs_hz, p_dbm, color="C0", linewidth=0.9)
    ax.set_xlim(0.0, min(25.0, fs_hz / 2.0))
    ax.set_xlabel("Frequenza (Hz)")
    ax.set_ylabel("Potenza (dBm)")
    ax.set_title("Spettro di potenza (Blackman, Δf ≈ 0.1 Hz, SNR = 20 dB)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = "plot_power_spectrum_dbm.png"
    fig.savefig(out, dpi=120)
    print(f"Grafico salvato in: {out}")
