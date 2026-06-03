"""
TDR visualization and interactive plotting module.

Provides functions for creating and displaying Time Domain Reflectometry (TDR) 
analysis plots with interactive features (markers, zoom, etc.).

Configuration:
    PLOT_BACKEND: Choose 'matplotlib' (default, full interactivity with markers/zoom)
                  or 'plotly' (web-based, built-in zoom/hover)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

# ============================================================================
# CONFIGURATION: Choose plotting backend
# ============================================================================
PLOT_BACKEND: Literal["matplotlib", "plotly"] = "plotly"
# To use Plotly: set to "plotly" or use --backend plotly from CLI
# ============================================================================

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class TdrResult(NamedTuple):
    """Time domain reflectometry analysis results."""
    t_s: np.ndarray
    h: np.ndarray
    step: np.ndarray
    z_ohm: np.ndarray
    df_hz: float
    n_uni: int
    nfft: int
    f_uniform_hz: np.ndarray


def _sync_twiny_from_tlim(ax0, ax_top, t_ps: np.ndarray, dist_mm: np.ndarray) -> None:
    """Synchronize top X-axis (distance) limits with bottom X-axis (time) limits."""
    lo, hi = ax0.get_xlim()
    if lo > hi:
        lo, hi = hi, lo
    d_lo = float(np.interp(lo, t_ps, dist_mm))
    d_hi = float(np.interp(hi, t_ps, dist_mm))
    if d_lo > d_hi:
        d_lo, d_hi = d_hi, d_lo
    ax_top.set_xlim(d_lo, d_hi)


def _data_dx_to_display_px(ax, dt: float) -> float:
    """Convert data distance to display pixels for marker proximity detection."""
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
    """
    Attach interactive features to TDR plot.
    
    Features:
    - Drag markers to measure impedance at different times
    - Right-click on impedance plot to zoom time window
    - Double-click to reset zoom
    - Display measurements (time, distance, impedance differences)
    """
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
        "Markers: left-click on all panels  |  Time zoom: right-click on Z plot  |  Double-click: reset  |  irfft + dt=1/(df·2NFFT)",
        ha="center",
        fontsize=8.5,
        transform=fig.transFigure,
        color="0.25",
    )


def plot_tdr_plotly(
    res: TdrResult,
    z0: float,
    er_eff: float,
    kaiser_beta: float,
    out_path: Path | None,
    show: bool,
) -> None:
    """
    Plot TDR results using Plotly (web-based, interactive).
    
    Creates a 3-panel figure showing:
    - Panel 1: Impulse response h(t) = irfft(S11)
    - Panel 2: Reflection coefficient γ(t) = cumsum(h)
    - Panel 3: Impedance Z(t) = Z0(1+γ)/(1-γ)
    
    Top axis shows one-way distance d = t·v_phase/2.
    Interactive features: hover to inspect values, zoom/pan, double-click to reset.
    
    Args:
        res: TdrResult from tdr_pipeline_irfft_steps_4_to_10()
        z0: Characteristic impedance [Ω]
        er_eff: Effective relative permittivity
        kaiser_beta: Kaiser window beta (for display in title)
        out_path: Path to save HTML (or None to skip saving)
        show: Display in browser
    """
    if not PLOTLY_AVAILABLE:
        raise ImportError(
            "plotly is required for PLOT_BACKEND='plotly'. "
            "Install it with: pip install plotly"
        )

    t_ps = res.t_s * 1e12
    c0 = 299792458.0
    v_phase = c0 / math.sqrt(er_eff)
    dist_m = res.t_s * v_phase / 2.0
    dist_mm = dist_m * 1e3

    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=(
            "Impulse response h (irfft S11)",
            "Reflection coefficient γ (cumsum h)",
            "Impedance Z (Z0(1+γ)/(1-γ))",
        ),
        vertical_spacing=0.08,
    )

    # Hover text helper
    def make_hover_text(t, d, val1, val2=None, val3=None):
        text = f"t: {t:.3f} ps<br>d: {d:.4f} mm<br>Value: {val1:.4g}"
        if val2 is not None:
            text += f"<br>Also: {val2:.4g}"
        if val3 is not None:
            text += f"<br>And: {val3:.4g}"
        return text

    # Panel 1: Impulse response
    fig.add_trace(
        go.Scatter(
            x=t_ps, y=res.h,
            mode="lines",
            name="h(t)",
            line=dict(color="blue", width=1.5),
            hovertemplate="<b>Impulse Response</b><br>t: %{x:.3f} ps<br>h: %{y:.4g}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Panel 2: Reflection coefficient
    fig.add_trace(
        go.Scatter(
            x=t_ps, y=res.step,
            mode="lines",
            name="γ(t)",
            line=dict(color="orange", width=1.5),
            hovertemplate="<b>Reflection Coefficient</b><br>t: %{x:.3f} ps<br>γ: %{y:.4g}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Panel 3: Impedance
    fig.add_trace(
        go.Scatter(
            x=t_ps, y=res.z_ohm,
            mode="lines",
            name="Z(t)",
            line=dict(color="green", width=1.5),
            hovertemplate="<b>Impedance</b><br>t: %{x:.3f} ps<br>Z: %{y:.1f} Ω<extra></extra>",
        ),
        row=3, col=1,
    )

    # Add characteristic impedance line
    fig.add_hline(
        y=z0,
        line_dash="dash",
        line_color="black",
        line_width=1,
        opacity=0.5,
        row=3, col=1,
        annotation_text=f"Z₀ = {z0:.1f} Ω",
        annotation_position="right",
    )

    # Update layout
    fig.update_layout(
        title_text=(
            f"TDR Analysis | N_uni={res.n_uni}, NFFT={res.nfft}, "
            f"df={res.df_hz / 1e6:.4g} MHz, Kaiser β={kaiser_beta:g}"
        ),
        height=900,
        hovermode="x unified",
        showlegend=True,
        template="plotly_white",
    )

    # Update axes
    fig.update_xaxes(title_text="Time t (ps)", row=3, col=1)
    fig.update_yaxes(title_text="h(t)", row=1, col=1)
    fig.update_yaxes(title_text="γ(t)", row=2, col=1)
    fig.update_yaxes(title_text="Z (Ω)", row=3, col=1)

    # Add distance information as text annotation
    fig.add_annotation(
        text=(
            f"One-way distance: d = t·v/2 | "
            f"εr,eff = {er_eff:g} | v = c/√εr"
        ),
        xref="paper", yref="paper",
        x=0.5, y=-0.05,
        showarrow=False,
        font=dict(size=10),
    )

    if out_path:
        # Save as HTML (Plotly format)
        if not str(out_path).endswith(".html"):
            out_path = out_path.with_suffix(".html")
        out_path = out_path.resolve()  # Convert to absolute path
        fig.write_html(str(out_path))
        print(f"Figure saved: {out_path}")
        if show:
            import webbrowser
            webbrowser.open(out_path.as_uri())
    elif show:
        # Display in browser without saving (uses temp file)
        import tempfile
        import webbrowser
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = Path(f.name).resolve()
        fig.write_html(str(temp_path))
        webbrowser.open(temp_path.as_uri())
        print(f"Figure saved to temporary file: {temp_path}")


def plot_tdr(
    res: TdrResult,
    z0: float,
    er_eff: float,
    kaiser_beta: float,
    out_path: Path | None,
    show: bool,
) -> None:
    """
    Plot TDR results using configured backend (matplotlib or plotly).
    
    Dispatches to plot_tdr_matplotlib() or plot_tdr_plotly() based on PLOT_BACKEND.
    
    Args:
        res: TdrResult from tdr_pipeline_irfft_steps_4_to_10()
        z0: Characteristic impedance [Ω]
        er_eff: Effective relative permittivity
        kaiser_beta: Kaiser window beta (for display in title)
        out_path: Path to save output (PNG for matplotlib, HTML for plotly)
        show: Display figure
    """
    if PLOT_BACKEND == "plotly":
        plot_tdr_plotly(res, z0, er_eff, kaiser_beta, out_path, show)
    else:
        plot_tdr_matplotlib(res, z0, er_eff, kaiser_beta, out_path, show)


def plot_tdr_matplotlib(
    res: TdrResult,
    z0: float,
    er_eff: float,
    kaiser_beta: float,
    out_path: Path | None,
    show: bool,
) -> None:
    """
    Plot TDR results using Matplotlib (static + interactive markers/zoom).
    
    Creates a 3-panel figure showing:
    - Panel 1: Impulse response h(t) = irfft(S11)
    - Panel 2: Reflection coefficient γ(t) = cumsum(h)
    - Panel 3: Impedance Z(t) = Z0(1+γ)/(1-γ)
    
    Top axis shows one-way distance d = t·v_phase/2.
    
    Interactive features (when show=True):
    - Drag markers to measure impedance at different times
    - Right-click on impedance panel to zoom time window
    - Double-click to reset zoom
    
    Args:
        res: TdrResult from tdr_pipeline_irfft_steps_4_to_10()
        z0: Characteristic impedance [Ω]
        er_eff: Effective relative permittivity
        kaiser_beta: Kaiser window beta (for display)
        out_path: Path to save PNG (or None to skip saving)
        show: Display interactive window
    """
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
    ax0.set_ylabel("Impulse h\nirfft(S11)")
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
    ax2.set_xlabel("Time t (ps)")
    ax2.grid(True, alpha=0.3)

    ax_top = ax0.twiny()
    ax_top.set_xlim(dist_mm[0], dist_mm[len(dist_mm) - 1])
    ax_top.set_xlabel(f"One-way distance d=t·v/2 (mm), εr,eff={er_eff:g}, v=c/√εr")

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
