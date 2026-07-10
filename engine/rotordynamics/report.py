"""
Report builders — decoupled from Streamlit.

Integration point A (docs/architecture.md). Pure functions that take a finished
`RotordynamicAnalysis` and return output, callable with no Streamlit / UI state:

    build_report(analysis)                    -> (html: str, plots: dict[str, bytes])
    build_wiki_page(analysis, run_id=None)    -> (slug: str, markdown: str)

`build_wiki_page` is what makes a simulation run citable: its output slots
straight into the same wiki the retriever reads, so past runs become part of
the knowledge base. The slug doubles as the citation token the assistant
emits, e.g. '(run: 2026-07-10-run-001)'.

The figure builders below are ports of the `build_*_fig` methods that lived in
`streamlit_app.py`; they depend only on matplotlib and the analysis object.
"""

from __future__ import annotations

import base64
import io
from datetime import date, datetime

import matplotlib

matplotlib.use("Agg")  # headless-safe; report generation never needs a display

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ======================================================================
#  Figure builders (pure ports of streamlit_app.build_*_fig)
# ======================================================================


def _fig_to_png_bytes(fig, dpi: int = 110) -> bytes:
    """Render a matplotlib figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _positions(analysis) -> tuple[float, float, float]:
    """(bearing1, disk, bearing2) axial positions [m] from the FE mesh."""
    bearing1_pos = float(analysis.coord[2, 1])   # node 3
    bearing2_pos = float(analysis.coord[14, 1])  # node 15
    disk_pos = float(analysis.coord[5, 1])       # node 6 (disk station)
    return bearing1_pos, disk_pos, bearing2_pos


def build_fe_model_fig(analysis):
    """FE model layout + static force diagram."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), height_ratios=[2, 1])

    bearing1_pos, disk_pos, bearing2_pos = _positions(analysis)
    bearing1_pos_mm = bearing1_pos * 1000
    disk_pos_mm = disk_pos * 1000
    bearing2_pos_mm = bearing2_pos * 1000
    shaft_length_mm = analysis.le * 1000

    shaft_radius = analysis.de * 1000 / 2
    bearing_radius = analysis.D * 1000 / 2
    disk_radius = analysis.dd * 1000 / 2

    # ---- main layout ----
    ax1.plot([0, shaft_length_mm], [shaft_radius, shaft_radius], "k-", linewidth=4, label="Shaft")
    ax1.plot([0, shaft_length_mm], [-shaft_radius, -shaft_radius], "k-", linewidth=4)
    ax1.fill_between([0, shaft_length_mm], [shaft_radius, shaft_radius],
                     [-shaft_radius, -shaft_radius], alpha=0.3, color="gray")

    bearing_height = bearing_radius * 2
    bearing_width = 8
    support_height = bearing_height * 0.6

    for pos_mm, label in ((bearing1_pos_mm, "Bearing 1"), (bearing2_pos_mm, "Bearing 2")):
        rect = Rectangle((pos_mm - bearing_width / 2, -bearing_height / 2),
                         bearing_width, bearing_height, facecolor="steelblue",
                         edgecolor="darkblue", linewidth=2, alpha=0.8,
                         label="Bearings" if label == "Bearing 1" else None)
        ax1.add_patch(rect)
        ax1.plot([pos_mm - bearing_width, pos_mm + bearing_width],
                 [-bearing_height / 2 - support_height, -bearing_height / 2 - support_height],
                 "k-", linewidth=3)
        ax1.plot([pos_mm, pos_mm],
                 [-bearing_height / 2, -bearing_height / 2 - support_height],
                 "k-", linewidth=2)
        ax1.text(pos_mm, -bearing_height / 2 - support_height - 20, label,
                 ha="center", fontsize=12, fontweight="bold")

    disk_width = analysis.ld * 1000
    disk_height = disk_radius * 2
    ax1.add_patch(Rectangle((disk_pos_mm - disk_width / 2, -disk_height / 2),
                            disk_width, disk_height, facecolor="crimson",
                            edgecolor="darkred", linewidth=2, alpha=0.8, label="Disk"))
    hub_height = shaft_radius * 4
    ax1.add_patch(Rectangle((disk_pos_mm - disk_width / 2, -hub_height / 2),
                            disk_width, hub_height, facecolor="darkred", alpha=0.9))
    ax1.text(disk_pos_mm, disk_height / 2 + 15, "Disk", ha="center",
             fontsize=12, fontweight="bold")

    dimension_y = -bearing_height / 2 - support_height - 50
    for (x0, x1, color) in ((0, bearing1_pos_mm, "blue"),
                            (bearing1_pos_mm, disk_pos_mm, "red"),
                            (disk_pos_mm, bearing2_pos_mm, "green")):
        ax1.annotate("", xy=(x1, dimension_y), xytext=(x0, dimension_y),
                     arrowprops=dict(arrowstyle="<->", color=color, lw=2))
        ax1.text((x0 + x1) / 2, dimension_y - 15, f"{x1 - x0:.0f} mm",
                 ha="center", fontsize=10, color=color, fontweight="bold")

    ax1.set_xlim(-30, shaft_length_mm + 30)
    ax1.set_ylim(dimension_y - 40, disk_height / 2 + 40)
    ax1.set_xlabel("Axial Position [mm]", fontsize=12)
    ax1.set_ylabel("Radial Distance [mm]", fontsize=12)
    ax1.set_title("Rotordynamic System Layout", fontsize=14, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect("equal")

    # ---- force diagram ----
    force_scale = 50
    beam_y = 0
    ax2.plot([0, shaft_length_mm], [beam_y, beam_y], "k-", linewidth=6, alpha=0.7)
    ax2.plot(bearing1_pos_mm, beam_y, "bs", markersize=12, label="Bearing 1")
    ax2.plot(disk_pos_mm, beam_y, "ro", markersize=12, label="Disk")
    ax2.plot(bearing2_pos_mm, beam_y, "bs", markersize=12, label="Bearing 2")

    weight_arrow_length = analysis.W / force_scale
    ax2.arrow(disk_pos_mm, beam_y, 0, -weight_arrow_length, head_width=15, head_length=8,
              fc="red", ec="red", linewidth=3, alpha=0.8)
    ax2.text(disk_pos_mm, -weight_arrow_length - 20, f"W = {analysis.W:.1f} N",
             ha="center", fontsize=11, color="red", fontweight="bold")

    fm1_arrow_length = analysis.FM1 / force_scale
    ax2.arrow(bearing1_pos_mm, beam_y, 0, fm1_arrow_length, head_width=15, head_length=8,
              fc="blue", ec="blue", linewidth=3, alpha=0.8)
    ax2.text(bearing1_pos_mm, fm1_arrow_length + 10, f"R1 = {analysis.FM1:.1f} N",
             ha="center", fontsize=11, color="blue", fontweight="bold")

    fm2_arrow_length = analysis.FM2 / force_scale
    ax2.arrow(bearing2_pos_mm, beam_y, 0, fm2_arrow_length, head_width=15, head_length=8,
              fc="blue", ec="blue", linewidth=3, alpha=0.8)
    ax2.text(bearing2_pos_mm, fm2_arrow_length + 10, f"R2 = {analysis.FM2:.1f} N",
             ha="center", fontsize=11, color="blue", fontweight="bold")

    max_force_arrow = max(weight_arrow_length, fm1_arrow_length, fm2_arrow_length)
    ax2.set_xlim(-30, shaft_length_mm + 30)
    ax2.set_ylim(-max_force_arrow - 40, max_force_arrow + 40)
    ax2.set_xlabel("Position [mm]", fontsize=12)
    ax2.set_ylabel("Force [N]", fontsize=12)
    ax2.set_title("Static Force Analysis", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="black", linewidth=1, alpha=0.5)

    force_sum = analysis.FM1 + analysis.FM2 - analysis.W
    ax2.text(shaft_length_mm / 2, -max_force_arrow - 25,
             f"Equilibrium Check: Sum F = R1 + R2 - W = {force_sum:.3f} N ~ 0",
             ha="center", fontsize=10, style="italic",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    return fig


def build_bearing_coefficients_fig(analysis):
    """Bearing stiffness/damping coefficient curves."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    curve_sets = (
        (axes[0, 0], (analysis.k_m1_11, analysis.k_m1_12, analysis.k_m1_21, analysis.k_m1_22),
         ("K_yy", "K_yz", "K_zy", "K_zz"), "Stiffness [N/m]",
         f"Bearing 1 Stiffness Coefficients ({analysis.bearing1_type.title()})", (-1e7, 1e7)),
        (axes[0, 1], (analysis.k_m2_11, analysis.k_m2_12, analysis.k_m2_21, analysis.k_m2_22),
         ("K_yy", "K_yz", "K_zy", "K_zz"), "Stiffness [N/m]",
         f"Bearing 2 Stiffness Coefficients ({analysis.bearing2_type.title()})", (-1e7, 1e7)),
        (axes[1, 0], (analysis.d_m1_11, analysis.d_m1_12, analysis.d_m1_21, analysis.d_m1_22),
         ("C_yy", "C_yz", "C_zy", "C_zz"), "Damping [N.s/m]",
         f"Bearing 1 Damping Coefficients ({analysis.bearing1_type.title()})", None),
        (axes[1, 1], (analysis.d_m2_11, analysis.d_m2_12, analysis.d_m2_21, analysis.d_m2_22),
         ("C_yy", "C_yz", "C_zy", "C_zz"), "Damping [N.s/m]",
         f"Bearing 2 Damping Coefficients ({analysis.bearing2_type.title()})", None),
    )
    colors = ("b-", "r-", "g-", "m-")
    for ax, curves, labels, ylabel, title, ylim in curve_sets:
        for curve, style, label in zip(curves, colors, labels):
            ax.plot(analysis.omega, curve, style, label=label, linewidth=2)
        if ylim:
            ax.set_ylim(ylim)
        ax.set_xlim([0, analysis.omega[-1]])
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Angular Velocity [rad/s]")
        ax.legend()
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def build_bearing_locus_fig(analysis):
    """Journal-center locus, or None if there are no journal bearings."""
    if analysis.bearing1_type != "journal" and analysis.bearing2_type != "journal":
        return None

    theta = np.linspace(0, 2 * np.pi, 1000)
    rho = 0.8
    fig, ax = plt.subplots(figsize=(8, 8))

    if analysis.bearing1_type == "journal":
        ax.plot(analysis.epsilon1 * np.sin(np.radians(analysis.phi1)),
                -analysis.epsilon1 * np.cos(np.radians(analysis.phi1)),
                "b-", linewidth=3, label="Bearing 1 (Journal)", alpha=0.8)
    if analysis.bearing2_type == "journal":
        ax.plot(analysis.epsilon2 * np.sin(np.radians(analysis.phi2)),
                -analysis.epsilon2 * np.cos(np.radians(analysis.phi2)),
                "r-", linewidth=3, label="Bearing 2 (Journal)", alpha=0.8)

    ax.plot(rho * np.cos(theta), rho * np.sin(theta), "k--", linewidth=2,
            label="Clearance circle", alpha=0.6)
    ax.set_xlim((-0.8, 0.8))
    ax.set_ylim((-0.8, 0.8))
    ax.set_xlabel("eps x sin(phi)", fontsize=12)
    ax.set_ylabel("eps x cos(phi)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_title("Bearing Locus (Journal Bearings Only)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    return fig


def build_frequency_response_fig(analysis):
    """FRFs at key nodes + Campbell diagram."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    frequency_hz = analysis.omega / (2 * np.pi)

    frf_specs = (
        (axes[0, 0], 8, 10, "FRF - Bearing 1 Node"),
        (axes[0, 1], 20, 22, "FRF - Disk Node"),
        (axes[1, 0], 56, 58, "FRF - Bearing 2 Node"),
    )
    for ax, iy, iz, title in frf_specs:
        if analysis.X.shape[0] > iz:
            ax.plot(frequency_hz, np.abs(analysis.X[iy, :]) * 1000, "b-",
                    label="Y displacement", linewidth=2)
            ax.plot(frequency_hz, np.abs(analysis.X[iz, :]) * 1000, "r-",
                    label="Z displacement", linewidth=2)
        ax.set_xlim([frequency_hz[0], frequency_hz[-1]])
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Amplitude [mm]")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Campbell diagram
    cax = axes[1, 1]
    cax.plot(frequency_hz, frequency_hz, "k-", linewidth=2, label="1X (Synchronous)")
    if hasattr(analysis, "natural_freq_matrix"):
        colors = ["blue", "orange", "green", "purple", "brown"]
        for mode_idx in range(min(5, analysis.natural_freq_matrix.shape[0])):
            nat_freq_line = analysis.natural_freq_matrix[mode_idx, :]
            valid = nat_freq_line > 0
            if np.any(valid):
                cax.plot(frequency_hz[valid], nat_freq_line[valid] / (2 * np.pi),
                         color=colors[mode_idx % len(colors)], linestyle="-",
                         linewidth=1.5, alpha=0.8, label=f"Mode {mode_idx + 1}")
    if hasattr(analysis, "critical_speeds") and len(analysis.critical_speeds) > 0:
        labels = ["1st Critical", "2nd Critical", "3rd Critical"]
        for i, cs in enumerate(analysis.critical_speeds[:3]):
            cs_hz = cs / (2 * np.pi)
            cax.plot(cs_hz, cs_hz, "ro", markersize=8, label=labels[i],
                     markerfacecolor="red", markeredgecolor="darkred", markeredgewidth=2)
            cax.axvline(x=cs_hz, color="r", linestyle="--", alpha=0.5)
    cax.set_xlabel("Rotor Speed [Hz]")
    cax.set_ylabel("Natural Frequency [Hz]")
    cax.set_title("Campbell Diagram")
    cax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    cax.grid(True, alpha=0.3)
    cax.set_xlim([frequency_hz[0], frequency_hz[-1]])
    cax.set_ylim([frequency_hz[0], frequency_hz[-1]])

    plt.tight_layout()
    return fig


def build_mode_shapes_fig(analysis):
    """Rotor deflection shapes at the identified critical speeds."""
    if hasattr(analysis, "critical_speeds") and len(analysis.critical_speeds) > 0:
        speeds = list(analysis.critical_speeds[:3])
    else:
        speeds = [160, 750, 1460]

    indices = [int(np.argmin(np.abs(analysis.omega - s))) for s in speeds]
    num_plots = min(3, len(speeds))
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
    if num_plots == 1:
        axes = [axes]

    for i, (speed, idx) in enumerate(zip(speeds[:num_plots], indices[:num_plots])):
        if idx < analysis.X.shape[1] and analysis.X.shape[0] > 4:
            # micrometers: avoids matplotlib's tiny "1e-5" offset multiplier that
            # makes micron deflections read like meters
            y_displacements = analysis.X[::4, idx].real * 1e6
            axes[i].plot(analysis.coord[:, 1], y_displacements, "b-o",
                         linewidth=3, markersize=6, label="Y deformation")
            axes[i].plot(analysis.coord[:, 1], analysis.coord[:, 2] * 1e6, "k--",
                         marker="s", markersize=4, label="Initial state", alpha=0.6)
        axes[i].ticklabel_format(axis="y", style="plain", useOffset=False)
        axes[i].legend()
        axes[i].set_xlabel("Axial Position [m]")
        axes[i].set_ylabel("Amplitude [um]")
        axes[i].set_title(f"Mode Shape at {speed:.0f} rad/s ({speed / (2 * np.pi):.1f} Hz)")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ======================================================================
#  Public API
# ======================================================================

# (plot key, title, description, builder) — order defines report order.
_PLOT_SPECS = (
    ("system-layout", "System Layout & Static Force Analysis",
     "Finite element layout of the shaft-disk-bearing system with the static "
     "weight distribution and the reaction equilibrium check.", build_fe_model_fig),
    ("bearing-coefficients", "Bearing Stiffness & Damping Coefficients",
     "Speed-dependent linearized stiffness (K) and damping (C) coefficients "
     "for both bearings.", build_bearing_coefficients_fig),
    ("bearing-locus", "Bearing Locus",
     "Journal center trajectory within the bearing clearance circle as a "
     "function of rotational speed.", build_bearing_locus_fig),
    ("frequency-response", "Frequency Response & Campbell Diagram",
     "Unbalance frequency response functions at key nodes and the Campbell "
     "diagram identifying critical speeds at the 1X synchronous intersections. "
     "Note: the Z-direction curves include the static disk weight in the rotating "
     "force vector (original MATLAB formulation), so they carry the static sag on "
     "top of the true vibration; the Y curves show the pure unbalance response. "
     "The model is linear - amplitudes at resonance peaks can exceed the physical "
     "bearing clearance and should be read as resonance indicators, not literal "
     "displacements.",
     build_frequency_response_fig),
    ("mode-shapes", "Mode Shapes at Critical Speeds",
     "Deflection patterns of the rotor at the identified critical speeds.",
     build_mode_shapes_fig),
)


def build_plots(analysis) -> dict[str, bytes]:
    """Render every applicable plot to PNG bytes, keyed by plot slug."""
    plots: dict[str, bytes] = {}
    for key, _title, _desc, builder in _PLOT_SPECS:
        fig = builder(analysis)
        if fig is not None:
            plots[key] = _fig_to_png_bytes(fig)
    return plots


def _critical_speed_rows(analysis) -> list[tuple[str, float]]:
    ordinals = ["1st", "2nd", "3rd", "4th", "5th"]
    crit = getattr(analysis, "critical_speeds", None)
    if crit is None or len(crit) == 0:
        return []
    return [(ordinals[i] if i < 5 else f"{i + 1}th", float(c))
            for i, c in enumerate(crit[:5])]


def _parameter_pairs(analysis) -> list[tuple[str, str]]:
    equilibrium = analysis.FM1 + analysis.FM2 - analysis.W
    return [
        ("Young's modulus, E", f"{analysis.E / 1e9:.1f} GPa"),
        ("Density, rho", f"{analysis.rho:.0f} kg/m3"),
        ("Shaft diameter, d", f"{analysis.de * 1e3:.1f} mm"),
        ("Shaft length, L", f"{analysis.le * 1e3:.1f} mm"),
        ("Disk diameter", f"{analysis.dd * 1e3:.1f} mm"),
        ("Disk length", f"{analysis.ld * 1e3:.1f} mm"),
        ("Disk mass", f"{analysis.md:.3f} kg"),
        ("Disk weight, W", f"{analysis.W:.2f} N"),
        ("Unbalance, m.e", f"{analysis.me:.2e} kg.m"),
        ("Bearing 1 type", analysis.bearing1_type.title()),
        ("Bearing 2 type", analysis.bearing2_type.title()),
        ("Bearing diameter", f"{analysis.D * 1e3:.1f} mm"),
        ("Bearing length", f"{analysis.C * 1e3:.1f} mm"),
        ("Radial clearance, delta", f"{analysis.delta * 1e6:.1f} um"),
        ("Oil viscosity, mu", f"{analysis.mi:.4f} Pa.s"),
        ("Speed range",
         f"{analysis.omega[0]:.0f} - {analysis.omega[-1]:.0f} rad/s ({analysis.n} points)"),
        ("Bearing 1 reaction, R1", f"{analysis.FM1:.2f} N"),
        ("Bearing 2 reaction, R2", f"{analysis.FM2:.2f} N"),
        ("Static equilibrium, Sum F", f"{equilibrium:.3e} N"),
    ]


def build_report(analysis) -> tuple[str, dict[str, bytes]]:
    """Return a self-contained HTML report + the rendered plot images.

    The HTML embeds every plot as base64 PNG, so the returned string is a
    single portable file. The same PNG bytes are also returned in the dict so
    callers (e.g. the wiki bridge) can persist them separately.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plots = build_plots(analysis)

    speed_min_hz = analysis.omega[0] / (2 * np.pi)
    speed_max_hz = analysis.omega[-1] / (2 * np.pi)

    crit_rows = "".join(
        f"<tr><td>{ord_} critical speed</td>"
        f"<td>{c:.1f} rad/s</td><td>{c / (2 * np.pi):.1f} Hz</td>"
        f"<td>{c / (2 * np.pi) * 60:.0f} RPM</td></tr>"
        for ord_, c in _critical_speed_rows(analysis)
    )
    param_rows = "".join(
        f"<tr><td>{name}</td><td>{val}</td></tr>" for name, val in _parameter_pairs(analysis)
    )

    sections = []
    for key, title, desc, _builder in _PLOT_SPECS:
        if key in plots:
            img = base64.b64encode(plots[key]).decode("utf-8")
            sections.append(
                f"<h2>{title.replace('&', '&amp;')}</h2><p class='desc'>{desc}</p>"
                f"<img src='data:image/png;base64,{img}' alt='{title}'/>"
            )
    figures_html = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Rotordynamic Analysis Report</title>
<style>
  body {{ font-family: 'Segoe UI', Arial, sans-serif; color: #1a1a1a;
         max-width: 1000px; margin: 0 auto; padding: 40px 24px; line-height: 1.55; }}
  h1 {{ color: #0b3d91; border-bottom: 3px solid #0b3d91; padding-bottom: 10px; }}
  h2 {{ color: #0b3d91; margin-top: 40px; }}
  .meta {{ color: #555; font-size: 0.9em; margin-bottom: 8px; }}
  .desc {{ color: #444; font-style: italic; }}
  table {{ border-collapse: collapse; width: 100%; margin: 14px 0; }}
  th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: left; font-size: 0.92em; }}
  th {{ background: #0b3d91; color: #fff; }}
  tr:nth-child(even) td {{ background: #f4f7fc; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px;
        margin: 12px 0; display: block; }}
  footer {{ margin-top: 48px; padding-top: 16px; border-top: 1px solid #ddd;
            color: #777; font-size: 0.85em; }}
</style>
</head>
<body>
  <h1>Finite Element Rotordynamic Analysis Report</h1>
  <p class="meta">Generated {timestamp}</p>
  <p>Vibration analysis of a rotating shaft-disk-bearing system using finite element
     methods with hydrodynamic bearing coefficients, gyroscopic effects and unbalance
     response.</p>

  <h2>Key Results</h2>
  <table>
    <tr><th>Quantity</th><th>Angular</th><th>Frequency</th><th>Speed</th></tr>
    <tr><td>Analysis speed range</td>
        <td>{analysis.omega[0]:.0f} - {analysis.omega[-1]:.0f} rad/s</td>
        <td>{speed_min_hz:.1f} - {speed_max_hz:.1f} Hz</td>
        <td>{speed_min_hz * 60:.0f} - {speed_max_hz * 60:.0f} RPM</td></tr>
    {crit_rows}
  </table>

  <h2>System Parameters</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    {param_rows}
  </table>

  {figures_html}

  <footer>
    Report produced by the Rotordynamics Copilot engine -
    <a href="https://github.com/tbgallinucci/rotordynamics">github.com/tbgallinucci/rotordynamics</a>
  </footer>
</body>
</html>"""
    return html, plots


def build_wiki_page(analysis, run_id: str | None = None) -> tuple[str, str]:
    """Render a finished analysis as a Markdown wiki page.

    Returns (slug, markdown). The slug is used both as the wiki filename
    (``assistant/wiki/<slug>.md``) and as the citation token the assistant
    emits, e.g. '(run: 2026-07-10-run-001)'.

    Plot references point at ``/wiki-files/runs/<page-id>/<plot>.png`` — the
    assistant serves ``assistant/wiki/`` under ``/wiki-files`` so the images
    render inline in the web UI. The caller that persists the page is
    responsible for writing those files (see assistant/app/tools.py).
    """
    run_id = run_id or "run-001"
    page_id = f"{date.today().isoformat()}-{run_id}"
    slug = f"runs/{page_id}"
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")

    crit_rows = _critical_speed_rows(analysis)
    crit_summary = ", ".join(f"{c:.1f} rad/s ({c / (2 * np.pi):.1f} Hz)" for _o, c in crit_rows)
    first_crit = (f"{crit_rows[0][1]:.1f} rad/s ({crit_rows[0][1] / (2 * np.pi):.1f} Hz)"
                  if crit_rows else "n/a")

    lines = [
        "---",
        "type: Analysis Run",
        f"title: Rotordynamic Analysis {page_id}",
        f"description: FEA run - {analysis.de * 1e3:.0f} mm x {analysis.le * 1e3:.0f} mm shaft, "
        f"{analysis.md:.2f} kg disk, {analysis.bearing1_type}/{analysis.bearing2_type} bearings; "
        f"first critical speed {first_crit}.",
        "tags: [rotordynamics, fea, simulation-run]",
        f"timestamp: {timestamp}",
        "---",
        "",
        "## Overview",
        "",
        f"Finite-element rotordynamic analysis of a shaft-disk-bearing system, executed by the "
        f"copilot engine (source: engine run {page_id}). Speed range "
        f"{analysis.omega[0]:.0f}-{analysis.omega[-1]:.0f} rad/s over {analysis.n} points, "
        f"with {analysis.bearing1_type} bearing 1 and {analysis.bearing2_type} bearing 2. "
        f"Computed critical speeds: {crit_summary or 'none found in range'}.",
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    lines += [f"| {name} | {val} |" for name, val in _parameter_pairs(analysis)]

    lines += [
        "",
        "## Critical Speeds",
        "",
        "| Order | Angular [rad/s] | Frequency [Hz] | Speed [RPM] |",
        "|---|---|---|---|",
    ]
    if crit_rows:
        lines += [
            f"| {ord_} | {c:.1f} | {c / (2 * np.pi):.1f} | {c / (2 * np.pi) * 60:.0f} |"
            for ord_, c in crit_rows
        ]
    else:
        lines.append("| - | no critical speeds found in the analysed range | - | - |")

    equilibrium = analysis.FM1 + analysis.FM2 - analysis.W
    lines += [
        "",
        "## Bearing Reactions",
        "",
        "| Support | Reaction [N] |",
        "|---|---|",
        f"| Bearing 1, R1 | {analysis.FM1:.2f} |",
        f"| Bearing 2, R2 | {analysis.FM2:.2f} |",
        f"| Disk weight, W | {analysis.W:.2f} |",
        f"| Equilibrium check, Sum F = R1 + R2 - W | {equilibrium:.3e} |",
        "",
        "## Plots",
        "",
    ]
    for key, title, _desc, builder in _PLOT_SPECS:
        # Embed every plot the run can produce (locus only for journal bearings).
        if key == "bearing-locus" and analysis.bearing1_type != "journal" \
                and analysis.bearing2_type != "journal":
            continue
        lines.append(f"**{title}**")
        lines.append("")
        lines.append(f"![{title}](/wiki-files/runs/{page_id}/{key}.png)")
        lines.append("")
    lines.append(f"[Full HTML report](/wiki-files/runs/{page_id}/report.html)")

    lines += [
        "",
        "## Provenance",
        "",
        f"* Generated by `engine.rotordynamics.analysis.RotordynamicAnalysis` on {timestamp}.",
        f"* Citation token: `(run: {page_id})`.",
        "* Method: FEM shaft model (Euler-Bernoulli beam elements, gyroscopic effects), "
        "short-bearing hydrodynamic coefficients via Newton-Raphson equilibrium, "
        "critical speeds from Campbell-diagram 1X intersections.",
        "",
        "## Interpretation Notes",
        "",
        "* Z-direction response amplitudes include the static disk weight in the "
        "rotating force vector (kept for parity with the original MATLAB model), so "
        "Z curves carry the static sag in addition to vibration; Y curves show the "
        "pure unbalance response.",
        "* The model is linear: amplitudes at resonance peaks can exceed the bearing "
        "radial clearance and are resonance indicators, not physically attainable "
        "displacements.",
        "",
    ]
    return slug, "\n".join(lines)
