import numpy as np
import pandas as pd

from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from scipy.signal import find_peaks
from typing import Iterable

from . import core
from .plotresult import PlotResult
from ..config import DataSeriesInfo
from ..data.experiment import Experiment, SimulatedExperiment


def _draw_markers(axes: Iterable[Axes], data: pd.DataFrame, region_data: pd.DataFrame, show_regions: bool | Iterable[str], x: str, kwargs: dict):
    if not show_regions:
        return
    elif isinstance(show_regions, bool):
        show_regions = ["Main Kink", "Secondary Kink", "HF Artefact", "LF Artefact"]

    styles = {
        "Main Kink":      {"color": "tab:blue",   "linestyle": "--"},
        "Secondary Kink": {"color": "tab:cyan",   "linestyle": ":"},
        "HF Artefact":    {"color": "tab:red",    "linestyle": "-."},
        "LF Artefact":    {"color": "tab:orange", "linestyle": "-."},
    }
    default_style = {"color": "k", "linestyle": "-."}

    grouped = core._prepare_groupby(region_data, {"tile": kwargs.get("tile")})

    ax: Axes
    for ax, (_, group) in zip(axes, grouped):
        view = data[data["Sample Name"].isin(group["Sample Name"].unique())]

        for region in show_regions:
            freqs = group[region].unique()
            mask = view["Frequency"].isin(freqs)

            vals = view[x][mask].unique()

            style = styles.get(region, default_style)
            for v in vals:
                ax.axvline(v, linewidth=0.75, zorder=0, **style)


# =================== DIAGNOSTIC PLOT FOR find_regions ===================

_DIAG_PANELS = ("Nyquist", r"$\varphi$", r"$|\mathrm{d}\varphi|$ / max", r"$\varphi$ Tail")

_DIAG_AXIS_LABELS = {
    _DIAG_PANELS[0]: (r"$Re(Z)$ [$\Omega$]", r"$-Im(Z)$ [$\Omega$]"),
    _DIAG_PANELS[1]: ("Index (HF → LF)", r"$\varphi$ [°]"),
    _DIAG_PANELS[2]: ("Index (HF → LF)", r"$|\mathrm{d}\varphi|$ / max"),
    _DIAG_PANELS[3]: ("Index (HF → LF)", r"$\varphi$ [°]"),
}

# point_key → (regions.data column, color, marker, legend label)
_DIAG_POINTS = {
    "kinetic_limit":              ("Kinetic Limit",              "tab:blue",   "o", "A — Kinetic Limit"),
    "diffusive_onset":            ("Diffusive Onset",            "tab:cyan",   "v", "C — Diffusive Onset"),
    "diffusive_capacitive_limit": ("Diffusive-Capacitive Onset", "tab:purple", "s", "B — Diff./Cap. Onset"),
    "hf_artefact_limit":          ("HF Artefact Limit",          "tab:red",    "X", "D — HF Artefact"),
    "inductive_limit":            ("Inductive Limit",            "tab:olive",  "P", "E — Inductive Limit"),
    "lf_artefact_limit":          ("LF Artefact Onset",          "tab:orange", "X", "F — LF Artefact"),
}


def _diag_collect(exps: list[Experiment]) -> tuple[pd.DataFrame, dict]:
    from ..analysis.regions import smooth

    records: list[tuple] = []
    by_name: dict[str, dict] = {}

    for exp_obj in exps:
        region_df = exp_obj.analysis.regions.data  # triggers __call__ if needed
        freqs = exp_obj.data["Frequency"].unique()
        sample_names = list(exp_obj.sample_names)
        ns, nf = len(sample_names), len(freqs)
        palette = exp_obj._palette

        R = smooth(exp_obj.data["Offset-Corrected Resistance"].to_numpy().reshape(ns, nf))
        X = smooth(exp_obj.data["Neg. Reactance"].to_numpy().reshape(ns, nf))
        phi = np.degrees(np.arctan2(np.gradient(-X, axis=1), np.gradient(R, axis=1)))
        dphi = np.gradient(phi, axis=1)

        freq_to_idx = pd.Series(np.arange(nf), index=freqs)
        ordered = region_df.set_index("Sample Name").reindex(sample_names).reset_index()

        for i, name in enumerate(sample_names):
            idx = {}
            for key, (col, *_) in _DIAG_POINTS.items():
                v = ordered.at[i, col]
                idx[key] = -1 if pd.isna(v) else int(freq_to_idx.get(v, -1))

            by_name[name] = {"R": R[i], "X": X[i], "phi": phi[i], "dphi": dphi[i], "idx": idx}

            # Nyquist + full φ traces (φ wird *-1 negiert für visuelle Nyquist-Konvention)
            records.extend((name, palette, _DIAG_PANELS[0], R[i, k], X[i, k]) for k in range(nf))
            records.extend((name, palette, _DIAG_PANELS[1], k, -phi[i, k]) for k in range(nf))

            # |dφ|/max — voller Index-Bereich, Normierung nach max im Suchbereich [A, B]
            A, B = idx["kinetic_limit"], idx["diffusive_capacitive_limit"]
            lo = A if 0 <= A < nf else 0
            hi = B if 0 <= B < nf else nf - 1
            absd_full = np.abs(dphi[i])
            mx = absd_full[lo:hi + 1].max() if hi > lo else absd_full.max()
            norm_full = absd_full / mx if mx > 0 else absd_full
            records.extend((name, palette, _DIAG_PANELS[2], k, float(norm_full[k])) for k in range(nf))

            # φ-Tail: B → F (oder Spektrumsende), nur wenn B existiert; phi auch negiert
            F = idx["lf_artefact_limit"]
            if 0 <= B < nf:
                t_hi = F if 0 <= F < nf else nf
                records.extend((name, palette, _DIAG_PANELS[3], k, -phi[i, k]) for k in range(B, t_hi))

    long_df = pd.DataFrame(records, columns=["Sample Name", "Palette", "Panel", "x", "y"])
    long_df["Panel"] = pd.Categorical(long_df["Panel"], categories=_DIAG_PANELS, ordered=True)
    return long_df, by_name


def _diag_decorate(grid, by_name: dict, cap_angle: float, dip_prominence: float) -> None:
    """Per (sample, panel) Achse: Punkt-Marker, Referenzlinien, Achsenbeschriftung."""
    for (row_val, col_val), ax in grid.axes_dict.items():
        s = by_name.get(row_val)
        if s is None:
            continue
        R, X, phi, dphi, idx = s["R"], s["X"], s["phi"], s["dphi"], s["idx"]
        n = len(R)

        ax.set(xlabel=_DIAG_AXIS_LABELS[col_val][0], ylabel=_DIAG_AXIS_LABELS[col_val][1])

        if col_val == _DIAG_PANELS[0]:  # Nyquist — Marker + axvlines A-F + Zoom
            for key, (_, color, marker, _label) in _DIAG_POINTS.items():
                i = idx[key]
                if 0 <= i < n:
                    ax.axvline(R[i], color=color, alpha=0.5, lw=1, zorder=2)
                    ax.scatter(R[i], X[i], s=110, color=color, marker=marker,
                               edgecolor="white", linewidths=0.8, zorder=5)

            # Auto-Zoom: Halbkreis + bisschen Diffusionsanlauf.
            A, B = idx["kinetic_limit"], idx["diffusive_capacitive_limit"]
            if 0 <= A < n:
                core_end = B if 0 <= B < n else min(n - 1, A + max(5, A // 3))
                end = min(n - 1, core_end + 8)
                start = max(0, A - 8)
                seg = slice(start, end + 1)
                xpad = (R[end] - R[start]) * 0.08
                ax.set_xlim(R[start] - xpad, R[end] + xpad)
                ax.set_ylim(0, max(X[seg].max(), X[end]) * 1.5)

        elif col_val == _DIAG_PANELS[1]:  # φ — Referenzlinien + axvlines (φ wurde negiert)
            ax.axhline(0,  ls=":",  color="#888",    alpha=0.4, lw=1)
            ax.axhline(45, ls="--", color="#10b981", alpha=0.5, lw=1)
            # cap_angle ist find_regions-Schwelle (Konvention: phi_internal >= cap_angle).
            # Nach phi *-1 entspricht das visuell -cap_angle.
            ax.axhline(-cap_angle, ls=":", color="tab:red", alpha=0.5, lw=1)
            for key, (_, color, *_rest) in _DIAG_POINTS.items():
                i = idx[key]
                if 0 <= i < n:
                    ax.axvline(i, color=color, alpha=0.7, lw=1.2, zorder=2)

        elif col_val == _DIAG_PANELS[2]:  # |dφ| / max — voller Bereich, Suchbereich shaded
            A, B = idx["kinetic_limit"], idx["diffusive_capacitive_limit"]
            B_valid = 0 <= B < n
            lo = A if 0 <= A < n else 0
            hi = B if B_valid else n - 1

            # Suchbereich [lo, hi] einfärben
            if hi > lo:
                ax.axvspan(lo, hi, color="#10b981", alpha=0.12, zorder=0)

            # Valleys auf der vollen normierten Reihe (matcht find_regions)
            absd_full = np.abs(dphi)
            mx = absd_full[lo:hi + 1].max() if hi > lo else absd_full.max()
            norm_full = absd_full / mx if mx > 0 else absd_full
            valleys, _ = find_peaks(-norm_full, prominence=dip_prominence)
            if len(valleys):
                ax.scatter(valleys, norm_full[valleys], s=130, color="#10b981",
                           marker="v", edgecolor="white", zorder=5)

            for key in ("kinetic_limit", "diffusive_onset", "diffusive_capacitive_limit"):
                i = idx[key]
                if 0 <= i < n:
                    ax.axvline(i, color=_DIAG_POINTS[key][1], alpha=0.6, lw=1, zorder=2)
            if not B_valid:
                ax.axvline(hi, color=_DIAG_POINTS["diffusive_capacitive_limit"][1],
                           alpha=0.4, lw=1, ls="--", zorder=2)
                ax.text(0.98, 0.97, "B fehlt → Suchbereich bis Ende",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=8, color="#888", style="italic")

        elif col_val == _DIAG_PANELS[3]:  # φ-Tail
            B = idx["diffusive_capacitive_limit"]
            if not (0 <= B < n):
                ax.text(0.5, 0.5, "Tail leer (B fehlt)", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="#888")
                continue
            ax.axhline(90, ls=":",  color="tab:red", alpha=0.5, lw=1)
            ax.axhline(70, ls=":",  color="#888",    alpha=0.4, lw=1)
            ax.axhline(45, ls="--", color="#10b981", alpha=0.5, lw=1)
            for key in ("diffusive_capacitive_limit", "lf_artefact_limit"):
                i = idx[key]
                if 0 <= i < n:
                    ax.axvline(i, color=_DIAG_POINTS[key][1], alpha=0.6, lw=1, zorder=2)


def regions_diagnostic(
    exp: Experiment | list[Experiment],
    title: str | None = None,
    *,
    cap_angle: float = 70.0,
    dip_prominence: float = 0.15,
    **kwargs,
) -> PlotResult:
    from .basics import _listify

    exps = [e for e in _listify(exp) if not isinstance(e, SimulatedExperiment)]


    long_df, by_name = _diag_collect(exps)
    if long_df.empty:
        raise ValueError("regions_diagnostic: keine Samples gefunden.")

    # Dummy-series_info: nötig für _coreplot, wird im Decorator pro Panel überschrieben.
    series_info = {
        "x": DataSeriesInfo("", None, "linear"),
        "y": DataSeriesInfo("", None, "linear"),
    }
    config = {
        "x": "x", "y": "y",
        "row": "Sample Name", "col": "Panel",
        "col_order": list(_DIAG_PANELS),
        "errorbar": None,
        "facet_kws": {"sharex": False, "sharey": False, "margin_titles": True},
    } | kwargs

    res = core.lineplot(long_df, title=title, series_info=series_info, **config)
    with res as (fig, _):
        grid = res.get_meta("grid")
        grid.set_titles(row_template="{row_name}", col_template="{col_name}")
        _diag_decorate(grid, by_name, cap_angle, dip_prominence)

        # Figure-Legende für die Punkt-Marker
        handles = [
            Line2D([], [], color=color, marker=marker, linestyle="None", markersize=9,
                   markeredgecolor="white", markeredgewidth=0.6, label=label)
            for _, color, marker, label in _DIAG_POINTS.values()
        ]
        handles.append(Line2D([], [], color="#10b981", marker="v", linestyle="None",
                              markersize=9, markeredgecolor="white", label="Delle (dφ)"))
        fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                   fontsize=9, frameon=False)

    return res
