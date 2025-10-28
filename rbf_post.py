# rbf_post.py
"""
Post-processing utilities for RBF-Comparisons.
Use cached NPZ results to (a) paint a morph mesh VTP without recomputing,
and (b) generate analysis plots (histograms, CDFs, summary text).
"""

from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Dict, Any, Iterable, Tuple, Optional

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


# --------------------------
# Cache I/O
# --------------------------

def load_cache(npz_path: str) -> Dict[str, Any]:
    z = np.load(npz_path, allow_pickle=True)
    return {
        "centers": z["centers"],
        "normals": z["normals"],
        "distances": z["distances"],
        "signed_values": z["signed_values"],
        "hits_meta": list(z["hits_meta"]),
        "glyph_length": float(z["glyph_length"][0]),
    }


# --------------------------
# Cast-from auto-detect helper
# --------------------------

def infer_cast_from(cache: Dict[str, Any],
                    morph_ncells: int,
                    ref_ncells: int) -> str:
    """
    Try to infer whether rays were cast FROM 'morph' or 'ref'
    based on the source index (column 0 of hits_meta).

    Heuristic:
      - If most source indices (ci) are < morph_ncells -> 'morph'
      - Else if most ci < ref_ncells -> 'ref'
      - Fallback to 'morph'
    """
    src_idx = np.array([row[0] for row in cache["hits_meta"] if row is not None], dtype=float)
    src_idx = src_idx[np.isfinite(src_idx)]
    if src_idx.size == 0:
        return "morph"

    frac_in_morph = np.mean((src_idx >= 0) & (src_idx < morph_ncells))
    frac_in_ref   = np.mean((src_idx >= 0) & (src_idx < ref_ncells))

    if frac_in_morph >= frac_in_ref:
        return "morph"
    return "ref"


# --------------------------
# Painting morph from cache
# --------------------------

def build_morph_distance_array_from_cache(
    morph_surf: pv.PolyData,
    cache: Dict[str, Any],
    cast_from: str = "auto",
    use_signed: bool = False,
    array_name: str = "batch_hit_distance",
    default_nan: bool = True,
) -> pv.PolyData:
    """
    Create/attach a cell-data array on morph_surf with distances from cache:
      - If cast_from=="morph": color the morph ORIGIN cell ids (column 0 of hits_meta).
      - If cast_from=="ref":   color the morph HIT cell ids    (column 1 of hits_meta).
      - If cast_from=="auto":  infer using mesh sizes.

    Non-touched cells are NaN by default (so ParaView can gray them out).
    """
    morph = morph_surf.copy()
    n = morph.n_cells

    values = cache["signed_values"] if use_signed else cache["distances"]
    values = np.asarray(values, dtype=float)

    # Prepare destination array (NaNs)
    arr = np.full(n, np.nan, dtype=float) if default_nan else np.zeros(n, dtype=float)

    # Decide mapping


    # hits_meta row: (ci, hit_cell, dist, signed, sign_dir, cx,cy,cz, nx,ny,nz, hx,hy,hz)

    # Safe origin ids (source cell indices)
    origin_ids_raw = np.array(
        [row[0] if (row is not None and row[0] is not None) else -1 for row in cache["hits_meta"]],
        dtype=int
    )

    # Safe hit ids (can be NaN in cache)
    hit_ids_raw = np.array(
        [row[1] if (row is not None and row[1] is not None) else np.nan for row in cache["hits_meta"]],
        dtype=float
    )
    hit_ids_int = np.full(hit_ids_raw.shape, -1, dtype=int)
    mask = np.isfinite(hit_ids_raw) & (hit_ids_raw >= 0)
    hit_ids_int[mask] = hit_ids_raw[mask].astype(int)

    # Auto-detect if requested
    if cast_from not in ("morph", "ref"):
        frac_in_morph = np.mean((origin_ids_raw >= 0) & (origin_ids_raw < n))
        cast_from = "morph" if frac_in_morph >= 0.5 else "ref"

    # Map values to morph cell ids
    idxs = origin_ids_raw if cast_from == "morph" else hit_ids_int
    valid = (idxs >= 0) & (idxs < n)
    arr[idxs[valid]] = values[valid]




    #####################################################
    morph.cell_data[array_name] = arr
    morph.set_active_scalars(array_name)
    return morph


# --------------------------
# Styled screenshot (optional)
# --------------------------

@dataclass
class RenderStyle:
    title: Optional[str] = None
    background: str | Tuple[float, float, float] = "white"
    show_edges: bool = False
    scalarbar_title: str = "Distance"
    scalarbar_fmt: str = "%.2f"
    window_size: Tuple[int, int] = (1280, 960)
    cmap: Optional[str] = None      # e.g., "coolwarm"
    reverse_cmap: bool = False
    nan_color: Tuple[float, float, float] = (0.7, 0.7, 0.7)  # light gray
    lighting: str = "light_kit"     # or "none"
    edge_color: Tuple[float, float, float] = (0.2, 0.2, 0.2)
    line_width: float = 1.0
    show_axes: bool = False


def save_styled_screenshot(
    morph_colored: pv.PolyData,
    screenshot_path: str,
    style: RenderStyle = RenderStyle(),
):
    """Render a quick screenshot with custom background, scalar bar, etc."""
    p = pv.Plotter(off_screen=True, window_size=style.window_size)
    p.set_background(style.background)

    # Configure colormap
    cm = style.cmap
    if cm is None:
        cm = "coolwarm"
    if style.reverse_cmap:
        cm = cm + "_r" if not cm.endswith("_r") else cm

    # NaN color via scalar_bar arguments
    actor = p.add_mesh(
        morph_colored,
        show_edges=style.show_edges,
        edge_color=style.edge_color,
        line_width=style.line_width,
        cmap=cm,
        nan_color=style.nan_color,
        lighting=style.lighting,
    )

    # Scalar bar
    p.add_scalar_bar(
        title=style.scalarbar_title,
        fmt=style.scalarbar_fmt,
        title_font_size=12,
        label_font_size=10,
        shadow=False,
        n_labels=5,
        italic=False,
        bold=False,
        position_x=0.85,
        position_y=0.1,
        width=0.12,
        height=0.8,
    )

    if style.show_axes:
        p.show_axes()

    if style.title:
        p.add_text(style.title, position="upper_edge", color="black", font_size=14)

    p.show(screenshot=True, auto_close=False)
    p.screenshot(screenshot_path)
    p.close()
    return screenshot_path


# --------------------------
# Analysis plots
# --------------------------

def save_histogram(
    distances: np.ndarray,
    out_png: str,
    bins: int = 60,
    title: str = "Distance Histogram",
    xlim: tuple[float, float] | None = None,
    percentile: float = 99.5,   # robust trim (ignored if xlim is provided)
):
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        logging.warning("No valid distances to plot histogram.")
        return out_png

    if xlim is None:
        p_low, p_high = np.percentile(d, [100 - percentile, percentile])
        if np.all(d >= 0):
            xmin = 0.0
            xmax = float(np.ceil(p_high * 1.05))
        else:
            xmin = float(np.floor(p_low * 1.05))
            xmax = float(np.ceil(p_high * 1.05))
        # hard guardrails (can be relaxed as you like)
        xmin = max(xmin, -50.0)
        xmax = min(xmax,  50.0)
    else:
        xmin, xmax = xlim

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=bins, edgecolor="black", alpha=0.75)
    plt.xlabel("Distance [mm]", fontsize=12)
    plt.ylabel("Triangle Count", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(xmin, xmax)
    plt.grid(True, alpha=0.3)

    mean = np.mean(d); std = np.std(d)
    plt.text(
        0.98, 0.95, f"Mean: {mean:.2f} mm\nStd: {std:.2f} mm",
        transform=plt.gca().transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.3")
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logging.info(f"Wrote histogram plot: {out_png}")
    return out_png


def save_cdf(
    distances: np.ndarray,
    out_png: str,
    title: str = "Distance CDF",
    xlim: tuple[float, float] | None = None,
    percentile: float = 99.5,
):
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        d = np.array([0.0])
    d_sorted = np.sort(d)
    y = np.linspace(0, 1, len(d_sorted), endpoint=True)

    if xlim is None:
        p_low, p_high = np.percentile(d, [100 - percentile, percentile])
        if np.all(d >= 0):
            xmin = 0.0
            xmax = float(np.ceil(p_high * 1.05))
        else:
            xmin = float(np.floor(p_low * 1.05))
            xmax = float(np.ceil(p_high * 1.05))
        xmin = max(xmin, -50.0)
        xmax = min(xmax,  50.0)
    else:
        xmin, xmax = xlim

    plt.figure(figsize=(7, 5))
    plt.plot(d_sorted, y)
    plt.xlabel("Distance [mm]", fontsize=12)
    plt.ylabel("Cumulative Probability", fontsize=12)
    plt.title(title, fontsize=14)
    plt.xlim(xmin, xmax)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    logging.info(f"Wrote CDF plot: {out_png}")
    return out_png


def save_boxplot(distances: np.ndarray, out_png: str, title: str = "Distance Boxplot"):
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    plt.figure()
    plt.boxplot(d, vert=True, whis=1.5, showfliers=True)
    plt.ylabel("Distance")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def summarize_distances(distances: np.ndarray) -> Dict[str, float]:
    d = np.asarray(distances, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(count=0, mean=np.nan, std=np.nan, min=np.nan, p50=np.nan, p95=np.nan, max=np.nan)
    return dict(
        count=int(d.size),
        mean=float(np.mean(d)),
        std=float(np.std(d)),
        min=float(np.min(d)),
        p50=float(np.percentile(d, 50)),
        p95=float(np.percentile(d, 95)),
        max=float(np.max(d)),
    )
