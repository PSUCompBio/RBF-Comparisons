# rbf_post_cli.py
"""
CLI front-end to post-process RBF cache (.npz) without recomputing:
- Paint morph VTP from cache (with signed/abs distances)
- Optional styled screenshot
- Analysis plots (hist, CDF, boxplot)
"""

import argparse
import logging
import os

import numpy as np
import pyvista as pv

from rbf_post import (
    load_cache,
    build_morph_distance_array_from_cache,
    RenderStyle,
    save_styled_screenshot,
    save_histogram,
    save_cdf,
    save_boxplot,
    summarize_distances,
)


def parse_args():
    p = argparse.ArgumentParser(description="Post-process RBF .npz cache (no recompute).")
    p.add_argument("--ref", required=True, help="Path to reference VTU (used only for size/inference)")
    p.add_argument("--target", required=True, help="Path to target/morph VTU (to paint)")
    p.add_argument("--load-batch", required=True, help="NPZ cache from previous batch")
    p.add_argument("--cast-from", choices=["auto", "morph", "ref"], default="auto",
                   help="Where rays originated (auto-detect if unknown)")
    p.add_argument("--use-signed", action="store_true", help="Use signed values instead of absolute distances")

    # Outputs

    p.add_argument("--export-morph-colored", dest="export_morph_colored",
               help="Write VTP of morph colored by distance")

    p.add_argument("--screenshot", help="Save a styled screenshot PNG of the colored morph")

    # Style knobs for screenshot
    p.add_argument("--bg", default="white", help="Background color (e.g., white, black, #222222)")
    p.add_argument("--cmap", default="coolwarm", help="Matplotlib colormap name (e.g., coolwarm, viridis)")
    p.add_argument("--reverse-cmap", action="store_true", help="Reverse the colormap")
    p.add_argument("--scalarbar-title", default="Distance", help="Scalar bar title")
    p.add_argument("--nan-gray", action="store_true", help="Show non-hit cells as gray in the screenshot")

    # Plots
    p.add_argument("--plots-prefix", help="Prefix for analysis plots (will produce _hist.png, _cdf.png, _box.png)")
    p.add_argument("--bins", type=int, default=60, help="Histogram bins")

    p.add_argument("--hist-xmin", type=float, help="Histogram x-axis min [mm]")
    p.add_argument("--hist-xmax", type=float, help="Histogram x-axis max [mm]")
    p.add_argument("--hist-percentile", type=float, default=99.5,
                   help="Histogram/CDF percentile cap (e.g., 99, 99.5). Ignored if x-lims are given.")
    p.add_argument("--cdf-xmin", type=float, help="CDF x-axis min [mm]")
    p.add_argument("--cdf-xmax", type=float, help="CDF x-axis max [mm]")



    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")

    # Load meshes
    ref = pv.read(args.ref)
    morph = pv.read(args.target)
    ref_surf = ref.extract_surface().triangulate()
    morph_surf = morph.extract_surface().triangulate()

    # Load cache
    cache = load_cache(args.load_batch)
    logging.info(f"Loaded cache: {args.load_batch}")

    # Decide cast_from if auto
    cast_from = args.cast_from
    if cast_from == "auto":
        # Simple inference using mesh sizes
        from rbf_post import infer_cast_from
        cast_from = infer_cast_from(cache, morph_surf.n_cells, ref_surf.n_cells)
        logging.info(f"Inferred cast-from: {cast_from}")

    # Build colored morph (cell data attached)
    morph_colored = build_morph_distance_array_from_cache(
        morph_surf=morph_surf,
        cache=cache,
        cast_from=cast_from,
        use_signed=args.use_signed,
        array_name="batch_hit_distance",
        default_nan=True if args.nan_gray else False,
    )


            # Export VTP if requested
    if args.export_morph_colored:

        morph_colored.save(args.export_morph_colored)  # native VTK writer; no meshio needed

        logging.info(f"Wrote colored morph: {args.export_morph_colored}")


    # Styled screenshot (optional)
    if args.screenshot:
        style = RenderStyle(
            background=args.bg,
            scalarbar_title=args.scalarbar_title,
            cmap=args.cmap,
            reverse_cmap=args.reverse_cmap,
            nan_color=(0.7, 0.7, 0.7) if args.nan_gray else (1.0, 1.0, 1.0),
            show_edges=False,
        )
        save_styled_screenshot(morph_colored, args.screenshot, style=style)
        logging.info(f"Wrote screenshot: {args.screenshot}")

    # Analysis plots
    if args.plots_prefix:
        d_abs = np.asarray(cache["distances"], dtype=float)
        d_signed = np.asarray(cache["signed_values"], dtype=float)
        d = d_signed if args.use_signed else d_abs

        hist_png = f"{args.plots_prefix}_hist.png"
        cdf_png  = f"{args.plots_prefix}_cdf.png"
        box_png  = f"{args.plots_prefix}_box.png"

        hist_xlim = None
        if args.hist_xmin is not None and args.hist_xmax is not None:
            hist_xlim = (args.hist_xmin, args.hist_xmax)

        cdf_xlim = None
        if args.cdf_xmin is not None and args.cdf_xmax is not None:
            cdf_xlim = (args.cdf_xmin, args.cdf_xmax)

        save_histogram(
            d, hist_png, bins=args.bins,
            title=("Signed Distance Histogram" if args.use_signed else "Absolute Distance Histogram"),
            xlim=hist_xlim,
            percentile=args.hist_percentile,
        )
        save_cdf(
            d, cdf_png,
            title=("Signed Distance CDF" if args.use_signed else "Absolute Distance CDF"),
            xlim=cdf_xlim,
            percentile=args.hist_percentile,
        )
        save_boxplot(
            d, box_png,
            title=("Signed Distance Boxplot" if args.use_signed else "Absolute Distance Boxplot"),
        )


        stats = summarize_distances(d)
        logging.info(f"Summary: n={stats['count']}  mean={stats['mean']:.3f}  std={stats['std']:.3f}  "
                     f"min={stats['min']:.3f}  p50={stats['p50']:.3f}  p95={stats['p95']:.3f}  max={stats['max']:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
