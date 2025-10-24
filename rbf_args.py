# rbf_args.py
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Normal-based mesh comparison and ParaView exports."
    )

    # ------------------------------------------------------------------
    # Required I/O
    # ------------------------------------------------------------------
    req = p.add_argument_group("Required inputs")
    req.add_argument("--ref", required=True, help="Path to reference VTU")
    req.add_argument("--target", required=True, help="Path to target VTU")

    # Optional legacy/placeholder output (no longer required)
    p.add_argument(
        "--out",
        required=False,
        help="(optional) legacy output path used by some modes; safe to omit"
    )

    # ------------------------------------------------------------------
    # General
    # ------------------------------------------------------------------
    gen = p.add_argument_group("General")
    gen.add_argument("--verbose", action="store_true", help="Verbose logs")

    # ------------------------------------------------------------------
    # Normals exports (reference / 50th)
    # ------------------------------------------------------------------
    g_norm = p.add_argument_group("Normals (reference exports)")
    g_norm.add_argument("--export-ref-with-normals", help="VTP: reference surface carrying 'Normals'")
    g_norm.add_argument("--export-normals-glyphs", help="VTP: arrow glyphs for normals (sampled)")
    g_norm.add_argument("--glyph-scale", type=float, default=1.0, help="Glyph overall size multiplier")
    g_norm.add_argument("--glyph-sample-step", type=int, default=50, help="Sample every Nth cell for glyphs")

    # ------------------------------------------------------------------
    # Single-normal debug mode
    # ------------------------------------------------------------------
    g_single = p.add_argument_group("Single-normal mode")
    g_single.add_argument("--single-cell-index", type=int, help="Reference triangle index to raycast")
    g_single.add_argument("--ray-len-factor", type=float, default=2.0, help="Ray length as bbox diagonal factor")
    g_single.add_argument("--no-both-directions", action="store_true", help="Cast only along +normal")
    g_single.add_argument("--export-single-ray", help="VTP: debug ray line/sphere")
    g_single.add_argument("--export-single-hit-morph", help="VTP: morph output for single-normal mode")

    # Single-normal extras
    g_single.add_argument("--export-single-normal-glyph", help="VTP: single arrow glyph at chosen normal on 50th")
    g_single.add_argument(
        "--single-glyph-length-frac",
        type=float,
        default=0.02,
        help="Glyph length as fraction of 50th bbox diagonal (default 0.02)",
    )
    g_single.add_argument("--export-hit-markers", help="VTP: outline + sphere marker on morph hit cell")
    g_single.add_argument("--marker-size-factor", type=float, default=0.01, help="Marker radius as fraction of bbox")
    g_single.add_argument(
        "--color-by-hit-distance",
        action="store_true",
        help="If set, color entire morph by distance from hit point; else paint only the hit cell."
    )

    # ------------------------------------------------------------------
    # Multi-normal batch
    # ------------------------------------------------------------------
    g_batch = p.add_argument_group("Multi-normal batch")
    g_batch.add_argument(
        "--multi-cell-indices",
        help="Comma-separated list of 50th triangle indices (e.g., 100,200,300)"
    )
    g_batch.add_argument(
        "--multi-auto",
        type=int,
        help="Auto-pick this many reference triangles for batch mode"
    )
    g_batch.add_argument(
        "--multi-auto-percent",
        type=float,
        help="Auto-pick this percent of reference triangles (e.g., 20 for 20%)"
    )
    g_batch.add_argument(
        "--multi-auto-mode",
        choices=["uniform", "random"],
        default="uniform",
        help="Selection mode for auto-pick (default: uniform)"
    )
    g_batch.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for --multi-auto-mode random (default: 42)"
    )
    g_batch.add_argument(
        "--export-multi-normals-glyph",
        help="VTP: 50th arrows for all requested normals, colored by absolute hit distance"
    )
    g_batch.add_argument(
        "--export-distances-csv",
        help="CSV: per-normal results (indices, distances, hit points, etc.)"
    )
    g_batch.add_argument(
        "--multi-glyph-length-frac",
        type=float,
        default=0.02,
        help="Arrow length as fraction of 50th bbox diagonal (default 0.02)"
    )
    g_batch.add_argument(
        "--export-batch-hit-morph",
        help="VTP: single morph mesh with ONLY batch-hit triangles colored by distance (others NaN)"
    )
    g_batch.add_argument(
        "--batch-use-signed",
        action="store_true",
        help="If set, color batch hits by signed distance; otherwise absolute distance"
    )
    g_batch.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Log progress every N completed rays (default: 100). Applies to parallel and sequential batch."
)


    # ------------------------------------------------------------------
    # Global signed distance / contours
    # ------------------------------------------------------------------
    g_global = p.add_argument_group("Global signed distance / contours")
    g_global.add_argument("--export-morph-distance", help="VTP: morph with signed distance at points")
    g_global.add_argument("--export-morph-contours", help="VTP: contour mesh from signed distance")
    g_global.add_argument("--contour-levels", help="Comma-separated levels, e.g. -3,-2,-1,0,1,2,3")
    g_global.add_argument("--n-iso", type=int, default=12, help="Number of contour isosurfaces if levels not provided")

    # ------------------------------------------------------------------
    # Caching / restyle (no recompute)
    # ------------------------------------------------------------------
    g_cache = p.add_argument_group("Caching / restyle")
    g_cache.add_argument("--save-batch", help="NPZ file to save batch results (for replotting later)")
    g_cache.add_argument("--load-batch", help="NPZ file to load prior batch results (skip ray casting)")
    g_cache.add_argument(
        "--recolor-batch-hit-morph",
        help="VTP: recolor the morph using cached results (no ray casting)"
    )
    g_cache.add_argument(
        "--recolor-use-signed",
        action="store_true",
        help="Use signed distances when recoloring from cache"
    )

    g_batch.add_argument("--parallel", type=int,
                     help="Number of worker processes for batch mode (parallel). If omitted, runs sequentially.")
    g_batch.add_argument("--chunksize", type=int, default=64,
                     help="Task chunk size per worker for map scheduling (default: 64)")


    return p

def parse_args(argv=None):
    return build_parser().parse_args(argv)
