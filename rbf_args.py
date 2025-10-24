# rbf_args.py
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Normal-based mesh comparison and ParaView exports.")

    # Required
    p.add_argument("--ref", required=True, help="Path to reference VTU")
    p.add_argument("--target", required=True, help="Path to target VTU")
    p.add_argument("--out", required=True, help="Placeholder/output path for full modes")

    # Verbosity
    p.add_argument("--verbose", action="store_true", help="Verbose logs")

    # Normals exports (reference / 50th)
    p.add_argument("--export-ref-with-normals", help="VTP: reference surface carrying 'Normals'")
    p.add_argument("--export-normals-glyphs", help="VTP: arrow glyphs for normals (sampled)")
    p.add_argument("--glyph-scale", type=float, default=1.0)
    p.add_argument("--glyph-sample-step", type=int, default=50)

    # Single-normal debug mode
    p.add_argument("--single-cell-index", type=int, help="Reference triangle index to raycast")
    p.add_argument("--ray-len-factor", type=float, default=2.0)
    p.add_argument("--no-both-directions", action="store_true")
    p.add_argument("--export-single-ray", help="VTP: debug rays")
    p.add_argument("--export-single-hit-morph", help="VTP: morph output for single-normal mode")

    # Single-normal extras
    p.add_argument("--export-single-normal-glyph", help="VTP: single arrow glyph at the chosen normal on 50th")
    p.add_argument("--single-glyph-length-frac", type=float, default=0.02,
                  help="Glyph length as fraction of 50th bbox diagonal (default 0.02)")
    p.add_argument("--export-hit-markers", help="VTP: outline + sphere marker on morph hit cell")
    p.add_argument("--marker-size-factor", type=float, default=0.01)

    # Coloring choice for single-normal output
    p.add_argument("--color-by-hit-distance", action="store_true",
                  help="If set, color the entire morph by distance from the hit point. Otherwise paint only the hit cell.")

    # Multi-normal batch (optional)
    p.add_argument("--multi-cell-indices",
                  help="Comma-separated list of 50th triangle indices to test (e.g., 100,200,300,...)")
    p.add_argument("--export-multi-normals-glyph",
                  help="VTP: 50th arrows for all requested normals, colored by absolute hit distance")
    p.add_argument("--export-distances-csv", help="CSV: per-normal results")
    p.add_argument("--multi-glyph-length-frac", type=float, default=0.02)

    # Global signed distance / contours (optional)
    p.add_argument("--export-morph-distance", help="VTP: morph with signed distance at points")
    p.add_argument("--export-morph-contours", help="VTP: contour mesh from signed distance")
    p.add_argument("--contour-levels", help="Comma-separated levels, e.g. -3,-2,-1,0,1,2,3")
    p.add_argument("--n-iso", type=int, default=12)

    p.add_argument("--export-batch-hit-morph",
              help="VTP: single morph mesh with ONLY the batch-hit triangles colored by distance (others NaN)")
    p.add_argument("--batch-use-signed", action="store_true",
              help="If set, color batch hits by signed distance; otherwise absolute distance.")


    p.add_argument("--multi-auto", type=int,
              help="Auto-pick this many reference triangles for batch mode")
    p.add_argument("--multi-auto-mode", choices=["uniform","random"], default="uniform",
              help="How to pick --multi-auto triangles (default: uniform)")


    return p

def parse_args(argv=None):
    return build_parser().parse_args(argv)
