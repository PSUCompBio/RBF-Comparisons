#!/usr/bin/env python
from __future__ import annotations
import os, sys, math, time, logging, argparse
from typing import Optional, Tuple, List

try:
    import numpy as np
    import pyvista as pv
    from pyvista import _vtk  # for vtkImplicitPolyDataDistance
except Exception as e:
    print("ERROR: Requires numpy and pyvista.\nInstall with: pip install numpy pyvista\n", e)
    sys.exit(1)

# -----------------------------
# Logging
# -----------------------------
def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

# -----------------------------
# I/O helpers
# -----------------------------
def resolve_path(path: str) -> str:
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"File not found: {ap}")
    return ap

def load_mesh(path: str) -> pv.DataSet:
    path = resolve_path(path)
    logging.info(f"Loading mesh: {path}")
    m = pv.read(path)
    logging.debug(f"type={type(m).__name__}, points={getattr(m,'n_points','n/a')}, cells={getattr(m,'n_cells','n/a')}")
    return m

def extract_surface(mesh: pv.DataSet, triangulate: bool = True) -> pv.PolyData:
    logging.info("Extracting surface...")
    s = mesh.extract_surface()
    if triangulate:
        s = s.triangulate()
    logging.debug(f"surface points={s.n_points} cells={s.n_cells}")
    return s

def save_poly(poly: pv.PolyData, path: str) -> str:
    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    poly.save(out)
    logging.info(f"Saved: {out}")
    return out

def bbox_diag(poly: pv.PolyData) -> float:
    x0, x1, y0, y1, z0, z1 = poly.bounds
    return float(np.linalg.norm([x1 - x0, y1 - y0, z1 - z0]))

def export_hit_markers_on_morph(
    morph_surf: pv.PolyData,
    hit_cell_id: int,
    hit_point: Optional[np.ndarray],
    out_path: str,
    size_factor: float = 0.01,
) -> str:
    """
    Writes a VTP with:
      • an outline of the hit cell
      • a small sphere at the hit point (or cell center)
    size_factor is relative to morph bounding-box diagonal.
    """
    if hit_cell_id is None or hit_cell_id < 0 or hit_cell_id >= morph_surf.n_cells:
        raise ValueError("Invalid hit_cell_id for markers.")

    # Extract the single hit cell
    cell_poly = morph_surf.extract_cells(hit_cell_id)

    # Outline (feature edges) so the triangle boundary is obvious
    outline = cell_poly.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=True,
        feature_edges=False,
        manifold_edges=False,
    )
    outline["marker_id"] = np.ones(outline.n_points)  # dummy scalar

    # Pick marker center
    if hit_point is None:
        center = cell_poly.cell_centers().points[0]
    else:
        center = np.array(hit_point, dtype=float)

    # Sphere size relative to model scale
    diag = bbox_diag(morph_surf)
    radius = max(1e-9, diag * float(size_factor))
    sphere = pv.Sphere(radius=radius, center=center)
    sphere["marker_id"] = np.full(sphere.n_points, 2.0)

    # Merge outline + sphere and save
    markers = outline.merge(sphere)
    return save_poly(markers, out_path)


# -----------------------------
# Geometry utilities
# -----------------------------
def compute_cell_normals(poly: pv.PolyData, auto_orient=True) -> pv.PolyData:
    logging.info("Computing per-cell normals...")
    poly = poly.copy()
    poly.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=auto_orient, inplace=True)
    if "Normals" not in poly.cell_data:
        raise RuntimeError("Cell normals missing after compute_normals()")
    return poly

def combined_bbox_diag(a: pv.PolyData, b: pv.PolyData) -> float:
    x0 = min(a.bounds[0], b.bounds[0]); x1 = max(a.bounds[1], b.bounds[1])
    y0 = min(a.bounds[2], b.bounds[2]); y1 = max(a.bounds[3], b.bounds[3])
    z0 = min(a.bounds[4], b.bounds[4]); z1 = max(a.bounds[5], b.bounds[5])
    return float(np.linalg.norm([x1-x0, y1-y0, z1-z0]))

# -----------------------------
# NEW: ParaView helpers
# -----------------------------
def write_ref_with_normals(ref_surf: pv.PolyData, out_path: str) -> str:
    """
    Save the 50th surface with a 'Normals' cell vector array.
    In ParaView: open this VTP, then use Filters → Glyph (Vectors='Normals', Glyph Type=Arrow).
    """
    if "Normals" not in ref_surf.cell_data:
        raise ValueError("Reference surface has no 'Normals'. Call compute_cell_normals() first.")
    return save_poly(ref_surf, out_path)

def write_normals_glyphs(
    ref_surf: pv.PolyData,
    out_path: str,
    glyph_scale: float = 1.0,
    sample_step: int = 50,
) -> str:
    """
    Create arrow glyph geometry for normals and save as a VTP (ready to load).
    """
    logging.info(f"Building normals glyphs (sample_step={sample_step}, scale={glyph_scale})...")
    centers = ref_surf.cell_centers().points[::sample_step]
    normals = ref_surf.cell_data["Normals"][::sample_step]
    pts = pv.PolyData(centers)
    pts.point_data.set_array(normals, "Normals", deep=True)
    arrows = pts.glyph(orient="Normals", scale="Normals", factor=glyph_scale, geom=pv.Arrow())
    return save_poly(arrows, out_path)

# -----------------------------
# Ray casting (single cell or many)
# -----------------------------
def _nearest_hit_distance(origin: np.ndarray, hit_points: Optional[np.ndarray]) -> Optional[float]:
    if hit_points is None or len(hit_points) == 0:
        return None
    d = np.linalg.norm(hit_points - origin, axis=1)
    return float(d.min())

def ray_length_from_diag(ref_surf: pv.PolyData, target_surf: pv.PolyData, factor: float) -> float:
    diag = combined_bbox_diag(ref_surf, target_surf)
    return max(1e-9, float(diag) * float(factor))


def cast_single_normal(
    ref_surf_with_normals: pv.PolyData,
    target_surf: pv.PolyData,
    cell_index: int,
    ray_len_factor: float = 2.0,
    both_directions: bool = True,
):
    """
    Cast rays for exactly one reference triangle.
    Returns (best_distance, hit_point, hit_cell_id, center, normal, sign_dir)
      - sign_dir = +1 if +normal ray produced the nearest hit
                   -1 if -normal ray produced the nearest hit
                    0 if no hit
    """
    if "Normals" not in ref_surf_with_normals.cell_data:
        raise ValueError("Reference surface missing 'Normals'")

    Nc = ref_surf_with_normals.n_cells
    if not (0 <= cell_index < Nc):
        raise IndexError(f"cell_index {cell_index} out of range [0, {Nc-1}]")

    centers = ref_surf_with_normals.cell_centers().points
    center = centers[cell_index]
    n = ref_surf_with_normals.cell_data["Normals"][cell_index]
    nmag = np.linalg.norm(n)
    if nmag == 0 or not math.isfinite(nmag):
        logging.warning("Selected normal is degenerate.")
        return None, None, None, center, n, 0
    n = n / nmag

    ray_len = ray_length_from_diag(ref_surf_with_normals, target_surf, ray_len_factor)

    best_d, best_pt, best_cell = None, None, None
    sign_dir = 0  # <-- initialize

    def pick_nearest(origin, pts, ids):
        if pts is None or len(pts) == 0:
            return None, None, None
        d = np.linalg.norm(pts - origin, axis=1)
        k = int(np.argmin(d))
        return float(d[k]), pts[k], int(ids[k])

    # +n
    pts_f, ids_f = target_surf.ray_trace(center, center + n * ray_len)
    d_f, pt_f, id_f = pick_nearest(center, pts_f, ids_f)

    # -n (optional)
    d_b = pt_b = id_b = None
    if both_directions:
        pts_b, ids_b = target_surf.ray_trace(center, center - n * ray_len)
        d_b, pt_b, id_b = pick_nearest(center, pts_b, ids_b)

    # choose winner
    if d_f is not None and (d_b is None or d_f <= d_b):
        best_d, best_pt, best_cell, sign_dir = d_f, pt_f, id_f, +1
    elif d_b is not None:
        best_d, best_pt, best_cell, sign_dir = d_b, pt_b, id_b, -1

    if best_d is None:
        logging.info("No intersection found for this normal.")
        return None, None, None, center, n, 0

    logging.info(f"Single-normal hit: cell={best_cell}, distance={best_d:.6g}, sign_dir={sign_dir:+d}")
    return best_d, best_pt, best_cell, center, n, sign_dir



def paint_single_hit_on_morph(
    morph_surf: pv.PolyData,
    hit_cell_id: int,
    distance_value: float,
    out_path: str,
    array_name: str = "single_normal_distance",
) -> str:
    """
    Create a cell scalar on the morphed surface where only the hit cell gets `distance_value`;
    all other cells are NaN (so ParaView will render them with NaN color or you can enable NaN color).
    """
    if hit_cell_id is None or hit_cell_id < 0 or hit_cell_id >= morph_surf.n_cells:
        raise ValueError("Invalid hit_cell_id for painting the morphed surface.")

    out = morph_surf.copy()
    vals = np.full(out.n_cells, np.nan, dtype=float)
    vals[hit_cell_id] = float(distance_value)
    out.cell_data[array_name] = vals
    return save_poly(out, out_path)


def write_debug_ray(center: np.ndarray, hit: Optional[np.ndarray], normal: np.ndarray, length: float, out_path: str) -> str:
    """
    Save a tiny debug scene: a line arrow for the normal (+n) and an optional line to the hit point.
    """
    geoms: List[pv.PolyData] = []
    # normal line (forward only, visual aid)
    a = center
    b = center + normal * length * 0.5
    normal_line = pv.Line(a, b)
    normal_line["id"] = np.array([1.0, 1.0])  # dummy scalar
    geoms.append(normal_line)
    # hit line if any
    if hit is not None:
        hit_line = pv.Line(center, hit)
        hit_line["id"] = np.array([2.0, 2.0])
        geoms.append(hit_line)
        # also add a little sphere at hit
        geoms.append(pv.Sphere(radius=length*0.01, center=hit))
    # merge
    scene = geoms[0]
    for g in geoms[1:]:
        scene = scene.merge(g)
    return save_poly(scene, out_path)

# -----------------------------
# NEW: Signed distance on MORPHED + optional contours
# -----------------------------
def compute_signed_distance_on_morph(morph_surf: pv.PolyData, ref_surf: pv.PolyData, array_name="signed_distance_to_ref") -> pv.PolyData:
    """
    Uses vtkImplicitPolyDataDistance to evaluate signed distance at each morph point to the reference surface.
    Positive/negative sign depends on ref surface normal orientation.
    """
    ipd = _vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(ref_surf)
    pts = morph_surf.points
    vals = np.empty(pts.shape[0], dtype=float)
    for i, p in enumerate(pts):
        vals[i] = ipd.EvaluateFunction(p)
    out = morph_surf.copy()
    out.point_data[array_name] = vals
    return out

def write_morph_contours(
    morph_with_distance: pv.PolyData,
    array_name: str,
    out_contour_path: str,
    levels: Optional[List[float]] = None,
    n_iso: int = 12,
) -> str:
    """
    Build a contour mesh from the signed-distance scalar and save as VTP.
    If 'levels' not given, auto-select n_iso evenly over scalar range.
    """
    scal = morph_with_distance.point_data[array_name]
    smin, smax = float(np.min(scal)), float(np.max(scal))
    if levels is None:
        levels = list(np.linspace(smin, smax, n_iso))
    logging.info(f"Creating contour mesh at {len(levels)} levels in [{smin:.6g}, {smax:.6g}]...")
    contour = morph_with_distance.contour(isosurfaces=levels, scalars=array_name)
    return save_poly(contour, out_contour_path)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Normal-based mesh comparison and ParaView exports.")

    # --- Required inputs ---
    p.add_argument("--ref", required=True, help="Path to reference VTU (e.g., Meshes/50th_Male.vtu)")
    p.add_argument("--target", required=True, help="Path to target VTU (e.g., Meshes/Skin_morphed.vtu)")
    p.add_argument("--out", required=True, help="Output file placeholder (not used in single-normal mode)")

    # --- Verbosity ---
    p.add_argument("--verbose", action="store_true", help="Enable detailed logging")

    # --- Visualization exports for reference (50th) mesh ---
    p.add_argument("--export-ref-with-normals", help="Save 50th surface (VTP) with 'Normals' for ParaView glyphs")
    p.add_argument("--export-normals-glyphs", help="Save VTP of actual arrow glyphs for normals")
    p.add_argument("--glyph-scale", type=float, default=1.0, help="Arrow size scale factor")
    p.add_argument("--glyph-sample-step", type=int, default=50, help="Sample every Nth triangle for glyphs")

    # --- Single-normal debug mode ---
    p.add_argument("--single-cell-index", type=int, help="Triangle index on reference mesh to test")
    p.add_argument("--ray-len-factor", type=float, default=2.0, help="Ray length = bounding box diagonal * factor")
    p.add_argument("--no-both-directions", action="store_true", help="Cast only along +normal (default casts ±normal)")
    p.add_argument("--export-single-ray", help="VTP: debug ray visualization (line + hit marker)")
    p.add_argument("--export-single-hit-morph", help="VTP: morphed mesh colored by the hit distance on one cell")

    # --- Optional full-field signed distance / contours ---
    p.add_argument("--export-morph-distance", help="VTP: morphed surface with signed distance to reference")
    p.add_argument("--export-morph-contours", help="VTP: contour mesh based on signed distance field")
    p.add_argument("--contour-levels", help="Comma-separated contour levels, e.g. -3,-2,-1,0,1,2,3")
    p.add_argument("--n-iso", type=int, default=12, help="Number of auto contour levels if none specified")

    return p.parse_args(argv)

def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    logging.info("=== RBF Normal-Based Mesh Comparison ===")
    logging.info(f"Reference mesh: {args.ref}")
    logging.info(f"Target mesh   : {args.target}")

    try:
        # 1️⃣ Load meshes
        ref = load_mesh(args.ref)
        morph = load_mesh(args.target)

        # 2️⃣ Extract triangulated surfaces
        ref_surf = extract_surface(ref, triangulate=True)
        morph_surf = extract_surface(morph, triangulate=True)

        # 3️⃣ Compute normals on reference (50th)
        ref_surf = compute_cell_normals(ref_surf, auto_orient=True)

        # 4️⃣ Export reference surface with normals for ParaView glyphs
        if args.export_ref_with_normals:
            write_ref_with_normals(ref_surf, args.export_ref_with_normals)

        if args.export_normals_glyphs:
            write_normals_glyphs(
                ref_surf,
                args.export_normals_glyphs,
                glyph_scale=args.glyph_scale,
                sample_step=max(1, args.glyph_sample_step),
            )

        # 5️⃣ SINGLE NORMAL DEBUG MODE
        if args.single_cell_index is not None:
            dist, hit_pt, hit_cell, center, n, sign_dir = cast_single_normal(
                ref_surf,
                morph_surf,
                cell_index=args.single_cell_index,
                ray_len_factor=args.ray_len_factor,
                both_directions=not args.no_both_directions,
            )

            # Optional debug ray visualization
            if args.export_single_ray:
                length = ray_length_from_diag(ref_surf, morph_surf, args.ray_len_factor)
                write_debug_ray(center, hit_pt, n, length, args.export_single_ray)

            # Paint the hit distance on the morphed surface (optional)
            if dist is None or hit_cell is None:
                logging.info("Single-normal: no intersection found; nothing to paint on the morphed surface.")
            else:
                signed_value = sign_dir * dist  # + for +n, - for -n
                logging.info(
                    f"Single-normal: hit morph cell {hit_cell} with distance {dist:.6g} (signed {signed_value:.6g})"
                )

                if args.export_single_hit_morph:
                    paint_single_hit_on_morph(
                        morph_surf,
                        hit_cell_id=hit_cell,
                        distance_value=signed_value,
                        out_path=args.export_single_hit_morph,
                        array_name="single_normal_distance",
                    )

                # NEW: export visible marker (outline + small sphere)
                if args.export_hit_markers:
                    export_hit_markers_on_morph(
                        morph_surf,
                        hit_cell_id=hit_cell,
                        hit_point=hit_pt,
                        out_path=args.export_hit_markers,
                        size_factor=args.marker_size_factor,
                    )


         

        # 6️⃣ Signed distance field on entire morph (optional)
        if args.export_morph_distance or args.export_morph_contours:
            morph_with_sd = compute_signed_distance_on_morph(
                morph_surf, ref_surf, array_name="signed_distance_to_ref"
            )

            if args.export_morph_distance:
                save_poly(morph_with_sd, args.export_morph_distance)

            if args.export_morph_contours:
                levels = None
                if args.contour_levels:
                    try:
                        levels = [float(x) for x in args.contour_levels.split(",")]
                    except Exception:
                        logging.warning("Invalid --contour-levels; using automatic levels.")
                write_morph_contours(
                    morph_with_sd,
                    "signed_distance_to_ref",
                    args.export_morph_contours,
                    levels=levels,
                    n_iso=args.n_iso,
                )

        logging.info("=== Done ===")
        return 0

    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
