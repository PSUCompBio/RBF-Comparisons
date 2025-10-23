# rbf_modes.py
import logging
import numpy as np
import pyvista as pv

from rbf_io import save_poly
from rbf_normals import write_ref_with_normals, write_normals_glyphs, write_single_normal_glyph
from rbf_raycast import cast_single_normal, ray_length_from_diag, write_debug_ray
from rbf_distance import compute_signed_distance_on_morph, write_morph_contours, paint_single_hit_on_morph
from rbf_markers import export_hit_markers_on_morph

def run_normals_exports(ref_surf: pv.PolyData, args) -> None:
    if args.export_ref_with_normals:
        write_ref_with_normals(ref_surf, args.export_ref_with_normals)
    if args.export_normals_glyphs:
        write_normals_glyphs(
            ref_surf,
            args.export_normals_glyphs,
            glyph_scale=args.glyph_scale,
            sample_step=max(1, args.glyph_sample_step),
        )

def run_single_normal_mode(ref_surf: pv.PolyData, morph_surf: pv.PolyData, args) -> None:
    if args.single_cell_index is None:
        return

    dist, hit_pt, hit_cell, center, n, sign_dir = cast_single_normal(
        ref_surf,
        morph_surf,
        cell_index=args.single_cell_index,
        ray_len_factor=args.ray_len_factor,
        both_directions=not args.no_both_directions,
    )

    if args.export_single_normal_glyph:
        x0, x1, y0, y1, z0, z1 = ref_surf.bounds
        diag = float(((x1 - x0)**2 + (y1 - y0)**2 + (z1 - z0)**2) ** 0.5)
        glyph_len = max(1e-9, diag * float(args.single_glyph_length_frac))
        write_single_normal_glyph(center, n, args.export_single_normal_glyph, length=glyph_len)

    if args.export_single_ray:
        length = ray_length_from_diag(ref_surf, morph_surf, args.ray_len_factor)
        write_debug_ray(center, hit_pt, n, length, args.export_single_ray)

    if dist is None or hit_cell is None:
        logging.info("Single-normal: no intersection found.")
        return

    signed_value = sign_dir * dist
    logging.info(f"Single-normal: hit morph cell {hit_cell} distance {dist:.6g} (signed {signed_value:.6g})")

    if args.export_single_hit_morph:
        if args.color_by_hit_distance and hit_pt is not None:
            # Shade entire morph by distance to the hit point (cell scalar)
            morph_colored = morph_surf.copy()
            centers = morph_colored.cell_centers().points
            dists = np.linalg.norm(centers - hit_pt, axis=1)
            morph_colored.cell_data["distance_from_hit"] = dists
            save_poly(morph_colored, args.export_single_hit_morph)
        else:
            # Paint only the hit cell (others stay NaN)
            paint_single_hit_on_morph(
                morph_surf,
                hit_cell_id=hit_cell,
                distance_value=signed_value,
                out_path=args.export_single_hit_morph,
                array_name="single_normal_distance",
            )

    if args.export_hit_markers:
        export_hit_markers_on_morph(
            morph_surf,
            hit_cell_id=hit_cell,
            hit_point=hit_pt,
            out_path=args.export_hit_markers,
            size_factor=args.marker_size_factor,
        )

def run_global_signed_distance(ref_surf: pv.PolyData, morph_surf: pv.PolyData, args) -> None:
    if not (args.export_morph_distance or args.export_morph_contours):
        return

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
                levels = None
        write_morph_contours(
            morph_with_sd,
            "signed_distance_to_ref",
            args.export_morph_contours,
            levels=levels,
            n_iso=args.n_iso,
        )
