# rbf_cli.py
import sys
import logging
import csv
import numpy as np
import pyvista as pv

from rbf_args import parse_args
from rbf_logging import setup_logging
from rbf_io import load_mesh, extract_surface
from rbf_normals import compute_cell_normals, write_multi_normal_glyphs
from rbf_batch import run_multi_normals_batch
from rbf_modes import run_normals_exports, run_single_normal_mode, run_global_signed_distance
from rbf_distance import paint_multi_hits_on_morph
from rbf_io import ensure_output_prefix


def main(argv=None) -> int:
    args = parse_args(argv)
    setup_logging(args.verbose)

    logging.info("=== RBF Normal-Based Mesh Comparison ===")
    logging.info(f"Reference: {args.ref}")
    logging.info(f"Target   : {args.target}")

    try:
        # Load meshes and surfaces
        ref = load_mesh(args.ref)
        morph = load_mesh(args.target)
        ref_surf = extract_surface(ref, triangulate=True)
        morph_surf = extract_surface(morph, triangulate=True)

        # Normals on reference
        ref_surf = compute_cell_normals(ref_surf, auto_orient=True)

        # Mode: normals exports
        run_normals_exports(ref_surf, args)

         # ----- Multi-normal batch (optional) -----
        idxs = None

        # 1️⃣ Manual list from --multi-cell-indices
        if args.multi_cell_indices:
            idxs = [int(s) for s in args.multi_cell_indices.split(",") if s.strip() != ""]

        # 2️⃣ Auto-pick mode (--multi-auto)
        if idxs is None and args.multi_auto:
            n_cells = ref_surf.n_cells
            k = max(1, int(args.multi_auto))
            if args.multi_auto_mode == "uniform":
                idxs = list(np.round(np.linspace(0, n_cells - 1, k)).astype(int))
                logging.info(f"Auto-picking {k} evenly spaced triangles out of {n_cells}")
            else:  # random mode
                rng = np.random.default_rng(42)  # fixed seed for reproducibility
                idxs = list(rng.choice(n_cells, size=k, replace=False))
                logging.info(f"Auto-picking {k} random triangles out of {n_cells}")

        # 3️⃣ Run batch if we have any indices
        if idxs:
            results = run_multi_normals_batch(
                ref_surf=ref_surf,
                morph_surf=morph_surf,
                indices=idxs,
                ray_len_factor=args.ray_len_factor,
                both_directions=not args.no_both_directions,
                glyph_length_frac=args.multi_glyph_length_frac,
            )

            # Export arrows on 50th
            if args.export_multi_normals_glyph:
                write_multi_normal_glyphs(
                    results["centers"],
                    results["normals"],
                    results["distances"],
                    args.export_multi_normals_glyph,
                    length=results["glyph_length"],
                )
                logging.info("ParaView: open VTP and color by 'hit_distance' (diverging map; invert so blue=min, red=max).")

            # Export morph surface with colored hit triangles
            if args.export_batch_hit_morph:
                hit_cell_ids = [row[1] for row in results["hits_meta"]]
                distance_values = list(results["signed_values"] if args.batch_use_signed else results["distances"])
                paint_multi_hits_on_morph(
                    morph_surf=morph_surf,
                    hit_cell_ids=hit_cell_ids,
                    distance_values=distance_values,
                    out_path=args.export_batch_hit_morph,
                    array_name="batch_hit_distance",
                )
                logging.info("ParaView: color by 'batch_hit_distance' (Cell Data), set NaN color to gray, Rescale to Data Range.")

            # Export CSV summary
            if args.export_distances_csv:
                header = ["ref_cell_index","morph_hit_cell","abs_distance","signed_distance","sign_dir",
                          "center_x","center_y","center_z","normal_x","normal_y","normal_z",
                          "hit_x","hit_y","hit_z"]
               
                csv_path = ensure_output_prefix(args.export_distances_csv)
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for row in results["hits_meta"]:
                        writer.writerow(row)
                logging.info(f"Wrote results CSV: {args.export_distances_csv}")


            # Build a morph file with ONLY the hit triangles colored by distance
        if args.export_batch_hit_morph:
            # extract morph hit cell ids from results["hits_meta"]
            # hits_meta rows: (ci, hit_cell, dist, signed, sign_dir, cx,cy,cz, nx,ny,nz, hx,hy,hz)
            hit_cell_ids = []
            if "hits_meta" in results:
                for row in results["hits_meta"]:
                    hit_cell_ids.append(row[1])  # morph hit cell id (or None)

            # choose distances (signed or absolute)
            if args.batch_use_signed:
                distance_values = list(results["signed_values"])
            else:
                distance_values = list(results["distances"])

            paint_multi_hits_on_morph(
                morph_surf=morph_surf,
                hit_cell_ids=hit_cell_ids,
                distance_values=distance_values,
                out_path=args.export_batch_hit_morph,
                array_name="batch_hit_distance",
            )
            logging.info("Wrote batch-hit morph. ParaView: color by 'batch_hit_distance' (Cell Data),"
                         " set NaN color to gray, Rescale to Data Range, choose diverging colormap"
                         " and invert if needed so blue=min (good) and red=max (bad).")


            if args.export_multi_normals_glyph:
                write_multi_normal_glyphs(
                    results["centers"],
                    results["normals"],
                    results["distances"],
                    args.export_multi_normals_glyph,
                    length=results["glyph_length"],
                )
                logging.info("ParaView: open VTP and color by 'hit_distance' (diverging map; invert so blue=min, red=max).")

            if args.export_distances_csv:
                header = ["ref_cell_index","morph_hit_cell","abs_distance","signed_distance","sign_dir",
                          "center_x","center_y","center_z","normal_x","normal_y","normal_z",
                          "hit_x","hit_y","hit_z"]
               
                csv_path = ensure_output_prefix(args.export_distances_csv)
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)
                    for row in results["hits_meta"]:
                        writer.writerow(row)
                logging.info(f"Wrote results CSV: {args.export_distances_csv}")

        # Mode: single-normal debug
        run_single_normal_mode(ref_surf, morph_surf, args)

        # Mode: global signed distance / contours
        run_global_signed_distance(ref_surf, morph_surf, args)

        logging.info("=== Done ===")
        return 0

    except Exception as e:
        logging.exception(f"Fatal error: {e}")
        return 2

if __name__ == "__main__":
    sys.exit(main())
