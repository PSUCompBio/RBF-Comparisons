# rbf_runner.py
import logging
import csv
import numpy as np

from rbf_io import load_mesh, extract_surface, ensure_output_prefix
from rbf_normals import compute_cell_normals, write_multi_normal_glyphs
from rbf_batch import (
    run_multi_normals_batch,
    run_multi_normals_batch_parallel,
    save_batch_results_npz,
    load_batch_results_npz,
)
from rbf_distance import paint_multi_hits_on_morph
from rbf_modes import (
    run_normals_exports,
    run_single_normal_mode,
    run_global_signed_distance,
)


def _build_index_list(ref_surf, args):
    """Return list of reference triangle indices based on CLI options, or None."""
    idxs = None

    # Manual list
    if args.multi_cell_indices:
        idxs = [int(s) for s in args.multi_cell_indices.split(",") if s.strip()]

    # Auto-pick (count or percent)
    n_cells = ref_surf.n_cells
    if idxs is None and (args.multi_auto or args.multi_auto_percent):
        if args.multi_auto_percent:
            k = max(1, int(n_cells * float(args.multi_auto_percent) / 100.0))
            logging.info(f"Auto-picking {k} triangles (~{args.multi_auto_percent}% of {n_cells})")
        else:
            k = max(1, int(args.multi_auto))

        if args.multi_auto_mode == "uniform":
            idxs = list(np.round(np.linspace(0, n_cells - 1, k)).astype(int))
            logging.info(f"Uniformly spaced selection of {k} indices across {n_cells}")
        else:
            rng = np.random.default_rng(42)  # reproducible
            idxs = list(rng.choice(n_cells, size=k, replace=False))
            logging.info(f"Randomly selected {k} indices out of {n_cells}")

    return idxs


def _emit_batch_outputs(morph_surf, results, args):
    """Write glyphs, morph painting, and CSV from an in-memory results dict."""
    if not results:
        return

    # 50th arrows (colored by absolute distance)
    if args.export_multi_normals_glyph:
        write_multi_normal_glyphs(
            results["centers"],
            results["normals"],
            results["distances"],
            args.export_multi_normals_glyph,
            length=results["glyph_length"],
        )
        logging.info("ParaView: color arrows by 'hit_distance' (diverging; invert for blue=min, red=max).")

    # Single morph mesh with ONLY hit triangles colored by distance
    if args.export_batch_hit_morph:
        hit_cell_ids = [row[1] for row in results["hits_meta"]]
        vals = results["signed_values"] if args.batch_use_signed else results["distances"]
        paint_multi_hits_on_morph(
            morph_surf=morph_surf,
            hit_cell_ids=hit_cell_ids,
            distance_values=list(vals),
            out_path=args.export_batch_hit_morph,
            array_name="batch_hit_distance",
        )
        logging.info("ParaView: color by 'batch_hit_distance' (Cell Data), set NaN color to gray, Rescale to Range.")

    # CSV summary
    if args.export_distances_csv:
        header = [
            "ref_cell_index","morph_hit_cell","abs_distance","signed_distance","sign_dir",
            "center_x","center_y","center_z","normal_x","normal_y","normal_z",
            "hit_x","hit_y","hit_z",
        ]
        csv_path = ensure_output_prefix(args.export_distances_csv)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for row in results["hits_meta"]:
                w.writerow(row)
        logging.info(f"Wrote results CSV: {csv_path}")


def run_all(args) -> int:
    logging.info("=== RBF Normal-Based Mesh Comparison ===")
    logging.info(f"Reference: {args.ref}")
    logging.info(f"Target   : {args.target}")

    # Load meshes and surfaces
    ref = load_mesh(args.ref)
    morph = load_mesh(args.target)
    ref_surf = extract_surface(ref, triangulate=True)
    morph_surf = extract_surface(morph, triangulate=True)

    # Normals on reference
    ref_surf = compute_cell_normals(ref_surf, auto_orient=True)

    # Mode: normals exports (glyphs of many normals, etc.)
    run_normals_exports(ref_surf, args)

    # ----- Batch mode (compute OR load cached) -----
    results = None

    # Load cached results if requested
    if getattr(args, "load_batch", None):
        results = load_batch_results_npz(args.load_batch)
        logging.info(f"Loaded cached batch results from {args.load_batch}")

    else:
        # Build index list to compute
        idxs = _build_index_list(ref_surf, args)
        if idxs:
            if getattr(args, "parallel", None):
                results = run_multi_normals_batch_parallel(
                    ref_path=args.ref,
                    morph_path=args.target,
                    indices=idxs,
                    ray_len_factor=args.ray_len_factor,
                    both_directions=not args.no_both_directions,
                    glyph_length_frac=args.multi_glyph_length_frac,
                    processes=int(args.parallel),
                    chunksize=int(getattr(args, "chunksize", 64)),
                    progress_every=int(getattr(args, "progress_every", 100)),
                )
            else:
                results = run_multi_normals_batch(
                    ref_surf=ref_surf,
                    morph_surf=morph_surf,
                    indices=idxs,
                    ray_len_factor=args.ray_len_factor,
                    both_directions=not args.no_both_directions,
                    glyph_length_frac=args.multi_glyph_length_frac,
                    progress_every=int(getattr(args, "progress_every", 100)),
                )
            # Save cache if requested
            if getattr(args, "save_batch", None):
                npz_path = ensure_output_prefix(args.save_batch)
                save_batch_results_npz(results, npz_path)
                logging.info(f"Saved batch results to {npz_path}")

    # Emit outputs from either cached or freshly computed results
    if results:
        _emit_batch_outputs(morph_surf, results, args)

    # Recolor-only from cache (no recompute)
    if getattr(args, "recolor_batch_hit_morph", None):
        if not results and getattr(args, "load_batch", None):
            results = load_batch_results_npz(args.load_batch)
        if results:
            hit_cell_ids = [row[1] for row in results["hits_meta"]]
            vals = results["signed_values"] if args.recolor_use_signed else results["distances"]
            paint_multi_hits_on_morph(
                morph_surf=morph_surf,
                hit_cell_ids=hit_cell_ids,
                distance_values=list(vals),
                out_path=args.recolor_batch_hit_morph,
                array_name="batch_hit_distance",
            )
            logging.info("Recolored morph from cache: 'batch_hit_distance' written.")

    # Mode: single-normal debug (independent of batch/cache)
    run_single_normal_mode(ref_surf, morph_surf, args)

    # Mode: global signed distance / contours (independent)
    run_global_signed_distance(ref_surf, morph_surf, args)

    logging.info("=== Done ===")
    return 0
