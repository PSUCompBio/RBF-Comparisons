# rbf_batch.py
import logging
import numpy as np
from typing import List, Tuple, Dict, Any

from rbf_raycast import cast_single_normal

def _bbox_diag_from_bounds(bounds: Tuple[float, float, float, float, float, float]) -> float:
    x0, x1, y0, y1, z0, z1 = bounds
    return float(((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) ** 0.5)

def run_multi_normals_batch(
    ref_surf,
    morph_surf,
    indices: List[int],
    ray_len_factor: float,
    both_directions: bool,
    glyph_length_frac: float,
) -> Dict[str, Any]:
    """
    Cast a normal ray for each reference triangle index in `indices`.

    Returns a dict with:
      centers        (N,3) float
      normals        (N,3) float (unit vectors)
      distances      (N,)  float (absolute hit distance; NaN if no hit)
      signed_values  (N,)  float (+/- distance; NaN if no hit)
      hits_meta      list of tuples per index for CSV
      glyph_length   float (absolute arrow length to use for glyphs)
    """
    logging.info(f"Batch normals: {len(indices)} indices -> {indices}")

    centers = []
    normals = []
    distances = []
    signed_values = []
    hits_meta = []

    # arrow absolute length based on 50th surface size
    diag = _bbox_diag_from_bounds(ref_surf.bounds)
    glyph_length = max(1e-9, diag * float(glyph_length_frac))

    for ci in indices:
        try:
            dist, hit_pt, hit_cell, center, n, sign_dir = cast_single_normal(
                ref_surf, morph_surf,
                cell_index=ci,
                ray_len_factor=ray_len_factor,
                both_directions=both_directions,
            )

            # always store center/normal (even if miss) so arrow list lines up
            centers.append(center)
            normals.append(n)

            if dist is None or hit_cell is None:
                logging.info(f"Index {ci}: no hit")
                distances.append(np.nan)
                signed_values.append(np.nan)
                hits_meta.append((ci, None, None, None, None, None, None, None, None, None, None, None, None))
                continue

            signed_value = sign_dir * dist
            distances.append(dist)
            signed_values.append(signed_value)

            # pack CSV row
            hx, hy, hz = (hit_pt if hit_pt is not None else (np.nan, np.nan, np.nan))
            cx, cy, cz = center
            nx, ny, nz = n
            hits_meta.append((ci, hit_cell, dist, signed_value, sign_dir, cx, cy, cz, nx, ny, nz, hx, hy, hz))
            logging.info(f"Index {ci}: hit cell {hit_cell}  abs={dist:.6g}  signed={signed_value:.6g}")

        except Exception as e:
            logging.exception(f"Index {ci}: error {e}")
            centers.append(np.array([np.nan, np.nan, np.nan]))
            normals.append(np.array([0.0, 0.0, 1.0]))
            distances.append(np.nan)
            signed_values.append(np.nan)
            hits_meta.append((ci, None, None, None, None, None, None, None, None, None, None, None, None))

    return {
        "centers": np.asarray(centers, float),
        "normals": np.asarray(normals, float),
        "distances": np.asarray(distances, float),
        "signed_values": np.asarray(signed_values, float),
        "hits_meta": hits_meta,
        "glyph_length": glyph_length,
    }
