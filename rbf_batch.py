# rbf_batch.py
import logging
import time
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
    progress_every: int = 100,   
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
    n = len(indices)
    logging.info(f"Batch normals (sequential): {n} indices")

    centers: list = []
    normals: list = []
    distances: list = []
    signed_values: list = []
    hits_meta: list = []

    # arrow absolute length based on 50th surface size
    diag = _bbox_diag_from_bounds(ref_surf.bounds)
    glyph_length = max(1e-9, diag * float(glyph_length_frac))

    t0 = time.time()  
    for k, ci in enumerate(indices, start=1):
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
                distances.append(np.nan)
                signed_values.append(np.nan)
                hits_meta.append((ci, None, None, None, None, None, None, None, None, None, None, None, None))
            else:
                signed_value = sign_dir * dist
                distances.append(float(dist))
                signed_values.append(float(signed_value))

                hx, hy, hz = (hit_pt if hit_pt is not None else (np.nan, np.nan, np.nan))
                cx, cy, cz = center
                nx, ny, nz = n
                hits_meta.append((ci, hit_cell, float(dist), float(signed_value), sign_dir,
                                  cx, cy, cz, nx, ny, nz, hx, hy, hz))

        except Exception as e:
            logging.exception(f"Index {ci}: error {e}")
            centers.append(np.array([np.nan, np.nan, np.nan]))
            normals.append(np.array([0.0, 0.0, 1.0]))
            distances.append(np.nan)
            signed_values.append(np.nan)
            hits_meta.append((ci, None, None, None, None, None, None, None, None, None, None, None, None))

        # 
        if progress_every and (k % progress_every == 0 or k == n):
            elapsed = time.time() - t0
            rate = k / elapsed if elapsed > 0 else 0.0
            eta = (n - k) / rate if rate > 0 else float("inf")
            logging.info(
                f"[batch-seq] {k}/{n} ({k*100.0/n:5.1f}%)  "
                f"elapsed={elapsed:6.1f}s  rate={rate:6.1f} r/s  eta={eta:6.1f}s"
            )

    return {
        "centers": np.asarray(centers, float),
        "normals": np.asarray(normals, float),
        "distances": np.asarray(distances, float),
        "signed_values": np.asarray(signed_values, float),
        "hits_meta": hits_meta,
        "glyph_length": glyph_length,
    }


def save_batch_results_npz(results: Dict[str, Any], path: str) -> str:
    """Save batch results compactly; everything needed to replot without re-casting."""
    np.savez_compressed(
        path,
        centers=results["centers"],
        normals=results["normals"],
        distances=results["distances"],
        signed_values=results["signed_values"],
        hits_meta=np.array(results["hits_meta"], dtype=object),
        glyph_length=np.array([results["glyph_length"]], dtype=float),
    )
    return path

def load_batch_results_npz(path: str) -> Dict[str, Any]:
    """Load previously saved batch results."""
    z = np.load(path, allow_pickle=True)
    return {
        "centers": z["centers"],
        "normals": z["normals"],
        "distances": z["distances"],
        "signed_values": z["signed_values"],
        "hits_meta": list(z["hits_meta"]),
        "glyph_length": float(z["glyph_length"][0]),
    }

# --- Parallel batch casting ---------------------------------------------------
import logging
import numpy as np
import multiprocessing as mp
import pyvista as pv
from rbf_raycast import cast_single_normal

# Globals inside worker processes
_G_REF = None
_G_MORPH = None

def _init_pool(ref_path: str, morph_path: str):
    """Load meshes once per worker process to avoid pickling large VTK objects."""
    global _G_REF, _G_MORPH
    ref = pv.read(ref_path).extract_surface().triangulate()
    ref.compute_normals(cell_normals=True, point_normals=False,
                        auto_orient_normals=True, inplace=True)
    morph = pv.read(morph_path).extract_surface().triangulate()
    _G_REF, _G_MORPH = ref, morph

def _worker_cast(args):
    """Cast one index using process-local meshes."""
    ci, ray_len_factor, both_directions = args
    try:
        dist, hit_pt, hit_cell, center, n, sign_dir = cast_single_normal(
            _G_REF, _G_MORPH, cell_index=ci,
            ray_len_factor=ray_len_factor, both_directions=both_directions
        )
        return (ci, dist, hit_pt, hit_cell, center, n, sign_dir)
    except Exception as e:
        logging.exception(f"Index {ci}: {e}")
        return (ci, None, None, None, np.array([np.nan, np.nan, np.nan]),
                np.array([0.0, 0.0, 1.0]), 0)

def _bbox_diag(bounds):
    x0,x1,y0,y1,z0,z1 = bounds
    return float(((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2) ** 0.5)


def run_multi_normals_batch_parallel(
    ref_path: str,
    morph_path: str,
    indices,
    ray_len_factor: float,
    both_directions: bool,
    glyph_length_frac: float,
    processes: int | None = None,
    chunksize: int = 64,
    progress_every: int = 100,
):
    ctx = mp.get_context("spawn")  # robust on Windows
    n = len(indices)
    logging.info(f"Spawning {processes or mp.cpu_count()} worker(s); submitting {n} jobs (chunksize={chunksize})")

    with ctx.Pool(processes=processes, initializer=_init_pool,
                  initargs=(ref_path, morph_path)) as pool:
        jobs = [(int(ci), ray_len_factor, both_directions) for ci in indices]
        it = pool.imap_unordered(_worker_cast, jobs, chunksize)

        results_list = []
        done = 0
        t0 = time.time()
        for res in it:
            results_list.append(res)
            done += 1
            if progress_every and (done % progress_every == 0 or done == n):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (n - done) / rate if rate > 0 else float("inf")
                logging.info(
                    f"[batch-par] {done}/{n} ({done*100.0/n:5.1f}%)  "
                    f"elapsed={elapsed:6.1f}s  rate={rate:6.1f} r/s  eta={eta:6.1f}s"
                )

    # Re-load ref once (in parent) just to compute glyph length
    ref = pv.read(ref_path).extract_surface().triangulate()
    ref.compute_normals(cell_normals=True, point_normals=False,
                        auto_orient_normals=True, inplace=True)
    glyph_length = max(1e-9, _bbox_diag(ref.bounds) * float(glyph_length_frac))

    centers, normals, distances, signed_values, hits_meta = [], [], [], [], []
    for (ci, dist, hit_pt, hit_cell, center, nvec, sign_dir) in results_list:
        centers.append(center)
        normals.append(nvec)
        if dist is None or hit_cell is None:
            distances.append(np.nan)
            signed_values.append(np.nan)
            hits_meta.append((ci, None, None, None, None, None, None, None, None, None, None, None, None))
        else:
            signed = sign_dir * float(dist)
            distances.append(float(dist))
            signed_values.append(signed)
            hx, hy, hz = (hit_pt if hit_pt is not None else (np.nan, np.nan, np.nan))
            cx, cy, cz = center
            nx, ny, nz = nvec
            hits_meta.append((ci, hit_cell, float(dist), signed, sign_dir, cx, cy, cz, nx, ny, nz, hx, hy, hz))

    return {
        "centers": np.asarray(centers, float),
        "normals": np.asarray(normals, float),
        "distances": np.asarray(distances, float),
        "signed_values": np.asarray(signed_values, float),
        "hits_meta": hits_meta,
        "glyph_length": glyph_length,
    }
