import math
import numpy as np
import pyvista as pv

def combined_bbox_diag(a: pv.PolyData, b: pv.PolyData) -> float:
    x0 = min(a.bounds[0], b.bounds[0]); x1 = max(a.bounds[1], b.bounds[1])
    y0 = min(a.bounds[2], b.bounds[2]); y1 = max(a.bounds[3], b.bounds[3])
    z0 = min(a.bounds[4], b.bounds[4]); z1 = max(a.bounds[5], b.bounds[5])
    return float(np.linalg.norm([x1-x0, y1-y0, z1-z0]))

def ray_length_from_diag(ref_surf: pv.PolyData, target_surf: pv.PolyData, factor: float) -> float:
    diag = combined_bbox_diag(ref_surf, target_surf)
    return max(1e-9, float(diag) * float(factor))

def cast_single_normal(ref_surf_with_normals: pv.PolyData,
                       target_surf: pv.PolyData,
                       cell_index: int,
                       ray_len_factor: float = 2.0,
                       both_directions: bool = True):
    """
    Returns: (best_distance, hit_point, hit_cell_id, center, normal, sign_dir)
      sign_dir = +1 if +normal won, -1 if -normal won, 0 if no hit
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
        return None, None, None, center, n, 0
    n = n / nmag

    ray_len = ray_length_from_diag(ref_surf_with_normals, target_surf, ray_len_factor)

    def pick_nearest(origin, pts, ids):
        if pts is None or len(pts) == 0:
            return None, None, None
        d = np.linalg.norm(pts - origin, axis=1)
        k = int(np.argmin(d))
        return float(d[k]), pts[k], int(ids[k])

    pts_f, ids_f = target_surf.ray_trace(center, center + n * ray_len)
    d_f, pt_f, id_f = pick_nearest(center, pts_f, ids_f)

    d_b = pt_b = id_b = None
    if both_directions:
        pts_b, ids_b = target_surf.ray_trace(center, center - n * ray_len)
        d_b, pt_b, id_b = pick_nearest(center, pts_b, ids_b)

    best_d = best_pt = best_cell = None
    sign_dir = 0
    if d_f is not None and (d_b is None or d_f <= d_b):
        best_d, best_pt, best_cell, sign_dir = d_f, pt_f, id_f, +1
    elif d_b is not None:
        best_d, best_pt, best_cell, sign_dir = d_b, pt_b, id_b, -1

    if best_d is None:
        return None, None, None, center, n, 0

    return best_d, best_pt, best_cell, center, n, sign_dir

def write_debug_ray(center: np.ndarray, hit: np.ndarray | None,
                    normal: np.ndarray, length: float, out_path: str) -> str:
    geoms: list[pv.PolyData] = []
    a = center
    b = center + normal * length * 0.5
    normal_line = pv.Line(a, b); normal_line["id"] = np.array([1.0, 1.0])
    geoms.append(normal_line)
    if hit is not None:
        hit_line = pv.Line(center, hit); hit_line["id"] = np.array([2.0, 2.0])
        geoms.extend([hit_line, pv.Sphere(radius=max(1e-9, length*0.01), center=hit)])
    scene = geoms[0]
    for g in geoms[1:]:
        scene = scene.merge(g)
    from rbf_io import save_poly
    return save_poly(scene, out_path)
