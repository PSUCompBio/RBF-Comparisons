import numpy as np
import pyvista as pv
from rbf_io import save_poly

def compute_cell_normals(poly: pv.PolyData, auto_orient=True) -> pv.PolyData:
    poly = poly.copy()
    poly.compute_normals(cell_normals=True, point_normals=False,
                         auto_orient_normals=auto_orient, inplace=True)
    if "Normals" not in poly.cell_data:
        raise RuntimeError("Cell normals missing after compute_normals()")
    return poly

def write_ref_with_normals(ref_surf: pv.PolyData, out_path: str) -> str:
    if "Normals" not in ref_surf.cell_data:
        raise ValueError("Reference surface has no 'Normals'.")
    return save_poly(ref_surf, out_path)

def write_normals_glyphs(ref_surf: pv.PolyData, out_path: str,
                         glyph_scale: float = 1.0, sample_step: int = 50) -> str:
    # sample centers and normals
    centers = ref_surf.cell_centers().points[::sample_step]
    normals = ref_surf.cell_data["Normals"][::sample_step]

    # ensure correct dtype/shape
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 2 or normals.shape[1] != 3:
        raise ValueError(f"Normals must be (N,3); got {normals.shape}")

    # attach vectors to points
    pts = pv.PolyData(centers)
    pts.point_data["Normals"] = normals      # <-- direct assignment (no 'deep' kwarg)

    # build glyphs; orient by vector, scale by vector magnitude
    arrows = pts.glyph(
        orient="Normals",
        scale="Normals",          # scales by the vector magnitude
        factor=glyph_scale,       # overall size multiplier
        geom=pv.Arrow(),
    )

    return save_poly(arrows, out_path)


import numpy as np
import pyvista as pv
from rbf_io import save_poly  # absolute import to match your setup

def write_single_normal_glyph(center: np.ndarray,
                              normal: np.ndarray,
                              out_path: str,
                              length: float) -> str:
    """
    Create a single arrow glyph at 'center' oriented by 'normal' with absolute length 'length'.
    'normal' will be normalized; 'length' is model-units long.
    """
    n = np.asarray(normal, dtype=float)
    nmag = np.linalg.norm(n)
    if not np.isfinite(nmag) or nmag == 0.0:
        raise ValueError("Normal vector is zero or invalid.")
    n = n / nmag

    pts = pv.PolyData(np.asarray(center, dtype=float).reshape(1, 3))
    # store as a vector so glyph() can orient and scale
    pts.point_data["Normals"] = n.reshape(1, 3)

    # factor sets the absolute length since the vector magnitude is 1
    arrow = pts.glyph(orient="Normals", scale="Normals", factor=float(length), geom=pv.Arrow())
    return save_poly(arrow, out_path)


import numpy as np
import pyvista as pv
from rbf_io import save_poly

def write_single_normal_glyph(center: np.ndarray, normal: np.ndarray, out_path: str, length: float) -> str:
    n = np.asarray(normal, dtype=float)
    nmag = np.linalg.norm(n)
    if not np.isfinite(nmag) or nmag == 0.0:
        raise ValueError("Normal vector is zero or invalid.")
    n = n / nmag
    pts = pv.PolyData(np.asarray(center, dtype=float).reshape(1, 3))
    pts.point_data["Normals"] = n.reshape(1, 3)
    arrow = pts.glyph(orient="Normals", scale="Normals", factor=float(length), geom=pv.Arrow())
    return save_poly(arrow, out_path)

def write_multi_normal_glyphs(centers: np.ndarray,
                              normals: np.ndarray,
                              distances: np.ndarray,
                              out_path: str,
                              length: float) -> str:
    """
    Build arrow glyphs at 50th centers, oriented by normals, colored by 'distances'.
    """
    centers = np.asarray(centers, float)
    normals = np.asarray(normals, float)
    distances = np.asarray(distances, float)
    if centers.shape[0] == 0:
        raise ValueError("No normals to write.")
    # normalize normals
    nmag = np.linalg.norm(normals, axis=1, keepdims=True)
    nmag[nmag == 0] = 1.0
    normals = normals / nmag

    pts = pv.PolyData(centers)
    pts.point_data["Normals"] = normals
    pts.point_data["hit_distance"] = distances  # scalar for coloring arrows
    arrows = pts.glyph(orient="Normals", scale="Normals", factor=float(length), geom=pv.Arrow())
    return save_poly(arrows, out_path)
