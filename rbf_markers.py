import numpy as np
import pyvista as pv
from rbf_io import save_poly

def bbox_diag(poly: pv.PolyData) -> float:
    x0, x1, y0, y1, z0, z1 = poly.bounds
    return float(np.linalg.norm([x1-x0, y1-y0, z1-z0]))

def export_hit_markers_on_morph(morph_surf: pv.PolyData,
                                hit_cell_id: int,
                                hit_point: np.ndarray | None,
                                out_path: str,
                                size_factor: float = 0.01) -> str:
    if hit_cell_id is None or hit_cell_id < 0 or hit_cell_id >= morph_surf.n_cells:
        raise ValueError("Invalid hit_cell_id for markers.")

    cell_poly = morph_surf.extract_cells(hit_cell_id)
    outline = cell_poly.extract_feature_edges(
        boundary_edges=True, non_manifold_edges=True,
        feature_edges=False, manifold_edges=False)
    outline["marker_id"] = np.ones(outline.n_points)

    center = cell_poly.cell_centers().points[0] if hit_point is None else np.array(hit_point, float)
    radius = max(1e-9, bbox_diag(morph_surf) * float(size_factor))
    sphere = pv.Sphere(radius=radius, center=center); sphere["marker_id"] = np.full(sphere.n_points, 2.0)

    markers = outline.merge(sphere)
    return save_poly(markers, out_path)
