import numpy as np
import pyvista as pv
from pyvista import _vtk
from rbf_io import save_poly

def compute_signed_distance_on_morph(morph_surf: pv.PolyData,
                                     ref_surf: pv.PolyData,
                                     array_name="signed_distance_to_ref") -> pv.PolyData:
    ipd = _vtk.vtkImplicitPolyDataDistance()
    ipd.SetInput(ref_surf)
    pts = morph_surf.points
    vals = np.empty(pts.shape[0], dtype=float)
    for i, p in enumerate(pts):
        vals[i] = ipd.EvaluateFunction(p)
    out = morph_surf.copy()
    out.point_data[array_name] = vals
    return out

def write_morph_contours(morph_with_distance: pv.PolyData,
                         array_name: str,
                         out_contour_path: str,
                         levels: list[float] | None = None,
                         n_iso: int = 12) -> str:
    scal = morph_with_distance.point_data[array_name]
    if levels is None:
        smin, smax = float(np.min(scal)), float(np.max(scal))
        levels = list(np.linspace(smin, smax, n_iso))
    contour = morph_with_distance.contour(isosurfaces=levels, scalars=array_name)
    return save_poly(contour, out_contour_path)

def paint_single_hit_on_morph(morph_surf: pv.PolyData,
                              hit_cell_id: int,
                              distance_value: float,
                              out_path: str,
                              array_name: str = "single_normal_distance") -> str:
    if hit_cell_id is None or hit_cell_id < 0 or hit_cell_id >= morph_surf.n_cells:
        raise ValueError("Invalid hit_cell_id for painting.")
    out = morph_surf.copy()
    vals = np.full(out.n_cells, np.nan, dtype=float)
    vals[hit_cell_id] = float(distance_value)
    out.cell_data[array_name] = vals
    return save_poly(out, out_path)


def paint_multi_hits_on_morph(
    morph_surf: pv.PolyData,
    hit_cell_ids: list[int],
    distance_values: list[float],
    out_path: str,
    array_name: str = "batch_hit_distance",
) -> str:
    """
    Color ONLY the given morph cells by the provided distances.
    All other cells are NaN (so they render with the NaN color in ParaView).

    hit_cell_ids and distance_values must be the same length.
    """
    if len(hit_cell_ids) != len(distance_values):
        raise ValueError("hit_cell_ids and distance_values must have the same length.")

    out = morph_surf.copy()
    vals = np.full(out.n_cells, np.nan, dtype=float)

    for cid, val in zip(hit_cell_ids, distance_values):
        if cid is None:
            continue
        if 0 <= int(cid) < out.n_cells:
            vals[int(cid)] = float(val)

    out.cell_data[array_name] = vals
    return save_poly(out, out_path)
