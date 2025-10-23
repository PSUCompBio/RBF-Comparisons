import os
import pyvista as pv

def resolve_path(path: str) -> str:
    ap = os.path.abspath(path)
    if not os.path.exists(ap):
        raise FileNotFoundError(f"File not found: {ap}")
    return ap

def load_mesh(path: str) -> pv.DataSet:
    path = resolve_path(path)
    return pv.read(path)

def extract_surface(mesh: pv.DataSet, triangulate: bool = True) -> pv.PolyData:
    surf = mesh.extract_surface()
    if triangulate:
        surf = surf.triangulate()
    return surf

def save_poly(poly: pv.PolyData, path: str) -> str:
    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    poly.save(out)
    return out
