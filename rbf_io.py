# -*- coding: utf-8 -*-
import os
import pyvista as pv

def ensure_output_prefix(path: str, prefix: str = "output_") -> str:
    """If basename doesn't start with prefix, prepend it."""
    d, b = os.path.dirname(path), os.path.basename(path)
    if not b.startswith(prefix):
        b = prefix + b
    return os.path.join(d, b)

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
    path = ensure_output_prefix(path)
    out = os.path.abspath(path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    poly.save(out)
    return out
