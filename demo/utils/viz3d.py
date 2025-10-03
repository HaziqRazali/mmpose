# utils/viz3d.py
# Open3D-based 3D visualization twin of utils.viz (2D).
# - 3D drawer registry
# - Vec-pair 3D lines (blue/green, matching 2D style)
# - Simple viewer helpers

from typing import Optional, Tuple, Dict, Callable, List
from dataclasses import dataclass
import numpy as np

try:
    import open3d as o3d
except Exception as e:
    o3d = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


def _require_o3d():
    if o3d is None:
        raise ImportError(
            "Open3D is required for utils.viz3d but is not available. "
            f"Original import error: {_IMPORT_ERR}"
        )


# ------------------------
# 3D drawer registry
# ------------------------

Drawer3DFn = Callable[['VizContext3D'], None]
_CUSTOM_DRAWERS_3D: Dict[str, Drawer3DFn] = {}


def register_drawer_3d(rom_name: str, fn: Drawer3DFn) -> None:
    """
    Register a 3D drawer function for a specific ROM.
    """
    _CUSTOM_DRAWERS_3D[rom_name] = fn


def draw_for_rom_3d(rom_name: str, ctx: 'VizContext3D') -> None:
    """
    Invoke the registered 3D drawer for `rom_name` if one exists.
    Drawer should append geometries to ctx.overlays list.
    """
    fn = _CUSTOM_DRAWERS_3D.get(rom_name, None)
    if fn is not None:
        fn(ctx)


# ------------------------
# Context & primitives
# ------------------------

@dataclass
class VizContext3D:
    """
    Everything a 3D drawer might need for one timestamp's scene.
    - rom_name: name of ROM
    - when_label: "t1" | "t2" | custom
    - kpts3d: List[Optional[np.ndarray(3,)]]  # camera-space 3D joints
    - vec_pair: optional [[P0,P1],[Q0,Q1]] for default vec overlay
    - pcd: open3d.geometry.PointCloud (for reference if needed)
    - median_k: same depth sampling patch size used elsewhere
    - overlays: List[o3d geometry] -> drawer should append here
    """
    rom_name: str
    when_label: str
    kpts3d: List[Optional[np.ndarray]]
    vec_pair: Optional[list]
    pcd: Optional[any]
    median_k: int = 5
    overlays: Optional[List[any]] = None

    def __post_init__(self):
        if self.overlays is None:
            self.overlays = []


def _point_ok(P: Optional[np.ndarray]) -> bool:
    return (P is not None) and np.all(np.isfinite(P))


def _make_lineset(points: np.ndarray,
                  lines: List[Tuple[int, int]],
                  color_rgb: Tuple[float, float, float]):

    _require_o3d()
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    cols = np.tile(np.array(color_rgb, dtype=np.float64)[None, :], (len(lines), 1))
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def draw_vectors3d(ctx: VizContext3D, thickness_ignored: int = 3):
    """
    Draw two 3D vectors defined by keypoints:
      vec_pair = [[P0, P1], [Q0, Q1]],
    colored to match 2D style: first vector BLUE, second GREEN.
    """
    if ctx.vec_pair is None:
        return
    (P0, P1), (Q0, Q1) = ctx.vec_pair

    def get3(j_spec):
        if isinstance(j_spec, (list, tuple)):
            pts = [ctx.kpts3d[int(j)] for j in j_spec if 0 <= int(j) < len(ctx.kpts3d)]
            pts = [p for p in pts if _point_ok(p)]
            if not pts:
                return None
            return np.mean(np.stack(pts, axis=0), axis=0)
        j = int(j_spec)
        if j < 0 or j >= len(ctx.kpts3d):
            return None
        return ctx.kpts3d[j]

    A = get3(P0); B = get3(P1)
    C = get3(Q0); D = get3(Q1)

    geoms: List[any] = []

    if _point_ok(A) and _point_ok(B):
        pts = np.stack([A, B], axis=0)
        geoms.append(_make_lineset(pts, [(0, 1)], (0.0, 0.0, 1.0)))  # BLUE

    if _point_ok(C) and _point_ok(D):
        pts = np.stack([C, D], axis=0)
        geoms.append(_make_lineset(pts, [(0, 1)], (0.0, 0.78, 0.0)))  # GREEN

    ctx.overlays.extend(geoms)


def make_axis(length: float = 0.1):
    """
    Small XYZ triad at origin (R,G,B = X,Y,Z).
    """
    _require_o3d()
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=length)


# ------------------------
# Simple viewer facade
# ------------------------

def show_in_one_window(geoms: List[any], title: str = "ROM 3D"):
    """
    Open a single Open3D window with the provided geometries.
    """
    _require_o3d()
    if not geoms:
        geoms = []
    o3d.visualization.draw_geometries(geoms, window_name=title)
