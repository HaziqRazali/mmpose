# depthzip.py
"""
Depth ZIP reader and visualization utilities.
Handles .zip archives of frame_*.bin depth frames.
"""

import re, zipfile
import numpy as np
import cv2
import logging
from pathlib import Path

_HEADER_RE = re.compile(
    rb'^(timestamp|width|height|data_length|fx|fy|ox|oy)\s*:\s*([0-9\.\-eE]+)\s*$',
    re.M
)

def _autoscale_intrinsics_to_frame(w, h, fx, fy, ox, oy):
    """
    If principal point is outside the image, try to infer anisotropic resize scales
    and scale (fx, fy, ox, oy) to the current (w, h).

    Returns: fx, fy, ox, oy, meta_dict
    """
    meta = {}
    inside = (0.0 < ox < w) and (0.0 < oy < h)
    if inside:
        meta['intrinsics_autoscaled'] = False
        return fx, fy, ox, oy, meta

    # Heuristic: assume original intrinsics were for some (W0,H0), then image was resized
    # non-uniformly to (w,h). If the resize roughly preserved the center, then:
    #   sx ≈ w / (2*ox_orig), sy ≈ h / (2*oy_orig)
    # Here we only have current (w,h) and header (ox,oy) that are clearly too big,
    # so estimate scale to bring the principal point near image center.
    sx = (w / (2.0 * ox)) if ox > 0 else np.nan
    sy = (h / (2.0 * oy)) if oy > 0 else np.nan

    ok = (np.isfinite(sx) and 0.2 <= sx <= 2.5) and (np.isfinite(sy) and 0.2 <= sy <= 2.5)
    if not ok:
        meta['intrinsics_autoscaled'] = False
        meta['intrinsics_warning'] = "Header intrinsics out of bounds; unable to auto-fix."
        logging.warning(meta['intrinsics_warning'] + f" (w={w},h={h}, fx={fx},fy={fy}, ox={ox},oy={oy})")
        return fx, fy, ox, oy, meta

    fx2, fy2 = fx * sx, fy * sy
    ox2, oy2 = ox * sx, oy * sy
    meta.update({
        'intrinsics_autoscaled': True,
        'scale_x': float(sx),
        'scale_y': float(sy),
        'fx_before': float(fx), 'fy_before': float(fy),
        'ox_before': float(ox), 'oy_before': float(oy),
    })
    logging.info(f"[ipad_depthio] autoscaled intrinsics: "
                 f"({fx},{fy},{ox},{oy}) -> ({fx2},{fy2},{ox2},{oy2}) for frame {w}x{h}")
    return fx2, fy2, ox2, oy2, meta

class DepthZip:
    """Stream depth frames from a ZIP of frame_*.bin files; normalize timestamps for reporting."""
    def __init__(self, zip_path: str):
        self.zf = zipfile.ZipFile(zip_path, 'r')
        names = [n for n in self.zf.namelist() if n.lower().endswith('.bin')]
        # sort by frame number if present
        def frame_id(s):
            import re
            m = re.search(r'(\d+)', Path(s).stem)
            return int(m.group(1)) if m else -1
        self.entries = sorted(names, key=frame_id)

        self._timestamps = []
        self._sizes = []
        self._intrinsics = []
        for n in self.entries:
            with self.zf.open(n, 'r') as f:
                head = f.read(2048)
            hmap = {}
            for m in _HEADER_RE.finditer(head):
                k = m.group(1).decode('ascii')
                v = float(m.group(2).decode('ascii'))
                hmap[k] = v
            w = int(hmap.get('width', 0)); h = int(hmap.get('height', 0))
            self._timestamps.append(float(hmap.get('timestamp', 0.0)))
            self._sizes.append((h, w))
            self._intrinsics.append({
                'fx': float(hmap.get('fx', 0.0)),
                'fy': float(hmap.get('fy', 0.0)),
                'ox': float(hmap.get('ox', 0.0)),
                'oy': float(hmap.get('oy', 0.0)),
            })

        self.count = len(self.entries)
        self.base_ts = self._timestamps[0] if self.count else 0.0
        # normalized timestamps (seconds since depth recording began)
        self.norm_ts = [(t - self.base_ts) for t in self._timestamps]

    def get_frame(self, index: int):
        """Return dict with depth array and header fields; includes normalized timestamp."""
        name = self.entries[index]
        with self.zf.open(name, 'r') as f:
            raw = f.read()

        m = list(_HEADER_RE.finditer(raw))
        if not m:
            raise ValueError(f'Bad header: {name}')
        header_end = m[-1].end()
        while header_end < len(raw) and raw[header_end:header_end+1] in (b'\r', b'\n', b'\t', b' '):
            header_end += 1
        head = raw[:header_end]
        payload = raw[header_end:]

        hmap = {}
        for m2 in _HEADER_RE.finditer(head):
            k = m2.group(1).decode('ascii')
            v = float(m2.group(2).decode('ascii'))
            hmap[k] = v
        w = int(hmap['width']); h = int(hmap['height'])
        dl = int(hmap['data_length'])
        if dl != w*h*4:  # fallback if header is inconsistent
            dl = w*h*4

        depth = np.frombuffer(payload[:dl], dtype='<f4', count=w*h).reshape(h, w).astype(np.float32)
        depth[~np.isfinite(depth)] = np.nan
        depth[depth <= 0] = np.nan

        ts_abs = float(hmap['timestamp'])
        ts_norm = ts_abs - self.base_ts

        # intrinsics from header
        fx = float(hmap['fx']); fy = float(hmap['fy'])
        ox = float(hmap['ox']); oy = float(hmap['oy'])

        # auto-fix intrinsics if they don't fit (w,h)
        fx, fy, ox, oy, meta = _autoscale_intrinsics_to_frame(w, h, fx, fy, ox, oy)

        return_data = {
            'depth': depth,
            'ts_sec_abs': ts_abs,
            'ts_sec_norm': ts_norm,
            'h': h, 'w': w,
            'fx': fx, 'fy': fy, 'ox': ox, 'oy': oy,
            #**meta
        }
        #print(return_data['fx'], return_data['fy'])
        #print(return_data['ox'], return_data['oy'])
        #sys.exit()
        return return_data

    @property
    def sizes(self): return self._sizes

    @property
    def intrinsics(self): return self._intrinsics


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    """Convert a depth array to a colorized visualization (jet colormap)."""
    d = depth.copy()
    mask = ~np.isfinite(d)
    if np.all(mask):
        return np.zeros((depth.shape[0], depth.shape[1], 3), np.uint8)
    v = d[~mask]
    mn, mx = float(np.nanpercentile(v, 2.0)), float(np.nanpercentile(v, 98.0))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
        mn, mx = np.nanmin(v), np.nanmax(v)
    d[mask] = mn
    d8 = np.clip((d - mn) / max(mx - mn, 1e-6) * 255.0, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(d8, cv2.COLORMAP_JET)
    vis[mask] = (0, 0, 0)
    return vis
