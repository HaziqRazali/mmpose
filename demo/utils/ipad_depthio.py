# depthzip.py
"""
Depth ZIP reader and visualization utilities.
Handles .zip archives of frame_*.bin depth frames.
"""

import re, zipfile
import numpy as np
import cv2
from pathlib import Path

_HEADER_RE = re.compile(
    rb'^(timestamp|width|height|data_length|fx|fy|ox|oy)\s*:\s*([0-9\.\-eE]+)\s*$',
    re.M
)

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

        return {
            'depth': depth,
            'ts_sec_abs': ts_abs,
            'ts_sec_norm': ts_norm,
            'h': h, 'w': w,
            'fx': float(hmap['fx']), 'fy': float(hmap['fy']),
            'ox': float(hmap['ox']), 'oy': float(hmap['oy']),
        }

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
