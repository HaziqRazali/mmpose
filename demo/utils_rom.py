import numpy as np

def angle(p_a, p_b, p_c, zero_at_extension=False):
    
    """
    Angle at vertex B formed by segments A-B and C-B.
    Returns degrees in [0, 180]. If zero_at_extension=True,
    returns 0° at full extension (180° raw) and increases with flexion.
    """
    
    # NaN guard
    if (np.any(np.isnan(p_a)) or np.any(np.isnan(p_b)) or np.any(np.isnan(p_c))):
        return np.nan

    u = p_a - p_b
    v = p_c - p_b
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return np.nan

    u = u / nu
    v = v / nv
    cosang = np.clip(np.dot(u, v), -1.0, 1.0)
    raw = np.degrees(np.arccos(cosang))  # 0° = colinear (same dir), 180° = opposite

    if zero_at_extension:
        # Clinical-style: 0° at full extension (raw~180), increases with flexion
        return 180.0 - raw
    else:
        return raw