from collections import deque
import numpy as np

# ---------- 1D filters ----------

class _Filter1DBase:
    def reset(self):
        pass
    def update(self, x: float) -> float:
        raise NotImplementedError

class MovingAverage1D(_Filter1DBase):
    def __init__(self, window=5, robust=False):
        self.window = max(1, int(window))
        self.robust = bool(robust)
        self.buf = deque(maxlen=self.window)
    def reset(self):
        self.buf.clear()
    def update(self, x: float) -> float:
        if not np.isfinite(x):
            return x
        self.buf.append(float(x))
        arr = np.array(self.buf, dtype=float)
        return float(np.median(arr)) if self.robust else float(np.mean(arr))

class Exponential1D(_Filter1DBase):
    def __init__(self, alpha=0.3):
        self.alpha = float(alpha)
        self.y = None
    def reset(self):
        self.y = None
    def update(self, x: float) -> float:
        if not np.isfinite(x):
            return x
        if self.y is None:
            self.y = float(x)
        else:
            self.y = self.alpha * float(x) + (1.0 - self.alpha) * self.y
        return float(self.y)

class SavitzkyGolay1D(_Filter1DBase):
    """
    Simple SG smoother without scipy:
    - Keep last W samples (W odd).
    - Fit poly of degree P to indices [0..W-1] each update and return fitted value at last index.
    """
    def __init__(self, window=7, poly=2):
        self.window = int(window if window % 2 == 1 else window + 1)
        self.poly = int(poly)
        if self.poly >= self.window:
            self.poly = max(0, self.window - 1)
        self.buf = deque(maxlen=self.window)
    def reset(self):
        self.buf.clear()
    def update(self, x: float) -> float:
        if not np.isfinite(x):
            return x
        self.buf.append(float(x))
        arr = np.array(self.buf, dtype=float)
        n = arr.shape[0]
        if n < max(3, self.poly + 1):
            return float(arr.mean())
        xs = np.arange(n, dtype=float)
        # Fit polynomial to current window and evaluate at last index
        try:
            coeffs = np.polyfit(xs, arr, deg=min(self.poly, n - 1))
            y_hat = np.polyval(coeffs, xs[-1])
            return float(y_hat)
        except Exception:
            return float(arr.mean())

class KalmanCV1D(_Filter1DBase):
    """
    Constant-velocity Kalman filter for 1D sequences.
    State x = [pos, vel].
    """
    def __init__(self, q=0.02, r=1.0, init_var=10.0, dt=1.0/30.0):
        self.q = float(q)
        self.r = float(r)
        self.dt = float(dt)
        self.init_var = float(init_var)
        self.reset()
    def reset(self):
        self.x = None  # state [pos, vel]
        self.P = None  # covariance 2x2
    def update(self, z: float) -> float:
        if not np.isfinite(z):
            return z
        z = float(z)
        dt = self.dt
        F = np.array([[1.0, dt],
                      [0.0, 1.0]], dtype=float)
        H = np.array([[1.0, 0.0]], dtype=float)
        Q = np.array([[self.q*dt*dt, 0.0],
                      [0.0, self.q]], dtype=float)
        R = np.array([[self.r]], dtype=float)

        if self.x is None:
            self.x = np.array([z, 0.0], dtype=float)
            self.P = np.eye(2, dtype=float) * self.init_var

        # Predict
        x_pred = F @ self.x
        P_pred = F @ self.P @ F.T + Q

        # Update
        y = np.array([[z]]) - (H @ x_pred).reshape(1,1)  # innovation
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)
        x_new = x_pred + (K @ y).reshape(2,)
        P_new = (np.eye(2) - K @ H) @ P_pred

        self.x, self.P = x_new, P_new
        return float(self.x[0])

# ---------- Vector wrappers (apply per dimension) ----------

class VectorFilter:
    def __init__(self, dim, base_factory):
        self.dim = int(dim)
        self.fs = [base_factory() for _ in range(self.dim)]
    def reset(self):
        for f in self.fs:
            f.reset()
    def update(self, v):
        v = np.asarray(v, dtype=float)
        out = np.array([self.fs[i].update(v[i]) if np.isfinite(v[i]) else v[i] for i in range(self.dim)], dtype=float)
        return out

# ---------- Factories ----------

def make_filter_1d(kind="moving_average",
                   ma_window=5, ma_robust=False,
                   exp_alpha=0.3,
                   sg_window=7, sg_poly=2,
                   kf_q=0.02, kf_r=1.0, kf_init_var=10.0, kf_dt=1.0/30.0):
    k = str(kind).lower()
    if k == "none":
        return _Noop1D()
    if k == "moving_average":
        return MovingAverage1D(window=ma_window, robust=ma_robust)
    if k == "exponential":
        return Exponential1D(alpha=exp_alpha)
    if k == "savitzky_golay":
        return SavitzkyGolay1D(window=sg_window, poly=sg_poly)
    if k == "kalman":
        return KalmanCV1D(q=kf_q, r=kf_r, init_var=kf_init_var, dt=kf_dt)
    return MovingAverage1D(window=ma_window, robust=ma_robust)

def make_filter_vec(kind="moving_average", dim=3,
                    ma_window=5, ma_robust=False,
                    exp_alpha=0.3,
                    sg_window=7, sg_poly=2,
                    kf_q=0.02, kf_r=1.0, kf_init_var=10.0, kf_dt=1.0/30.0):
    def factory():
        return make_filter_1d(
            kind=kind,
            ma_window=ma_window, ma_robust=ma_robust,
            exp_alpha=exp_alpha,
            sg_window=sg_window, sg_poly=sg_poly,
            kf_q=kf_q, kf_r=kf_r, kf_init_var=kf_init_var, kf_dt=kf_dt,
        )
    return VectorFilter(dim=dim, base_factory=factory)

class _Noop1D(_Filter1DBase):
    def update(self, x: float) -> float:
        return x
