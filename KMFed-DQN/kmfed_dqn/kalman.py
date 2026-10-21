import numpy as np
from scipy import stats
from scipy.signal import savgol_filter


def kalman_cv(z, q, r, dt=1.0):

    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("z must be a nonempty finite one-dimensional series.")
    if not np.isfinite([q, r, dt]).all() or q < 0 or r <= 0 or dt <= 0:
        raise ValueError("Require q >= 0, r > 0 and dt > 0.")
    n = len(z)
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = q * np.array([[dt**3 / 3.0, dt**2 / 2.0], [dt**2 / 2.0, dt]])
    R = np.array([[r]])

    x = np.array([[z[0]], [0.0]])
    P = np.eye(2) * r
    x_filt = np.zeros(n)
    innov = np.zeros(n)

    for k in range(n):

        x = F @ x
        P = F @ P @ F.T + Q

        y = z[k] - (H @ x)[0, 0]
        innov[k] = y

        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T) / S
        x = x + K * y
        P = (np.eye(2) - K @ H) @ P
        x_filt[k] = x[0, 0]
    return x_filt, innov


def reference_proxy(z, win=31, poly=2):

    z = np.asarray(z, dtype=float)
    if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Expected a nonempty finite series.")
    if poly < 0 or win < 1 or int(poly) != poly or int(win) != win:
        raise ValueError("Invalid smoothing window or polynomial order.")
    win = min(win, len(z) - (1 - len(z) % 2))
    if win < poly + 2:
        return z.copy()
    if win % 2 == 0:
        win += 1
    return savgol_filter(z, win, poly)


def estimate_R(z):

    ref = reference_proxy(z)
    return max(np.var(z - ref), 1e-9)


def ljung_box(x, lags=20):

    x = np.asarray(x, dtype=float)
    if (
        x.ndim != 1
        or not np.isfinite(x).all()
        or not 1 <= lags < len(x)
        or int(lags) != lags
    ):
        raise ValueError("Require a finite series and integer 1 <= lags < n.")
    x = x - x.mean()
    n = len(x)
    c0 = np.sum(x * x)
    if c0 == 0:
        return np.nan, np.nan
    q = (
        n
        * (n + 2)
        * sum((np.dot(x[k:], x[:-k]) / c0) ** 2 / (n - k) for k in range(1, lags + 1))
    )
    return float(q), float(stats.chi2.sf(q, lags))
