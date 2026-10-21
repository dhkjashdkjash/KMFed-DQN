import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR

from .common import IND_COLS, NONNEGATIVE, validate_station, validate_values
from .kalman import kalman_cv

STEP_HOURS = 4
HORIZONS = [1, 2, 3, 4, 5, 6, 12]
HORIZON_HOURS = {h: h * STEP_HOURS for h in HORIZONS}
LOOKBACK = 6
TF_LOG = {"turbidity", "nh4_n"}


def contiguous_segments(df, step_hours=STEP_HOURS, min_length=1):
    t = validate_station(df)
    if step_hours <= 0 or min_length < 1:
        raise ValueError("Positive step_hours and min_length are required.")
    breaks = np.flatnonzero(np.diff(t.asi8) != pd.Timedelta(hours=step_hours).value) + 1
    bounds = np.r_[0, breaks, len(df)]
    return [
        df.iloc[a:b].reset_index(drop=True)
        for a, b in zip(bounds[:-1], bounds[1:])
        if b - a >= min_length
    ]


def physical_projection(values):

    out = np.asarray(values, dtype=float).copy()
    if not np.isfinite(out).all():
        raise FloatingPointError(
            "Nonfinite model output; no replacement value was used."
        )
    before = out.copy()
    for c in NONNEGATIVE:
        j = IND_COLS.index(c)
        out[..., j] = np.maximum(out[..., j], 0)
    out[..., 1] = np.clip(out[..., 1], 0, 14)
    return out, out != before


def kalman_filter_df(df, q_over_r=0.01):

    parts = []
    for segment, seg in enumerate(contiguous_segments(df)):
        out = seg.copy()
        raw = np.column_stack(
            [kalman_cv(seg[c].to_numpy(), q=q_over_r, r=1)[0] for c in IND_COLS]
        )
        fitted, flags = physical_projection(raw)
        for j, c in enumerate(IND_COLS):
            out[c] = fitted[:, j]
            out[c + "_unconstrained"] = raw[:, j]
            out[c + "_projected"] = flags[:, j]
        out["segment"] = segment
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def to_model(vals):
    out = np.asarray(vals, dtype=float).copy()
    validate_values(out.reshape(-1, len(IND_COLS)))
    for j, c in enumerate(IND_COLS):
        if c in TF_LOG:
            out[..., j] = np.log1p(out[..., j])
    return out


def from_model(vals):
    out = np.asarray(vals, dtype=float).copy()
    with np.errstate(over="raise", invalid="raise"):
        for j, c in enumerate(IND_COLS):
            if c in TF_LOG:
                out[..., j] = np.expm1(out[..., j])
    return out


def _time_feats(ts):
    doy, hod = ts.dayofyear.values, ts.hour.values
    return np.column_stack(
        [
            np.sin(2 * np.pi * doy / 365.25),
            np.cos(2 * np.pi * doy / 365.25),
            np.sin(2 * np.pi * hod / 24),
            np.cos(2 * np.pi * hod / 24),
        ]
    )


def build_supervised(df, feat_df=None, lookback=LOOKBACK, horizons=HORIZONS):

    if (
        lookback < 1
        or not horizons
        or len(set(horizons)) != len(horizons)
        or any(h < 1 or int(h) != h for h in horizons)
    ):
        raise ValueError(
            "Positive lookback and distinct positive integer horizons are required."
        )
    feat_df = df if feat_df is None else feat_df
    times = validate_station(df)
    if not times.equals(validate_station(feat_df)):
        raise ValueError("Feature and target timestamps must match exactly.")
    if (
        "station" in df
        and "station" in feat_df
        and not np.array_equal(df.station, feat_df.station)
    ):
        raise ValueError("Feature and target station labels differ.")
    sf = contiguous_segments(feat_df, min_length=lookback + max(horizons))
    st = contiguous_segments(df, min_length=lookback + max(horizons))
    X, seq, last, physical, origins = [], [], [], [], []
    D, Y = {h: [] for h in horizons}, {h: [] for h in horizons}
    for f, t in zip(sf, st):
        fv, tv = to_model(f[IND_COLS].values), to_model(t[IND_COLS].values)
        ts = pd.DatetimeIndex(f.time)
        for i in range(lookback - 1, len(f) - max(horizons)):
            win = fv[i - lookback + 1 : i + 1]
            X.append(np.r_[win.ravel(), _time_feats(ts[i : i + 1])[0]])
            seq.append(win)
            last.append(tv[i])
            physical.append(t[IND_COLS].iloc[i].values)
            origins.append(ts[i])
            for h in horizons:
                D[h].append(tv[i + h] - tv[i])
                Y[h].append(t[IND_COLS].iloc[i + h].values)
    n, d = len(X), len(IND_COLS)
    return (
        np.asarray(X).reshape(n, lookback * d + 4),
        np.asarray(seq).reshape(n, lookback, d),
        {h: np.asarray(v).reshape(n, d) for h, v in D.items()},
        {h: np.asarray(v).reshape(n, d) for h, v in Y.items()},
        np.asarray(last).reshape(n, d),
        np.asarray(physical).reshape(n, d),
        pd.DatetimeIndex(origins),
    )


def chrono_split(origins, train_end, validation_end, max_horizon=12):

    t = pd.DatetimeIndex(origins)
    a, b = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if not a < b:
        raise ValueError("train_end must precede validation_end.")
    last_target = t + pd.Timedelta(hours=STEP_HOURS * max_horizon)
    return (
        np.flatnonzero(last_target < a),
        np.flatnonzero((t >= a) & (last_target < b)),
        np.flatnonzero(t >= b),
    )


def risk_level(values, spec=None):
    if spec is None:
        raise ValueError(
            "Supply an evidence-backed RiskSpec; no default thresholds are provided."
        )
    return spec.levels(values)


def make_regressor(name):
    if name == "RF":
        return RandomForestRegressor(
            n_estimators=120, max_depth=14, n_jobs=-1, random_state=0
        )
    if name == "GBDT":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_iter=200, max_depth=6, learning_rate=0.08, random_state=0
            )
        )
    if name == "SVR":
        return MultiOutputRegressor(SVR(C=10.0, gamma="scale", epsilon=0.05))
    if name in ("Linear-AR", "AR"):
        return LinearRegression()
    if name == "Ridge":
        return RidgeCV(alphas=np.logspace(-2, 3, 12))
    raise ValueError(name)


def build_dl(name, lookback, n_feat, n_out):

    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.random.set_seed(0)
    inp = layers.Input(shape=(lookback, n_feat))
    if name == "LSTM":
        x = layers.LSTM(32)(inp)
    elif name == "GRU":
        x = layers.GRU(32)(inp)
    elif name == "CausalCNN":
        x = layers.Conv1D(32, 3, padding="causal", dilation_rate=1, activation="relu")(
            inp
        )
        x = layers.Conv1D(32, 3, padding="causal", dilation_rate=2, activation="relu")(
            x
        )
        x = layers.GlobalAveragePooling1D()(x)
    else:
        raise ValueError(name)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(n_out)(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    return model
