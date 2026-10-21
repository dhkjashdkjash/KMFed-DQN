import hashlib
import json

import numpy as np
import pandas as pd

from .common import IND_COLS, INDICATORS, load_all_stations
from .forecasting import (
    HORIZONS,
    LOOKBACK,
    STEP_HOURS,
    build_supervised,
    chrono_split,
    from_model,
    kalman_filter_df,
    physical_projection,
)

HSET = [1, 2, 3, 4, 5, 6]
WARN_LEVEL = 2


def station_data(label, df, train_end, validation_end, kalman=True, risk_spec=None):
    X, _, D, Yabs, Slast, Plast, times = build_supervised(df)
    if kalman:
        Xf, seq, *_ = build_supervised(df, feat_df=kalman_filter_df(df))
        X = np.concatenate([X, Xf, seq[:, -1] - seq[:, -2]], axis=1)
    tr, va, te = chrono_split(times, train_end, validation_end)
    if min(len(tr), len(va), len(te)) < 2:
        raise ValueError(
            f"{label}: each purged train/validation/test partition needs at least two samples."
        )
    Dall = np.concatenate([D[h] for h in HORIZONS], axis=1)
    sd = Plast[tr].std(0)
    checksum = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=False).values.tobytes()
    ).hexdigest()
    return dict(
        label=label,
        feat=X,
        Dall=Dall,
        tr=tr,
        va=va,
        te=te,
        times=times,
        Slast=Slast,
        Yabs=Yabs,
        ystd=np.where(sd > 1e-12, sd, 1),
        constant_indicators=[IND_COLS[i] for i in np.flatnonzero(sd <= 1e-12)],
        risk_spec=risk_spec,
        n_train=len(tr),
        kalman=kalman,
        checksum=checksum,
        train_end=str(pd.Timestamp(train_end)),
        validation_end=str(pd.Timestamp(validation_end)),
    )


def load_clients(
    data_dir, train_end, validation_end, kalman=True, stations=None, risk_spec=None
):
    ps, _ = load_all_stations(data_dir)
    return {
        lb: station_data(lb, ps[lb], train_end, validation_end, kalman, risk_spec)
        for lb in (list(ps) if stations is None else stations)
    }


def shared_scaler(clients):

    out = {}
    for field, prefix in [("feat", "f"), ("Dall", "t")]:
        parts = [c[field][c["tr"]] for c in clients.values()]
        counts = np.array([len(a) for a in parts])
        mean = np.average([a.mean(0) for a in parts], axis=0, weights=counts)
        var = np.average(
            [a.var(0) + (a.mean(0) - mean) ** 2 for a in parts], axis=0, weights=counts
        )
        out[prefix + "mean"] = mean
        out[prefix + "std"] = np.where(var > 1e-24, np.sqrt(var), 1)
    return out


class LinearForecaster:
    def __init__(self, n_feat, n_out, l2=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_feat, n_out))
        self.b = np.zeros(n_out)
        self.l2 = l2

    def predict(self, Xs):
        return Xs @ self.W + self.b

    def get(self):
        return [self.W.copy(), self.b.copy()]

    def set(self, w):
        self.W, self.b = w[0].copy(), w[1].copy()

    def sgd(self, Xs, Y, epochs=5, lr=0.02, batch=128, seed=0, gclip=5.0):
        rng = np.random.default_rng(seed)
        n = len(Xs)
        for _ in range(epochs):
            idx = rng.permutation(n)
            for s in range(0, n, batch):
                b = idx[s : s + batch]
                pred = Xs[b] @ self.W + self.b
                err = pred - Y[b]
                gW = Xs[b].T @ err / len(b) + self.l2 * self.W
                gb = err.mean(0)
                gn = np.sqrt((gW**2).sum() + (gb**2).sum())
                if gn > gclip:
                    scale = gclip / (gn + 1e-12)
                    gW *= scale
                    gb *= scale
                self.W -= lr * gW
                self.b -= lr * gb


def _macro_f1(yt, yp):
    scores = []
    for cls in (False, True):
        tp = np.sum((yp == cls) & (yt == cls))
        fp = np.sum((yp == cls) & (yt != cls))
        fn = np.sum((yp != cls) & (yt == cls))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def _stdX(feat, scaler):
    return (feat - scaler["fmean"]) / scaler["fstd"]


def predict(W, b, client, scaler, partition="te"):
    if partition not in ("tr", "va", "te"):
        raise ValueError("Unknown partition.")
    idx = client[partition]
    increments = (_stdX(client["feat"][idx], scaler) @ W + b) * scaler["tstd"] + scaler[
        "tmean"
    ]
    result = {}
    for k, h in enumerate(HORIZONS):
        raw = from_model(client["Slast"][idx] + increments[:, k * 5 : (k + 1) * 5])
        projected, flags = physical_projection(raw)
        result[h] = (raw, projected, flags)
    return result


def evaluate(W, b, client, scaler, partition="te"):
    idx = client[partition]
    predictions = predict(W, b, client, scaler, partition)
    rmse, raw_rmse, mae, flags = [], [], [], []
    for h, (raw, pred, mask) in predictions.items():
        target = client["Yabs"][h][idx]
        rmse.append(np.sqrt(np.mean((pred - target) ** 2, axis=0)) / client["ystd"])
        raw_rmse.append(np.sqrt(np.mean((raw - target) ** 2, axis=0)) / client["ystd"])
        mae.append(np.mean(abs(pred - target), axis=0) / client["ystd"])
        flags.append(mask)
    f1 = None
    if client["risk_spec"] is not None:
        risk = client["risk_spec"].levels
        truth = np.any(
            [risk(client["Yabs"][h][idx]) >= WARN_LEVEL for h in HSET], axis=0
        )
        alarm = np.any([risk(predictions[h][1]) >= WARN_LEVEL for h in HSET], axis=0)
        f1 = _macro_f1(truth, alarm)
    return dict(
        nrmse=float(np.mean(rmse)),
        raw_nrmse=float(np.mean(raw_rmse)),
        nmae=float(np.mean(mae)),
        macro_f1=f1,
        projected_values=int(np.sum(flags)),
        n_samples=len(idx),
        constant_indicators=client["constant_indicators"],
    )


def prediction_table(W, b, client, scaler, partition="te"):
    idx = client[partition]
    rows = []
    for h, (raw, pred, mask) in predict(W, b, client, scaler, partition).items():
        for j, col in enumerate(IND_COLS):
            for i, k in enumerate(idx):
                rows.append(
                    dict(
                        station=client["label"],
                        origin=client["times"][k],
                        target_time=client["times"][k]
                        + pd.Timedelta(hours=h * STEP_HOURS),
                        horizon_hours=h * STEP_HOURS,
                        indicator=col,
                        unit=INDICATORS[col]["unit"],
                        observed=client["Yabs"][h][k, j],
                        prediction_raw=raw[i, j],
                        prediction=pred[i, j],
                        projected=bool(mask[i, j]),
                        partition=partition,
                    )
                )
    return pd.DataFrame(rows)


def _dims(clients):
    if not clients:
        raise ValueError("No clients.")
    c = next(iter(clients.values()))
    boundaries = {(x["train_end"], x["validation_end"]) for x in clients.values()}
    if len(boundaries) != 1:
        raise ValueError("All clients must use the same calendar split boundaries.")
    return c["feat"].shape[1], c["Dall"].shape[1]


def _std_xy(c, scaler):
    return _stdX(c["feat"][c["tr"]], scaler), (
        c["Dall"][c["tr"]] - scaler["tmean"]
    ) / scaler["tstd"]


def train_local(clients, scaler, epochs=60, lr=0.05, l2=1e-3, seed=0):

    nf, no = _dims(clients)
    models = {}
    for lb, c in clients.items():
        m = LinearForecaster(nf, no, l2=l2, seed=seed)
        Xs, Y = _std_xy(c, scaler)
        m.sgd(Xs, Y, epochs=epochs, lr=lr, seed=seed)
        models[lb] = m.get()
    return models


def train_centralized(clients, scaler, epochs=60, lr=0.05, l2=1e-3, seed=0):

    nf, no = _dims(clients)
    parts = [_std_xy(c, scaler) for c in clients.values()]
    Xs = np.concatenate([p[0] for p in parts])
    Y = np.concatenate([p[1] for p in parts])
    m = LinearForecaster(nf, no, l2=l2, seed=seed)
    m.sgd(Xs, Y, epochs=epochs, lr=lr, seed=seed)
    return m.get()


def train_federated(
    clients,
    scaler,
    rounds=50,
    local_epochs=5,
    lr=0.05,
    l2=1e-3,
    mode="fedavg",
    seed=0,
    track=True,
):
    if mode not in ("fedavg", "quality"):
        raise ValueError(
            "Supported modes: fedavg, quality. Unverified DP code was removed."
        )
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must be positive.")
    nf, no = _dims(clients)
    model = LinearForecaster(nf, no, l2=l2, seed=seed)
    history = []
    for rd in range(rounds):
        params, weights = [], []
        for i, c in enumerate(clients.values()):
            local = LinearForecaster(nf, no, l2=l2, seed=seed)
            local.set(model.get())
            local.sgd(
                *_std_xy(c, scaler),
                epochs=local_epochs,
                lr=lr,
                seed=seed + rd * len(clients) + i,
            )
            params.append(local.get())
            weights.append(
                1 / (evaluate(local.W, local.b, c, scaler, "va")["nrmse"] + 1e-3)
                if mode == "quality"
                else c["n_train"]
            )
        weights = np.asarray(weights) / np.sum(weights)
        model.set(
            [
                np.sum([w * p[j] for w, p in zip(weights, params)], axis=0)
                for j in (0, 1)
            ]
        )
        if track:
            history.append(
                dict(
                    round=rd + 1,
                    aggregation_weights=weights.tolist(),
                    partition="va",
                    validation={
                        lb: evaluate(model.W, model.b, c, scaler, "va")
                        for lb, c in clients.items()
                    },
                )
            )
    return model.get(), history


def save_forecaster(path, weights, scaler, clients, training):

    _dims(clients)
    settings = {c["kalman"] for c in clients.values()}
    if len(settings) != 1:
        raise ValueError("Mixed feature configurations.")
    metadata = dict(
        schema_version=2,
        indicators=IND_COLS,
        horizons=HORIZONS,
        lookback=LOOKBACK,
        step_hours=STEP_HOURS,
        kalman=settings.pop(),
        q_over_r=0.01,
        log1p=["turbidity", "nh4_n"],
        projection="nonnegative EC/turbidity/NH4-N; pH in [0,14]; raw outputs retained",
        feature_order="raw lags then time; optional filtered lags then time then filtered slope",
        training=training,
        clients={
            lb: {
                k: c[k]
                for k in (
                    "checksum",
                    "train_end",
                    "validation_end",
                    "constant_indicators",
                )
            }
            for lb, c in clients.items()
        },
        risk_specs={
            lb: c["risk_spec"].to_dict() if c["risk_spec"] else None
            for lb, c in clients.items()
        },
    )
    np.savez(
        path,
        W=weights[0],
        b=weights[1],
        **scaler,
        metadata=json.dumps(metadata, ensure_ascii=False),
    )


def load_forecaster(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if (
            metadata["schema_version"] != 2
            or metadata["indicators"] != IND_COLS
            or metadata["horizons"] != HORIZONS
        ):
            raise ValueError("Unsupported forecaster schema.")
        return (
            [data["W"].copy(), data["b"].copy()],
            {k: data[k].copy() for k in ("fmean", "fstd", "tmean", "tstd")},
            metadata,
        )


def forecast_latest(path, observations):

    from .forecasting import _time_feats, contiguous_segments, to_model

    weights, scaler, metadata = load_forecaster(path)
    segments = contiguous_segments(observations)
    latest = segments[-1]
    if len(latest) < metadata["lookback"]:
        raise ValueError("Insufficient contiguous history at the forecast origin.")
    win = to_model(latest[IND_COLS].values[-LOOKBACK:])
    origin = pd.Timestamp(latest.time.iloc[-1])
    tf = _time_feats(pd.DatetimeIndex([origin]))[0]
    features = np.r_[win.ravel(), tf]
    if metadata["kalman"]:
        filtered = kalman_filter_df(latest, metadata["q_over_r"])
        fwin = to_model(filtered[IND_COLS].values[-LOOKBACK:])
        features = np.r_[features, fwin.ravel(), tf, fwin[-1] - fwin[-2]]
    increments = (_stdX(features[None], scaler) @ weights[0] + weights[1]) * scaler[
        "tstd"
    ] + scaler["tmean"]
    rows = []
    for k, h in enumerate(HORIZONS):
        raw = from_model(win[-1] + increments[0, k * 5 : (k + 1) * 5])
        pred, mask = physical_projection(raw)
        for j, c in enumerate(IND_COLS):
            rows.append(
                dict(
                    origin=origin,
                    target_time=origin + pd.Timedelta(hours=h * STEP_HOURS),
                    horizon_hours=h * STEP_HOURS,
                    indicator=c,
                    unit=INDICATORS[c]["unit"],
                    prediction_raw=raw[j],
                    prediction=pred[j],
                    projected=bool(mask[j]),
                )
            )
    return pd.DataFrame(rows)
