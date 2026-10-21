import numpy as np
import pandas as pd

HORIZONS = (1, 2, 3, 6, 12)
N_LAGS = 6
MODELS = ("persistence", "federated_ar", "kalman_federated_ar")


def aggregate_monthly(
    observations, *, min_count, variance_ddof, quantile_interpolation
):

    required = ["basin", "station", "param", "time", "value", "unit"]
    if observations.empty or not set(required) <= set(observations):
        raise ValueError(
            "Expected nonempty basin,station,param,time,value,unit observations."
        )
    if (
        not isinstance(min_count, (int, np.integer))
        or min_count < 1
        or variance_ddof not in (0, 1)
        or quantile_interpolation
        not in {"linear", "lower", "higher", "midpoint", "nearest"}
    ):
        raise ValueError(
            "Specify min_count >= 1, variance_ddof 0/1 and a quantile convention."
        )
    data = observations[required].copy()
    if data.isna().any().any():
        raise ValueError(
            "Resolve missing values using source QC before monthly aggregation."
        )
    for col in ("basin", "station", "param", "unit"):
        if data[col].astype(str).str.strip().eq("").any():
            raise ValueError("Empty identifiers or units.")
    data["time"] = pd.to_datetime(data["time"], errors="raise")
    if data["time"].isna().any():
        raise ValueError("Invalid observation time.")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if not np.isfinite(data["value"]).all():
        raise ValueError("Nonfinite observations.")
    keys = ["basin", "station", "param"]
    if data.duplicated(keys + ["time"]).any():
        raise ValueError(
            "Duplicate observations require source resolution, not aggregation."
        )
    if (data.groupby("param")["unit"].nunique() != 1).any():
        raise ValueError("Convert and document units before aggregation.")
    if ((data["param"] == "NH4N") & (data["value"] < 0)).any():
        raise ValueError("Negative NH4N observations.")
    if ((data["param"] == "pH") & ~data["value"].between(0, 14)).any():
        raise ValueError("pH outside the supported [0,14] modelling range.")
    data["month"] = data["time"].dt.strftime("%Y-%m")
    group = data.groupby(keys + ["month", "unit"], sort=True)["value"]
    result = group.agg(value="median", n="size").reset_index()
    result["variance"] = group.var(ddof=variance_ddof).to_numpy()
    result["p95"] = group.quantile(
        0.95, interpolation=quantile_interpolation
    ).to_numpy()
    result["eligible"] = (result["n"] >= min_count) & (result["n"] > variance_ddof)
    result["min_count"] = min_count
    result["variance_ddof"] = variance_ddof
    result["quantile_interpolation"] = quantile_interpolation
    result["aggregation"] = "monthly median"
    return result


def make_windows(monthly, horizons=HORIZONS, log_params=("NH4N",), min_valid_lags=3):

    required = ["basin", "param", "station", "month", "value", "unit"]
    if not set(required) <= set(monthly) or monthly.empty:
        raise ValueError(
            "Monthly table requires nonempty basin,param,station,month,value,unit columns."
        )
    monthly = monthly.copy()
    if "eligible" in monthly and not monthly["eligible"].eq(True).all():
        raise ValueError(
            "Select eligible monthly rows explicitly before making windows."
        )
    if monthly[required].isna().any().any():
        raise ValueError(
            "Missing monthly values require source QC; absent months may be omitted."
        )
    monthly["month"] = pd.PeriodIndex(monthly["month"], freq="M")
    if monthly["month"].isna().any():
        raise ValueError("Missing or invalid monthly timestamps.")
    if (
        (
            monthly[["basin", "station", "param", "unit"]]
            .astype(str)
            .applymap(lambda x: not x.strip())
        )
        .any()
        .any()
    ):
        raise ValueError("Empty identifiers or units.")
    monthly["value"] = pd.to_numeric(monthly["value"], errors="raise")
    if not np.isfinite(monthly["value"]).all():
        raise ValueError("Nonfinite monthly observations.")
    if monthly.duplicated(["basin", "param", "station", "month"]).any():
        raise ValueError(
            "Duplicate station-indicator-month keys; no first/last value is selected."
        )
    if (monthly.groupby("param")["unit"].nunique() != 1).any():
        raise ValueError("Convert and document units consistently before analysis.")
    if ((monthly["param"].isin(log_params)) & (monthly["value"] < 0)).any():
        raise ValueError("Negative concentrations are invalid for log1p modelling.")
    ph = monthly.loc[monthly["param"] == "pH", "value"]
    if ((ph < 0) | (ph > 14)).any():
        raise ValueError("pH outside the supported [0,14] modelling range.")
    if (
        not horizons
        or len(set(horizons)) != len(horizons)
        or any(h < 1 or int(h) != h for h in horizons)
        or not 1 <= min_valid_lags <= N_LAGS
    ):
        raise ValueError("Invalid horizons or min_valid_lags.")
    rows = []
    for (basin, param, station), g in monthly.groupby(
        ["basin", "param", "station"], sort=False
    ):
        months = pd.PeriodIndex(g["month"], freq="M").asi8
        lo, hi = months.min(), months.max()
        series = np.full(hi - lo + 1, np.nan)
        series[months - lo] = g["value"].values
        if param in log_params:
            series = np.log1p(series)
        for i in np.flatnonzero(np.isfinite(series)):
            if i < N_LAGS - 1:
                continue
            lag = series[i - N_LAGS + 1 : i + 1]
            if np.isfinite(lag).sum() < min_valid_lags:
                continue
            for h in horizons:
                if i + h >= len(series) or not np.isfinite(series[i + h]):
                    continue
                row = dict(
                    basin=basin,
                    param=param,
                    station=station,
                    origin=int(i + lo),
                    target_month=int(i + lo + h),
                    horizon=h,
                    y=float(series[i + h]),
                    unit=g["unit"].iloc[0],
                    transform="log1p" if param in log_params else "identity",
                )
                row.update({f"lag{k}": lag[k] for k in range(N_LAGS)})
                rows.append(row)
    windows = pd.DataFrame(rows)
    if len(windows):
        windows["year"] = windows["origin"] // 12 + 1970
    return windows


def design_matrix(lags, origin, horizon, mean, sd, kalman=False, q=0.05):

    if not np.isfinite([mean, sd, q]).all() or sd <= 0 or q < 0:
        raise ValueError("Require finite mean, positive sd and nonnegative q.")
    z = (lags - mean) / sd
    mask = np.isfinite(z)
    if kalman:
        smoothed = np.zeros_like(z)
        state = np.zeros(len(z))
        p = np.ones(len(z))
        started = np.zeros(len(z), dtype=bool)
        for j in range(z.shape[1]):
            first = mask[:, j] & ~started
            state[first] = z[first, j]
            p[first] = 1.0
            started |= first
            existing = mask[:, j] & ~first
            p += q
            k = p / (p + 1.0)
            state[existing] += k[existing] * (z[existing, j] - state[existing])
            p[existing] *= 1 - k[existing]
            smoothed[:, j] = state
        z = smoothed
    else:
        z = np.nan_to_num(z, nan=0.0)
    angle = 2 * np.pi * ((origin + horizon) % 12) / 12
    return np.column_stack(
        [np.ones(len(z)), z, mask.astype(float), np.sin(angle), np.cos(angle)]
    )


def federated_ridge(X, y, clients, ridge=1e-3):

    if ridge < 0 or not np.isfinite(ridge):
        raise ValueError("ridge must be finite and nonnegative.")
    clients = np.asarray(clients)
    A = np.zeros((X.shape[1], X.shape[1]))
    b = np.zeros(X.shape[1])
    for c in sorted(set(clients)):
        Xc, yc = X[clients == c], y[clients == c]
        A += Xc.T @ Xc
        b += Xc.T @ yc
    reg = np.eye(X.shape[1]) * ridge * len(X)
    reg[0, 0] = 1e-10
    return np.linalg.solve(A + reg, b)


def leave_one_basin_out(
    windows, min_train=200, min_clients=3, ridge=1e-3, nonneg_params=("NH4N",)
):

    if windows.empty:
        raise ValueError("No forecasting windows; check continuity and record length.")
    out = []
    for (param, h), block in windows.groupby(["param", "horizon"]):
        y = block["y"].values
        lags = block[[f"lag{k}" for k in range(N_LAGS)]].values
        origin = block["origin"].values
        basins = block["basin"].values
        target = block["target_month"].values
        years = block["year"].values
        for held in pd.unique(basins):
            for year in sorted(np.unique(years[basins == held])):
                cutoff = (year - 1970) * 12
                train = (basins != held) & (target < cutoff)
                test = (basins == held) & (years == year)
                clients = basins[train]
                if train.sum() < min_train or len(set(clients)) < min_clients:
                    continue
                mean = float(np.mean(y[train]))
                sd = float(np.std(y[train], ddof=1))
                if sd <= 1e-8:
                    continue
                base = block.loc[
                    test,
                    [
                        "basin",
                        "param",
                        "station",
                        "origin",
                        "target_month",
                        "horizon",
                        "year",
                        "y",
                        "unit",
                        "transform",
                    ],
                ].copy()
                for model in MODELS:
                    if model == "persistence":
                        pred = lags[test, -1]
                    else:
                        kal = model.startswith("kalman")
                        X = design_matrix(lags[train], origin[train], h, mean, sd, kal)
                        Xt = design_matrix(lags[test], origin[test], h, mean, sd, kal)
                        delta = (y[train] - lags[train, -1]) / sd
                        coef = federated_ridge(X, delta, clients, ridge)
                        pred = lags[test, -1] + Xt @ coef * sd
                    res = base.copy()
                    res["model"] = model
                    res["prediction_model_raw"] = pred
                    raw = (
                        np.expm1(pred)
                        if base["transform"].iloc[0] == "log1p"
                        else pred.copy()
                    )
                    truth = (
                        np.expm1(base["y"])
                        if base["transform"].iloc[0] == "log1p"
                        else base["y"]
                    )
                    if not np.isfinite(raw).all():
                        raise FloatingPointError("Nonfinite cross-basin prediction.")
                    physical = (
                        np.maximum(raw, 0) if param in nonneg_params else raw.copy()
                    )
                    if param == "pH":
                        physical = np.clip(physical, 0, 14)
                    res["prediction_raw"] = raw
                    res["prediction"] = physical
                    res["projected"] = physical != raw
                    res["observed"] = truth
                    res = res.rename(columns={"y": "observed_model"})
                    res["training_sd_model"] = sd
                    res["origin_month"] = [
                        str(pd.Period(ordinal=int(v), freq="M")) for v in base["origin"]
                    ]
                    res["target_month_label"] = [
                        str(pd.Period(ordinal=int(v), freq="M"))
                        for v in base["target_month"]
                    ]
                    res["training_target_before"] = str(
                        pd.Period(ordinal=int(cutoff), freq="M")
                    )
                    res["n_train"] = int(train.sum())
                    res["n_source_basins"] = len(set(clients))
                    out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
