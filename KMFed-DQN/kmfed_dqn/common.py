from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

STATIONS = {
    "\u705e\u6cb3\u53e3": {"label": "Bahekou", "color": "#1f77b4"},
    "\u5317\u5cea\u53e3": {"label": "Beiyukou", "color": "#ff7f0e"},
    "\u78a7\u6751": {"label": "Bicun", "color": "#2ca02c"},
    "\u5c94\u53e3": {"label": "Chakou", "color": "#d62728"},
}
INDICATORS = {
    "water_temperature": {"display": "Water temperature", "short": "WT", "unit": "°C"},
    "ph": {"display": "pH", "short": "pH", "unit": "1"},
    "conductivity": {"display": "Conductivity", "short": "EC", "unit": "µS/cm"},
    "turbidity": {"display": "Turbidity", "short": "Turb", "unit": "NTU"},
    "nh4_n": {"display": "Ammonia nitrogen", "short": "NH4-N", "unit": "mg N/L"},
}
IND_COLS = list(INDICATORS)
NONNEGATIVE = ("conductivity", "turbidity", "nh4_n")


def validate_values(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != len(IND_COLS) or not np.isfinite(a).all():
        raise ValueError(
            "Expected a finite (n, 5) matrix; missing values require source QC."
        )
    if any((a[:, IND_COLS.index(c)] < 0).any() for c in NONNEGATIVE):
        raise ValueError(
            "Negative conductivity, turbidity or ammonia requires source QC."
        )
    if ((a[:, 1] < 0) | (a[:, 1] > 14)).any():
        raise ValueError("pH outside [0, 14] requires source QC.")
    return a


def validate_station(df):
    missing = {"time", *IND_COLS} - set(df)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Empty station table.")
    t = pd.DatetimeIndex(pd.to_datetime(df["time"], errors="raise"))
    if t.hasnans or t.has_duplicates or not t.is_monotonic_increasing:
        raise ValueError(
            "Station times must be unique, present and sorted. Resolve duplicates at source."
        )
    validate_values(df[IND_COLS].to_numpy())
    return t


def _clean_one(df, station_label):

    out = df.copy()
    source = "time" if "time" in out else "Monitoring Time"
    out["time"] = pd.to_datetime(out[source], errors="raise")
    for c in IND_COLS:
        out[c] = pd.to_numeric(out[c], errors="raise")
    out = out.sort_values("time").reset_index(drop=True)
    validate_station(out)
    out["station"] = station_label
    return out[["time", "station", *IND_COLS]]


def load_all_stations(data_dir=None):
    if data_dir is None:
        raise ValueError(
            "Supply an external data_dir. Monitoring data are not distributed with this code."
        )
    paths = sorted(Path(data_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No station CSV files in {data_dir}")
    stations = {}
    for path in paths:
        label = STATIONS.get(path.stem, {}).get("label", path.stem)
        if label in stations:
            raise ValueError(f"Duplicate station label: {label}")
        stations[label] = _clean_one(pd.read_csv(path), label)
    return stations, pd.concat(stations.values(), ignore_index=True)


def station_order():
    return [m["label"] for m in STATIONS.values()]


def station_colors():
    return {m["label"]: m["color"] for m in STATIONS.values()}
