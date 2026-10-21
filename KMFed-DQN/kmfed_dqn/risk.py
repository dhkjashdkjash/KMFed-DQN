import json
from pathlib import Path

import numpy as np

from .common import IND_COLS, validate_values


class RiskSpec:


    def __init__(self, sensors, source):
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                "A threshold source and applicability description is required."
            )
        if set(sensors) != set(IND_COLS):
            raise ValueError(f"Risk bands must cover exactly {IND_COLS}.")
        self.source = source
        self.sensors = sensors
        self.bands = {}
        for col, entry in sensors.items():
            bands = []
            for name in ("normal_range", "warning_threshold", "critical_threshold"):
                pair = entry[name]
                if len(pair) != 2:
                    raise ValueError(f"{col}/{name}: expected two bounds.")
                lo = -np.inf if pair[0] is None else float(pair[0])
                hi = np.inf if pair[1] is None else float(pair[1])
                if (
                    np.isnan(lo)
                    or np.isnan(hi)
                    or lo >= hi
                    or (lo == -np.inf and hi == np.inf)
                ):
                    raise ValueError(
                        f"{col}/{name}: invalid or wholly unbounded interval."
                    )
                bands.append((lo, hi))
            if not all(
                bands[i + 1][0] <= bands[i][0] and bands[i][1] <= bands[i + 1][1]
                for i in (0, 1)
            ):
                raise ValueError(f"{col}: risk bands must be nested.")
            self.bands[col] = bands

    @classmethod
    def load(cls, path):
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(obj["sensors"], obj["source"])

    def levels(self, values):
        a = validate_values(values)
        sub = np.zeros(a.shape, dtype=int)
        for j, col in enumerate(IND_COLS):
            for level, (lo, hi) in enumerate(self.bands[col], 1):
                sub[:, j][(a[:, j] < lo) | (a[:, j] > hi)] = level
        return sub.max(axis=1)

    def to_dict(self):
        return {"sensors": self.sensors, "source": self.source}
