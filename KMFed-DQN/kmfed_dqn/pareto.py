import numpy as np
import pandas as pd

REQUIRED = ["basin", "policy_id", "family", "cost", "risk"]


def nondominated(values):

    a = np.asarray(values, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError("Expected finite [cost, residual risk] pairs.")
    return np.array(
        [not np.any(np.all(a <= p, axis=1) & np.any(a < p, axis=1)) for p in a],
        dtype=bool,
    )


def validate(df, expected=None):

    df = df.copy()
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError("Missing columns: " + ",".join(sorted(missing)))
    if len(df) == 0:
        raise ValueError("No policy evaluation data.")
    if df[REQUIRED].isna().any().any():
        raise ValueError("Missing required values; no implicit imputation.")
    if df.duplicated(["basin", "policy_id"]).any():
        raise ValueError(
            "Aggregate evaluation windows to one row per basin and policy first."
        )
    for c in ["cost", "risk"]:
        if not np.isfinite(pd.to_numeric(df[c], errors="coerce")).all():
            raise ValueError("Nonfinite objective: " + c)
        df[c] = pd.to_numeric(df[c], errors="raise")
        if (df[c] < 0).any():
            raise ValueError("Expected nonnegative cost and residual risk.")
    if expected is not None:
        absent = set(expected) - set(df.basin)
        extra = set(df.basin) - set(expected)
        if absent or extra:
            raise ValueError(
                f"Basin mismatch. Missing: {sorted(absent)}. Extra: {sorted(extra)}."
            )
    return df


def analyze(df):

    df = validate(df)
    parts, fronts, representatives = [], [], []
    for basin, g in df.groupby("basin", sort=False):
        g = g.sort_values(["cost", "risk", "policy_id"]).copy().reset_index(drop=True)
        g["nondominated"] = nondominated(g[["cost", "risk"]].values)
        f = g.loc[g.nondominated].copy()
        lo = f[["cost", "risk"]].min().values
        span = f[["cost", "risk"]].max().values - lo
        z = (f[["cost", "risk"]].values - lo) / np.where(span > 0, span, 1.0)
        f["ideal_distance"] = np.sqrt((z * z).sum(axis=1))
        reps = {
            "Minimum cost": f.sort_values(["cost", "risk", "policy_id"]).iloc[0],
            "Balanced": f.sort_values(
                ["ideal_distance", "cost", "risk", "policy_id"]
            ).iloc[0],
            "Minimum risk": f.sort_values(["risk", "cost", "policy_id"]).iloc[0],
        }
        for role, row in reps.items():
            representatives.append(
                dict(
                    basin=basin,
                    role=role,
                    policy_id=row.policy_id,
                    cost=row.cost,
                    risk=row.risk,
                )
            )
        g["representative_roles"] = g.policy_id.map(
            lambda p, reps=reps: "|".join(
                k for k, v in reps.items() if v.policy_id == p
            )
        )
        parts.append(g)
        fronts.append(f)
    return (
        pd.concat(parts, ignore_index=True),
        pd.concat(fronts, ignore_index=True),
        pd.DataFrame(representatives),
    )


def self_check():

    x = np.array([[1, 4], [2, 2], [3, 3], [4, 1], [2, 2], [0, 5], [1, 5]])
    assert nondominated(x).tolist() == [True, True, False, True, True, True, False]
    assert nondominated([[0, 0], [0, 0]]).all()
    assert nondominated([[0, 2], [1, 2], [2, 2]]).tolist() == [True, False, False]
    assert nondominated([[2, 0], [2, 1], [2, 2]]).tolist() == [True, False, False]
    assert nondominated([[1, 1]]).tolist() == [True]
    return "PASS"


if __name__ == "__main__":
    print(self_check())
