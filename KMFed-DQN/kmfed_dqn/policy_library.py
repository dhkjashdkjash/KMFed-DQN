import numpy as np
import pandas as pd

from .pareto import analyze


def aggregate_policy_evaluations(evaluations):

    metadata = ["evaluation_source", "cost_unit", "risk_definition", "time_step"]
    keys = ["basin", "policy_id", "station", "month"]
    required = keys + ["family", "cost", "risk"] + metadata
    if evaluations.empty or not set(required) <= set(evaluations):
        raise ValueError("Missing policy evaluation identifiers, values or provenance.")
    data = evaluations[required].copy()
    if data.isna().any().any():
        raise ValueError("Missing evaluation values.")
    text_cols = keys + ["family"] + metadata
    if data[text_cols].applymap(lambda x: not str(x).strip()).any().any():
        raise ValueError("Empty evaluation identifiers or provenance.")
    data["month"] = pd.PeriodIndex(data["month"], freq="M").astype(str)
    if data["month"].eq("NaT").any() or data.duplicated(keys).any():
        raise ValueError("Invalid or duplicate station-month-policy keys.")
    for col in ("cost", "risk"):
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col]).all() or (data[col] < 0).any():
            raise ValueError("Cost and residual risk must be finite and nonnegative.")
    rows = []
    for basin, block in data.groupby("basin", sort=False):
        if any(block[c].nunique() != 1 for c in metadata):
            raise ValueError(
                "Compared policies require consistent evaluation provenance and units."
            )
        expected_cohort = None
        for policy, group in block.groupby("policy_id", sort=False):
            cohort = set(zip(group["station"], group["month"]))
            if expected_cohort is None:
                expected_cohort = cohort
            if cohort != expected_cohort or group["family"].nunique() != 1:
                raise ValueError(
                    "Policy cohorts differ or family labels are inconsistent."
                )
            stations = group.groupby("station")[["cost", "risk"]].mean()
            rows.append(
                dict(
                    basin=basin,
                    policy_id=policy,
                    family=group["family"].iloc[0],
                    cost=float(stations["cost"].mean()),
                    risk=float(stations["risk"].mean()),
                    n_stations=len(stations),
                    n_station_months=len(group),
                    aggregation="time mean within station; equal station mean",
                    **{key: group[key].iloc[0] for key in metadata},
                )
            )
    return pd.DataFrame(rows)


def paired_observations(monthly):

    keys = ["basin", "station", "month", "param"]
    if not set(keys + ["value", "unit"]) <= set(monthly):
        raise ValueError("Missing monthly-table columns including unit.")
    data = monthly.loc[monthly["param"].isin(["pH", "NH4N"])].copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M").astype(str)
    if (data["month"] == "NaT").any():
        raise ValueError("Missing monthly timestamps.")
    if data[keys + ["value", "unit"]].isna().any().any() or data.duplicated(keys).any():
        raise ValueError("Missing values or duplicate station-month-indicator keys.")
    data["value"] = pd.to_numeric(data["value"], errors="raise")
    if (
        not np.isfinite(data["value"]).all()
        or (data.groupby("param")["unit"].nunique() != 1).any()
    ):
        raise ValueError("Nonfinite values or inconsistent units.")
    wide = data.pivot(index=keys[:3], columns="param", values="value").reset_index()
    if not {"pH", "NH4N"} <= set(wide) or wide[["pH", "NH4N"]].isna().any().any():
        raise ValueError("Unpaired pH/NH4N records; no rows were silently dropped.")
    if ((wide["pH"] < 0) | (wide["pH"] > 14) | (wide["NH4N"] < 0)).any():
        raise ValueError("Invalid pH or NH4N observations.")
    return wide


def policy_frontier(evaluations):

    metadata = [
        "evaluation_source",
        "cost_unit",
        "risk_definition",
        "aggregation",
        "time_step",
    ]
    if not set(metadata) <= set(evaluations):
        raise ValueError("Policy evaluations require " + ", ".join(metadata))
    if (
        evaluations[metadata].isna().any().any()
        or evaluations[metadata].applymap(lambda x: not str(x).strip()).any().any()
    ):
        raise ValueError("Empty policy evaluation metadata.")
    for _, block in evaluations.groupby("basin"):
        if any(block[c].nunique() != 1 for c in metadata[1:]):
            raise ValueError(
                "Compared policies must have identical units, risk definition, aggregation and time step."
            )
    return analyze(evaluations)
