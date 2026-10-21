from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .common import IND_COLS, validate_station, validate_values
from .forecasting import contiguous_segments
from .risk import RiskSpec

OBS_DIM = 2 * len(IND_COLS)


@dataclass
class ControlSpec:
    action_names: tuple
    no_action: int
    costs: tuple
    cost_unit: str
    risk_penalties: tuple
    step_hours: float
    source: str
    transition: Callable
    risk: RiskSpec

    def __post_init__(self):
        if len(self.action_names) < 2 or len(set(self.action_names)) != len(
            self.action_names
        ):
            raise ValueError("Provide at least two distinct, documented actions.")
        if (
            not 0 <= self.no_action < len(self.action_names)
            or int(self.no_action) != self.no_action
        ):
            raise ValueError("Invalid no_action index.")
        if (
            len(self.costs) != len(self.action_names)
            or not np.isfinite(self.costs).all()
            or min(self.costs) < 0
        ):
            raise ValueError(
                "Provide finite nonnegative per-step costs in one stated unit."
            )
        if (
            len(self.risk_penalties) != 4
            or not np.isfinite(self.risk_penalties).all()
            or min(self.risk_penalties) < 0
        ):
            raise ValueError(
                "Provide four nonnegative ordinal-class penalties, with a decision rationale."
            )
        if not np.isfinite(self.step_hours) or self.step_hours <= 0:
            raise ValueError("Specify the calibrated time step in hours.")
        if (
            not self.source.strip()
            or not self.cost_unit.strip()
            or not callable(self.transition)
            or not isinstance(self.risk, RiskSpec)
        ):
            raise ValueError("Transition, source, cost unit and RiskSpec are required.")

    def metadata(self):
        return dict(
            action_names=list(self.action_names),
            no_action=self.no_action,
            costs=list(self.costs),
            cost_unit=self.cost_unit,
            risk_penalties=list(self.risk_penalties),
            step_hours=self.step_hours,
            source=self.source,
            risk=self.risk.to_dict(),
        )


class WaterControlEnv:


    def __init__(
        self,
        series,
        norm,
        spec,
        max_steps=200,
        cost_weight=1.0,
        seed=0,
        label="station",
    ):
        validate_station(series)
        self.frame = series.copy()
        self.series = series[IND_COLS].to_numpy(dtype=float).copy()
        self.norm = tuple(np.asarray(x, dtype=float).copy() for x in norm)
        if (
            any(x.shape != (5,) for x in self.norm)
            or not np.isfinite(self.norm).all()
            or (self.norm[1] <= 0).any()
        ):
            raise ValueError(
                "Expected finite five-indicator training mean and positive scale."
            )
        if not isinstance(spec, ControlSpec):
            raise ValueError(
                "An external ControlSpec is required; no action-response defaults exist."
            )
        if (
            int(max_steps) != max_steps
            or max_steps < 1
            or not np.isfinite(cost_weight)
            or cost_weight < 0
        ):
            raise ValueError("Invalid episode length or cost weight.")
        self.spec, self.max_steps, self.cost_weight = spec, max_steps, cost_weight
        self.label, self.rng = label, np.random.default_rng(seed)
        self.obs_dim, self.n_actions = OBS_DIM, len(spec.action_names)
        self.segments = contiguous_segments(self.frame, step_hours=spec.step_hours)
        self.starts = []
        offset = 0
        for segment in self.segments:
            self.starts.extend(range(offset, offset + len(segment) - max_steps))
            offset += len(segment)
        if not self.starts:
            raise ValueError(
                "No continuous segment is long enough for the requested episode."
            )
        self.done = True

    def _obs(self):
        mean, scale = self.norm
        return np.r_[(self.state - mean) / scale, (self.state - self.previous) / scale]

    def reset(self, start=None):
        start = int(self.rng.choice(self.starts)) if start is None else start
        if start not in self.starts:
            raise ValueError("Episode would cross a gap or run past the record.")
        if hasattr(self.spec.transition, "reset"):
            self.spec.transition.reset()
        self.index, self.steps, self.done = start, 0, False
        self.state = self.series[start].copy()
        self.previous = self.state.copy()
        self.initial_level = int(self.spec.risk.levels(self.state[None])[0])
        return self._obs()

    def step(self, action):
        if self.done:
            raise RuntimeError(
                "Call reset before stepping or after episode completion."
            )
        if int(action) != action or not 0 <= action < self.n_actions:
            raise ValueError("Invalid action index.")
        i = self.index
        value = self.spec.transition(
            self.state.copy(),
            self.series[i].copy(),
            self.series[i + 1].copy(),
            int(action),
            self.spec.step_hours,
        )
        value = np.asarray(value, dtype=float)
        if value.shape != (5,):
            raise ValueError("Transition must return five physical indicator values.")
        validate_values(value[None])
        self.previous, self.state = self.state.copy(), value.copy()
        self.index += 1
        self.steps += 1
        self.done = self.steps == self.max_steps
        level = int(self.spec.risk.levels(self.state[None])[0])
        cost = float(self.spec.costs[action])
        reward = -self.spec.risk_penalties[level] - self.cost_weight * cost
        info = dict(
            level=level,
            total_cost=cost,
            time=self.frame.time.iloc[self.index],
            state=self.state.copy(),
        )
        return self._obs(), float(reward), self.done, info


def run_episode(env, policy, start=None):
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset(start)
    result = dict(
        levels=[],
        actions=[],
        costs=[],
        rewards=[],
        times=[],
        states=[],
        initial_level=env.initial_level,
        step_hours=env.spec.step_hours,
        no_action=env.spec.no_action,
        cost_unit=env.spec.cost_unit,
    )
    done = False
    while not done:
        action = policy(obs, env.state.copy())
        obs, reward, done, info = env.step(action)
        for k, value in [
            ("levels", info["level"]),
            ("actions", action),
            ("costs", info["total_cost"]),
            ("rewards", reward),
            ("times", info["time"]),
            ("states", info["state"]),
        ]:
            result[k].append(value)
    return result


def episode_metrics(roll, threshold=1):

    bad = np.r_[roll["initial_level"], roll["levels"]] >= threshold
    durations, left, right = [], 0, 0
    start = 0 if bad[0] else None
    for i in range(1, len(bad)):
        if bad[i] and not bad[i - 1]:
            start = i
        if not bad[i] and bad[i - 1]:
            if start == 0:
                left += 1
            else:
                durations.append((i - start) * roll["step_hours"])
            start = None
    if bad[-1]:
        right += 1
        left += int(start == 0)
    return dict(
        n_steps=len(roll["levels"]),
        within_band_rate=float(np.mean(~bad[1:])),
        exceedance_pct=float(np.mean(bad[1:]) * 100),
        complete_recovery_mean_h=float(np.mean(durations)) if durations else None,
        complete_recoveries=len(durations),
        left_censored_runs=left,
        right_censored_runs=right,
        intervention_frequency=float(
            np.mean(np.asarray(roll["actions"]) != roll["no_action"])
        ),
        total_cost=float(np.sum(roll["costs"])),
        cost_unit=roll["cost_unit"],
    )


def station_series(per_station, train_end, validation_end):

    a, b = pd.Timestamp(train_end), pd.Timestamp(validation_end)
    if not a < b:
        raise ValueError("train_end must precede validation_end.")
    split = {}
    for label, df in per_station.items():
        t = validate_station(df)
        parts = dict(
            train=df.loc[t < a].reset_index(drop=True),
            validation=df.loc[(t >= a) & (t < b)].reset_index(drop=True),
            test=df.loc[t >= b].reset_index(drop=True),
        )
        if any(len(v) < 2 for v in parts.values()):
            raise ValueError(
                f"{label}: insufficient train/validation/test observations."
            )
        split[label] = parts
    train = np.concatenate([x["train"][IND_COLS].values for x in split.values()])
    sd = train.std(0)
    norm = train.mean(0), np.where(sd > 1e-12, sd, 1.0)
    for parts in split.values():
        parts["norm"] = norm
    return split
