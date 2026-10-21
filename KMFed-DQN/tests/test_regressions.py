import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from kmfed_dqn.common import IND_COLS, _clean_one, load_all_stations
from kmfed_dqn.control_env import (
    ControlSpec,
    WaterControlEnv,
    episode_metrics,
    station_series,
)
from kmfed_dqn.cross_basin import federated_ridge, leave_one_basin_out, make_windows
from kmfed_dqn.dqn import (
    NumpyDQN,
    Replay,
    evaluate_agent,
    fedavg_train,
    load_agent,
    save_agent,
    train_centralized_dqn,
    train_dqn,
)
from kmfed_dqn.fed_control import get_paradigm_agents
from kmfed_dqn.fed_forecast import (
    evaluate,
    forecast_latest,
    load_forecaster,
    predict,
    save_forecaster,
    shared_scaler,
    station_data,
    train_federated,
)
from kmfed_dqn.forecasting import (
    build_supervised,
    chrono_split,
    contiguous_segments,
    kalman_filter_df,
    physical_projection,
)
from kmfed_dqn.kalman import kalman_cv, ljung_box
from kmfed_dqn.pareto import analyze, self_check
from kmfed_dqn.policy_library import paired_observations, policy_frontier
from kmfed_dqn.risk import RiskSpec


def frame(n=300, offset=0, start="2020-01-01"):
    t = np.arange(n)
    a = np.column_stack(
        [
            15 + np.sin(t / 7),
            7 + 0.1 * np.sin(t / 9),
            300 + 20 * np.sin(t / 8),
            10 + np.sin(t / 5),
            0.3 + 0.1 * np.sin(t / 10),
        ]
    )
    a[:, 2] += offset
    out = pd.DataFrame(a, columns=IND_COLS)
    out["time"] = pd.date_range(start, periods=n, freq="4h")
    out["station"] = "TEST_ONLY"
    return out


def risk():

    sensors = {
        c: dict(
            normal_range=[None, 1000],
            warning_threshold=[None, 2000],
            critical_threshold=[None, 3000],
        )
        for c in IND_COLS
    }
    return RiskSpec(sensors, "TEST FIXTURE ONLY: arbitrary bands for branch coverage")


def transition(state, observed_now, observed_next, action, dt_hours):
    return observed_next.copy()


def control_spec():
    return ControlSpec(
        ("noop", "test_action"),
        0,
        (0.0, 1.0),
        "test units",
        (0.0, 1.0, 2.0, 3.0),
        4.0,
        "TEST ONLY: identity response, not an intervention model",
        transition,
        risk(),
    )


def env(label="x", start="2020-01-01", n=40):
    return WaterControlEnv(
        frame(n, start=start),
        (np.zeros(5), np.ones(5)),
        control_spec(),
        max_steps=4,
        seed=2,
        label=label,
    )


class DataTests(unittest.TestCase):
    def test_no_fill_or_duplicate_selection(self):
        for broken in [
            pd.concat([frame(4), frame(4).iloc[:1]]),
            frame(4).assign(nh4_n=np.nan),
        ]:
            with self.assertRaises(ValueError):
                _clean_one(broken, "x")
        f = frame(5)
        before = f.copy(deep=True)
        _clean_one(f, "x")
        pd.testing.assert_frame_equal(f, before)

    def test_external_file_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.csv"
            frame(30).to_csv(path, index=False)
            before = path.read_bytes()
            load_all_stations(folder)
            self.assertEqual(before, path.read_bytes())
        with self.assertRaises(ValueError):
            load_all_stations()

    def test_exact_grid_and_minimum_window(self):
        f = frame(5)
        f["time"] = pd.date_range("2020", periods=5, freq="270min")
        self.assertEqual([len(s) for s in contiguous_segments(f)], [1] * 5)
        self.assertEqual(
            len(build_supervised(frame(3), lookback=2, horizons=[1])[0]), 1
        )
        self.assertEqual(
            build_supervised(frame(2), lookback=2, horizons=[1])[0].shape, (0, 14)
        )

    def test_timestamp_alignment(self):
        a, b = frame(30), frame(30)
        b["time"] += pd.Timedelta(minutes=1)
        with self.assertRaises(ValueError):
            build_supervised(a, b)

    def test_risk_missing_is_error_and_low_pollution_is_safe(self):
        self.assertEqual(risk().levels([[15, 7, 0, 0, 0]]).tolist(), [0])
        with self.assertRaises(ValueError):
            risk().levels([[15, 7, 100, 10, np.nan]])

    def test_calendar_purge(self):
        ts = pd.date_range("2020", periods=100, freq="4h")
        tr, va, te = chrono_split(ts, ts[40], ts[75])
        self.assertTrue((ts[tr] + pd.Timedelta(hours=48) < ts[40]).all())
        self.assertTrue(
            ((ts[va] >= ts[40]) & (ts[va] + pd.Timedelta(hours=48) < ts[75])).all()
        )
        self.assertTrue((ts[te] >= ts[75]).all())

    def test_filter_causal_and_resets_at_gaps(self):
        a = frame(40)
        full = kalman_filter_df(a)
        short = kalman_filter_df(a.iloc[:20])
        np.testing.assert_allclose(full[IND_COLS].values[:20], short[IND_COLS])
        a.loc[20:, "time"] += pd.Timedelta(days=10)
        f = kalman_filter_df(a)
        np.testing.assert_allclose(f[IND_COLS].iloc[20], a[IND_COLS].iloc[20])
        self.assertEqual(f.segment.nunique(), 2)
        self.assertTrue(
            all(c + "_unconstrained" in f and c + "_projected" in f for c in IND_COLS)
        )

    def test_kalman_and_short_ljung_box(self):
        for a in ([], [1, np.nan]):
            with self.assertRaises(ValueError):
                kalman_cv(a, 0.01, 1)
        with self.assertRaises(ValueError):
            ljung_box([1, 2, 3], lags=3)
        q, p = ljung_box(np.sin(np.arange(50)), lags=10)
        self.assertTrue(np.isfinite([q, p]).all())


class ForecastTests(unittest.TestCase):
    def clients(self):
        return {
            str(i): station_data(
                str(i), frame(offset=i * 10), "2020-01-18", "2020-02-06"
            )
            for i in range(2)
        }

    def test_test_labels_cannot_change_fitted_weights_or_history(self):
        clients = self.clients()
        scaler = shared_scaler(clients)
        changed = copy.deepcopy(clients)
        for c in changed.values():
            c["Dall"][c["te"]] += 100
            for y in c["Yabs"].values():
                y[c["te"], 2:] += 100
        w1, h1 = train_federated(
            clients, scaler, rounds=2, local_epochs=1, mode="quality"
        )
        w2, h2 = train_federated(
            changed, scaler, rounds=2, local_epochs=1, mode="quality"
        )
        for a, b in zip(w1, w2):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(h1, h2)
        self.assertNotEqual(
            evaluate(*w1, clients["0"], scaler), evaluate(*w2, changed["0"], scaler)
        )

    def test_calendar_boundaries_must_match(self):
        c = self.clients()
        c["0"]["train_end"] = "1999-01-01"
        with self.assertRaises(ValueError):
            train_federated(c, shared_scaler(c), rounds=1)

    def test_scaler_matches_pooled_train(self):
        c = self.clients()
        sc = shared_scaler(c)
        x = np.concatenate([v["feat"][v["tr"]] for v in c.values()])
        np.testing.assert_allclose(sc["fmean"], x.mean(0), atol=1e-12)
        expected = np.where(x.std(0) > 1e-12, x.std(0), 1)
        np.testing.assert_allclose(sc["fstd"], expected, rtol=1e-9, atol=1e-12)

    def test_projection_is_explicit(self):
        raw = np.array([[15, 20, -1, -2, -3.0]])
        projected, flags = physical_projection(raw)
        np.testing.assert_array_equal(raw, [[15, 20, -1, -2, -3]])
        self.assertEqual(flags.sum(), 4)
        self.assertTrue((projected[0, 2:] == 0).all())

    def test_bundle_inference_matches_batch_prediction(self):
        clients = self.clients()
        sc = shared_scaler(clients)
        weights, _ = train_federated(clients, sc, rounds=1, local_epochs=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "m.npz"
            save_forecaster(path, weights, sc, clients, {"purpose": "test"})
            got, scales, meta = load_forecaster(path)
            np.testing.assert_array_equal(got[0], weights[0])
            self.assertIn("checksum", meta["clients"]["0"])
            client = clients["0"]
            origin = client["times"][client["te"][-1]]
            history = frame().loc[lambda x: x.time <= origin]
            table = forecast_latest(path, history)
            batch = predict(*weights, client, sc)
            for h, (raw, pred, flags) in batch.items():
                rows = table[table.horizon_hours == h * 4]
                np.testing.assert_allclose(rows.prediction_raw, raw[-1], atol=1e-10)
                np.testing.assert_allclose(rows.prediction, pred[-1], atol=1e-10)
            self.assertEqual(
                sorted(table.horizon_hours.unique()), [4, 8, 12, 16, 20, 24, 48]
            )

    def test_unverified_dp_rejected(self):
        clients = self.clients()
        with self.assertRaises(ValueError):
            train_federated(clients, shared_scaler(clients), mode="dp")


class ControlTests(unittest.TestCase):
    def test_dimensions_and_exploration(self):
        a = NumpyDQN(3, 2, hidden=(4, 4))
        actions = [a.act(np.zeros(3), 1) for _ in range(200)]
        self.assertEqual(set(actions), {0, 1})
        b = Replay(3, cap=4)
        b.push([1, 2, 3], 1, 0, [2, 3, 4], False)
        self.assertEqual(b.sample(1)[0].shape, (1, 3))

    def test_lr_and_replay_persist(self):
        e = env()
        a = NumpyDQN(e.obs_dim, e.n_actions, lr=0.1)
        a, _ = train_dqn(
            e, episodes=1, batch=2, warmup=2, agent=a, lr=0.0005, log=False
        )
        self.assertEqual(a.lr, 0.0005)
        self.assertEqual(a._training.steps, 4)
        a, _ = train_dqn(e, episodes=1, batch=2, warmup=2, agent=a, log=False)
        self.assertEqual(len(a._training.replay), 8)
        self.assertEqual(a._training.steps, 8)
        self.assertGreater(a.adam_t, 0)
        a.set_weights(a.get_weights())
        self.assertEqual(a.adam_t, 0)
        self.assertTrue(all((x == 0).all() for x in a.mW))

    def test_centralized_shared_replay(self):
        a = train_centralized_dqn(
            {"x": env(), "y": env("y")},
            cycles=2,
            local_episodes=1,
            batch=2,
            warmup=2,
            log=False,
        )
        self.assertEqual(len(a._training.replay), 16)
        self.assertEqual(a._training.steps, 16)

    def test_federated_evaluates_aggregated_agent(self):
        train = {"x": env(), "y": env("y")}
        val = {"x": env(start="2021"), "y": env("y", start="2021")}
        a, history = fedavg_train(
            train,
            validation_envs=val,
            rounds=2,
            local_episodes=1,
            batch=2,
            warmup=2,
            lr=0.0005,
            log=False,
        )
        self.assertEqual(a.lr, 0.0005)
        self.assertEqual(history[-1]["global_validation"], evaluate_agent(a, val))
        self.assertEqual(history[-1]["client_train_steps"], [4, 4])

    def test_validation_overlap_rejected(self):
        with self.assertRaises(ValueError):
            train_dqn(env(), episodes=1, validation_env=env(), log=False)

    def test_control_rejects_gap_and_preserves_input(self):
        f = frame(20)
        f.loc[10:, "time"] += pd.Timedelta(days=1)
        before = f.copy(deep=True)
        e = WaterControlEnv(f, (np.zeros(5), np.ones(5)), control_spec(), max_steps=4)
        with self.assertRaises(ValueError):
            e.reset(8)
        e.reset(10)
        e.step(0)
        pd.testing.assert_frame_equal(f, before)
        with self.assertRaises(ValueError):
            e.step(2)

    def test_censored_recovery(self):
        base = dict(
            initial_level=1,
            levels=[1, 0, 1, 1, 0, 1],
            actions=[0] * 6,
            costs=[0] * 6,
            no_action=0,
            step_hours=4,
            cost_unit="test",
        )
        m = episode_metrics(base)
        self.assertEqual(
            (
                m["complete_recoveries"],
                m["left_censored_runs"],
                m["right_censored_runs"],
            ),
            (1, 1, 1),
        )
        self.assertEqual(m["complete_recovery_mean_h"], 8)
        base.update(initial_level=0, levels=[0] * 6)
        self.assertIsNone(episode_metrics(base)["complete_recovery_mean_h"])

    def test_matched_paradigm_budget(self):
        models, logs = get_paradigm_agents(
            {"x": env(), "y": env("y")},
            rounds=2,
            local_episodes=1,
            batch=2,
            warmup=2,
            log=False,
        )
        self.assertEqual(
            [sum(r["train_steps"] for r in log) for log in logs["local"].values()],
            [8, 8],
        )
        self.assertEqual(models["centralized"]._training.steps, 16)
        self.assertEqual(
            np.sum([r["client_train_steps"] for r in logs["federated"]]), 16
        )

    def test_shared_train_only_normalization(self):
        f = frame()
        parts = station_series(
            {"x": f, "y": frame(offset=20)}, "2020-01-18", "2020-02-06"
        )
        np.testing.assert_array_equal(parts["x"]["norm"], parts["y"]["norm"])
        pool = np.concatenate([v["train"][IND_COLS] for v in parts.values()])
        np.testing.assert_allclose(parts["x"]["norm"][0], pool.mean(0))

    def test_checkpoint_round_trip(self):
        a = NumpyDQN(3, 2, hidden=(4, 5), lr=0.0005)
        metadata = dict(
            normalization=[[0] * 3, [1] * 3],
            control_spec={"source": "test"},
            input_fingerprints={"x": "test"},
            training={"test": True},
            transition_sha256="test",
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "a.npz"
            save_agent(path, a, metadata)
            b, meta = load_agent(path)
            np.testing.assert_array_equal(a.q([1, 2, 3]), b.q([1, 2, 3]))
            self.assertEqual(b.lr, 0.0005)
            self.assertEqual(meta["checkpoint_type"], "inference")


class CrossBasinTests(unittest.TestCase):
    def monthly(self):
        return pd.DataFrame(
            [
                dict(
                    basin=b,
                    station=b,
                    month=str(m),
                    param=p,
                    value=(0.3 + i * 0.001 if p == "NH4N" else 7 + 0.01 * np.sin(i)),
                    unit="mg N/L" if p == "NH4N" else "1",
                )
                for b in ["a", "b"]
                for p in ["NH4N", "pH"]
                for i, m in enumerate(pd.period_range("2015-01", periods=60, freq="M"))
            ]
        )

    def test_duplicates_and_units_rejected(self):
        a = self.monthly()
        with self.assertRaises(ValueError):
            make_windows(pd.concat([a, a.iloc[:1]]))
        with self.assertRaises(ValueError):
            paired_observations(pd.concat([a, a.iloc[:1]]))
        a.loc[0, "unit"] = "wrong"
        with self.assertRaises(ValueError):
            make_windows(a)

    def test_physical_scale_and_dates(self):
        w = make_windows(self.monthly(), horizons=[1])
        r = leave_one_basin_out(w, min_train=10, min_clients=1)
        self.assertGreater(len(r), 0)
        sub = r[r.param == "NH4N"]
        np.testing.assert_allclose(sub.observed, np.expm1(sub.observed_model))
        self.assertTrue(
            {
                "prediction_raw",
                "projected",
                "unit",
                "origin_month",
                "target_month_label",
            }
            <= set(r)
        )
        self.assertTrue(
            (
                pd.to_datetime(r.training_target_before)
                <= pd.to_datetime(r.origin_month)
            ).all()
        )

    def test_ridge_matches_pooled_solution(self):
        rng = np.random.default_rng(1)
        x = np.c_[np.ones(30), rng.normal(size=(30, 3))]
        y = rng.normal(size=30)
        reg = np.eye(4) * 0.01 * len(x)
        reg[0, 0] = 1e-10
        expected = np.linalg.solve(x.T @ x + reg, x.T @ y)
        np.testing.assert_allclose(
            federated_ridge(x, y, ["a"] * 15 + ["b"] * 15, ridge=0.01), expected
        )

    def test_pareto_ties_numeric_strings_and_provenance(self):
        self.assertEqual(self_check(), "PASS")
        f = pd.DataFrame(
            dict(
                basin=["a"] * 3,
                policy_id=["a", "b", "c"],
                family=["x"] * 3,
                cost=["1", "1", "2"],
                risk=["2", "2", "3"],
            )
        )
        self.assertEqual(analyze(f)[1].policy_id.tolist(), ["a", "b"])
        with self.assertRaises(ValueError):
            policy_frontier(f)
        f = f.assign(
            evaluation_source="TEST ONLY",
            cost_unit="u",
            risk_definition="test",
            aggregation="test",
            time_step="4h",
        )
        self.assertEqual(len(policy_frontier(f)[1]), 2)
        f.loc[0, "time_step"] = "1 month"
        with self.assertRaises(ValueError):
            policy_frontier(f)


if __name__ == "__main__":
    unittest.main()
