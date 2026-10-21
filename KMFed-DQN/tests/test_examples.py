import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from test_regressions import frame

ROOT = Path(__file__).resolve().parents[1]


class ExampleTests(unittest.TestCase):
    def run_example(self, name, *args, expected=0):
        run = subprocess.run(
            [sys.executable, str(ROOT / "examples" / name), *map(str, args)],
            cwd=ROOT,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(run.returncode, expected, run.stdout + "\n" + run.stderr)
        return run

    def test_all_help_and_missing_input(self):
        for path in sorted((ROOT / "examples").glob("*.py")):
            self.run_example(path.name, "--help")
            self.run_example(path.name, expected=2)

    def test_external_data_workflows_and_outputs(self):
        with tempfile.TemporaryDirectory(prefix="kmfed-test-only-") as folder:
            temp = Path(folder)
            data, out = temp / "external", temp / "derived"
            data.mkdir()
            for i in range(2):
                frame(offset=i * 10).to_csv(data / f"TEST_ONLY_{i}.csv", index=False)
            checksums = {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in data.iterdir()
            }
            self.run_example(
                "01_kalman_state_estimation.py", "--data-dir", data, "--output-dir", out
            )
            filtered = pd.read_csv(out / "filtered_TEST_ONLY_0.csv")
            self.assertEqual(len(filtered), 300)
            self.assertIn("nh4_n_projected", filtered)
            common = [
                "--data-dir",
                data,
                "--output-dir",
                out,
                "--train-end",
                "2020-01-18",
                "--validation-end",
                "2020-02-06",
            ]
            for mode in ["fedavg", "quality"]:
                self.run_example(
                    "02_federated_forecasting.py",
                    *common,
                    "--mode",
                    mode,
                    "--rounds",
                    "2",
                    "--local-epochs",
                    "1",
                )
                predictions = pd.read_csv(out / f"predictions_{mode}.csv")
                self.assertTrue(
                    np.isfinite(
                        predictions[["observed", "prediction_raw", "prediction"]]
                    )
                    .all()
                    .all()
                )
                self.assertEqual(
                    set(predictions.horizon_hours), {4, 8, 12, 16, 20, 24, 48}
                )
                self.assertTrue(
                    (
                        pd.to_datetime(predictions.target_time)
                        > pd.to_datetime(predictions.origin)
                    ).all()
                )
                report = json.loads((out / f"evaluation_{mode}.json").read_text())
                self.assertEqual(len(report["validation_history"]), 2)
                self.assertTrue(
                    all(v["macro_f1"] is None for v in report["final_test"].values())
                )
            module = temp / "test_only_identity_response.py"
            module.write_text("""from kmfed_dqn.control_env import ControlSpec
from kmfed_dqn.risk import RiskSpec
from kmfed_dqn.common import IND_COLS

def transition(state, observed_now, observed_next, action, dt_hours):
    return observed_next.copy()

def make_control_spec():
    bands={c:{'normal_range':[None,1000], 'warning_threshold':[None,2000], 'critical_threshold':[None,3000]} for c in IND_COLS}
    risk=RiskSpec(bands,'TEST ONLY - arbitrary bands')
    return ControlSpec(('noop','test_action'),0,(0.,1.),'test units',(0.,1.,2.,3.),4.,'TEST ONLY - identity transition',transition,risk)
""")
            self.run_example(
                "03_train_control_agents.py",
                *common,
                "--control-module",
                module,
                "--rounds",
                "2",
                "--local-episodes",
                "2",
                "--max-steps",
                "8",
            )
            report = json.loads((out / "control_evaluation.json").read_text())
            self.assertEqual(len(report["validation_history"]), 2)
            self.assertEqual(
                report["validation_history"][-1]["client_train_steps"], [16, 16]
            )
            self.assertTrue((out / "federated_dqn.npz").is_file())
            for p in data.iterdir():
                self.assertEqual(
                    hashlib.sha256(p.read_bytes()).hexdigest(), checksums[p.name]
                )

    def test_monthly_and_policy_exports(self):
        with tempfile.TemporaryDirectory(prefix="kmfed-test-only-") as folder:
            temp = Path(folder)
            monthly = pd.DataFrame(
                [
                    dict(
                        basin=b,
                        station=b,
                        month=str(m),
                        param=p,
                        value=(
                            0.3 + i * 0.001 if p == "NH4N" else 7 + 0.01 * np.sin(i)
                        ),
                        unit="mg N/L" if p == "NH4N" else "1",
                    )
                    for b in ["a", "b", "c", "d"]
                    for p in ["NH4N", "pH"]
                    for i, m in enumerate(
                        pd.period_range("2010-01", periods=120, freq="M")
                    )
                ]
            )
            path = temp / "TEST_ONLY_monthly.csv"
            monthly.to_csv(path, index=False)
            before = path.read_bytes()
            self.run_example(
                "04_cross_basin_transfer.py", path, "--output-dir", temp / "out"
            )
            predictions = pd.read_csv(temp / "out/cross_basin_predictions.csv")
            self.assertFalse(predictions.empty)
            self.assertTrue(
                np.isfinite(predictions[["observed", "prediction_raw", "prediction"]])
                .all()
                .all()
            )
            self.assertTrue(
                {"unit", "target_month_label", "transform", "projected"}
                <= set(predictions)
            )
            self.assertEqual(path.read_bytes(), before)
            policies = pd.DataFrame(
                dict(
                    basin=["test"] * 3,
                    policy_id=["a", "b", "c"],
                    family=["test"] * 3,
                    cost=[1, 2, 3],
                    risk=[3, 2, 4],
                    evaluation_source=["TEST ONLY"] * 3,
                    cost_unit=["test units"] * 3,
                    risk_definition=["test"] * 3,
                    aggregation=["test"] * 3,
                    time_step=["4h"] * 3,
                )
            )
            path = temp / "TEST_ONLY_policies.csv"
            policies.to_csv(path, index=False)
            self.run_example(
                "05_policy_frontier.py", path, "--output-dir", temp / "out"
            )
            front = pd.read_csv(temp / "out/policy_front.csv")
            self.assertEqual(set(front.policy_id), {"a", "b"})
            self.assertIn("evaluation_source", front)


if __name__ == "__main__":
    unittest.main()
