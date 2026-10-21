import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmfed_dqn.fed_forecast import (
    evaluate,
    load_clients,
    prediction_table,
    save_forecaster,
    shared_scaler,
    train_federated,
)
from kmfed_dqn.risk import RiskSpec


def main():
    p = argparse.ArgumentParser(description="Train and evaluate a federated water-quality forecaster.")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--train-end", required=True)
    p.add_argument("--validation-end", required=True)
    p.add_argument("--risk-config")
    p.add_argument("--rounds", type=int, default=50)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mode", choices=["fedavg", "quality"], default="fedavg")
    p.add_argument("--output-dir", default=str(ROOT / "outputs"))
    a = p.parse_args()
    risk = RiskSpec.load(a.risk_config) if a.risk_config else None
    clients = load_clients(a.data_dir, a.train_end, a.validation_end, risk_spec=risk)
    scaler = shared_scaler(clients)
    training = dict(
        rounds=a.rounds, local_epochs=a.local_epochs, mode=a.mode, seed=a.seed
    )
    weights, history = train_federated(clients, scaler, **training)
    results = {lb: evaluate(*weights, c, scaler) for lb, c in clients.items()}
    out = Path(a.output_dir)
    if out.resolve() == Path(a.data_dir).resolve():
        p.error("Output directory must differ from the observation directory.")
    out.mkdir(parents=True, exist_ok=True)
    save_forecaster(
        out / f"forecaster_{a.mode}.npz", weights, scaler, clients, training
    )
    pd.concat([prediction_table(*weights, c, scaler) for c in clients.values()]).to_csv(
        out / f"predictions_{a.mode}.csv", index=False
    )
    (out / f"evaluation_{a.mode}.json").write_text(
        json.dumps(
            dict(validation_history=history, final_test=results),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
