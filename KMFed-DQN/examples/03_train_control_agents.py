import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmfed_dqn.control_train import get_federated_agent, make_environments
from kmfed_dqn.dqn import evaluate_agent, save_agent


def main():
    p = argparse.ArgumentParser(description="Train control agents using an external response model.")
    p.add_argument("--data-dir", required=True)
    p.add_argument(
        "--control-module",
        required=True,
        help="Trusted local Python file with the documented transition model and parameters",
    )
    p.add_argument("--train-end", required=True)
    p.add_argument("--validation-end", required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=str(ROOT / "outputs"))
    a = p.parse_args()
    module_path = Path(a.control_module)
    module_spec = importlib.util.spec_from_file_location(
        "external_control_spec", module_path
    )
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    spec = module.make_control_spec()
    envs = make_environments(
        a.data_dir,
        a.train_end,
        a.validation_end,
        spec,
        max_steps=a.max_steps,
        seed=a.seed,
    )
    training = dict(
        rounds=a.rounds, local_episodes=a.local_episodes, seed=a.seed, lr=5e-4
    )
    agent, history = get_federated_agent(envs["train"], envs["validation"], **training)
    result = evaluate_agent(agent, envs["test"])
    out = Path(a.output_dir)
    if out.resolve() == Path(a.data_dir).resolve():
        p.error("Output directory must differ from the observation directory.")
    out.mkdir(parents=True, exist_ok=True)
    first = next(iter(envs["train"].values()))
    metadata = dict(
        normalization=[v.tolist() for v in first.norm],
        control_spec=spec.metadata(),
        transition_sha256=hashlib.sha256(module_path.read_bytes()).hexdigest(),
        training=training,
        max_steps=a.max_steps,
        train_end=a.train_end,
        validation_end=a.validation_end,
        input_fingerprints={
            part: {
                lb: hashlib.sha256(
                    pd.util.hash_pandas_object(e.frame, index=False).values.tobytes()
                ).hexdigest()
                for lb, e in group.items()
            }
            for part, group in envs.items()
        },
    )
    save_agent(out / "federated_dqn.npz", agent, metadata)
    (out / "control_evaluation.json").write_text(
        json.dumps(
            dict(validation_history=history, final_test=result),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved inference checkpoint and separate validation/test evaluation.")


if __name__ == "__main__":
    main()
