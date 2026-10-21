import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmfed_dqn.policy_library import policy_frontier


def main():
    p = argparse.ArgumentParser(description="Calculate Pareto fronts from externally evaluated policies.")
    p.add_argument("evaluations_csv")
    p.add_argument("--output-dir", default=str(ROOT / "outputs"))
    a = p.parse_args()
    tables = policy_frontier(pd.read_csv(a.evaluations_csv))
    out = Path(a.output_dir)
    names = ("policy_points.csv", "policy_front.csv", "representative_policies.csv")
    if any(
        (out / name).resolve() == Path(a.evaluations_csv).resolve() for name in names
    ):
        p.error("Output must not overwrite the input table.")
    out.mkdir(parents=True, exist_ok=True)
    for name, table in zip(names, tables):
        table.to_csv(out / name, index=False)
    print(f"{len(tables[0])} evaluated policies; {len(tables[1])} nondominated points")


if __name__ == "__main__":
    main()
