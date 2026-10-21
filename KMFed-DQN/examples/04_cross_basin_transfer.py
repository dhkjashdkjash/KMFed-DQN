import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmfed_dqn.cross_basin import leave_one_basin_out, make_windows


def main():
    p = argparse.ArgumentParser(description="Evaluate monthly forecasts on held-out basins.")
    p.add_argument("monthly_csv")
    p.add_argument("--output-dir", default=str(ROOT / "outputs"))
    a = p.parse_args()
    monthly = pd.read_csv(a.monthly_csv)
    windows = make_windows(monthly)
    predictions = leave_one_basin_out(windows)
    if predictions.empty:
        p.error("No eligible folds; check source-basin counts and training history.")
    out = Path(a.output_dir)
    destination = out / "cross_basin_predictions.csv"
    if destination.resolve() == Path(a.monthly_csv).resolve():
        p.error("Output must not overwrite the input table.")
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    print(
        f"{len(windows)} windows; {len(predictions)} prediction rows with units and projection flags"
    )


if __name__ == "__main__":
    main()
