import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kmfed_dqn.common import load_all_stations
from kmfed_dqn.forecasting import kalman_filter_df


def main():
    parser = argparse.ArgumentParser(description="Filter external station observations with a Kalman filter.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs"))
    parser.add_argument("--q-over-r", type=float, default=0.01)
    args = parser.parse_args()
    stations, _ = load_all_stations(args.data_dir)
    out = Path(args.output_dir)
    if out.resolve() == Path(args.data_dir).resolve():
        parser.error("Output directory must differ from the observation directory.")
    out.mkdir(parents=True, exist_ok=True)
    for label, df in stations.items():
        filtered = kalman_filter_df(df, args.q_over_r)
        filtered.to_csv(out / f"filtered_{label}.csv", index=False)
        print(f"{label}: {len(filtered)} rows; gaps reset filter state")


if __name__ == "__main__":
    main()
