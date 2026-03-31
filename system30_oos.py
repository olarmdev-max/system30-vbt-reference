from __future__ import annotations

import argparse
from pathlib import Path

from system30.config import System30Config
from system30.engine import System30Engine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run only the System30 OOS execution pass.")
    parser.add_argument("--daily-dir", type=Path, default=None)
    parser.add_argument("--universe-csv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--first-oos-year", type=int, default=2020)
    parser.add_argument("--last-oos-year", type=int, default=2025)
    parser.add_argument("--max-tickers", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    defaults = System30Config()
    cfg = System30Config(
        data_source='tiingo_local',
        local_daily_dir=args.daily_dir if args.daily_dir is not None else defaults.local_daily_dir,
        universe_csv=args.universe_csv if args.universe_csv is not None else defaults.universe_csv,
        output_dir=args.output_dir if args.output_dir is not None else defaults.output_dir,
        max_tickers=args.max_tickers,
        first_oos_year=args.first_oos_year,
        last_oos_year=args.last_oos_year,
    )
    engine = System30Engine(cfg)
    print(f'Progress log: {engine.progress_logger.path}')
    universe = engine.load_and_filter_universe()
    outputs = engine.run_oos_execution(universe=universe)
    print('OOS System30 outputs:')
    for key, path in outputs.items():
        print(f'  {key}: {path}')


if __name__ == '__main__':
    main()
