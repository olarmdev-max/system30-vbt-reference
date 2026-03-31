from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from system30.config import System30Config
from system30.data import load_local_ticker_history
from system30.simulation import build_branch_state, compute_branch_metrics

BRANCH_RE = re.compile(r"^(\d+)d RSI ([A-Z0-9._-]+) LT(\d+) - L ([A-Z0-9._-]+)$")
DEFAULT_OUT_CSV = Path("system30_output/branch_compare.csv")


def parse_branch(branch_id: str) -> tuple[str, int, int]:
    m = BRANCH_RE.match(branch_id.strip())
    if not m:
        raise ValueError(f"Unsupported branch format: {branch_id}")
    period = int(m.group(1))
    signal_ticker = m.group(2)
    threshold = int(m.group(3))
    invest_ticker = m.group(4)
    if signal_ticker != invest_ticker:
        raise ValueError(f"Cross-ticker branch unsupported: {branch_id}")
    return invest_ticker, period, threshold


def _detect_header_skiprows(sheet_csv: Path) -> int:
    lines = sheet_csv.read_text(encoding='utf-8', errors='ignore').splitlines()
    candidates = lines[:3]
    for idx, line in enumerate(candidates):
        if re.search(r'(^|,)Ticker( \(\d+\))?(,|$)', line) and re.search(r'(^|,)(Condition|Branch to Use)( \(\d+\))?(,|$)', line):
            return idx
    return 1


def load_sheet(sheet_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(sheet_csv, skiprows=_detect_header_skiprows(sheet_csv), engine='python')
    df.columns = [str(c).strip() for c in df.columns]

    renamed: dict[str, str] = {
        'Ticker': 'ticker',
        'Branch to Use': 'sheet_branch',
        'Condition': 'sheet_branch',
        'IS Profit %': 'sheet_is_profit_pct',
        'IS Time in market%': 'sheet_is_tim_pct',
        'IS Max DD%': 'sheet_is_dd_pct',
        'IS Total Trades': 'sheet_is_total_trades',
    }
    for col in list(df.columns):
        if re.fullmatch(r'Ticker \(\d+\)', col):
            renamed[col] = 'ticker'
        elif re.fullmatch(r'Condition \(\d+\)', col):
            renamed[col] = 'sheet_branch'
    df = df.rename(columns=renamed)

    required = {'ticker', 'sheet_branch', 'sheet_is_profit_pct', 'sheet_is_tim_pct', 'sheet_is_dd_pct', 'sheet_is_total_trades'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'Unsupported sheet schema in {sheet_csv}: missing {sorted(missing)}; columns={list(df.columns)}')

    for col in ['ticker', 'sheet_branch']:
        df[col] = df[col].astype(str).str.strip()
    df = df[(df['ticker'] != '') & (df['sheet_branch'] != '')].copy()
    for col in ['sheet_is_profit_pct', 'sheet_is_tim_pct', 'sheet_is_dd_pct', 'sheet_is_total_trades']:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace('%', '', regex=False).str.replace(' ', '', regex=False),
            errors='coerce',
        )
    df['sheet_is_dd_pct_abs'] = df['sheet_is_dd_pct'].abs()
    return df[['ticker', 'sheet_branch', 'sheet_is_profit_pct', 'sheet_is_tim_pct', 'sheet_is_dd_pct_abs', 'sheet_is_total_trades']]


def compare_branches(branches: list[str], sheet_df: pd.DataFrame, start: str, end: str, daily_dir: Path) -> pd.DataFrame:
    cfg = System30Config(local_daily_dir=daily_dir)
    rows: list[dict] = []
    for branch_id in branches:
        ticker, period, threshold = parse_branch(branch_id)
        df = load_local_ticker_history(ticker, cfg.local_daily_dir)
        if df.empty:
            rows.append({'ticker': ticker, 'branch_id': branch_id, 'error': 'missing_local_history'})
            continue
        state = build_branch_state(
            df=df,
            start=start,
            end=end,
            rsi_period=period,
            rsi_threshold=threshold,
            rsi_wtype='wilder',
        )
        metrics = compute_branch_metrics(state)
        sheet_row = sheet_df[(sheet_df['ticker'] == ticker) & (sheet_df['sheet_branch'] == branch_id)]
        sheet = sheet_row.iloc[0].to_dict() if not sheet_row.empty else {}
        rows.append(
            {
                'ticker': ticker,
                'branch_id': branch_id,
                'vbt_profit_pct': metrics['profit_pct'] * 100.0,
                'vbt_tim_pct': metrics['time_in_market_fraction'] * 100.0,
                'vbt_dd_pct_abs': metrics['max_drawdown_pct'] * 100.0,
                'vbt_total_trades': metrics['total_trades'],
                'sheet_is_profit_pct': sheet.get('sheet_is_profit_pct'),
                'sheet_is_tim_pct': sheet.get('sheet_is_tim_pct'),
                'sheet_is_dd_pct_abs': sheet.get('sheet_is_dd_pct_abs'),
                'sheet_is_total_trades': sheet.get('sheet_is_total_trades'),
            }
        )
    out = pd.DataFrame(rows)
    for metric in ['profit_pct', 'tim_pct', 'dd_pct_abs', 'total_trades']:
        out[f'delta_{metric}'] = out[f'vbt_{metric}'] - out[f'sheet_is_{metric}']
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare exact sheet branches using canonical System30 metric logic.')
    parser.add_argument('--sheet-csv', type=Path, required=True, help='Workbook branch-list CSV to compare against')
    parser.add_argument('--daily-dir', type=Path, required=True, help='Directory containing LEAN-format daily ZIP files')
    parser.add_argument('--start', required=True, help='IS window start date, e.g. 2023-01-01')
    parser.add_argument('--end', required=True, help='IS window end date, e.g. 2023-12-31')
    parser.add_argument('--branch', action='append', help='Branch label. Can be passed multiple times.')
    parser.add_argument('--out-csv', type=Path, default=DEFAULT_OUT_CSV)
    args = parser.parse_args()

    sheet_df = load_sheet(args.sheet_csv)
    branches = args.branch if args.branch else sheet_df['sheet_branch'].dropna().astype(str).str.strip().tolist()
    out = compare_branches(branches, sheet_df=sheet_df, start=args.start, end=args.end, daily_dir=args.daily_dir)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    print(f'\nWrote {args.out_csv}')


if __name__ == '__main__':
    main()
