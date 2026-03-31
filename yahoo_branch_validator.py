from __future__ import annotations

import argparse
import csv
import io
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd
import yfinance as yf

from system30.config import System30Config

BRANCH_RE = re.compile(r'^(\d+)d RSI ([A-Z0-9._-]+) LT(\d+) - L ([A-Z0-9._-]+)$')
WARMUP_DAYS = 365


@dataclass(slots=True)
class BranchSpec:
    ticker: str
    branch_id: str
    period: int
    threshold: int


@dataclass(slots=True)
class MetricParams:
    metric_style: str = 'reference_6rolling'
    slippage_bps_per_side: float = 2.0


@dataclass(slots=True)
class BranchMetrics:
    profit_pct: float
    tim_pct: float
    dd_pct_abs: float
    total_trades: int
    realized_trades: int
    unrealized_trades: int


DEFAULT_BRANCHES = [
    '4d RSI SLV LT11 - L SLV',
    '5d RSI SPY LT20 - L SPY',
    '4d RSI VTI LT15 - L VTI',
    '14d RSI XLV LT30 - L XLV',
    '19d RSI XLE LT37 - L XLE',
]


def parse_branch(branch_id: str) -> BranchSpec:
    m = BRANCH_RE.match(branch_id.strip())
    if not m:
        raise ValueError(f'Unsupported branch format: {branch_id}')
    period = int(m.group(1))
    signal_ticker = m.group(2)
    threshold = int(m.group(3))
    invest_ticker = m.group(4)
    if signal_ticker != invest_ticker:
        raise ValueError(f'Cross-ticker branches not yet supported: {branch_id}')
    return BranchSpec(ticker=invest_ticker, branch_id=branch_id.strip(), period=period, threshold=threshold)


def fetch_yahoo_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    warmup_start = (pd.Timestamp(start) - pd.Timedelta(days=WARMUP_DAYS)).strftime('%Y-%m-%d')
    df = yf.download(ticker, start=warmup_start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    expected = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f'Yahoo missing expected columns for {ticker}: {missing}')
    return df[expected].copy()


def load_local_lean_history(ticker: str, daily_dir: Path) -> pd.DataFrame:
    path = daily_dir / f'{ticker.lower()}.zip'
    if not path.exists():
        raise FileNotFoundError(f'Local LEAN zip not found for {ticker}: {path}')
    with ZipFile(path) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode('utf-8')
    rows = list(csv.reader(io.StringIO(text)))
    parsed = []
    for row in rows:
        if len(row) < 6:
            continue
        dt, o, h, l, c, v = row[:6]
        date_str = dt.split()[0]
        parsed.append(
            {
                'Date': pd.to_datetime(date_str, format='%Y%m%d'),
                'Open': float(o) / 10000.0,
                'High': float(h) / 10000.0,
                'Low': float(l) / 10000.0,
                'Close': float(c) / 10000.0,
                'Volume': float(v),
            }
        )
    df = pd.DataFrame(parsed).set_index('Date').sort_index()
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


def compute_rsi_wilder(prices: pd.Series, period: int) -> pd.Series:
    close = np.asarray(prices.to_numpy(dtype=float), dtype=float)
    n = len(close)
    rsi = np.full(n, np.nan, dtype=float)
    if n < period + 1:
        return pd.Series(rsi, index=prices.index)

    delta = np.diff(close)
    gains = np.where(np.isnan(delta), 0.0, np.maximum(delta, 0.0))
    losses = np.where(np.isnan(delta), 0.0, np.maximum(-delta, 0.0))
    alpha = 1.0 / period
    avg_g = float(np.nanmean(gains[:period]))
    avg_l = float(np.nanmean(losses[:period]))

    rsi[period] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)
    for i in range(period, n - 1):
        avg_g = alpha * gains[i] + (1.0 - alpha) * avg_g
        avg_l = alpha * losses[i] + (1.0 - alpha) * avg_l
        rsi[i + 1] = 100.0 if avg_l == 0 else 100.0 - 100.0 / (1.0 + avg_g / avg_l)

    return pd.Series(rsi, index=prices.index)


def simulate_reference_6rolling(price: pd.DataFrame, spec: BranchSpec, params: MetricParams, start: str, end: str) -> BranchMetrics:
    px = price.loc[start:end, ['High', 'Low', 'Close']].copy()
    if px.empty:
        return BranchMetrics(0.0, 0.0, 0.0, 0, 0, 0)

    rsi_full = compute_rsi_wilder(price['Close'], spec.period)
    rsi = rsi_full.loc[start:end]
    exits_full = price['Close'] > price['High'].shift(1)
    exits = exits_full.loc[start:end]
    close_vals = px['Close'].to_numpy(dtype=float)
    low_vals = px['Low'].to_numpy(dtype=float)
    rsi_vals = rsi.to_numpy(dtype=float)
    exits_vals = exits.fillna(False).to_numpy(dtype=bool)

    slip = params.slippage_bps_per_side / 10_000.0
    initial_capital = 1.0
    capital = initial_capital
    capital_at_entry = 0.0
    entry_price = 0.0
    in_trade = False
    active_bars = 0
    realized_trades = 0
    equity_marks = [initial_capital]

    for i in range(1, len(px)):
        if in_trade:
            active_bars += 1
            low_mark = low_vals[i]
            low_pnl = (low_mark - entry_price) / entry_price
            equity_marks.append(capital_at_entry * (1.0 + low_pnl))

            close_mark = close_vals[i]
            close_pnl = (close_mark - entry_price) / entry_price
            equity_marks.append(capital_at_entry * (1.0 + close_pnl))

            if exits_vals[i]:
                exit_price = close_vals[i] * (1.0 - slip)
                pnl = (exit_price - entry_price) / entry_price
                capital *= (1.0 + pnl)
                equity_marks.append(capital)
                realized_trades += 1
                in_trade = False
                capital_at_entry = 0.0
        else:
            r = rsi_vals[i]
            if not np.isnan(r) and r < spec.threshold:
                entry_price = close_vals[i] * (1.0 + slip)
                in_trade = True
                active_bars += 1
                capital_at_entry = capital
                entry_mark_pnl = (close_vals[i] - entry_price) / entry_price
                equity_marks.append(capital_at_entry * (1.0 + entry_mark_pnl))

    eq = np.asarray(equity_marks, dtype=float)
    peak = np.maximum.accumulate(eq)
    drawdown = eq / peak - 1.0
    max_dd_pct = float(abs(np.nanmin(drawdown))) if len(drawdown) else 0.0

    return BranchMetrics(
        profit_pct=float((capital - initial_capital) * 100.0),
        tim_pct=float(active_bars / len(px) * 100.0) if len(px) else 0.0,
        dd_pct_abs=float(max_dd_pct * 100.0),
        total_trades=int(realized_trades),
        realized_trades=int(realized_trades),
        unrealized_trades=int(in_trade),
    )


def compute_metrics_for_source(df: pd.DataFrame, spec: BranchSpec, params: MetricParams, start: str, end: str) -> BranchMetrics:
    if params.metric_style == 'reference_6rolling':
        return simulate_reference_6rolling(df, spec, params, start, end)
    raise ValueError(f'Unknown metric_style: {params.metric_style}')


def compare_branch(spec: BranchSpec, params: MetricParams, start: str, end: str, include_local: bool, local_daily_dir: Path) -> dict:
    yahoo_df = fetch_yahoo_history(spec.ticker, start, end)
    yahoo_metrics = compute_metrics_for_source(yahoo_df, spec, params, start, end)
    result = {
        'ticker': spec.ticker,
        'branch_id': spec.branch_id,
        'start': start,
        'end': end,
        'metric_style': params.metric_style,
        'slippage_bps_per_side': params.slippage_bps_per_side,
        'yahoo': asdict(yahoo_metrics),
    }
    if include_local:
        local_df = load_local_lean_history(spec.ticker, local_daily_dir)
        local_metrics = compute_metrics_for_source(local_df, spec, params, start, end)
        result['local'] = asdict(local_metrics)
        result['delta'] = {
            'profit_pct_points': local_metrics.profit_pct - yahoo_metrics.profit_pct,
            'tim_pct_points': local_metrics.tim_pct - yahoo_metrics.tim_pct,
            'dd_pct_points': local_metrics.dd_pct_abs - yahoo_metrics.dd_pct_abs,
            'total_trades': local_metrics.total_trades - yahoo_metrics.total_trades,
            'realized_trades': local_metrics.realized_trades - yahoo_metrics.realized_trades,
            'unrealized_trades': local_metrics.unrealized_trades - yahoo_metrics.unrealized_trades,
        }
    return result


def main(branches: Iterable[str], params: MetricParams, start: str, end: str, out_json: Path | None, out_csv: Path | None, include_local: bool, local_daily_dir: Path) -> None:
    specs = [parse_branch(b) for b in branches]
    results = [compare_branch(spec, params, start, end, include_local, local_daily_dir) for spec in specs]

    rows = []
    for r in results:
        row = {
            'ticker': r['ticker'],
            'branch_id': r['branch_id'],
            'metric_style': r['metric_style'],
            'slippage_bps_per_side': r['slippage_bps_per_side'],
            'start': r['start'],
            'end': r['end'],
            'yahoo_profit_pct': r['yahoo']['profit_pct'],
            'yahoo_tim_pct': r['yahoo']['tim_pct'],
            'yahoo_dd_pct_abs': r['yahoo']['dd_pct_abs'],
            'yahoo_total_trades': r['yahoo']['total_trades'],
        }
        if include_local and 'local' in r:
            row.update({
                'local_profit_pct': r['local']['profit_pct'],
                'delta_profit_pct_points': r['delta']['profit_pct_points'],
                'local_tim_pct': r['local']['tim_pct'],
                'delta_tim_pct_points': r['delta']['tim_pct_points'],
                'local_dd_pct_abs': r['local']['dd_pct_abs'],
                'delta_dd_pct_points': r['delta']['dd_pct_points'],
                'local_total_trades': r['local']['total_trades'],
                'delta_total_trades': r['delta']['total_trades'],
            })
        rows.append(row)

    frame = pd.DataFrame(rows)
    print(frame.to_string(index=False))

    if out_json:
        out_json.write_text(json.dumps(results, indent=2) + '\n')
        print(f'Wrote {out_json}')
    if out_csv:
        frame.to_csv(out_csv, index=False)
        print(f'Wrote {out_csv}')


if __name__ == '__main__':
    defaults = System30Config()
    parser = argparse.ArgumentParser(description='Standalone Yahoo Finance branch validator for System30-style RSI branches.')
    parser.add_argument('--branch', action='append', help='Branch label. Can be passed multiple times.')
    parser.add_argument('--start', default='2023-01-01')
    parser.add_argument('--end', default='2024-01-01')
    parser.add_argument('--metric-style', default='reference_6rolling', choices=['reference_6rolling'])
    parser.add_argument('--slippage-bps', type=float, default=2.0)
    parser.add_argument('--include-local', action='store_true', help='Also compare against local LEAN-format data if available.')
    parser.add_argument('--local-daily-dir', type=Path, default=defaults.local_daily_dir)
    parser.add_argument('--out-json', type=Path, default=Path('yahoo_branch_validator.json'))
    parser.add_argument('--out-csv', type=Path, default=Path('yahoo_branch_validator.csv'))
    args = parser.parse_args()
    branches = args.branch if args.branch else DEFAULT_BRANCHES
    params = MetricParams(metric_style=args.metric_style, slippage_bps_per_side=args.slippage_bps)
    main(branches, params, args.start, args.end, args.out_json, args.out_csv, args.include_local, Path(args.local_daily_dir))
