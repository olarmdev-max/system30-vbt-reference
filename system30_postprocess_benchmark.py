from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

from system30.config import System30Config
from system30.data import load_local_ticker_history
from system30.simulation import (
    INITIAL_EQUITY,
    SLIPPAGE_FRACTION_PER_SIDE,
    build_branch_state,
    compute_branch_metrics,
    compute_fast_metrics_from_signals,
)


DEFAULT_OUT_DIR = Path("system30_output/benchmarks")
DEFAULT_START = "2023-01-01"
DEFAULT_END = "2023-12-31"
DEFAULT_WARMUPS = 3
DEFAULT_RUNS = 25

GOLDEN_BRANCHES: tuple[tuple[str, int, int], ...] = (
    ("AGG", 3, 11),
    ("BND", 3, 10),
    ("DIA", 6, 21),
    ("EEM", 14, 34),
    ("EFA", 8, 26),
    ("GLD", 21, 46),
    ("HYG", 18, 38),
    ("IEF", 3, 10),
    ("IVV", 5, 20),
    ("LQD", 4, 13),
    ("QQQ", 7, 29),
    ("SLV", 4, 11),
    ("SPY", 5, 20),
    ("TLT", 3, 10),
    ("VOO", 5, 20),
    ("VTI", 4, 15),
    ("XLE", 19, 37),
    ("XLV", 14, 30),
)

METRIC_FIELDS = ("profit_pct", "time_in_market_fraction", "max_drawdown_pct", "total_trades")


@dataclass(frozen=True)
class BranchCase:
    ticker: str
    rsi_period: int
    rsi_threshold: int
    branch_id: str
    price: pd.DataFrame
    state: pd.DataFrame
    entry_signal: pd.Series
    close_vals: np.ndarray
    high_vals: np.ndarray
    low_vals: np.ndarray
    entry_vals: np.ndarray
    active_open_vals: np.ndarray
    active_close_vals: np.ndarray
    exit_vals: np.ndarray
    prev_close_vals: np.ndarray
    canonical_metrics: dict


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    fn: callable


def branch_id(ticker: str, period: int, threshold: int) -> str:
    return f"{period}d RSI {ticker} LT{threshold} - L {ticker}"


@njit(cache=True)
def _numba_exact_metrics(close_vals, high_vals, low_vals, entry_vals):
    n = len(close_vals)
    if n == 0:
        return 0.0, 0.0, 0, 0.0, 0, 0

    capital = INITIAL_EQUITY
    capital_at_entry = 0.0
    entry_price = 0.0
    in_position = False
    active_bars = 0
    realized_trades = 0

    marks = np.empty(1 + n * 3, dtype=np.float64)
    mark_count = 1
    marks[0] = INITIAL_EQUITY

    prev_high = np.empty(n, dtype=np.float64)
    prev_high[0] = np.nan
    for i in range(1, n):
        prev_high[i] = high_vals[i - 1]

    for i in range(n):
        if in_position:
            active_bars += 1

            low_mark = low_vals[i]
            if not np.isnan(low_mark) and entry_price != 0.0:
                low_pnl = (low_mark - entry_price) / entry_price
                marks[mark_count] = capital_at_entry * (1.0 + low_pnl)
                mark_count += 1

            close_mark = close_vals[i]
            if not np.isnan(close_mark) and entry_price != 0.0:
                close_pnl = (close_mark - entry_price) / entry_price
                marks[mark_count] = capital_at_entry * (1.0 + close_pnl)
                mark_count += 1

            exit_signal = (not np.isnan(prev_high[i])) and (not np.isnan(close_vals[i])) and (close_vals[i] > prev_high[i])
            if exit_signal:
                exit_price = close_vals[i] * (1.0 - SLIPPAGE_FRACTION_PER_SIDE)
                pnl = (exit_price - entry_price) / entry_price
                capital *= 1.0 + pnl
                marks[mark_count] = capital
                mark_count += 1
                in_position = False
                capital_at_entry = 0.0
                realized_trades += 1
        else:
            if entry_vals[i] and (not np.isnan(close_vals[i])):
                entry_price = close_vals[i] * (1.0 + SLIPPAGE_FRACTION_PER_SIDE)
                capital_at_entry = capital
                in_position = True
                active_bars += 1
                entry_mark_pnl = (close_vals[i] - entry_price) / entry_price
                marks[mark_count] = capital_at_entry * (1.0 + entry_mark_pnl)
                mark_count += 1

    peak = marks[0]
    min_drawdown = 0.0
    for i in range(mark_count):
        val = marks[i]
        if val > peak:
            peak = val
        dd = val / peak - 1.0
        if dd < min_drawdown:
            min_drawdown = dd

    return (
        capital - INITIAL_EQUITY,
        abs(min_drawdown),
        realized_trades,
        active_bars / n,
        realized_trades,
        1 if in_position else 0,
    )


@njit(cache=True)
def _numba_compact_exact_metrics(close_vals, high_vals, low_vals, entry_vals):
    n = len(close_vals)
    if n == 0:
        return 0.0, 0.0, 0, 0.0

    capital = INITIAL_EQUITY
    capital_at_entry = 0.0
    entry_price = 0.0
    in_position = False
    active_bars = 0
    realized_trades = 0
    peak = INITIAL_EQUITY
    min_drawdown = 0.0

    prev_high = np.empty(n, dtype=np.float64)
    prev_high[0] = np.nan
    for i in range(1, n):
        prev_high[i] = high_vals[i - 1]

    for i in range(n):
        if in_position:
            active_bars += 1

            low_mark = low_vals[i]
            if not np.isnan(low_mark) and entry_price != 0.0:
                val = capital_at_entry * (1.0 + (low_mark - entry_price) / entry_price)
                if val > peak:
                    peak = val
                dd = val / peak - 1.0
                if dd < min_drawdown:
                    min_drawdown = dd

            close_mark = close_vals[i]
            if not np.isnan(close_mark) and entry_price != 0.0:
                val = capital_at_entry * (1.0 + (close_mark - entry_price) / entry_price)
                if val > peak:
                    peak = val
                dd = val / peak - 1.0
                if dd < min_drawdown:
                    min_drawdown = dd

            exit_signal = (not np.isnan(prev_high[i])) and (not np.isnan(close_vals[i])) and (close_vals[i] > prev_high[i])
            if exit_signal:
                exit_price = close_vals[i] * (1.0 - SLIPPAGE_FRACTION_PER_SIDE)
                pnl = (exit_price - entry_price) / entry_price
                capital *= 1.0 + pnl
                if capital > peak:
                    peak = capital
                dd = capital / peak - 1.0
                if dd < min_drawdown:
                    min_drawdown = dd
                in_position = False
                capital_at_entry = 0.0
                realized_trades += 1
        else:
            if entry_vals[i] and (not np.isnan(close_vals[i])):
                entry_price = close_vals[i] * (1.0 + SLIPPAGE_FRACTION_PER_SIDE)
                capital_at_entry = capital
                in_position = True
                active_bars += 1
                val = capital_at_entry * (1.0 + (close_vals[i] - entry_price) / entry_price)
                if val > peak:
                    peak = val
                dd = val / peak - 1.0
                if dd < min_drawdown:
                    min_drawdown = dd

    return capital - INITIAL_EQUITY, abs(min_drawdown), realized_trades, active_bars / n


def _as_metrics_tuple(res) -> dict:
    profit_pct, max_drawdown_pct, total_trades, time_in_market_fraction = res[:4]
    return {
        "profit_pct": float(profit_pct),
        "max_drawdown_pct": float(max_drawdown_pct),
        "total_trades": int(total_trades),
        "time_in_market_fraction": float(time_in_market_fraction),
    }


def variant_fast_python_arrays(case: BranchCase) -> dict:
    return compute_fast_metrics_from_signals(case.price, case.entry_signal)


def variant_numba_exact(case: BranchCase) -> dict:
    return _as_metrics_tuple(_numba_exact_metrics(case.close_vals, case.high_vals, case.low_vals, case.entry_vals))


def variant_numba_compact(case: BranchCase) -> dict:
    return _as_metrics_tuple(_numba_compact_exact_metrics(case.close_vals, case.high_vals, case.low_vals, case.entry_vals))


def variant_state_mask_close_only(case: BranchCase) -> dict:
    active_open = case.active_open_vals
    active_close = case.active_close_vals
    touched = np.logical_or(active_open, active_close)
    n = len(case.close_vals)

    equity = np.empty(n, dtype=float)
    equity[0] = INITIAL_EQUITY
    for i in range(1, n):
        if active_open[i]:
            prev_close = case.prev_close_vals[i]
            if np.isfinite(prev_close) and prev_close != 0.0:
                ret = case.close_vals[i] / prev_close - 1.0
                equity[i] = equity[i - 1] * (1.0 + ret)
            else:
                equity[i] = equity[i - 1]
        else:
            equity[i] = equity[i - 1]
    if n > 0:
        peak = np.maximum.accumulate(equity)
        drawdown = equity / peak - 1.0
        dd = float(abs(np.nanmin(drawdown)))
        profit = float(equity[-1] - INITIAL_EQUITY)
    else:
        dd = 0.0
        profit = 0.0

    total_trades = int(case.exit_vals.sum())
    tim = float(touched.mean()) if n else 0.0
    return {
        "profit_pct": profit,
        "max_drawdown_pct": dd,
        "total_trades": total_trades,
        "time_in_market_fraction": tim,
    }


def variant_trade_span_inclusive(case: BranchCase) -> dict:
    n = len(case.close_vals)
    in_pos = False
    starts: list[int] = []
    ends: list[int] = []
    entry_price = math.nan
    compounded = INITIAL_EQUITY

    for i in range(n):
        if not in_pos and case.entry_vals[i]:
            starts.append(i)
            entry_price = case.close_vals[i] * (1.0 + SLIPPAGE_FRACTION_PER_SIDE)
            in_pos = True
        elif in_pos and case.exit_vals[i]:
            exit_price = case.close_vals[i] * (1.0 - SLIPPAGE_FRACTION_PER_SIDE)
            compounded *= 1.0 + (exit_price - entry_price) / entry_price
            ends.append(i)
            in_pos = False

    trade_count = len(ends)
    active_bars = 0
    for j in range(trade_count):
        active_bars += ends[j] - starts[j] + 1
    if trade_count > 0:
        eq = np.empty(trade_count + 1, dtype=float)
        eq[0] = INITIAL_EQUITY
        compounded = INITIAL_EQUITY
        for j in range(trade_count):
            entry_price = case.close_vals[starts[j]] * (1.0 + SLIPPAGE_FRACTION_PER_SIDE)
            exit_price = case.close_vals[ends[j]] * (1.0 - SLIPPAGE_FRACTION_PER_SIDE)
            compounded *= 1.0 + (exit_price - entry_price) / entry_price
            eq[j + 1] = compounded
        peak = np.maximum.accumulate(eq)
        drawdown = eq / peak - 1.0
        dd = float(abs(np.nanmin(drawdown)))
        profit = float(eq[-1] - INITIAL_EQUITY)
    else:
        dd = 0.0
        profit = 0.0

    return {
        "profit_pct": profit,
        "max_drawdown_pct": dd,
        "total_trades": trade_count,
        "time_in_market_fraction": float(active_bars / n) if n else 0.0,
    }


def warm_numba() -> None:
    close_vals = np.array([100.0, 99.0, 101.0, 102.0], dtype=np.float64)
    high_vals = np.array([101.0, 100.0, 102.0, 103.0], dtype=np.float64)
    low_vals = np.array([99.0, 98.0, 100.0, 101.0], dtype=np.float64)
    entry_vals = np.array([True, False, False, False], dtype=np.bool_)
    _numba_exact_metrics(close_vals, high_vals, low_vals, entry_vals)
    _numba_compact_exact_metrics(close_vals, high_vals, low_vals, entry_vals)


def load_cases(start: str, end: str) -> list[BranchCase]:
    cfg = System30Config()
    cases: list[BranchCase] = []
    for ticker, period, threshold in GOLDEN_BRANCHES:
        df = load_local_ticker_history(ticker, cfg.local_daily_dir)
        if df.empty:
            raise FileNotFoundError(f"Missing local history for {ticker} under {cfg.local_daily_dir}")
        state = build_branch_state(
            df=df,
            start=start,
            end=end,
            rsi_period=period,
            rsi_threshold=threshold,
            rsi_wtype="wilder",
        )
        if state.empty:
            raise ValueError(f"Empty state for {ticker} {period}d LT{threshold} in {start}..{end}")
        price = df.loc[start:end, ["Open", "High", "Low", "Close", "Volume"]].copy()
        entry_signal = state["entry_signal"].astype(bool)
        canonical_metrics = compute_branch_metrics(state)
        cases.append(
            BranchCase(
                ticker=ticker,
                rsi_period=period,
                rsi_threshold=threshold,
                branch_id=branch_id(ticker, period, threshold),
                price=price,
                state=state,
                entry_signal=entry_signal,
                close_vals=state["Close"].to_numpy(dtype=float),
                high_vals=state["High"].to_numpy(dtype=float),
                low_vals=state["Low"].to_numpy(dtype=float),
                entry_vals=state["entry_signal"].fillna(False).astype(bool).to_numpy(),
                active_open_vals=state["active_at_open"].fillna(False).astype(bool).to_numpy(),
                active_close_vals=state["active_after_close"].fillna(False).astype(bool).to_numpy(),
                exit_vals=state["exit_executed"].fillna(False).astype(bool).to_numpy(),
                prev_close_vals=price["Close"].shift(1).to_numpy(dtype=float),
                canonical_metrics=canonical_metrics,
            )
        )
    return cases


def time_variant(case: BranchCase, fn: callable, warmups: int, runs: int) -> tuple[dict, list[float]]:
    last_metrics = None
    for _ in range(warmups):
        last_metrics = fn(case)
    timings = []
    for _ in range(runs):
        t0 = time.perf_counter()
        last_metrics = fn(case)
        timings.append((time.perf_counter() - t0) * 1e6)
    return last_metrics, timings


def summarize_timings(timings: list[float]) -> dict:
    return {
        "mean_us": float(statistics.mean(timings)),
        "median_us": float(statistics.median(timings)),
        "min_us": float(min(timings)),
        "max_us": float(max(timings)),
        "stdev_us": float(statistics.pstdev(timings)),
    }


def compare_metrics(actual: dict, expected: dict) -> dict:
    return {
        "delta_profit_pct_points": (actual["profit_pct"] - expected["profit_pct"]) * 100.0,
        "delta_tim_pct_points": (actual["time_in_market_fraction"] - expected["time_in_market_fraction"]) * 100.0,
        "delta_dd_pct_points": (actual["max_drawdown_pct"] - expected["max_drawdown_pct"]) * 100.0,
        "delta_total_trades": int(actual["total_trades"]) - int(expected["total_trades"]),
    }


def build_variants() -> list[Variant]:
    return [
        Variant(
            name="fast_python_arrays_exact",
            description="Existing exact array path via compute_fast_metrics_from_signals",
            fn=variant_fast_python_arrays,
        ),
        Variant(
            name="numba_exact_marks",
            description="Numba-compiled exact replica with explicit equity marks array",
            fn=variant_numba_exact,
        ),
        Variant(
            name="numba_exact_compact",
            description="Numba exact replica with streaming peak/DD updates and no marks array",
            fn=variant_numba_compact,
        ),
        Variant(
            name="state_mask_close_only_approx",
            description="Uses active-state masks and close-to-close equity only; faster but DD/profit semantics drift",
            fn=variant_state_mask_close_only,
        ),
        Variant(
            name="trade_span_inclusive_approx",
            description="Trade-span inclusive TIM + closed-trade equity only; intentionally approximate for DD/open trades",
            fn=variant_trade_span_inclusive,
        ),
    ]


def run_benchmark(start: str, end: str, warmups: int, runs: int, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    warm_numba()
    cases = load_cases(start=start, end=end)
    variants = build_variants()

    branch_rows: list[dict] = []
    summary_rows: list[dict] = []

    for variant in variants:
        variant_branch_rows = []
        for case in cases:
            metrics, timings = time_variant(case=case, fn=variant.fn, warmups=warmups, runs=runs)
            timing_stats = summarize_timings(timings)
            deltas = compare_metrics(metrics, case.canonical_metrics)
            row = {
                "variant": variant.name,
                "variant_description": variant.description,
                "ticker": case.ticker,
                "branch_id": case.branch_id,
                "bars": len(case.state),
                **timing_stats,
                "canonical_profit_pct": case.canonical_metrics["profit_pct"] * 100.0,
                "canonical_tim_pct": case.canonical_metrics["time_in_market_fraction"] * 100.0,
                "canonical_dd_pct": case.canonical_metrics["max_drawdown_pct"] * 100.0,
                "canonical_total_trades": int(case.canonical_metrics["total_trades"]),
                "variant_profit_pct": metrics["profit_pct"] * 100.0,
                "variant_tim_pct": metrics["time_in_market_fraction"] * 100.0,
                "variant_dd_pct": metrics["max_drawdown_pct"] * 100.0,
                "variant_total_trades": int(metrics["total_trades"]),
                **deltas,
            }
            branch_rows.append(row)
            variant_branch_rows.append(row)

        vdf = pd.DataFrame(variant_branch_rows)
        summary_rows.append(
            {
                "variant": variant.name,
                "variant_description": variant.description,
                "branches": len(vdf),
                "mean_median_us": float(vdf["median_us"].mean()),
                "median_median_us": float(vdf["median_us"].median()),
                "max_median_us": float(vdf["median_us"].max()),
                "speedup_vs_canonical_state_path": math.nan,
                "mean_abs_profit_delta_pct_points": float(vdf["delta_profit_pct_points"].abs().mean()),
                "mean_abs_tim_delta_pct_points": float(vdf["delta_tim_pct_points"].abs().mean()),
                "mean_abs_dd_delta_pct_points": float(vdf["delta_dd_pct_points"].abs().mean()),
                "max_abs_profit_delta_pct_points": float(vdf["delta_profit_pct_points"].abs().max()),
                "max_abs_tim_delta_pct_points": float(vdf["delta_tim_pct_points"].abs().max()),
                "max_abs_dd_delta_pct_points": float(vdf["delta_dd_pct_points"].abs().max()),
                "max_abs_trade_delta": int(vdf["delta_total_trades"].abs().max()),
            }
        )

    canonical_timing_rows = []
    for case in cases:
        metrics, timings = time_variant(case=case, fn=lambda c: compute_branch_metrics(c.state), warmups=warmups, runs=runs)
        timing_stats = summarize_timings(timings)
        canonical_timing_rows.append({
            "ticker": case.ticker,
            "branch_id": case.branch_id,
            **timing_stats,
            "profit_pct": metrics["profit_pct"] * 100.0,
            "tim_pct": metrics["time_in_market_fraction"] * 100.0,
            "dd_pct": metrics["max_drawdown_pct"] * 100.0,
            "total_trades": int(metrics["total_trades"]),
        })

    # Canonical end-to-end path benchmark: state build + canonical metrics.
    e2e_rows = []
    cfg = System30Config()
    for case in cases:
        df = load_local_ticker_history(case.ticker, cfg.local_daily_dir)
        for _ in range(warmups):
            state = build_branch_state(df, start, end, case.rsi_period, case.rsi_threshold, "wilder")
            compute_branch_metrics(state)
        e2e_timings = []
        for _ in range(runs):
            t0 = time.perf_counter()
            state = build_branch_state(df, start, end, case.rsi_period, case.rsi_threshold, "wilder")
            metrics = compute_branch_metrics(state)
            e2e_timings.append((time.perf_counter() - t0) * 1e6)
        timing_stats = summarize_timings(e2e_timings)
        e2e_rows.append({
            "ticker": case.ticker,
            "branch_id": case.branch_id,
            **timing_stats,
            "profit_pct": metrics["profit_pct"] * 100.0,
            "tim_pct": metrics["time_in_market_fraction"] * 100.0,
            "dd_pct": metrics["max_drawdown_pct"] * 100.0,
            "total_trades": int(metrics["total_trades"]),
        })

    branch_df = pd.DataFrame(branch_rows).sort_values(["variant", "ticker"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows).sort_values("mean_median_us").reset_index(drop=True)
    canonical_df = pd.DataFrame(canonical_timing_rows).sort_values("ticker").reset_index(drop=True)
    canonical_e2e_df = pd.DataFrame(e2e_rows).sort_values("ticker").reset_index(drop=True)

    canonical_postprocess_mean = float(canonical_df["median_us"].mean())
    summary_df["canonical_postprocess_mean_median_us"] = canonical_postprocess_mean
    summary_df["speedup_vs_canonical_state_metrics_only"] = canonical_postprocess_mean / summary_df["mean_median_us"]
    summary_df["canonical_e2e_mean_median_us"] = float(canonical_e2e_df["median_us"].mean())
    summary_df["speedup_vs_canonical_e2e"] = summary_df["canonical_e2e_mean_median_us"] / summary_df["mean_median_us"]

    out_dir.mkdir(parents=True, exist_ok=True)
    branch_df.to_csv(out_dir / "system30_postprocess_benchmark_per_branch.csv", index=False)
    summary_df.to_csv(out_dir / "system30_postprocess_benchmark_summary.csv", index=False)
    canonical_df.to_csv(out_dir / "system30_postprocess_canonical_metrics_only_timing.csv", index=False)
    canonical_e2e_df.to_csv(out_dir / "system30_postprocess_canonical_e2e_timing.csv", index=False)

    write_markdown_report(
        out_dir=out_dir,
        start=start,
        end=end,
        warmups=warmups,
        runs=runs,
        summary_df=summary_df,
        canonical_df=canonical_df,
        canonical_e2e_df=canonical_e2e_df,
    )
    return branch_df, summary_df


def _markdown_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = [["" if pd.isna(v) else str(v) for v in row] for row in df.to_numpy()]
    widths = [len(col) for col in cols]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def fmt(row_vals: list[str]) -> str:
        return "| " + " | ".join(val.ljust(widths[i]) for i, val in enumerate(row_vals)) + " |"

    header = fmt(cols)
    sep = "| " + " | ".join("-" * widths[i] for i in range(len(cols))) + " |"
    body = [fmt(row) for row in rows]
    return "\n".join([header, sep, *body])


def write_markdown_report(
    out_dir: Path,
    start: str,
    end: str,
    warmups: int,
    runs: int,
    summary_df: pd.DataFrame,
    canonical_df: pd.DataFrame,
    canonical_e2e_df: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# System30 post-processing benchmark")
    lines.append("")
    lines.append(f"- Window: {start} .. {end}")
    lines.append(f"- Golden branches: {len(GOLDEN_BRANCHES)}")
    lines.append(f"- Warmups per branch/variant: {warmups}")
    lines.append(f"- Timed runs per branch/variant: {runs}")
    lines.append("")
    lines.append("## Variants ranked by speed")
    lines.append("")
    lines.append(_markdown_table(summary_df))
    lines.append("")
    lines.append("## Canonical baselines")
    lines.append("")
    lines.append("### Metrics-only canonical path (compute_branch_metrics on prebuilt state)")
    lines.append("")
    lines.append(_markdown_table(canonical_df[["ticker", "median_us", "profit_pct", "tim_pct", "dd_pct", "total_trades"]]))
    lines.append("")
    lines.append("### End-to-end canonical path (build_branch_state + compute_branch_metrics)")
    lines.append("")
    lines.append(_markdown_table(canonical_e2e_df[["ticker", "median_us", "profit_pct", "tim_pct", "dd_pct", "total_trades"]]))
    lines.append("")
    (out_dir / "system30_postprocess_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark System30 post-processing metric variants against the canonical custom metric path.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    branch_df, summary_df = run_benchmark(
        start=args.start,
        end=args.end,
        warmups=args.warmups,
        runs=args.runs,
        out_dir=args.out_dir,
    )

    print("Per-branch results:")
    print(branch_df.head(20).to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
    print(f"\nWrote outputs under {args.out_dir}")


if __name__ == "__main__":
    main()
