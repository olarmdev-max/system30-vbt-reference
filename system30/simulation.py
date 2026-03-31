from __future__ import annotations

import numpy as np
import pandas as pd

from .rsi import compute_wilder_rsi

try:
    from numba import njit
except Exception:  # pragma: no cover - fallback if numba is unavailable in some environment
    njit = None

SLIPPAGE_BPS_PER_SIDE = 2.0
SLIPPAGE_FRACTION_PER_SIDE = SLIPPAGE_BPS_PER_SIDE / 10_000.0
INITIAL_EQUITY = 1.0


if njit is not None:
    @njit(cache=True)
    def _numba_reference_trade_metrics_from_arrays(
        close_vals: np.ndarray,
        high_vals: np.ndarray,
        low_vals: np.ndarray,
        entry_vals: np.ndarray,
    ) -> tuple[float, float, int, float, int, int]:
        n = len(close_vals)
        if n == 0:
            return 0.0, 0.0, 0, 0.0, 0, 0

        capital = INITIAL_EQUITY
        capital_at_entry = 0.0
        entry_price = 0.0
        in_position = False
        active_bars = 0
        realized_trades = 0

        # Why keep explicit marks instead of a more compact streaming update?
        # This version is nearly as fast as the compact winner in benchmarks, but it is
        # easier to audit because it mirrors the readable Python reference path one-for-one.
        # If Hassan later asks for feature changes or deeper parity debugging, this structure
        # is much safer to reason about than a highly compressed kernel.
        marks = np.empty(1 + n * 3, dtype=np.float64)
        mark_count = 1
        marks[0] = INITIAL_EQUITY

        prev_high_vals = np.empty_like(high_vals)
        prev_high_vals[0] = np.nan
        prev_high_vals[1:] = high_vals[:-1]

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

                exit_signal = (not np.isnan(prev_high_vals[i])) and (not np.isnan(close_vals[i])) and (close_vals[i] > prev_high_vals[i])
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


if njit is not None:
    @njit(cache=True)
    def _numba_reference_trade_metrics_batch_from_rsi_thresholds(
        close_vals: np.ndarray,
        high_vals: np.ndarray,
        low_vals: np.ndarray,
        rsi_vals: np.ndarray,
        thresholds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        threshold_count = len(thresholds)
        n = len(close_vals)

        profit_pcts = np.zeros(threshold_count, dtype=np.float64)
        max_drawdown_pcts = np.zeros(threshold_count, dtype=np.float64)
        total_trades = np.zeros(threshold_count, dtype=np.int64)
        time_in_market_fractions = np.zeros(threshold_count, dtype=np.float64)
        realized_trades = np.zeros(threshold_count, dtype=np.int64)
        unrealized_trades = np.zeros(threshold_count, dtype=np.int64)
        if n == 0 or threshold_count == 0:
            return (
                profit_pcts,
                max_drawdown_pcts,
                total_trades,
                time_in_market_fractions,
                realized_trades,
                unrealized_trades,
            )

        prev_high_vals = np.empty_like(high_vals)
        prev_high_vals[0] = np.nan
        prev_high_vals[1:] = high_vals[:-1]

        for j in range(threshold_count):
            threshold = thresholds[j]
            capital = INITIAL_EQUITY
            capital_at_entry = 0.0
            entry_price = 0.0
            in_position = False
            active_bars = 0
            realized = 0

            marks = np.empty(1 + n * 3, dtype=np.float64)
            mark_count = 1
            marks[0] = INITIAL_EQUITY

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

                    exit_signal = (not np.isnan(prev_high_vals[i])) and (not np.isnan(close_vals[i])) and (close_vals[i] > prev_high_vals[i])
                    if exit_signal:
                        exit_price = close_vals[i] * (1.0 - SLIPPAGE_FRACTION_PER_SIDE)
                        pnl = (exit_price - entry_price) / entry_price
                        capital *= 1.0 + pnl
                        marks[mark_count] = capital
                        mark_count += 1
                        in_position = False
                        capital_at_entry = 0.0
                        realized += 1
                else:
                    rsi_val = rsi_vals[i]
                    if (not np.isnan(rsi_val)) and (rsi_val < threshold) and (not np.isnan(close_vals[i])):
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

            profit_pcts[j] = capital - INITIAL_EQUITY
            max_drawdown_pcts[j] = abs(min_drawdown)
            total_trades[j] = realized
            time_in_market_fractions[j] = active_bars / n
            realized_trades[j] = realized
            unrealized_trades[j] = 1 if in_position else 0

        return (
            profit_pcts,
            max_drawdown_pcts,
            total_trades,
            time_in_market_fractions,
            realized_trades,
            unrealized_trades,
        )


def compute_any_touch_time_in_market_fraction(
    active_at_open: pd.Series | np.ndarray,
    active_after_close: pd.Series | np.ndarray,
) -> float:
    """Fraction of bars touched by an active position.

    A bar counts as time-in-market if the position was active at the open or
    became/remained active after the close. This cleanly captures same-bar
    close entries without overstating fully flat days.
    """
    open_mask = np.asarray(active_at_open, dtype=bool)
    close_mask = np.asarray(active_after_close, dtype=bool)
    if len(open_mask) == 0:
        return 0.0
    return float(np.logical_or(open_mask, close_mask).mean())


def _compute_reference_trade_metrics_from_arrays(
    close_vals: np.ndarray,
    high_vals: np.ndarray,
    low_vals: np.ndarray,
    entry_vals: np.ndarray,
) -> dict:
    """Mirror the yearly-rolling reference branch accounting as closely as practical.

    Key semantics:
    - entry on same close when RSI < threshold
    - exit on same close when already in trade and Close > prior High
    - 2 bps slippage per side
    - profit compounds closed trades only
    - total trades counts closed trades only
    - time in market counts entry bar plus every bar held while open
    - max DD is based on intratrade equity marks (low mark + close mark while open)

    Performance note:
    - if Numba is available, we use the jitted exact-marks replica by default
    - if not, we fall back to the readable Python reference implementation below
    - the exact-marks kernel was chosen because it matched the golden branches with
      zero observed drift while remaining easier to audit than the more compact kernel
    """
    if njit is not None:
        profit_pct, max_drawdown_pct, total_trades, time_in_market_fraction, realized_trades, unrealized_trades = _numba_reference_trade_metrics_from_arrays(
            close_vals.astype(np.float64, copy=False),
            high_vals.astype(np.float64, copy=False),
            low_vals.astype(np.float64, copy=False),
            entry_vals.astype(np.bool_, copy=False),
        )
        return {
            "profit_pct": float(profit_pct),
            "max_drawdown_pct": float(max_drawdown_pct),
            "total_trades": int(total_trades),
            "time_in_market_fraction": float(time_in_market_fraction),
            "realized_trades": int(realized_trades),
            "unrealized_trades": int(unrealized_trades),
        }

    n = len(close_vals)
    if n == 0:
        return {
            "profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "time_in_market_fraction": 0.0,
            "realized_trades": 0,
            "unrealized_trades": 0,
        }

    capital = INITIAL_EQUITY
    capital_at_entry = 0.0
    entry_price = 0.0
    in_position = False
    active_bars = 0
    realized_trades = 0
    equity_marks = [INITIAL_EQUITY]

    prev_high_vals = np.empty_like(high_vals)
    prev_high_vals[0] = np.nan
    prev_high_vals[1:] = high_vals[:-1]

    for i in range(n):
        if in_position:
            active_bars += 1

            # Important parity rule:
            # C5/reference-style max DD is not a simple close-to-close portfolio DD.
            # While a trade is open, we mark both the bar low and the bar close into
            # the synthetic equity path. This is why DD can stay close to C5 even when
            # native portfolio-style DD would look artificially smaller. If Hassan asks
            # for fresh VBT-vs-C5 DD stats, this custom path is the one to trust.
            low_mark = low_vals[i]
            if np.isfinite(low_mark) and entry_price != 0.0:
                low_pnl = (low_mark - entry_price) / entry_price
                equity_marks.append(capital_at_entry * (1.0 + low_pnl))

            close_mark = close_vals[i]
            if np.isfinite(close_mark) and entry_price != 0.0:
                close_pnl = (close_mark - entry_price) / entry_price
                equity_marks.append(capital_at_entry * (1.0 + close_pnl))

            exit_signal = bool(np.isfinite(prev_high_vals[i]) and np.isfinite(close_vals[i]) and close_vals[i] > prev_high_vals[i])
            if exit_signal:
                exit_price = close_vals[i] * (1.0 - SLIPPAGE_FRACTION_PER_SIDE)
                pnl = (exit_price - entry_price) / entry_price
                capital *= 1.0 + pnl
                equity_marks.append(capital)
                in_position = False
                capital_at_entry = 0.0
                realized_trades += 1
        else:
            if bool(entry_vals[i]) and np.isfinite(close_vals[i]):
                entry_price = close_vals[i] * (1.0 + SLIPPAGE_FRACTION_PER_SIDE)
                capital_at_entry = capital
                in_position = True
                active_bars += 1
                entry_mark_pnl = (close_vals[i] - entry_price) / entry_price
                equity_marks.append(capital_at_entry * (1.0 + entry_mark_pnl))

    eq = np.asarray(equity_marks, dtype=float)
    if len(eq) == 0:
        max_drawdown_pct = 0.0
    else:
        peak = np.maximum.accumulate(eq)
        drawdown = eq / peak - 1.0
        max_drawdown_pct = float(abs(np.nanmin(drawdown))) if len(drawdown) else 0.0

    return {
        "profit_pct": float(capital - INITIAL_EQUITY),
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": realized_trades,
        "time_in_market_fraction": float(active_bars / n) if n > 0 else 0.0,
        "realized_trades": realized_trades,
        "unrealized_trades": int(in_position),
    }


def build_branch_state(
    df: pd.DataFrame,
    start: str,
    end: str,
    rsi_period: int,
    rsi_threshold: int,
    rsi_wtype: str = "wilder",
) -> pd.DataFrame:
    """Build branch state using trade-on-close semantics.

    Important System30 assumption in this project:
    - RSI is computed on the bar close.
    - If RSI triggers on bar t, the entry is assumed to happen on that same bar's close.
    - Therefore the position is *not* active at the open of bar t, and the first P&L contribution
      appears on bar t+1 via close-to-close return.
    - Exit is also evaluated on bar close using Close[t] > High[t-1], and if true the position is
      closed on that same close.
    """
    full_price = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    price = full_price.loc[start:end].copy()
    if price.empty:
        return pd.DataFrame()

    if rsi_wtype != "wilder":
        raise ValueError(f"Unsupported rsi_wtype for reference path: {rsi_wtype}")
    rsi_full = compute_wilder_rsi(full_price["Close"], period=rsi_period)
    rsi = rsi_full.loc[price.index]
    prev_high = full_price["High"].shift(1).loc[price.index]
    prev_close = full_price["Close"].shift(1).loc[price.index]

    in_position = False
    rows = []
    for dt in price.index:
        entry_signal = bool(pd.notna(rsi.loc[dt]) and float(rsi.loc[dt]) < rsi_threshold)
        exit_signal = bool(pd.notna(prev_high.loc[dt]) and float(price.at[dt, "Close"]) > float(prev_high.loc[dt]))
        active_at_open = in_position
        asset_return = 0.0
        if active_at_open and pd.notna(prev_close.loc[dt]) and float(prev_close.loc[dt]) != 0:
            asset_return = float(price.at[dt, "Close"]) / float(prev_close.loc[dt]) - 1.0

        entry_executed = False
        exit_executed = False

        if not in_position and entry_signal:
            in_position = True
            entry_executed = True

        if active_at_open and exit_signal:
            in_position = False
            exit_executed = True

        rows.append(
            {
                "Date": dt,
                "entry_signal": entry_signal,
                "exit_signal": exit_signal,
                "entry_executed": entry_executed,
                "exit_executed": exit_executed,
                "active_at_open": active_at_open,
                "active_after_close": in_position,
                "asset_return": asset_return,
                "Close": float(price.at[dt, "Close"]),
                "High": float(price.at[dt, "High"]),
                "Low": float(price.at[dt, "Low"]),
                "RSI": float(rsi.loc[dt]) if pd.notna(rsi.loc[dt]) else None,
            }
        )

    out = pd.DataFrame(rows).set_index("Date")
    return out


def compute_branch_metrics(state: pd.DataFrame) -> dict:
    if state.empty:
        return {
            "profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "time_in_market_fraction": 0.0,
            "realized_trades": 0,
            "unrealized_trades": 0,
        }

    return _compute_reference_trade_metrics_from_arrays(
        close_vals=state["Close"].to_numpy(dtype=float),
        high_vals=state["High"].to_numpy(dtype=float),
        low_vals=state["Low"].to_numpy(dtype=float),
        entry_vals=state["entry_signal"].fillna(False).astype(bool).to_numpy(),
    )


def compute_fast_metrics_from_signals(price: pd.DataFrame, entry_signal: pd.Series) -> dict:
    if price.empty or entry_signal.empty:
        return {
            "profit_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "total_trades": 0,
            "time_in_market_fraction": 0.0,
            "realized_trades": 0,
            "unrealized_trades": 0,
        }

    return _compute_reference_trade_metrics_from_arrays(
        close_vals=price["Close"].to_numpy(dtype=float),
        high_vals=price["High"].to_numpy(dtype=float),
        low_vals=price["Low"].to_numpy(dtype=float),
        entry_vals=entry_signal.reindex(price.index).fillna(False).astype(bool).to_numpy(),
    )


def compute_fast_metrics_batch_from_rsi_thresholds(
    price: pd.DataFrame,
    rsi: pd.Series | np.ndarray,
    thresholds: tuple[int, ...] | list[int] | np.ndarray,
) -> dict[str, np.ndarray]:
    if price.empty or len(thresholds) == 0:
        empty_float = np.empty(0, dtype=float)
        empty_int = np.empty(0, dtype=int)
        return {
            "profit_pct": empty_float,
            "max_drawdown_pct": empty_float,
            "total_trades": empty_int,
            "time_in_market_fraction": empty_float,
            "realized_trades": empty_int,
            "unrealized_trades": empty_int,
        }

    rsi_vals = np.asarray(rsi, dtype=np.float64)
    threshold_vals = np.asarray(thresholds, dtype=np.int64)
    close_vals = price["Close"].to_numpy(dtype=float)
    high_vals = price["High"].to_numpy(dtype=float)
    low_vals = price["Low"].to_numpy(dtype=float)

    if njit is not None:
        profit_pct, max_drawdown_pct, total_trades, time_in_market_fraction, realized_trades, unrealized_trades = _numba_reference_trade_metrics_batch_from_rsi_thresholds(
            close_vals=close_vals.astype(np.float64, copy=False),
            high_vals=high_vals.astype(np.float64, copy=False),
            low_vals=low_vals.astype(np.float64, copy=False),
            rsi_vals=rsi_vals.astype(np.float64, copy=False),
            thresholds=threshold_vals.astype(np.int64, copy=False),
        )
        return {
            "profit_pct": profit_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "total_trades": total_trades,
            "time_in_market_fraction": time_in_market_fraction,
            "realized_trades": realized_trades,
            "unrealized_trades": unrealized_trades,
        }

    out = {
        "profit_pct": np.empty(len(threshold_vals), dtype=float),
        "max_drawdown_pct": np.empty(len(threshold_vals), dtype=float),
        "total_trades": np.empty(len(threshold_vals), dtype=int),
        "time_in_market_fraction": np.empty(len(threshold_vals), dtype=float),
        "realized_trades": np.empty(len(threshold_vals), dtype=int),
        "unrealized_trades": np.empty(len(threshold_vals), dtype=int),
    }
    rsi_series = pd.Series(rsi_vals, index=price.index)
    for idx, threshold in enumerate(threshold_vals):
        metrics = compute_fast_metrics_from_signals(price, rsi_series < int(threshold))
        out["profit_pct"][idx] = metrics["profit_pct"]
        out["max_drawdown_pct"][idx] = metrics["max_drawdown_pct"]
        out["total_trades"][idx] = metrics["total_trades"]
        out["time_in_market_fraction"][idx] = metrics["time_in_market_fraction"]
        out["realized_trades"][idx] = metrics["realized_trades"]
        out["unrealized_trades"][idx] = metrics["unrealized_trades"]
    return out
