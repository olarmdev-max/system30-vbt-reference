from __future__ import annotations

import pandas as pd

from .labels import branch_id
from .rsi_cache import TickerRSICache
from .rolling import RollingWindow
from .simulation import compute_fast_metrics_batch_from_rsi_thresholds


def _build_period_metrics_frame(
    ticker: str,
    price: pd.DataFrame,
    rsi: pd.Series,
    period: int,
    wtype: str,
    threshold_values: tuple[int, ...],
) -> pd.DataFrame:
    batch_metrics = compute_fast_metrics_batch_from_rsi_thresholds(
        price=price,
        rsi=rsi,
        thresholds=threshold_values,
    )
    return pd.DataFrame(
        {
            "ticker": ticker,
            "branch_id": [branch_id(ticker, period, threshold) for threshold in threshold_values],
            "rsi_period": period,
            "rsi_threshold": threshold_values,
            "rsi_wtype": wtype,
            "is_profit_pct": batch_metrics["profit_pct"],
            "is_max_drawdown_pct": batch_metrics["max_drawdown_pct"],
            "is_total_trades": batch_metrics["total_trades"],
            "is_time_in_market_fraction": batch_metrics["time_in_market_fraction"],
            "is_realized_trades": batch_metrics["realized_trades"],
            "is_unrealized_trades": batch_metrics["unrealized_trades"],
        }
    )


def compute_branch_metrics_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    is_start: str,
    is_end: str,
    rsi_periods: tuple[int, ...],
    rsi_thresholds: tuple[int, ...],
    rsi_wtypes: tuple[str, ...] = ("wilder",),
) -> pd.DataFrame:
    full_price = df.loc[:, ["High", "Low", "Close"]]
    price = full_price.loc[is_start:is_end]
    if price.empty:
        return pd.DataFrame()

    out_frames: list[pd.DataFrame] = []
    threshold_values = tuple(int(threshold) for threshold in rsi_thresholds)
    rsi_cache = TickerRSICache(ticker=ticker, close=full_price["Close"])
    for wtype in rsi_wtypes:
        if wtype != "wilder":
            raise ValueError(f"Unsupported rsi_wtype for reference path: {wtype}")
        for period in rsi_periods:
            rsi_full = rsi_cache.get_full_rsi(period=period, wtype=wtype)
            rsi = rsi_full.loc[price.index]
            out_frames.append(
                _build_period_metrics_frame(
                    ticker=ticker,
                    price=price,
                    rsi=rsi,
                    period=period,
                    wtype=wtype,
                    threshold_values=threshold_values,
                )
            )
    if not out_frames:
        return pd.DataFrame()
    return pd.concat(out_frames, ignore_index=True)


def compute_branch_metrics_for_ticker_across_windows(
    ticker: str,
    df: pd.DataFrame,
    windows: list[RollingWindow],
    rsi_periods: tuple[int, ...],
    rsi_thresholds: tuple[int, ...],
    rsi_wtypes: tuple[str, ...] = ("wilder",),
) -> dict[int, pd.DataFrame]:
    full_price = df.loc[:, ["High", "Low", "Close"]]
    threshold_values = tuple(int(threshold) for threshold in rsi_thresholds)
    price_by_oos_year = {
        window.oos_year: full_price.loc[window.is_start:window.is_end]
        for window in windows
    }
    out_frames_by_oos_year: dict[int, list[pd.DataFrame]] = {window.oos_year: [] for window in windows}
    rsi_cache = TickerRSICache(ticker=ticker, close=full_price["Close"])

    for wtype in rsi_wtypes:
        if wtype != "wilder":
            raise ValueError(f"Unsupported rsi_wtype for reference path: {wtype}")
        for period in rsi_periods:
            rsi_full = rsi_cache.get_full_rsi(period=period, wtype=wtype)
            for window in windows:
                price = price_by_oos_year[window.oos_year]
                if price.empty:
                    continue
                rsi = rsi_full.loc[price.index]
                out_frames_by_oos_year[window.oos_year].append(
                    _build_period_metrics_frame(
                        ticker=ticker,
                        price=price,
                        rsi=rsi,
                        period=period,
                        wtype=wtype,
                        threshold_values=threshold_values,
                    )
                )

    out: dict[int, pd.DataFrame] = {}
    for window in windows:
        frames = out_frames_by_oos_year[window.oos_year]
        out[window.oos_year] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out
