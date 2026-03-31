from __future__ import annotations

import pandas as pd


def passes_inception_filter(price_df: pd.DataFrame, start: str, end: str) -> bool:
    if price_df.empty:
        return False
    first_date = pd.Timestamp(price_df.index.min())
    return pd.Timestamp(start) <= first_date <= pd.Timestamp(end)


def average_dollar_volume_for_window(price_df: pd.DataFrame, start: str, end: str) -> float | None:
    if price_df.empty:
        return None
    window_df = price_df.loc[start:end]
    if window_df.empty:
        return None
    dollar_volume = window_df["Close"] * window_df["Volume"]
    valid = dollar_volume.dropna()
    if valid.empty:
        return None
    return float(valid.mean())


def passes_liquidity_filter_for_window(
    price_df: pd.DataFrame,
    start: str,
    end: str,
    min_avg_dollar_volume: float,
) -> bool:
    avg_dollar_volume = average_dollar_volume_for_window(price_df, start=start, end=end)
    return avg_dollar_volume is not None and avg_dollar_volume >= min_avg_dollar_volume


def trailing_avg_dollar_volume_90d(price_df: pd.DataFrame, asof: str) -> float | None:
    """Diagnostic helper kept for audits/back-compat.

    Active selection logic now uses average dollar volume over the full IS window via
    passes_liquidity_filter_for_window(...), not a trailing 90D snapshot.
    """
    if price_df.empty:
        return None
    hist = price_df.loc[:asof]
    if len(hist) < 90:
        return None
    dollar_volume = hist["Close"] * hist["Volume"]
    adv_90 = dollar_volume.rolling(90, min_periods=90).mean()
    valid = adv_90.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def passes_liquidity_filter_asof(
    price_df: pd.DataFrame,
    asof: str,
    min_avg_dollar_volume_90d: float,
) -> bool:
    adv_90 = trailing_avg_dollar_volume_90d(price_df, asof=asof)
    return adv_90 is not None and adv_90 >= min_avg_dollar_volume_90d


def passes_liquidity_filter(price_df: pd.DataFrame, min_avg_dollar_volume_90d: float) -> bool:
    """Legacy full-history liquidity check kept for diagnostics/back-compat."""
    if price_df.empty or len(price_df) < 90:
        return False
    dollar_volume = price_df["Close"] * price_df["Volume"]
    adv_90 = dollar_volume.rolling(90, min_periods=90).mean()
    valid = adv_90.dropna()
    if valid.empty:
        return False
    return bool((valid >= min_avg_dollar_volume_90d).all())


def compute_time_in_market_fraction(position_mask: pd.Series) -> float:
    if len(position_mask) == 0:
        return 0.0
    return float(position_mask.astype(bool).mean())
