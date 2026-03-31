from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi_from_averages(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0.0:
        if avg_gain == 0.0:
            return 50.0
        return 100.0
    if avg_gain == 0.0:
        return 0.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    if period <= 0:
        raise ValueError("period must be positive")

    close = pd.to_numeric(close, errors="coerce").astype(float)
    values = close.to_numpy(dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) <= period:
        return pd.Series(out, index=close.index, name="RSI")

    deltas = np.diff(values)
    gains = np.where(deltas > 0.0, deltas, 0.0)
    losses = np.where(deltas < 0.0, -deltas, 0.0)

    seed_gains = gains[:period]
    seed_losses = losses[:period]
    if np.isnan(seed_gains).any() or np.isnan(seed_losses).any():
        return pd.Series(out, index=close.index, name="RSI")

    avg_gain = float(seed_gains.mean())
    avg_loss = float(seed_losses.mean())
    out[period] = _rsi_from_averages(avg_gain, avg_loss)

    for idx in range(period + 1, len(values)):
        gain = gains[idx - 1]
        loss = losses[idx - 1]
        if np.isnan(gain) or np.isnan(loss):
            avg_gain = np.nan
            avg_loss = np.nan
            out[idx] = np.nan
            continue
        if np.isnan(avg_gain) or np.isnan(avg_loss):
            out[idx] = np.nan
            continue
        avg_gain = ((avg_gain * (period - 1)) + gain) / period
        avg_loss = ((avg_loss * (period - 1)) + loss) / period
        out[idx] = _rsi_from_averages(avg_gain, avg_loss)

    return pd.Series(out, index=close.index, name="RSI")
