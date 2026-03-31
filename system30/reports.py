from __future__ import annotations

import pandas as pd


def build_period_rollups(oos_daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if oos_daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    df = oos_daily.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')

    weekly = df.resample('W-FRI').agg(
        active_positions_mean=('active_positions', 'mean'),
        entries=('entries', 'sum'),
        exits=('exits', 'sum'),
        daily_pnl=('daily_pnl', 'sum'),
        daily_return=('daily_return', 'sum'),
        realized_closes=('realized_positions_closed', 'sum') if 'realized_positions_closed' in df.columns else ('exits', 'sum'),
        unrealized_open_positions=('unrealized_open_positions', 'last') if 'unrealized_open_positions' in df.columns else ('active_positions', 'last'),
    ).reset_index()

    monthly = df.resample('ME').agg(
        active_positions_mean=('active_positions', 'mean'),
        entries=('entries', 'sum'),
        exits=('exits', 'sum'),
        daily_pnl=('daily_pnl', 'sum'),
        daily_return=('daily_return', 'sum'),
        realized_closes=('realized_positions_closed', 'sum') if 'realized_positions_closed' in df.columns else ('exits', 'sum'),
        unrealized_open_positions=('unrealized_open_positions', 'last') if 'unrealized_open_positions' in df.columns else ('active_positions', 'last'),
    ).reset_index()

    return weekly, monthly
