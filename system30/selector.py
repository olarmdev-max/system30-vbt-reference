from __future__ import annotations

import pandas as pd


def select_one_branch_per_ticker(passing_df: pd.DataFrame) -> pd.DataFrame:
    """Apply System30 tie-break logic exactly.

    Order:
    1) lowest IS max drawdown
    2) highest IS profit
    3) highest IS total trades
    4) alphabetical branch id
    """
    if passing_df.empty:
        return passing_df.copy()

    required = {
        "ticker",
        "branch_id",
        "is_max_drawdown_pct",
        "is_profit_pct",
        "is_total_trades",
    }
    missing = required - set(passing_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for selection: {sorted(missing)}")

    ranked = passing_df.sort_values(
        by=[
            "ticker",
            "is_max_drawdown_pct",
            "is_profit_pct",
            "is_total_trades",
            "branch_id",
        ],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    return ranked.groupby("ticker", as_index=False, sort=False).head(1).reset_index(drop=True)
