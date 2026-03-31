from __future__ import annotations

import pandas as pd

from .progress_log import ProgressLogger
from .simulation import build_branch_state


def run_oos_portfolio(
    universe: dict[str, pd.DataFrame],
    selected_df: pd.DataFrame,
    oos_start: str,
    oos_end: str,
    daily_capital_base: float,
    progress_logger: ProgressLogger | None = None,
    progress_context: dict[str, object] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    trade_rows = []
    total = len(selected_df)
    started = pd.Timestamp.now().timestamp()
    context = progress_context or {}
    for idx, row in enumerate(selected_df.itertuples(index=False), start=1):
        ticker = row.ticker
        state = build_branch_state(
            df=universe[ticker],
            start=oos_start,
            end=oos_end,
            rsi_period=int(row.rsi_period),
            rsi_threshold=int(row.rsi_threshold),
            rsi_wtype=str(row.rsi_wtype),
        )
        elapsed = pd.Timestamp.now().timestamp() - started
        avg = elapsed / idx if idx else 0.0
        eta = avg * (total - idx)
        if progress_logger is not None:
            progress_logger.maybe_log(
                'oos_portfolio_progress',
                idx=f'{idx}/{total}',
                ticker=ticker,
                avg_sec_per_branch=f'{avg:.3f}',
                eta_sec=f'{eta:.1f}',
                **context,
            )
        if state.empty:
            continue
        one = pd.DataFrame(index=state.index)
        one[f"{ticker}_active"] = state["active_at_open"].astype(bool)
        one[f"{ticker}_ret"] = state["asset_return"].astype(float)
        one[f"{ticker}_entry"] = state["entry_executed"].astype(bool)
        one[f"{ticker}_exit"] = state["exit_executed"].astype(bool)
        frames.append(one)

        trade_rows.append(
            {
                'ticker': ticker,
                'branch_id': row.branch_id,
                'oos_start': oos_start,
                'oos_end': oos_end,
                'realized_trades': int(state['exit_executed'].sum()),
                'unrealized_trades': int(state['active_after_close'].iloc[-1]),
                'days_active': int(state['active_at_open'].sum()),
                'entries': int(state['entry_executed'].sum()),
                'exits': int(state['exit_executed'].sum()),
            }
        )

    if not frames:
        return pd.DataFrame(), pd.DataFrame(trade_rows)

    panel = pd.concat(frames, axis=1).sort_index()
    active_cols = [c for c in panel.columns if c.endswith('_active')]
    ret_cols = [c for c in panel.columns if c.endswith('_ret')]
    entry_cols = [c for c in panel.columns if c.endswith('_entry')]
    exit_cols = [c for c in panel.columns if c.endswith('_exit')]

    active_mask = panel[active_cols].astype('boolean').fillna(False).astype(bool)
    active_count = active_mask.sum(axis=1)

    renamed_rets = panel[ret_cols].copy()
    renamed_rets.columns = [c[:-4] + '_active' for c in ret_cols]
    aligned_rets = renamed_rets.reindex(columns=active_cols).astype(float).fillna(0.0)
    weighted_rets = aligned_rets.where(active_mask, 0.0)
    daily_return = weighted_rets.sum(axis=1) / active_count.where(active_count > 0, 1)
    daily_return = daily_return.where(active_count > 0, 0.0)

    entries = panel[entry_cols].astype('boolean').fillna(False).astype(bool).sum(axis=1).astype(int)
    exits = panel[exit_cols].astype('boolean').fillna(False).astype(bool).sum(axis=1).astype(int)

    out = pd.DataFrame(index=panel.index)
    out['active_positions'] = active_count.astype(int)
    out['entries'] = entries
    out['exits'] = exits
    out['realized_positions_closed'] = exits
    out['unrealized_open_positions'] = active_count.astype(int)
    out['daily_return'] = daily_return.astype(float)
    out['daily_pnl'] = out['daily_return'] * float(daily_capital_base)
    out['equity_non_compounding'] = float(daily_capital_base) + out['daily_pnl'].cumsum()
    out['daily_mismatch'] = 0.0
    out['daily_slippage'] = 0.0
    return out, pd.DataFrame(trade_rows)
