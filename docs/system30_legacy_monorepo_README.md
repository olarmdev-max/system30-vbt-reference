# System30 Prototype

Current implementation status:

- **Canonical data source:** local Tiingo-derived adjusted daily ETF dataset in LEAN format
- **Canonical adjusted bars path:**
  - `/home/hassanchaudhary/.openclaw/workspace/quant-multicharts/research/LEAN/data/equity/usa/daily/`
- **Shared raw Tiingo archive:**
  - `/home/hassanchaudhary/.openclaw/workspace/data/tiingo/raw/equity/usa/daily/`
- **Universe file:**
  - `/home/hassanchaudhary/.openclaw/workspace/quant-multicharts/research/LEAN/all_2926_etfs_gid1128886323_raw.csv`

## Important

System30 should use the **local Tiingo dataset**, not Alpaca.

- Use adjusted Tiingo-derived daily bars for backtest math
- Keep shared raw Tiingo files for reuse, audits, and rebuilds
- Alpaca is not part of the active implementation path

## Execution timing note

For System30 in this subtree, entries are modeled as **same-bar close executions**:

- RSI is computed on the adjusted daily **close**
- if the RSI condition is true on day `t`, the buy is assumed to happen at **day `t` close**
- the first P&L contribution therefore begins on the next bar's close-to-close move
- exits are also evaluated on close using `Close[t] > High[t-1]` and are closed on that same close

This is intentionally a little different from the more conservative "signal on close, trade next bar" convention, and it is documented here because the Google Sheet appears to use trade-on-close behavior.

## Current production sweep constants

The production runner now uses this L1 RSI sweep:

- RSI periods: `3..21`
- operator: `LT`
- thresholds: `10..70`
- RSI type: `wilder`

So the branch factory is no longer using the earlier toy subset like `(3, 5, 10, 20)` and `5..40 step 5` unless we intentionally override it for debugging.

## Run

### Full two-pass run

```bash
cd /home/hassanchaudhary/.openclaw/workspace/quant-multicharts/research/vbt-projects
.venv/bin/python system30_run.py
```

### Offline branch ensemble only

```bash
.venv/bin/python system30_offline.py
```

### OOS execution only

```bash
.venv/bin/python system30_oos.py
```

## Output structure

- `system30_output/offline_ensemble/`
  - per-year IS branch metrics
  - per-year frozen selected branches for the next OOS year
- `system30_output/oos_execution/`
  - per-year OOS daily results
  - per-year weekly rollups
  - per-year monthly rollups
  - per-year trade summaries
- combined files at `system30_output/`
  - `branch_metrics.csv`
  - `selected_branches.csv`
  - `oos_daily_combined.csv`
  - `oos_weekly_combined.csv`
  - `oos_monthly_combined.csv`
  - `oos_trade_summary_combined.csv`

## Current prototype scope

This is a first executable prototype that currently covers:

- universe loading
- inception + liquidity filters
- RSI branch generation
- exact branch tie-break selection
- rolling IS prior-year / OOS current-year orchestration
- **two-pass structure:**
  1. offline branch ensemble / culling pass
  2. separate OOS execution pass using frozen yearly winners
- OOS daily equal-weight active-position aggregation with a fixed daily capital base
- daily / weekly / monthly OOS rollups
- realized / unrealized trade summary per ticker-year

It does **not** yet claim full historical parity with the original System30 implementation.
