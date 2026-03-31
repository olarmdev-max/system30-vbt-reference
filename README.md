# System30 VBT Reference

A cleaned shareable copy of the System30 yearly-rolling branch-selection prototype.

This repo contains the core backtest code only:
- branch generation
- in-sample filtering
- one-branch-per-ticker selection
- yearly rolling OOS execution
- simple compare script for checking exact sheet branches against local results

It does **not** include:
- market data
- workbook exports
- local outputs
- private research notes
- environment-specific paths

## Strategy shape

Current reference behavior in this share repo:
- adjusted daily bars in LEAN-style ZIP files
- RSI family: Wilder RSI
- branch family: `Xd RSI TICKER LTy - L TICKER`
- entry: same-bar close when `RSI < threshold`
- exit: same-bar close when `Close[t] > High[t-1]`
- slippage: 2 bps per side
- yearly rolling process: IS year = `N-1`, OOS year = `N`
- branch filter 3:
  - IS profit > 0
  - IS time in market >= 5%
  - IS max drawdown <= 15%
- ticker selector tie-break order:
  1. lowest IS max drawdown
  2. highest IS profit
  3. highest IS total trades
  4. alphabetical branch id
- liquidity filter:
  - average **dollar volume** over the full IS year
  - computed as `Close * Volume`
  - threshold default: `$100,000`

## Expected inputs

### 1) Daily bars directory
Pass a directory containing LEAN-style ETF ZIP files such as:

- `spy.zip`
- `qqq.zip`
- `gld.zip`

Each ZIP should contain one CSV-like file with rows in this format:

```text
YYYYMMDD 00:00,open,high,low,close,volume
```

Prices are expected to be stored in LEAN's scaled integer format and are divided by `10000` on load.

### 2) Universe CSV
Pass a CSV containing a `Ticker(2926)` column, matching the format used in the original project universe file.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

### Full run

```bash
python system30_run.py \
  --daily-dir /path/to/daily \
  --universe-csv /path/to/universe.csv \
  --output-dir ./system30_output \
  --first-oos-year 2020 \
  --last-oos-year 2026
```

### Offline only

```bash
python system30_offline.py \
  --daily-dir /path/to/daily \
  --universe-csv /path/to/universe.csv \
  --output-dir ./system30_output
```

### OOS only

```bash
python system30_oos.py \
  --daily-dir /path/to/daily \
  --universe-csv /path/to/universe.csv \
  --output-dir ./system30_output
```

## Compare against a workbook branch list

```bash
python system30_compare_c5_canonical.py \
  --sheet-csv /path/to/C5_or_C7.csv \
  --daily-dir /path/to/daily \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --out-csv ./system30_output/branch_compare.csv
```

If `--branch` is omitted, the compare script evaluates every branch listed in the sheet CSV.

## Output files

Main outputs are written under the chosen output directory:
- `offline_ensemble/`
- `oos_execution/`
- `branch_metrics.csv`
- `selected_branches.csv`
- `oos_daily_combined.csv`
- `oos_trade_summary_combined.csv`
- `oos_weekly_combined.csv`
- `oos_monthly_combined.csv`
- `system30_progress.log`

## Notes

This repo is intended as a readable reference implementation for debugging and comparison. It is not presented as the original production System30 codebase.
