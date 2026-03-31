# System30 post-processing benchmark harness

Local benchmark harness for the 18-branch C5 2024 IS golden set:

- AGG 3d LT11
- BND 3d LT10
- DIA 6d LT21
- EEM 14d LT34
- EFA 8d LT26
- GLD 21d LT46
- HYG 18d LT38
- IEF 3d LT10
- IVV 5d LT20
- LQD 4d LT13
- QQQ 7d LT29
- SLV 4d LT11
- SPY 5d LT20
- TLT 3d LT10
- VOO 5d LT20
- VTI 4d LT15
- XLE 19d LT37
- XLV 14d LT30

## What it benchmarks

Canonical comparison target:

- `build_branch_state(...) + compute_branch_metrics(...)` for end-to-end canonical timing
- `compute_branch_metrics(state)` for canonical metrics-only timing

Five post-processing variants are compared against the canonical custom metric result for each branch:

1. `fast_python_arrays_exact`
   - existing exact array path via `compute_fast_metrics_from_signals`
2. `numba_exact_marks`
   - exact numba replica with explicit equity-mark array
3. `numba_exact_compact`
   - exact numba replica with streaming peak/DD updates
4. `state_mask_close_only_approx`
   - approximate close-to-close active-mask path
5. `trade_span_inclusive_approx`
   - approximate trade-span/TIM path using closed-trade equity only

Accuracy deltas are emitted for:

- profit
- TIM
- max DD
- total trades

## Run

```bash
cd /path/to/system30-vbt-reference
python system30_postprocess_benchmark.py
```

Useful knobs:

```bash
python system30_postprocess_benchmark.py --warmups 3 --runs 25
python system30_postprocess_benchmark.py --start 2023-01-01 --end 2023-12-31
```

## Outputs

Written under:

- `system30_output/benchmarks/system30_postprocess_benchmark_per_branch.csv`
- `system30_output/benchmarks/system30_postprocess_benchmark_summary.csv`
- `system30_output/benchmarks/system30_postprocess_canonical_metrics_only_timing.csv`
- `system30_output/benchmarks/system30_postprocess_canonical_e2e_timing.csv`
- `system30_output/benchmarks/system30_postprocess_benchmark_report.md`

## Current quick-check result

Using `--warmups 2 --runs 8`:

- `numba_exact_compact` was the fastest exact path
- `numba_exact_marks` was a close second
- both exact numba paths matched canonical metrics with zero observed deltas on all 18 branches
- `fast_python_arrays_exact` was exact but much slower than numba
- the two approximate paths were faster than the Python exact path but drifted on DD, and one also drifted on profit

## Constraints / notes

- Uses local adjusted LEAN/Tiingo daily data from `system30.config.System30Config.local_daily_dir`
- Assumes RSI type = `wilder`
- No `talib` dependency
- Markdown report generation is self-contained and does not require `tabulate`
- This is benchmark-oriented research code, not productionized API surface
