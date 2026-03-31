from __future__ import annotations

from pathlib import Path

import pandas as pd

from .branching import compute_branch_metrics_for_ticker, compute_branch_metrics_for_ticker_across_windows
from .config import System30Config
from .data import load_local_histories
from .execution import run_oos_portfolio
from .filters import passes_inception_filter, passes_liquidity_filter, passes_liquidity_filter_for_window
from .reports import build_period_rollups
from .rolling import build_rolling_windows
from .selector import select_one_branch_per_ticker
from .progress_log import ProgressLogger


class System30Engine:
    def __init__(self, cfg: System30Config):
        self.cfg = cfg
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_logger = ProgressLogger(
            path=self.cfg.progress_log_path,
            interval_seconds=self.cfg.progress_log_interval_seconds,
            enabled=self.cfg.progress_log_enabled,
        )
        self.progress_logger.reset(
            output_dir=self.cfg.output_dir,
            first_oos_year=self.cfg.first_oos_year,
            last_oos_year=self.cfg.last_oos_year,
            max_tickers=self.cfg.max_tickers,
        )

    def load_and_filter_universe(self) -> dict[str, pd.DataFrame]:
        self.progress_logger.log("load_and_filter_universe_start", force=True)
        histories = load_local_histories(self.cfg, progress_logger=self.progress_logger)
        kept: dict[str, pd.DataFrame] = {}
        rows = []
        total = len(histories)
        for idx, (ticker, df) in enumerate(histories.items(), start=1):
            inception_ok = passes_inception_filter(df, self.cfg.inception_start, self.cfg.inception_end)
            legacy_liquidity_ok = passes_liquidity_filter(df, self.cfg.min_avg_dollar_volume_90d)
            rows.append({
                'ticker': ticker,
                'rows': len(df),
                'first_date': df.index.min() if not df.empty else None,
                'last_date': df.index.max() if not df.empty else None,
                'passes_inception': inception_ok,
                'passes_liquidity_full_history_legacy': legacy_liquidity_ok,
                'kept_for_branch_generation': inception_ok,
            })
            if inception_ok:
                kept[ticker] = df
            self.progress_logger.maybe_log(
                "filter_universe_progress",
                idx=f"{idx}/{total}",
                ticker=ticker,
                kept=len(kept),
            )
        report_path = self.cfg.output_dir / 'universe_filter_report.csv'
        pd.DataFrame(rows).sort_values('ticker').to_csv(report_path, index=False)
        self.progress_logger.log("load_and_filter_universe_done", force=True, kept=len(kept), report_path=report_path)
        return kept

    def _passes_window_liquidity(self, df: pd.DataFrame, is_start: str, is_end: str) -> bool:
        return passes_liquidity_filter_for_window(
            df,
            start=is_start,
            end=is_end,
            min_avg_dollar_volume=self.cfg.min_avg_dollar_volume_90d,
        )

    def _filter_branch_metrics(self, metrics: pd.DataFrame, is_year: int, oos_year: int) -> pd.DataFrame:
        if metrics.empty:
            return metrics
        metrics = metrics.copy()
        metrics['is_year'] = is_year
        metrics['oos_year'] = oos_year
        metrics = metrics[
            (metrics['is_profit_pct'] > self.cfg.min_is_profit_pct)
            & (metrics['is_time_in_market_fraction'] >= self.cfg.min_is_time_in_market_fraction)
            & (metrics['is_max_drawdown_pct'] <= self.cfg.max_is_drawdown_pct)
        ].copy()
        return metrics

    def _finalize_offline_outputs(
        self,
        windows,
        yearly_metrics_by_oos_year: dict[int, list[pd.DataFrame]],
        ensemble_dir: Path,
    ) -> dict[str, Path]:
        all_selected = []
        all_metrics = []
        for window in windows:
            yearly_metrics = yearly_metrics_by_oos_year.get(window.oos_year, [])
            if not yearly_metrics:
                continue
            yearly_metrics_df = pd.concat(yearly_metrics, ignore_index=True)
            yearly_metrics_df['selection_rank_year'] = window.oos_year
            metrics_path = ensemble_dir / f'branch_metrics_is_{window.is_year}_for_oos_{window.oos_year}.csv'
            yearly_metrics_df.to_csv(metrics_path, index=False)
            selected = select_one_branch_per_ticker(yearly_metrics_df)
            selected['selected_for_oos_year'] = window.oos_year
            selected_path = ensemble_dir / f'selected_branches_for_oos_{window.oos_year}.csv'
            selected.to_csv(selected_path, index=False)
            self.progress_logger.log(
                'offline_window_done',
                force=True,
                is_year=window.is_year,
                oos_year=window.oos_year,
                branch_rows=len(yearly_metrics_df),
                selected_rows=len(selected),
                metrics_path=metrics_path,
                selected_path=selected_path,
            )
            all_metrics.append(yearly_metrics_df)
            all_selected.append(selected)

        outputs: dict[str, Path] = {}
        if all_metrics:
            metrics_df = pd.concat(all_metrics, ignore_index=True)
            metrics_path = self.cfg.output_dir / 'branch_metrics.csv'
            metrics_df.to_csv(metrics_path, index=False)
            outputs['branch_metrics'] = metrics_path
        if all_selected:
            selected_df = pd.concat(all_selected, ignore_index=True)
            selected_path = self.cfg.output_dir / 'selected_branches.csv'
            selected_df.to_csv(selected_path, index=False)
            outputs['selected_branches'] = selected_path
        return outputs

    def _run_offline_branch_ensemble_window_major(self, universe: dict[str, pd.DataFrame], windows, ensemble_dir: Path) -> dict[str, Path]:
        yearly_metrics_by_oos_year: dict[int, list[pd.DataFrame]] = {window.oos_year: [] for window in windows}
        for window in windows:
            self.progress_logger.log(
                'offline_window_start',
                force=True,
                is_year=window.is_year,
                oos_year=window.oos_year,
                universe_size=len(universe),
                mode='window_major',
            )
            total = len(universe)
            window_started = pd.Timestamp.now().timestamp()
            for idx, (ticker, df) in enumerate(universe.items(), start=1):
                if not self._passes_window_liquidity(df, window.is_start, window.is_end):
                    elapsed = pd.Timestamp.now().timestamp() - window_started
                    avg = elapsed / idx if idx else 0.0
                    eta = avg * (total - idx)
                    self.progress_logger.maybe_log(
                        'offline_window_progress',
                        is_year=window.is_year,
                        oos_year=window.oos_year,
                        idx=f'{idx}/{total}',
                        ticker=ticker,
                        kept_metric_frames=len(yearly_metrics_by_oos_year[window.oos_year]),
                        avg_sec_per_ticker=f'{avg:.3f}',
                        eta_sec=f'{eta:.1f}',
                        mode='window_major',
                    )
                    continue
                metrics = compute_branch_metrics_for_ticker(
                    ticker=ticker,
                    df=df,
                    is_start=window.is_start,
                    is_end=window.is_end,
                    rsi_periods=self.cfg.rsi_periods,
                    rsi_thresholds=self.cfg.rsi_thresholds,
                    rsi_wtypes=self.cfg.rsi_wtypes,
                )
                elapsed = pd.Timestamp.now().timestamp() - window_started
                avg = elapsed / idx if idx else 0.0
                eta = avg * (total - idx)
                self.progress_logger.maybe_log(
                    'offline_window_progress',
                    is_year=window.is_year,
                    oos_year=window.oos_year,
                    idx=f'{idx}/{total}',
                    ticker=ticker,
                    kept_metric_frames=len(yearly_metrics_by_oos_year[window.oos_year]),
                    avg_sec_per_ticker=f'{avg:.3f}',
                    eta_sec=f'{eta:.1f}',
                    mode='window_major',
                )
                filtered = self._filter_branch_metrics(metrics, is_year=window.is_year, oos_year=window.oos_year)
                if filtered.empty:
                    continue
                yearly_metrics_by_oos_year[window.oos_year].append(filtered)
        return self._finalize_offline_outputs(windows, yearly_metrics_by_oos_year, ensemble_dir)

    def _run_offline_branch_ensemble_ticker_major(self, universe: dict[str, pd.DataFrame], windows, ensemble_dir: Path) -> dict[str, Path]:
        yearly_metrics_by_oos_year: dict[int, list[pd.DataFrame]] = {window.oos_year: [] for window in windows}
        total = len(universe)
        started = pd.Timestamp.now().timestamp()
        self.progress_logger.log(
            'offline_multiwindow_start',
            force=True,
            mode='ticker_major_rsi_reuse',
            universe_size=total,
            window_count=len(windows),
        )
        for idx, (ticker, df) in enumerate(universe.items(), start=1):
            eligible_windows = [window for window in windows if self._passes_window_liquidity(df, window.is_start, window.is_end)]
            metrics_by_oos_year = compute_branch_metrics_for_ticker_across_windows(
                ticker=ticker,
                df=df,
                windows=eligible_windows,
                rsi_periods=self.cfg.rsi_periods,
                rsi_thresholds=self.cfg.rsi_thresholds,
                rsi_wtypes=self.cfg.rsi_wtypes,
            ) if eligible_windows else {}
            for window in eligible_windows:
                filtered = self._filter_branch_metrics(
                    metrics_by_oos_year.get(window.oos_year, pd.DataFrame()),
                    is_year=window.is_year,
                    oos_year=window.oos_year,
                )
                if filtered.empty:
                    continue
                yearly_metrics_by_oos_year[window.oos_year].append(filtered)
            elapsed = pd.Timestamp.now().timestamp() - started
            avg = elapsed / idx if idx else 0.0
            eta = avg * (total - idx)
            self.progress_logger.maybe_log(
                'offline_ticker_progress',
                idx=f'{idx}/{total}',
                ticker=ticker,
                avg_sec_per_ticker=f'{avg:.3f}',
                eta_sec=f'{eta:.1f}',
                mode='ticker_major_rsi_reuse',
            )
        return self._finalize_offline_outputs(windows, yearly_metrics_by_oos_year, ensemble_dir)

    def run_offline_branch_ensemble(self, universe: dict[str, pd.DataFrame] | None = None) -> dict[str, Path]:
        universe = universe or self.load_and_filter_universe()
        windows = build_rolling_windows(self.cfg.first_oos_year, self.cfg.last_oos_year)
        ensemble_dir = self.cfg.output_dir / 'offline_ensemble'
        ensemble_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.offline_reuse_rsi_across_windows:
            return self._run_offline_branch_ensemble_ticker_major(universe, windows, ensemble_dir)
        return self._run_offline_branch_ensemble_window_major(universe, windows, ensemble_dir)

    def run_oos_execution(self, universe: dict[str, pd.DataFrame] | None = None) -> dict[str, Path]:
        universe = universe or self.load_and_filter_universe()
        windows = build_rolling_windows(self.cfg.first_oos_year, self.cfg.last_oos_year)
        ensemble_dir = self.cfg.output_dir / 'offline_ensemble'
        oos_dir = self.cfg.output_dir / 'oos_execution'
        oos_dir.mkdir(parents=True, exist_ok=True)

        oos_outputs = []
        trade_summaries = []
        weekly_outputs = []
        monthly_outputs = []

        for window in windows:
            selected_path = ensemble_dir / f'selected_branches_for_oos_{window.oos_year}.csv'
            if not selected_path.exists():
                continue
            self.progress_logger.log('oos_window_start', force=True, oos_year=window.oos_year, selected_path=selected_path)
            selected = pd.read_csv(selected_path)
            oos_daily, trade_summary = run_oos_portfolio(
                progress_logger=self.progress_logger,
                progress_context={'oos_year': window.oos_year},
                universe=universe,
                selected_df=selected,
                oos_start=window.oos_start,
                oos_end=window.oos_end,
                daily_capital_base=self.cfg.daily_capital_base,
            )
            if oos_daily.empty:
                continue
            oos_daily = oos_daily.copy()
            oos_daily['oos_year'] = window.oos_year
            oos_daily = oos_daily.reset_index(names='Date')
            oos_daily.to_csv(oos_dir / f'oos_daily_{window.oos_year}.csv', index=False)
            weekly, monthly = build_period_rollups(oos_daily)
            if not weekly.empty:
                weekly['oos_year'] = window.oos_year
                weekly.to_csv(oos_dir / f'oos_weekly_{window.oos_year}.csv', index=False)
                weekly_outputs.append(weekly)
            if not monthly.empty:
                monthly['oos_year'] = window.oos_year
                monthly.to_csv(oos_dir / f'oos_monthly_{window.oos_year}.csv', index=False)
                monthly_outputs.append(monthly)
            if not trade_summary.empty:
                trade_summary['oos_year'] = window.oos_year
                trade_summary.to_csv(oos_dir / f'trade_summary_{window.oos_year}.csv', index=False)
                trade_summaries.append(trade_summary)
            self.progress_logger.log(
                'oos_window_done',
                force=True,
                oos_year=window.oos_year,
                daily_rows=len(oos_daily),
                trade_rows=len(trade_summary),
            )
            oos_outputs.append(oos_daily)

        outputs: dict[str, Path] = {}
        if oos_outputs:
            all_oos = pd.concat(oos_outputs, ignore_index=True)
            oos_path = self.cfg.output_dir / 'oos_daily_combined.csv'
            all_oos.to_csv(oos_path, index=False)
            outputs['oos_daily_combined'] = oos_path
        if trade_summaries:
            trades_path = self.cfg.output_dir / 'oos_trade_summary_combined.csv'
            pd.concat(trade_summaries, ignore_index=True).to_csv(trades_path, index=False)
            outputs['oos_trade_summary_combined'] = trades_path
        if weekly_outputs:
            weekly_path = self.cfg.output_dir / 'oos_weekly_combined.csv'
            pd.concat(weekly_outputs, ignore_index=True).to_csv(weekly_path, index=False)
            outputs['oos_weekly_combined'] = weekly_path
        if monthly_outputs:
            monthly_path = self.cfg.output_dir / 'oos_monthly_combined.csv'
            pd.concat(monthly_outputs, ignore_index=True).to_csv(monthly_path, index=False)
            outputs['oos_monthly_combined'] = monthly_path
        return outputs

    def run(self) -> dict[str, Path]:
        self.progress_logger.log('full_run_start', force=True)
        universe = self.load_and_filter_universe()
        outputs = {}
        outputs.update(self.run_offline_branch_ensemble(universe=universe))
        outputs.update(self.run_oos_execution(universe=universe))
        self.progress_logger.log('full_run_done', force=True, output_keys=','.join(sorted(outputs.keys())))
        return outputs
