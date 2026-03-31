from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "daily"
DEFAULT_UNIVERSE_CSV = REPO_ROOT / "data" / "universe.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "system30_output"


@dataclass(slots=True)
class System30Config:
    # Data source
    data_source: str = "tiingo_local"
    local_daily_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    universe_csv: Path = field(default_factory=lambda: DEFAULT_UNIVERSE_CSV)

    # Output
    output_dir: Path = field(default_factory=lambda: DEFAULT_OUTPUT_DIR)

    # Universe filters
    inception_start: str = "1993-01-01"
    inception_end: str = "2018-12-31"
    # Threshold is applied to average dollar volume over the full IS year.
    min_avg_dollar_volume_90d: float = 100_000.0

    # Branch generation
    rsi_periods: tuple[int, ...] = tuple(range(3, 22))
    rsi_thresholds: tuple[int, ...] = tuple(range(10, 71))
    rsi_wtypes: tuple[str, ...] = ("wilder",)

    # Rolling IS/OOS
    first_oos_year: int = 2020
    last_oos_year: int = 2025

    # Filter 3
    min_is_profit_pct: float = 0.0
    min_is_time_in_market_fraction: float = 0.05
    max_is_drawdown_pct: float = 0.15

    # Portfolio logic
    daily_capital_base: float = 100_000.0
    non_compounding: bool = True

    # Runtime
    max_tickers: int | None = None
    save_branch_metrics: bool = True
    save_daily_panels: bool = True
    verbose: bool = True

    # In-memory optimization toggles
    offline_reuse_rsi_across_windows: bool = True

    # Progress logging
    progress_log_enabled: bool = True
    progress_log_path: Path | None = None
    progress_log_interval_seconds: float = 30.0

    # Research shortcuts
    debug_tickers: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        self.local_daily_dir = Path(self.local_daily_dir)
        self.universe_csv = Path(self.universe_csv)
        self.output_dir = Path(self.output_dir)
        self.progress_log_path = Path(self.progress_log_path) if self.progress_log_path is not None else self.output_dir / "system30_progress.log"
