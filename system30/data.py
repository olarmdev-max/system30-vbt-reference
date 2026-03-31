from __future__ import annotations

from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from .config import System30Config
from .progress_log import ProgressLogger


def load_universe_from_csv(path: Path) -> list[str]:
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if 'Ticker(2926)' in line:
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find 'Ticker(2926)' in {path}")
    df = pd.read_csv(path, skiprows=header_idx)
    ticker_col = None
    for col in df.columns:
        if str(col).strip() == 'Ticker(2926)':
            ticker_col = col
            break
    if ticker_col is None:
        raise ValueError(f"Could not parse 'Ticker(2926)' in {path}")
    tickers = (
        df[ticker_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )
    tickers = [t for t in tickers if t and t not in {'NAN', 'NONE', 'TICKER(2926)'}]
    return list(dict.fromkeys(tickers))


def _read_lean_zip_csv(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        name = zf.namelist()[0]
        text = zf.read(name).decode('utf-8')
    df = pd.read_csv(
        StringIO(text),
        header=None,
        names=['DateTime', 'Open', 'High', 'Low', 'Close', 'Volume'],
    )
    df['Date'] = pd.to_datetime(df['DateTime'].str.split().str[0], format='%Y%m%d')
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce') / 10000.0
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    df = df.drop(columns=['DateTime']).set_index('Date').sort_index()
    return df


def load_local_ticker_history(ticker: str, daily_dir: Path) -> pd.DataFrame:
    path = daily_dir / f"{ticker.lower()}.zip"
    if not path.exists():
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    return _read_lean_zip_csv(path)


def load_local_histories(cfg: System30Config, progress_logger: ProgressLogger | None = None) -> dict[str, pd.DataFrame]:
    tickers = list(cfg.debug_tickers) if cfg.debug_tickers else load_universe_from_csv(cfg.universe_csv)
    if cfg.max_tickers is not None:
        tickers = tickers[: cfg.max_tickers]
    out: dict[str, pd.DataFrame] = {}
    total = len(tickers)
    started = pd.Timestamp.now().timestamp()
    for idx, ticker in enumerate(tickers, start=1):
        df = load_local_ticker_history(ticker, cfg.local_daily_dir)
        if not df.empty:
            out[ticker] = df
        if progress_logger is not None:
            elapsed = pd.Timestamp.now().timestamp() - started
            avg = elapsed / idx if idx else 0.0
            eta = avg * (total - idx)
            progress_logger.maybe_log(
                'load_histories_progress',
                idx=f'{idx}/{total}',
                ticker=ticker,
                loaded=len(out),
                avg_sec_per_ticker=f'{avg:.3f}',
                eta_sec=f'{eta:.1f}',
            )
    return out
