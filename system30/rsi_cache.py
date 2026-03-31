from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .rsi import compute_wilder_rsi


@dataclass(slots=True)
class TickerRSICache:
    ticker: str
    close: pd.Series
    _store: dict[tuple[int, str], pd.Series] = field(default_factory=dict, init=False)

    def get_full_rsi(self, period: int, wtype: str = "wilder") -> pd.Series:
        if wtype != "wilder":
            raise ValueError(f"Unsupported rsi_wtype: {wtype}")
        key = (int(period), str(wtype))
        cached = self._store.get(key)
        if cached is not None:
            return cached
        rsi = compute_wilder_rsi(self.close, period=int(period))
        self._store[key] = rsi
        return rsi

    def clear(self) -> None:
        self._store.clear()
