from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RollingWindow:
    is_year: int
    oos_year: int

    @property
    def is_start(self) -> str:
        return f"{self.is_year}-01-01"

    @property
    def is_end(self) -> str:
        return f"{self.is_year}-12-31"

    @property
    def oos_start(self) -> str:
        return f"{self.oos_year}-01-01"

    @property
    def oos_end(self) -> str:
        return f"{self.oos_year}-12-31"


def build_rolling_windows(first_oos_year: int, last_oos_year: int) -> list[RollingWindow]:
    return [RollingWindow(is_year=year - 1, oos_year=year) for year in range(first_oos_year, last_oos_year + 1)]
