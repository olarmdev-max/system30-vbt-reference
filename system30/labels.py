from __future__ import annotations


def branch_id(ticker: str, rsi_period: int, threshold: int, side: str = "L") -> str:
    return f"{rsi_period}d RSI {ticker} LT{threshold} - {side} {ticker}"
