from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class StrategyConfig:
    entry_top_n: int = 10
    keep_top_n: int = 25
    rebalance_frequency: str = "Weekly"
    w6m: float = 0.50
    w3m: float = 0.30
    w12m: float = 0.20
    volatility_adjusted: bool = True
    vol_lookback: int = 60
    stale_tolerance_days: int = 1
    universe_size: Optional[int] = 80
    include_former_constituents: bool = False
    signal_fallback_sessions: int = 1
    valuation_fallback_sessions: int = 3
    pending_order_max_sessions: int = 5

    def momentum_weights(self) -> dict[str, float]:
        total = self.w6m + self.w3m + self.w12m
        if total <= 0:
            raise ValueError("Momentum weights must sum to a positive value.")
        return {
            "w6m": self.w6m / total,
            "w3m": self.w3m / total,
            "w12m": self.w12m / total,
        }


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10_000.0
    transaction_cost: float = 0.0
    risk_free_rate: float = 0.0
    analysis_start_date: Optional[str] = None
    analysis_end_date: Optional[str] = None
    warmup_days: int = 252
