from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_sharpe(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    daily_returns = returns.dropna()
    if daily_returns.empty:
        return np.nan

    daily_excess = daily_returns - (risk_free_rate / annualization_factor)
    if daily_excess.std(ddof=0) == 0 or np.isnan(daily_excess.std(ddof=0)):
        return np.nan

    return np.sqrt(annualization_factor) * daily_excess.mean() / daily_excess.std(ddof=0)


def annualized_sortino(returns: pd.Series, risk_free_rate: float = 0.0, annualization_factor: int = 252) -> float:
    daily_returns = returns.dropna()
    if daily_returns.empty:
        return np.nan

    downside = daily_returns[daily_returns < 0]
    if downside.empty:
        return np.nan

    daily_excess = daily_returns - (risk_free_rate / annualization_factor)
    downside_vol = downside.std(ddof=0)
    if downside_vol == 0 or np.isnan(downside_vol):
        return np.nan

    return np.sqrt(annualization_factor) * daily_excess.mean() / downside_vol


def compute_position_attribution(position_snapshots: list[dict]) -> pd.DataFrame:
    if not position_snapshots:
        return pd.DataFrame(columns=["Company", "CumulativeContribution", "LastPositionValue", "LastWeight"])

    aggregated: dict[str, dict[str, float | str]] = {}

    for snapshot in position_snapshots:
        company = snapshot.get("Company")
        if not company:
            continue

        if company not in aggregated:
            aggregated[company] = {
                "Company": company,
                "CumulativeContribution": 0.0,
                "LastPositionValue": 0.0,
                "LastWeight": 0.0,
            }

        position_value = float(snapshot.get("PositionValue", 0.0) or 0.0)
        aggregated[company]["CumulativeContribution"] = float(aggregated[company]["CumulativeContribution"]) + position_value
        aggregated[company]["LastPositionValue"] = position_value
        aggregated[company]["LastWeight"] = float(snapshot.get("Weight", 0.0) or 0.0)

    attribution_df = pd.DataFrame(list(aggregated.values()))
    if attribution_df.empty:
        return attribution_df

    attribution_df = attribution_df.sort_values("CumulativeContribution", ascending=False)
    return attribution_df[["Company", "CumulativeContribution", "LastPositionValue", "LastWeight"]].reset_index(drop=True)


def compute_diagnostics(
    strategy_equity: pd.Series,
    benchmark_equity: pd.Series,
    trade_events: list,
    transaction_costs_paid: float,
    turnover_notional: float,
    invested_fraction_history: list[float],
    risk_free_rate: float = 0.0,
):
    if strategy_equity.empty:
        return {}

    strategy_returns = strategy_equity.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()

    years = (strategy_equity.index[-1] - strategy_equity.index[0]).days / 365.25

    total_return = strategy_equity.iloc[-1] / strategy_equity.iloc[0] - 1 if len(strategy_equity) > 0 else np.nan
    benchmark_return = benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1 if len(benchmark_equity) > 0 else np.nan
    excess_return = total_return - benchmark_return

    if years > 0:
        strategy_cagr = (strategy_equity.iloc[-1] / strategy_equity.iloc[0]) ** (1 / years) - 1
        benchmark_cagr = (benchmark_equity.iloc[-1] / benchmark_equity.iloc[0]) ** (1 / years) - 1
    else:
        strategy_cagr = np.nan
        benchmark_cagr = np.nan

    rolling_peak = strategy_equity.cummax()
    drawdown = (strategy_equity / rolling_peak) - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else np.nan

    annualized_vol = strategy_returns.std(ddof=0) * np.sqrt(252)
    sharpe = annualized_sharpe(strategy_returns, risk_free_rate=risk_free_rate)
    sortino = annualized_sortino(strategy_returns, risk_free_rate=risk_free_rate)

    average_equity = strategy_equity.mean()
    annual_turnover = turnover_notional / max(average_equity * years, 1e-9) if years > 0 else np.nan

    return {
        "strategy_return": float(total_return),
        "benchmark_return": float(benchmark_return),
        "excess_return": float(excess_return),
        "strategy_cagr": float(strategy_cagr),
        "benchmark_cagr": float(benchmark_cagr),
        "max_drawdown": float(max_drawdown),
        "annualized_volatility": float(annualized_vol),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "trades": len(trade_events),
        "transaction_costs_paid": float(transaction_costs_paid),
        "annual_turnover": float(annual_turnover),
        "percent_time_invested": float(np.nanmean(invested_fraction_history)) if len(invested_fraction_history) else 0.0,
        "yearly_returns": strategy_returns.groupby(strategy_returns.index.year).apply(lambda s: (s + 1).prod() - 1),
    }
