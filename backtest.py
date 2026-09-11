from __future__ import annotations

import pandas as pd

from data import get_last_observation
from metrics import compute_diagnostics
from portfolio import PortfolioState, execute_rebalance, portfolio_value_at_date
from strategy import rank_universe


def generate_rebalance_dates(benchmark_dates, rebalance_frequency):
    dates = []

    if rebalance_frequency == "Weekly":
        week_keys = {}
        for dt in benchmark_dates:
            iso_year, iso_week, _ = dt.isocalendar()
            key = (iso_year, iso_week)
            week_keys[key] = dt
        dates = list(week_keys.values())
    else:
        month_keys = {}
        for dt in benchmark_dates:
            key = (dt.year, dt.month)
            month_keys[key] = dt
        dates = list(month_keys.values())

    return sorted(dates)


def backtest_momentum_selection(
    all_stock_data,
    benchmark_data,
    strategy_config,
    backtest_config,
    analysis_start_date=None,
    analysis_end_date=None,
):
    if benchmark_data is None or benchmark_data.empty:
        return None

    benchmark_dates = benchmark_data.index
    start_date = analysis_start_date or benchmark_dates[0]
    end_date = analysis_end_date or benchmark_dates[-1]

    analysis_dates = benchmark_data.loc[start_date:end_date].index
    if len(analysis_dates) < 2:
        return None

    strategy_values = pd.Series(index=analysis_dates, dtype=float)
    benchmark_values = pd.Series(index=analysis_dates, dtype=float)
    strategy_values.iloc[0] = backtest_config.initial_capital
    benchmark_values.iloc[0] = backtest_config.initial_capital

    portfolio = PortfolioState(cash=float(backtest_config.initial_capital), positions={})
    ranking_history = []
    trade_events = []
    holdings_history = []
    eligible_counts = []
    invested_fraction_history = []
    daily_audit = []
    position_snapshots = []
    transaction_costs_paid = 0.0
    turnover_notional = 0.0
    quality_issues = []

    rebalance_dates = generate_rebalance_dates(analysis_dates, strategy_config.rebalance_frequency)
    pending_target_holdings = []
    pending_order_age = 0

    for idx in range(1, len(analysis_dates)):
        current_date = analysis_dates[idx]
        prev_date = analysis_dates[idx - 1]

        if pending_target_holdings:
            portfolio, new_events, extra_costs, extra_turnover, order_completed = execute_rebalance(
                portfolio=portfolio,
                current_date=current_date,
                target_holdings=pending_target_holdings,
                all_stock_data=all_stock_data,
                transaction_cost=backtest_config.transaction_cost,
            )
            trade_events.extend(new_events)
            transaction_costs_paid += extra_costs
            turnover_notional += extra_turnover

            if order_completed:
                pending_target_holdings = []
                pending_order_age = 0
            else:
                pending_order_age += 1
                if pending_order_age > strategy_config.pending_order_max_sessions:
                    quality_issues.append(
                        {
                            "Date": current_date,
                            "Company": "multiple",
                            "Issue": "pending_order_cancelled",
                            "PendingTargets": pending_target_holdings,
                            "AgeSessions": pending_order_age,
                        }
                    )
                    pending_target_holdings = []
                    pending_order_age = 0

        _, prev_close = get_last_observation(benchmark_data, prev_date, "Close")
        _, curr_close = get_last_observation(benchmark_data, current_date, "Close")
        if prev_close is None or curr_close is None:
            benchmark_values.iloc[idx] = benchmark_values.iloc[idx - 1]
        else:
            benchmark_values.iloc[idx] = benchmark_values.iloc[idx - 1] * (curr_close / prev_close)

        strategy_values.iloc[idx] = portfolio_value_at_date(
            portfolio=portfolio,
            current_date=current_date,
            all_stock_data=all_stock_data,
            valuation_fallback_sessions=strategy_config.valuation_fallback_sessions,
        )

        invested_value = portfolio_value_at_date(
            portfolio=PortfolioState(cash=0.0, positions=portfolio.positions),
            current_date=current_date,
            all_stock_data=all_stock_data,
            valuation_fallback_sessions=strategy_config.valuation_fallback_sessions,
        )
        total_value = strategy_values.iloc[idx]
        invested_fraction = invested_value / total_value if total_value > 0 else 0.0
        invested_fraction_history.append(invested_fraction)

        snapshot_rows = []
        for company, shares in portfolio.positions.items():
            stock_data = all_stock_data.get(company)
            if stock_data is None:
                continue

            last_date, last_price = get_last_observation(stock_data, current_date, "Close")
            if last_price is None or last_price <= 0:
                continue

            position_value = shares * last_price
            snapshot_rows.append(
                {
                    "Date": current_date,
                    "Company": company,
                    "Shares": shares,
                    "Price": float(last_price),
                    "PositionValue": float(position_value),
                    "Weight": float(position_value / total_value) if total_value > 0 else 0.0,
                }
            )

        daily_audit.append(
            {
                "Date": current_date,
                "Cash": float(portfolio.cash),
                "PortfolioValue": float(strategy_values.iloc[idx]),
                "InvestedValue": float(invested_value),
                "InvestedFraction": float(invested_fraction),
                "Holdings": sorted(list(portfolio.positions.keys())),
                "PositionCount": len(portfolio.positions),
                "PositionSnapshots": snapshot_rows,
            }
        )
        position_snapshots.extend(snapshot_rows)

        if current_date in rebalance_dates and not pending_target_holdings:
            ranked, quality_issues = rank_universe(
                current_date=current_date,
                all_stock_data=all_stock_data,
                strategy_config=strategy_config,
                all_issues=quality_issues,
            )

            if ranked:
                eligible_counts.append(len(ranked))
                rank_lookup = {row["Company"]: row["Rank"] for row in ranked}

                if not portfolio.positions:
                    target_holdings = [row["Company"] for row in ranked[:strategy_config.entry_top_n]]
                else:
                    keep_holdings = [company for company in portfolio.positions if rank_lookup.get(company, strategy_config.keep_top_n + 1) <= strategy_config.keep_top_n]
                    target_holdings = keep_holdings[:]
                    held_set = set(target_holdings)

                    for row in ranked:
                        if len(target_holdings) >= strategy_config.entry_top_n:
                            break
                        if row["Company"] not in held_set:
                            target_holdings.append(row["Company"])
                            held_set.add(row["Company"])

                target_holdings = target_holdings[:strategy_config.entry_top_n]
                pending_target_holdings = target_holdings
                pending_order_age = 0
            else:
                eligible_counts.append(0)
                target_holdings = []

            ranking_history.append(
                {
                    "Date": current_date,
                    "Holdings": target_holdings,
                    "Ranked": ranked[:strategy_config.keep_top_n] if ranked else [],
                }
            )

            holdings_history.append(
                {
                    "Date": current_date,
                    "Held": sorted(list(portfolio.positions.keys())),
                    "Target": sorted(target_holdings),
                }
            )

    diagnostics = compute_diagnostics(
        strategy_equity=strategy_values,
        benchmark_equity=benchmark_values,
        trade_events=trade_events,
        transaction_costs_paid=transaction_costs_paid,
        turnover_notional=turnover_notional,
        invested_fraction_history=invested_fraction_history,
        risk_free_rate=backtest_config.risk_free_rate,
    )

    return {
        "strategy_equity": strategy_values,
        "benchmark_equity": benchmark_values,
        "latest_rankings": ranking_history[-1]["Ranked"] if ranking_history else [],
        "ranking_history": ranking_history,
        "trade_events": trade_events,
        "holdings_history": holdings_history,
        "eligible_counts": eligible_counts,
        "daily_audit": daily_audit,
        "position_snapshots": position_snapshots,
        "quality_issues": quality_issues,
        "diagnostics": diagnostics,
        "portfolio": portfolio,
    }
