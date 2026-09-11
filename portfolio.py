from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data import get_last_observation, get_price_on_session


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, float] = field(default_factory=dict)


def portfolio_value_at_date(
    portfolio: PortfolioState,
    current_date: pd.Timestamp,
    all_stock_data,
    valuation_fallback_sessions: int = 3,
) -> float:
    total = float(portfolio.cash)

    for company, shares in portfolio.positions.items():
        stock_data = all_stock_data.get(company)
        if stock_data is None:
            continue

        last_date, last_price = get_last_observation(stock_data, current_date, field="Close")
        if last_price is None or last_price <= 0:
            continue

        if last_date is None:
            continue

        session_gap = len(stock_data.loc[last_date:current_date].index) - 1 if last_date is not None else 0
        if session_gap > valuation_fallback_sessions:
            continue

        total += shares * last_price

    return total


def execute_rebalance(
    portfolio: PortfolioState,
    current_date: pd.Timestamp,
    target_holdings: list[str],
    all_stock_data,
    transaction_cost: float,
):
    remaining_positions = {company: shares for company, shares in portfolio.positions.items() if company in set(target_holdings)}
    remaining_cash = float(portfolio.cash)
    trade_events = []
    transaction_costs_paid = 0.0
    turnover_notional = 0.0
    target_set = set(target_holdings)

    for company, shares in list(portfolio.positions.items()):
        if company in target_set:
            continue

        stock_data = all_stock_data.get(company)
        if stock_data is None:
            trade_events.append(
                {
                    "Date": current_date,
                    "Type": "Sell",
                    "Company": company,
                    "Shares": shares,
                    "Notional": 0.0,
                    "Status": "failed_missing_data",
                }
            )
            remaining_positions[company] = shares
            continue

        execution_price, status = get_price_on_session(stock_data, current_date, field="Open")
        if execution_price is None:
            trade_events.append(
                {
                    "Date": current_date,
                    "Type": "Sell",
                    "Company": company,
                    "Shares": shares,
                    "Notional": 0.0,
                    "Status": "failed_unavailable_price",
                }
            )
            remaining_positions[company] = shares
            continue

        sale_notional = shares * execution_price
        proceeds = sale_notional * (1 - transaction_cost)
        remaining_cash += proceeds
        transaction_costs_paid += sale_notional * transaction_cost
        turnover_notional += sale_notional

        trade_events.append(
            {
                "Date": current_date,
                "Type": "Sell",
                "Company": company,
                "Shares": shares,
                "Price": execution_price,
                "Notional": sale_notional,
                "Status": "executed",
            }
        )

    non_target_positions = [company for company in target_holdings if company not in remaining_positions]

    if non_target_positions:
        budget_per_name = remaining_cash / max(len(non_target_positions), 1)
        for company in non_target_positions:
            if remaining_cash <= 0:
                break

            stock_data = all_stock_data.get(company)
            if stock_data is None:
                trade_events.append(
                    {
                        "Date": current_date,
                        "Type": "Buy",
                        "Company": company,
                        "Shares": 0.0,
                        "Notional": 0.0,
                        "Status": "failed_missing_data",
                    }
                )
                continue

            execution_price, status = get_price_on_session(stock_data, current_date, field="Open")
            if execution_price is None:
                trade_events.append(
                    {
                        "Date": current_date,
                        "Type": "Buy",
                        "Company": company,
                        "Shares": 0.0,
                        "Notional": 0.0,
                        "Status": "failed_unavailable_price",
                    }
                )
                continue

            buy_notional = min(budget_per_name, remaining_cash)
            shares = buy_notional / (execution_price * (1 + transaction_cost))

            if shares <= 0:
                trade_events.append(
                    {
                        "Date": current_date,
                        "Type": "Buy",
                        "Company": company,
                        "Shares": 0.0,
                        "Notional": 0.0,
                        "Status": "failed_invalid_trade",
                    }
                )
                continue

            remaining_positions[company] = shares
            remaining_cash -= buy_notional
            transaction_costs_paid += buy_notional * transaction_cost
            turnover_notional += buy_notional

            trade_events.append(
                {
                    "Date": current_date,
                    "Type": "Buy",
                    "Company": company,
                    "Shares": shares,
                    "Price": execution_price,
                    "Notional": buy_notional,
                    "Status": "executed",
                }
            )

    order_completed = set(target_holdings) == set(remaining_positions.keys())

    return (
        PortfolioState(cash=remaining_cash, positions=remaining_positions),
        trade_events,
        transaction_costs_paid,
        turnover_notional,
        order_completed,
    )
