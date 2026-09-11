from __future__ import annotations

import numpy as np
import pandas as pd

from data import has_fresh_observation


def cumulative_return_from_close(close_series: pd.Series, lookback_days: int) -> float:
    if len(close_series) <= lookback_days:
        return np.nan

    current_close = float(close_series.iloc[-1])
    past_close = float(close_series.iloc[-(lookback_days + 1)])

    if current_close <= 0 or past_close <= 0:
        return np.nan

    return current_close / past_close - 1


def compute_momentum_score(data_up_to: pd.DataFrame, weights: dict[str, float], use_vol_adjustment: bool, vol_lookback: int):
    close_series = data_up_to["Close"].dropna()

    if len(close_series) < 252:
        return np.nan, np.nan, np.nan, np.nan

    ret_6m = cumulative_return_from_close(close_series, 126)
    ret_3m = cumulative_return_from_close(close_series, 63)
    ret_12m = cumulative_return_from_close(close_series, 252)

    if any(pd.isna(value) for value in [ret_6m, ret_3m, ret_12m]):
        return np.nan, np.nan, np.nan, np.nan

    raw_score = weights["w6m"] * ret_6m + weights["w3m"] * ret_3m + weights["w12m"] * ret_12m

    if use_vol_adjustment:
        returns = close_series.pct_change().dropna().tail(vol_lookback)
        vol = returns.std(ddof=0)
        if vol <= 0 or pd.isna(vol):
            return ret_6m, ret_3m, ret_12m, np.nan
        raw_score = raw_score / vol

    return ret_6m, ret_3m, ret_12m, raw_score


def rank_universe(current_date: pd.Timestamp, all_stock_data, strategy_config, all_issues=None):
    candidates = []
    quality_issues = [] if all_issues is None else all_issues

    for company, stock_data in all_stock_data.items():
        if stock_data is None:
            quality_issues.append({"Date": current_date, "Company": company, "Issue": "missing_data"})
            continue

        is_fresh, last_date, last_price = has_fresh_observation(
            stock_data,
            current_date,
            field="Close",
            tolerance_days=strategy_config.stale_tolerance_days,
        )

        if not is_fresh:
            quality_issues.append(
                {
                    "Date": current_date,
                    "Company": company,
                    "Issue": "stale_security",
                    "LastObservationDate": last_date,
                    "LastPrice": last_price,
                }
            )
            continue

        data_up_to = stock_data.loc[:current_date].copy()
        if len(data_up_to) < 252:
            quality_issues.append({"Date": current_date, "Company": company, "Issue": "insufficient_history"})
            continue

        close_series = data_up_to["Close"].dropna()
        if close_series.empty:
            quality_issues.append({"Date": current_date, "Company": company, "Issue": "no_close_price"})
            continue

        current_close = float(close_series.iloc[-1])
        moving_average_200 = close_series.rolling(200).mean().iloc[-1]

        if pd.isna(moving_average_200) or current_close <= moving_average_200:
            continue

        ret_6m, ret_3m, ret_12m, score = compute_momentum_score(
            data_up_to,
            strategy_config.momentum_weights(),
            use_vol_adjustment=strategy_config.volatility_adjusted,
            vol_lookback=strategy_config.vol_lookback,
        )

        if np.isnan(score):
            continue

        candidates.append(
            {
                "Company": company,
                "Ticker": stock_data.attrs.get("ticker", company),
                "Score": float(score),
                "Ret_6m": float(ret_6m),
                "Ret_3m": float(ret_3m),
                "Ret_12m": float(ret_12m),
                "Close": float(current_close),
                "MA200": float(moving_average_200),
            }
        )

    ranked = sorted(candidates, key=lambda row: row["Score"], reverse=True)
    return [
        {
            "Rank": idx,
            "Company": row["Company"],
            "Ticker": row["Ticker"],
            "Score": row["Score"],
            "Ret_6m": row["Ret_6m"],
            "Ret_3m": row["Ret_3m"],
            "Ret_12m": row["Ret_12m"],
            "Close": row["Close"],
            "MA200": row["MA200"],
        }
        for idx, row in enumerate(ranked, start=1)
    ], quality_issues
