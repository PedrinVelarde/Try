from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_FRESHNESS_TOLERANCE_DAYS = 1


def download_price_data(ticker: str, period: str = "20y") -> pd.DataFrame | None:
    data = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        else:
            data.columns = data.columns.get_level_values(0)

    needed_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]
    data = data[needed_cols].copy()
    data = data.sort_index()

    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].fillna(0)

    data = data.dropna(subset=["Open", "Close"], how="all")
    data.attrs["ticker"] = ticker
    return data


def download_benchmark_data(period: str = "20y") -> pd.DataFrame | None:
    data = yf.download(
        "SPY",
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )

    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    needed_cols = [col for col in ["Open", "High", "Low", "Close", "Volume"] if col in data.columns]
    data = data[needed_cols].copy()
    data = data.sort_index()

    if "Volume" in data.columns:
        data["Volume"] = data["Volume"].fillna(0)

    data = data.dropna(subset=["Open", "Close"], how="all")
    return data


def _session_gap(data: pd.DataFrame | None, last_date: pd.Timestamp | None, current_date: pd.Timestamp) -> int:
    if data is None or data.empty or last_date is None or current_date < last_date:
        return 0

    relevant_index = data.loc[last_date:current_date].index
    if len(relevant_index) <= 1:
        return 0

    return len(relevant_index) - 1


def get_last_observation(data: pd.DataFrame | None, current_date: pd.Timestamp, field: str = "Close") -> tuple[pd.Timestamp | None, float | None]:
    if data is None or data.empty:
        return None, None

    hist = data.loc[:current_date, field].dropna()
    if hist.empty:
        return None, None

    last_date = hist.index[-1]
    return last_date, float(hist.iloc[-1])


def has_fresh_observation(
    data: pd.DataFrame | None,
    current_date: pd.Timestamp,
    field: str = "Close",
    tolerance_days: int = DEFAULT_FRESHNESS_TOLERANCE_DAYS,
) -> tuple[bool, pd.Timestamp | None, float | None]:
    last_date, last_price = get_last_observation(data, current_date, field)
    if last_date is None or last_price is None:
        return False, None, None

    session_gap = _session_gap(data, last_date, current_date)
    is_fresh = session_gap <= tolerance_days
    return is_fresh, last_date, last_price


def get_price_on_session(data: pd.DataFrame | None, current_date: pd.Timestamp, field: str = "Open") -> tuple[float | None, str | None]:
    if data is None or data.empty:
        return None, None

    if current_date in data.index:
        value = data.loc[current_date, field]
        if pd.notna(value):
            return float(value), "on_session"

    return None, "missing"


def get_signal_price(
    data: pd.DataFrame | None,
    current_date: pd.Timestamp,
    field: str = "Close",
    stale_tolerance_sessions: int = DEFAULT_FRESHNESS_TOLERANCE_DAYS,
) -> tuple[float | None, pd.Timestamp | None, str]:
    price, status = get_price_on_session(data, current_date, field)
    if price is not None:
        return price, current_date, "exact"

    last_date, last_price = get_last_observation(data, current_date, field)
    if last_date is None or last_price is None:
        return None, None, "missing"

    session_gap = _session_gap(data, last_date, current_date)
    if session_gap <= stale_tolerance_sessions:
        return last_price, last_date, "fallback_previous_close"

    return None, last_date, "stale_security"


def get_valuation_price(
    data: pd.DataFrame | None,
    current_date: pd.Timestamp,
    field: str = "Close",
    max_missing_sessions: int = 3,
) -> tuple[float | None, pd.Timestamp | None, str]:
    price, status = get_price_on_session(data, current_date, field)
    if price is not None:
        return price, current_date, "exact"

    last_date, last_price = get_last_observation(data, current_date, field)
    if last_date is None or last_price is None:
        return None, None, "missing"

    session_gap = _session_gap(data, last_date, current_date)
    if session_gap <= max_missing_sessions:
        return last_price, last_date, "fallback_previous_close"

    return None, last_date, "stale_security"


def get_benchmark_return(benchmark_data: pd.DataFrame, prev_date: pd.Timestamp, current_date: pd.Timestamp) -> tuple[float, float]:
    _, prev_close = get_last_observation(benchmark_data, prev_date, "Close")
    _, curr_close = get_last_observation(benchmark_data, current_date, "Close")

    if prev_close is None or curr_close is None:
        return np.nan, np.nan

    return prev_close, curr_close
