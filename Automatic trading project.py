from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from backtest import backtest_momentum_selection
from config import BacktestConfig, StrategyConfig
from data import download_benchmark_data, download_price_data
from universe import get_universe


INITIAL_CAPITAL = 10_000.0


def _downsample_daily_audit(daily_audit, frequency):
    if not daily_audit:
        return pd.DataFrame(columns=["Date", "Cash", "PortfolioValue", "Holdings", "PositionSnapshots"])

    audit_df = pd.DataFrame(daily_audit)
    audit_df["Date"] = pd.to_datetime(audit_df["Date"])
    audit_df = audit_df.sort_values("Date").reset_index(drop=True)

    if frequency == "Weekly":
        audit_df["Period"] = audit_df["Date"].dt.to_period("W-SUN")
    else:
        audit_df["Period"] = audit_df["Date"].dt.to_period("M")

    sampled = (
        audit_df.groupby("Period")
        .tail(1)
        .drop(columns=["Period"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return sampled


def _build_company_max_weights(selected_result):
    snapshots = selected_result.get("position_snapshots") or []
    if not snapshots:
        return pd.Series(dtype=float)

    snapshot_df = pd.DataFrame(snapshots)
    snapshot_df["Date"] = pd.to_datetime(snapshot_df["Date"])
    return snapshot_df.groupby("Company")["Weight"].max().sort_values(ascending=False)


def _build_composition_dataframe(selected_result, frequency, show_mode):
    daily_audit = selected_result.get("daily_audit") or []
    if not daily_audit:
        return pd.DataFrame(columns=["Date", "Company", "Weight", "PositionValue"])

    sampled_audit = _downsample_daily_audit(daily_audit, frequency)
    company_max_weights = _build_company_max_weights(selected_result)

    if show_mode == "All holdings":
        included_companies = list(company_max_weights.index)
    elif show_mode == "Top 10 by maximum historical weight":
        included_companies = list(company_max_weights.head(10).index)
    elif show_mode == "Top 15 by maximum historical weight":
        included_companies = list(company_max_weights.head(15).index)
    else:
        included_companies = list(company_max_weights[company_max_weights > 0.05].index)

    included_set = set(included_companies)
    composition_rows = []

    for _, audit_row in sampled_audit.iterrows():
        portfolio_value = float(audit_row.get("PortfolioValue") or 0.0)
        if portfolio_value <= 0:
            continue

        cash_value = float(audit_row.get("Cash") or 0.0)
        position_snapshots = audit_row.get("PositionSnapshots") or []
        snapshot_lookup = {
            snapshot.get("Company"): snapshot
            for snapshot in position_snapshots
            if snapshot.get("Company")
        }

        excluded_value = 0.0
        for company, snapshot in snapshot_lookup.items():
            position_value = float(snapshot.get("PositionValue") or 0.0)
            if company in included_set:
                composition_rows.append(
                    {
                        "Date": audit_row["Date"],
                        "Company": company,
                        "Weight": float(snapshot.get("Weight") or 0.0),
                        "PositionValue": position_value,
                    }
                )
            else:
                excluded_value += position_value

        if excluded_value > 0:
            composition_rows.append(
                {
                    "Date": audit_row["Date"],
                    "Company": "Other",
                    "Weight": excluded_value / portfolio_value,
                    "PositionValue": excluded_value,
                }
            )

        if cash_value > 0:
            composition_rows.append(
                {
                    "Date": audit_row["Date"],
                    "Company": "Cash",
                    "Weight": cash_value / portfolio_value,
                    "PositionValue": cash_value,
                }
            )

    composition_df = pd.DataFrame(composition_rows)
    if composition_df.empty:
        return composition_df

    composition_df["Date"] = pd.to_datetime(composition_df["Date"])
    return composition_df.sort_values(["Date", "Company"]).reset_index(drop=True)


def _build_presence_heatmap(selected_result, frequency):
    daily_audit = selected_result.get("daily_audit") or []
    if not daily_audit:
        return pd.DataFrame(columns=["Date", "Company", "Held"])

    sampled_audit = _downsample_daily_audit(daily_audit, frequency)
    company_max_weights = _build_company_max_weights(selected_result)
    companies = list(company_max_weights.index)

    heatmap_rows = []
    for _, audit_row in sampled_audit.iterrows():
        held_companies = {
            snapshot.get("Company")
            for snapshot in (audit_row.get("PositionSnapshots") or [])
            if snapshot.get("Company")
        }

        for company in companies:
            heatmap_rows.append(
                {
                    "Date": audit_row["Date"],
                    "Company": company,
                    "Held": int(company in held_companies),
                }
            )

    heatmap_df = pd.DataFrame(heatmap_rows)
    if heatmap_df.empty:
        return heatmap_df

    heatmap_df["Date"] = pd.to_datetime(heatmap_df["Date"])
    return heatmap_df.sort_values(["Date", "Company"]).reset_index(drop=True)


def run_screening(all_stock_data, benchmark_data, strategy_config, backtest_config, analysis_start_date=None):
    benchmark_dates = benchmark_data.index
    latest_year = int(benchmark_dates.year.max())
    train_end_year = max(1, latest_year - 8)
    train_end_date = pd.Timestamp(f"{train_end_year}-12-31")
    train_start_date = pd.Timestamp(analysis_start_date) if analysis_start_date is not None else benchmark_dates[0]
    train_start_date = max(train_start_date, benchmark_dates[0])

    eval_start_date = pd.Timestamp(f"{train_end_year + 1}-01-01")
    eval_end_date = benchmark_dates[-1]

    train_benchmark = benchmark_data.loc[train_start_date:train_end_date]
    train_result = backtest_momentum_selection(
        all_stock_data=all_stock_data,
        benchmark_data=train_benchmark,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        analysis_start_date=train_start_date,
        analysis_end_date=train_end_date,
    )

    if train_result is None:
        return pd.DataFrame(), None

    scenario_rows = []
    for entry_top_n in [5, 10, 15]:
        for keep_top_n in [15, 25, 35]:
            for rebalance_frequency in ["Weekly", "Monthly"]:
                scenario_rows.append(
                    {
                        "entry_top_n": entry_top_n,
                        "keep_top_n": keep_top_n,
                        "rebalance_frequency": rebalance_frequency,
                        "strategy_return": train_result["diagnostics"]["strategy_return"],
                        "benchmark_return": train_result["diagnostics"]["benchmark_return"],
                        "performance": train_result["diagnostics"]["excess_return"],
                    }
                )

    screening_df = pd.DataFrame(scenario_rows)
    screening_df = screening_df.sort_values("performance", ascending=False)
    best = screening_df.iloc[0]

    eval_benchmark = benchmark_data.loc[eval_start_date:eval_end_date]
    if eval_benchmark.empty:
        return screening_df, None

    eval_strategy = StrategyConfig(
        entry_top_n=int(best["entry_top_n"]),
        keep_top_n=int(best["keep_top_n"]),
        rebalance_frequency=str(best["rebalance_frequency"]),
        w6m=strategy_config.w6m,
        w3m=strategy_config.w3m,
        w12m=strategy_config.w12m,
        volatility_adjusted=strategy_config.volatility_adjusted,
        vol_lookback=strategy_config.vol_lookback,
        stale_tolerance_days=strategy_config.stale_tolerance_days,
        universe_size=strategy_config.universe_size,
        include_former_constituents=strategy_config.include_former_constituents,
    )

    holdout_result = backtest_momentum_selection(
        all_stock_data=all_stock_data,
        benchmark_data=eval_benchmark,
        strategy_config=eval_strategy,
        backtest_config=backtest_config,
        analysis_start_date=eval_start_date,
        analysis_end_date=eval_end_date,
    )

    return screening_df, holdout_result


st.set_page_config(layout="wide")
st.title("Momentum Backtester (Research-Oriented)")

st.caption(
    "Method note: the benchmark remains SPY for comparison, while the investable universe can be expanded with European companies. "
    "Former S&P 500 constituents are retained as an approximate extension, but they carry survivorship and stale-membership bias."
)

with st.container():
    col1, col2, col3 = st.columns(3)
    universe_size_choice = col1.selectbox("Universe size", ["40", "80", "120", "240", "All"], index=4)
    universe_choice = col2.selectbox(
        "Universe",
        ["S&P 500", "Europe", "S&P 500 + Europe"],
        index=2,
    )
    include_former_constituents = col3.checkbox(
        "Include former S&P 500 constituents (approximate, more bias)",
        value=False,
    )

include_european = universe_choice in ["Europe", "S&P 500 + Europe"]
include_sp500 = universe_choice != "Europe"

max_universe_size = None if universe_size_choice == "All" else int(universe_size_choice)

with st.container():
    col1, col2, col3 = st.columns(3)
    entry_top_n = col1.slider("Entry: Top N stocks to buy", min_value=5, max_value=20, value=10, step=1)
    keep_top_n = col2.slider("Keep: Top N ranks to retain", min_value=10, max_value=40, value=25, step=1)
    rebalance_frequency = col3.selectbox("Rebalance frequency", ["Weekly", "Monthly"], index=0)

vol_lookback = st.slider("Volatility lookback (days)", min_value=30, max_value=120, value=60, step=5)

with st.container():
    col4, col5, col6 = st.columns(3)
    w6m = col4.slider("6m momentum weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    w3m = col5.slider("3m momentum weight", min_value=0.0, max_value=1.0, value=0.3, step=0.05)
    w12m = col6.slider("12m momentum weight", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

if abs(w6m + w3m + w12m) < 1e-9:
    st.error("Momentum weights must sum to a positive value.")
    st.stop()

strategy_config = StrategyConfig(
    entry_top_n=entry_top_n,
    keep_top_n=keep_top_n,
    rebalance_frequency=rebalance_frequency,
    w6m=w6m,
    w3m=w3m,
    w12m=w12m,
    volatility_adjusted=st.checkbox("Scale score by volatility", value=True),
    vol_lookback=vol_lookback,
    stale_tolerance_days=1,
    universe_size=max_universe_size,
    include_former_constituents=include_former_constituents,
)

transaction_cost_percent = st.number_input(
    "Transaction cost per rebalance (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)
backtest_config = BacktestConfig(
    initial_capital=INITIAL_CAPITAL,
    transaction_cost=transaction_cost_percent / 100,
    risk_free_rate=0.0,
)

st.info(
    "Research design: rankings are computed using data available up to each signal date, execution occurs on the next executable market session, and retained positions are preserved unless explicitly rotated out."
)

universe = get_universe(
    include_former_constituents=strategy_config.include_former_constituents,
    max_size=strategy_config.universe_size,
    include_european=include_european,
    include_sp500=include_sp500,
)

all_stock_data = {company: download_price_data(ticker) for company, ticker in universe.items()}
benchmark_data = download_benchmark_data()

if benchmark_data is None:
    st.error("Benchmark data could not be loaded. Please retry.")
    st.stop()

analysis_start_date = st.date_input(
    "Analysis begin",
    value=pd.Timestamp("2018-01-01").date(),
    min_value=benchmark_data.index[0].date(),
    max_value=benchmark_data.index[-1].date(),
)

run_screening_checkbox = st.checkbox("Run coarse parameter screening", value=False)

result = backtest_momentum_selection(
    all_stock_data=all_stock_data,
    benchmark_data=benchmark_data,
    strategy_config=strategy_config,
    backtest_config=backtest_config,
    analysis_start_date=pd.Timestamp(analysis_start_date),
)

if result is None:
    st.error("Backtest could not be completed.")
    st.stop()

selected_result = result
screening_df = pd.DataFrame()
best_screening_result = None

if run_screening_checkbox:
    screening_df, selected_result = run_screening(
        all_stock_data=all_stock_data,
        benchmark_data=benchmark_data,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
        analysis_start_date=pd.Timestamp(analysis_start_date),
    )
    if not screening_df.empty:
        best_screening_result = screening_df.iloc[0]

    st.subheader("Coarse screening results")
    if screening_df.empty:
        st.info("Screening did not produce results.")
    else:
        screening_display = screening_df.copy()
        screening_display["Performance (Strategy - SPY)"] = screening_display["performance"].map(lambda x: f"{x * 100:+.2f}%")
        screening_display["Strategy return"] = screening_display["strategy_return"].map(lambda x: f"{x * 100:.2f}%")
        screening_display["SPY return"] = screening_display["benchmark_return"].map(lambda x: f"{x * 100:.2f}%")
        screening_display = screening_display[
            [
                "entry_top_n",
                "keep_top_n",
                "rebalance_frequency",
                "Strategy return",
                "SPY return",
                "Performance (Strategy - SPY)",
            ]
        ]
        st.dataframe(screening_display, use_container_width=True, hide_index=True)

st.subheader("Performance summary")
metric_cols = st.columns(6)
metric_cols[0].metric("Strategy return", f"{selected_result['diagnostics']['strategy_return'] * 100:.2f}%")
metric_cols[1].metric("SPY return", f"{selected_result['diagnostics']['benchmark_return'] * 100:.2f}%")
metric_cols[2].metric("Excess return", f"{selected_result['diagnostics']['excess_return'] * 100:+.2f}%")
metric_cols[3].metric("CAGR", f"{selected_result['diagnostics']['strategy_cagr'] * 100:.2f}%")
metric_cols[4].metric("Max drawdown", f"{selected_result['diagnostics']['max_drawdown'] * 100:.2f}%")
metric_cols[5].metric("Sharpe", f"{selected_result['diagnostics']['sharpe_ratio']:.2f}")

if best_screening_result is not None:
    st.caption(
        "Best coarse-screening scenario: Entry Top N = "
        f"{best_screening_result['entry_top_n']}, Keep Top N = {best_screening_result['keep_top_n']}, "
        f"Rebalance = {best_screening_result['rebalance_frequency']}"
    )

st.subheader("Equity curves")

chart_df = pd.DataFrame(
    {
        "Momentum portfolio": selected_result["strategy_equity"].values,
        "SPY": selected_result["benchmark_equity"].values,
    },
    index=selected_result["strategy_equity"].index,
)
st.line_chart(chart_df)

normalized_start = float(selected_result["strategy_equity"].iloc[0])
normalized_benchmark_start = float(selected_result["benchmark_equity"].iloc[0])
normalized_df = pd.DataFrame(
    {
        "Momentum portfolio (normalized)": (selected_result["strategy_equity"] / normalized_start) * 100,
        "SPY (normalized)": (selected_result["benchmark_equity"] / normalized_benchmark_start) * 100,
    },
    index=selected_result["strategy_equity"].index,
)
st.subheader("Normalized equity curves")
st.caption("Each series starts at 100 and shows relative performance from the beginning of the backtest.")
st.line_chart(normalized_df)

st.subheader("Portfolio composition over time")
composition_frequency = st.selectbox("Frequency", ["Weekly", "Monthly"], index=0)
composition_show = st.selectbox(
    "Show",
    [
        "All holdings",
        "Top 10 by maximum historical weight",
        "Top 15 by maximum historical weight",
        "Holdings that ever exceeded 5%",
    ],
    index=0,
)

composition_df = _build_composition_dataframe(selected_result, composition_frequency, composition_show)

if composition_df.empty:
    st.info("No position history is available for the composition chart.")
else:
    company_order = list(_build_company_max_weights(selected_result).index)
    chart = (
        alt.Chart(composition_df)
        .mark_area(opacity=0.85)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Weight:Q", stack="normalize", title="Portfolio weight", axis=alt.Axis(format="%")),
            color=alt.Color(
                "Company:N",
                title="Holding",
                sort=company_order + ["Other", "Cash"],
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Company:N", title="Company"),
                alt.Tooltip("Weight:Q", title="Portfolio weight", format=".2%"),
                alt.Tooltip("PositionValue:Q", title="Position value", format="$,.2f"),
            ],
        )
        .properties(height=500)
    )
    st.caption("Derived from the historical replay state. This only down-samples the simulated portfolio snapshots; it does not recalculate holdings or returns.")
    st.altair_chart(chart, use_container_width=True)

    show_heatmap = st.checkbox("Show holdings-presence heatmap", value=False)
    if show_heatmap:
        heatmap_df = _build_presence_heatmap(selected_result, composition_frequency)
        if heatmap_df.empty:
            st.info("No heatmap data is available.")
        else:
            heatmap = (
                alt.Chart(heatmap_df)
                .mark_rect()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Company:N", title="Company", sort=list(_build_company_max_weights(selected_result).index)),
                    color=alt.Color(
                        "Held:Q",
                        title="Held",
                        scale=alt.Scale(domain=[0, 1], range=["#ecf0f1", "#2563eb"]),
                    ),
                    tooltip=[
                        alt.Tooltip("Date:T", title="Date"),
                        alt.Tooltip("Company:N", title="Company"),
                        alt.Tooltip("Held:Q", title="Held / not held", format=".0f"),
                    ],
                )
                .properties(height=max(260, min(700, len(_build_company_max_weights(selected_result).index) * 14)))
            )
            st.altair_chart(heatmap, use_container_width=True)

st.subheader("Diagnostics")
summary_rows = [
    {"Metric": "Total return", "Value": f"{selected_result['diagnostics']['strategy_return'] * 100:.2f}%"},
    {"Metric": "Benchmark return", "Value": f"{selected_result['diagnostics']['benchmark_return'] * 100:.2f}%"},
    {"Metric": "Excess return", "Value": f"{selected_result['diagnostics']['excess_return'] * 100:+.2f}%"},
    {"Metric": "CAGR", "Value": f"{selected_result['diagnostics']['strategy_cagr'] * 100:.2f}%"},
    {"Metric": "Max drawdown", "Value": f"{selected_result['diagnostics']['max_drawdown'] * 100:.2f}%"},
    {"Metric": "Annualized volatility", "Value": f"{selected_result['diagnostics']['annualized_volatility'] * 100:.2f}%"},
    {"Metric": "Sharpe ratio", "Value": f"{selected_result['diagnostics']['sharpe_ratio']:.2f}"},
    {"Metric": "Rotations / trades", "Value": int(selected_result['diagnostics']['trades'])},
    {"Metric": "Annual turnover", "Value": f"{selected_result['diagnostics']['annual_turnover'] * 100:.2f}%"},
    {"Metric": "Transaction costs paid", "Value": f"${selected_result['diagnostics']['transaction_costs_paid']:.2f}"},
    {"Metric": "Time invested", "Value": f"{selected_result['diagnostics']['percent_time_invested'] * 100:.2f}%"},
]
st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.subheader("Eligible stock counts")
eligible_df = pd.DataFrame(
    {
        "Rebalance date": [row["Date"] for row in selected_result["ranking_history"]],
        "Eligible stocks": selected_result["eligible_counts"],
    }
)
st.dataframe(eligible_df, use_container_width=True, hide_index=True)

st.subheader("Latest rankings")
if selected_result["latest_rankings"]:
    latest_df = pd.DataFrame(selected_result["latest_rankings"])
    latest_df = latest_df[["Rank", "Company", "Ticker", "Score", "Ret_6m", "Ret_3m", "Ret_12m"]]
    latest_df["Score"] = latest_df["Score"].map(lambda x: f"{x:.4f}")
    latest_df["Ret_6m"] = latest_df["Ret_6m"].map(lambda x: f"{x * 100:.2f}%")
    latest_df["Ret_3m"] = latest_df["Ret_3m"].map(lambda x: f"{x * 100:.2f}%")
    latest_df["Ret_12m"] = latest_df["Ret_12m"].map(lambda x: f"{x * 100:.2f}%")
    st.dataframe(latest_df, use_container_width=True, hide_index=True)
else:
    st.info("No rankings were available yet.")

st.subheader("Holdings history")
if selected_result.get("holdings_history"):
    holdings_df = pd.DataFrame(selected_result["holdings_history"])
    holdings_df["Held"] = holdings_df["Held"].map(lambda x: ", ".join(x))
    holdings_df["Target"] = holdings_df["Target"].map(lambda x: ", ".join(x))
    st.dataframe(holdings_df, use_container_width=True, hide_index=True)
else:
    st.info("No holdings history captured.")

st.markdown(
    """
    Notes:
    - This backtest is designed for research and should not be interpreted as live trading advice.
    - The benchmark remains SPY for comparison, while the investable universe can be restricted to S&P 500, expanded to Europe, or combined with both regions.
    - Former constituents are available as an optional approximation, but they should be interpreted with caution because they can introduce survivorship bias.
    - Rankings are computed using only information available up to each signal date, and execution occurs on the next available market session.
    - Transaction costs are charged only on actual turnover, and the strategy rotates holdings instead of fully liquidating and rebuilding the portfolio every rebalance.
    """
)
