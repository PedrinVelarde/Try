import pandas as pd

from backtest import backtest_momentum_selection
from config import BacktestConfig, StrategyConfig
from data import download_price_data
from metrics import compute_position_attribution


def build_sample_data(periods: int = 5):
    base_prices = [100 + idx * 0.5 for idx in range(periods)]
    data = pd.DataFrame(
        {
            "Open": base_prices,
            "High": [price + 1 for price in base_prices],
            "Low": [max(price - 1, 0) for price in base_prices],
            "Close": [price + 0.5 for price in base_prices],
            "Volume": [1000] * periods,
        },
        index=pd.date_range("2020-01-01", periods=periods, freq="D"),
    )
    data.attrs["ticker"] = "TEST"
    return data


def test_no_capital_destruction_on_zero_eligible():
    sample_data = build_sample_data()
    all_stock_data = {"Test": sample_data}
    benchmark_data = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
            "Volume": [1000, 1000, 1000, 1000, 1000],
        },
        index=pd.date_range("2020-01-01", periods=5, freq="D"),
    )

    strategy_config = StrategyConfig(
        entry_top_n=5,
        keep_top_n=5,
        rebalance_frequency="Weekly",
        w6m=0.5,
        w3m=0.3,
        w12m=0.2,
    )
    backtest_config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)

    result = backtest_momentum_selection(
        all_stock_data=all_stock_data,
        benchmark_data=benchmark_data,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
    )

    assert result["diagnostics"]["strategy_return"] >= -1.0


def test_freshness_policy_blocks_stale_security():
    sample_data = build_sample_data()
    sample_data = sample_data.loc[: pd.Timestamp("2020-01-05")]
    all_stock_data = {"Test": sample_data}
    benchmark_data = sample_data.copy()

    strategy_config = StrategyConfig(
        entry_top_n=1,
        keep_top_n=1,
        rebalance_frequency="Weekly",
        w6m=0.5,
        w3m=0.3,
        w12m=0.2,
        stale_tolerance_days=0,
    )
    result = backtest_momentum_selection(
        all_stock_data=all_stock_data,
        benchmark_data=benchmark_data,
        strategy_config=strategy_config,
        backtest_config=BacktestConfig(initial_capital=10000, transaction_cost=0.0),
    )

    assert result["quality_issues"]


def test_daily_audit_and_attribution_are_available():
    sample_data = build_sample_data(periods=260)
    all_stock_data = {"Test": sample_data}
    benchmark_data = sample_data.copy()

    strategy_config = StrategyConfig(
        entry_top_n=1,
        keep_top_n=1,
        rebalance_frequency="Weekly",
        w6m=0.5,
        w3m=0.3,
        w12m=0.2,
    )
    backtest_config = BacktestConfig(initial_capital=10000, transaction_cost=0.0)

    result = backtest_momentum_selection(
        all_stock_data=all_stock_data,
        benchmark_data=benchmark_data,
        strategy_config=strategy_config,
        backtest_config=backtest_config,
    )

    assert result["daily_audit"]
    assert result["position_snapshots"]

    attribution_df = compute_position_attribution(result["position_snapshots"])
    assert not attribution_df.empty
    assert {"Company", "CumulativeContribution"}.issubset(set(attribution_df.columns))
