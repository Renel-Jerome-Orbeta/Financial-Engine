from pathlib import Path
import polars as pl
from engine.backtester import VectorizedBacktester

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def test_backtester_initialization():
    """Verify the backtester initializes with correct default values."""
    engine = VectorizedBacktester(parquet_path=PARQUET_PATH)
    assert engine.initial_capital == 100000.0
    assert engine.fee_rate == 0.0005
    assert engine.slippage_rate == 0.0001


def test_run_strategy_output_structure():
    """Verify that run_strategy returns a Polars DataFrame with the expected schema."""
    if not PARQUET_PATH.exists():
        pytest.skip("Parquet data file not found.")

    engine = VectorizedBacktester(parquet_path=PARQUET_PATH)
    df_results = engine.run_strategy(fast_span=50, slow_span=200)

    # Check that it's a Polars DataFrame
    assert isinstance(df_results, pl.DataFrame)
    
    # Check for expected columns produced by the backtest pipeline
    expected_columns = [
        "timestamp", "close", "ema_fast", "ema_slow", 
        "position", "trade_executed", "net_strategy_return", "portfolio_value_net"
    ]
    for col in expected_columns:
        assert col in df_results.columns


def test_compute_metrics_keys():
    """Verify that compute_metrics returns all required institutional performance keys."""
    if not PARQUET_PATH.exists():
        pytest.skip("Parquet data file not found.")

    engine = VectorizedBacktester(parquet_path=PARQUET_PATH)
    df_results = engine.run_strategy(fast_span=50, slow_span=200)
    metrics = engine.compute_metrics(df_results)

    expected_keys = [
        "Starting Capital",
        "Ending Portfolio Value",
        "Total Return (%)",
        "CAGR (%)",
        "Annualized Volatility (%)",
        "Sharpe Ratio",
        "Maximum Drawdown (%)"
    ]
    for key in expected_keys:
        assert key in metrics
        assert isinstance(metrics[key], float)