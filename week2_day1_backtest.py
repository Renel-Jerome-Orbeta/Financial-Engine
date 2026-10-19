from pathlib import Path
import polars as pl

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def run_vectorized_backtest(file_path: Path, initial_capital: float = 100000.0) -> pl.DataFrame:
    """Runs a vectorized backtest based on a Dual Moving Average Crossover strategy."""
    
    # 1. Load data and compute indicators / signals
    lazy_df = pl.scan_parquet(file_path)

    backtest_plan = (
        lazy_df
        .sort("timestamp")
        .with_columns([
            # Calculate EMAs
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_slow"),
            
            # Daily Simple Return of the underlying asset
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))
            .fill_null(0.0)
            .alias("asset_return"),
        ])
        .with_columns([
            # Position State: 1 (Long) when Fast > Slow, else 0 (Cash/Flat)
            pl.when(pl.col("ema_fast") > pl.col("ema_slow"))
            .then(1)
            .otherwise(0)
            .alias("position")
        ])
        .with_columns([
            # Strategy Return: Yesterday's position * Today's asset return
            # (Using .shift(1) prevents look-ahead bias!)
            (pl.col("position").shift(1).fill_null(0) * pl.col("asset_return"))
            .alias("strategy_return")
        ])
        .with_columns([
            # Cumulative Equity Curve
            (initial_capital * (1.0 + pl.col("strategy_return")).cum_prod())
            .alias("portfolio_value")
        ])
    )

    return backtest_plan.collect()


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first!")

    df_backtest = run_vectorized_backtest(PARQUET_PATH)

    print("✅ Successfully executed vectorized backtest.")
    print(f"Starting Capital: $100,000.00")
    final_value = df_backtest.select("portfolio_value").tail(1).item()
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    
    print("\nSample Backtest Performance Data (Last 5 rows):")
    print(
        df_backtest.select([
            "timestamp",
            "close",
            "position",
            "strategy_return",
            "portfolio_value",
        ])
        .tail(5)
    )