from pathlib import Path
import polars as pl

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def run_backtest_with_costs(
    file_path: Path, 
    initial_capital: float = 100000.0,
    fee_rate: float = 0.0005,    # 0.05% per trade (5 bps)
    slippage_rate: float = 0.0001 # 0.01% price slippage (1 bp)
) -> pl.DataFrame:
    """Runs a vectorized backtest including transaction fees and slippage costs."""
    
    lazy_df = pl.scan_parquet(file_path)

    backtest_plan = (
        lazy_df
        .sort("timestamp")
        .with_columns([
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_slow"),
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))
            .fill_null(0.0)
            .alias("asset_return"),
        ])
        .with_columns([
            # Position State: 1 (Long) or 0 (Flat)
            pl.when(pl.col("ema_fast") > pl.col("ema_slow"))
            .then(1)
            .otherwise(0)
            .alias("position")
        ])
        .with_columns([
            # Detect trades: Absolute change in position indicates a buy or sell event
            (pl.col("position") - pl.col("position").shift(1))
            .fill_null(0)
            .abs()
            .alias("trade_executed")
        ])
        .with_columns([
            # Raw strategy return from the asset movement
            (pl.col("position").shift(1).fill_null(0) * pl.col("asset_return"))
            .alias("raw_strategy_return"),
            
            # Friction cost applied only on days a trade execution occurs
            # Cost = Trade Executed * (Fee Rate + Slippage Rate)
            (pl.col("trade_executed") * (fee_rate + slippage_rate))
            .alias("friction_cost")
        ])
        .with_columns([
            # Net strategy return = Raw return minus friction costs on transaction days
            (pl.col("raw_strategy_return") - pl.col("friction_cost"))
            .alias("net_strategy_return")
        ])
        .with_columns([
            # Cumulative Equity Curve with friction
            (initial_capital * (1.0 + pl.col("net_strategy_return")).cum_prod())
            .alias("portfolio_value_net")
        ])
    )

    return backtest_plan.collect()


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first!")

    df_costs = run_backtest_with_costs(PARQUET_PATH)

    print("✅ Successfully executed backtest with transaction costs & slippage.")
    final_value_net = df_costs.select("portfolio_value_net").tail(1).item()
    print(f"Final Portfolio Value (Net of Costs): ${final_value_net:,.2f}")
    
    print("\nSample Trade Execution & Cost Rows:")
    print(
        df_costs
        .filter(pl.col("trade_executed") > 0)
        .select([
            "timestamp",
            "close",
            "position",
            "trade_executed",
            "friction_cost",
            "portfolio_value_net",
        ])
        .head(5)
    )