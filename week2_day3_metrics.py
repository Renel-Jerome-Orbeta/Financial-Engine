from pathlib import Path
import numpy as np
import polars as pl

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def compute_performance_metrics(file_path: Path) -> dict:
    """Calculates professional quantitative risk and performance metrics from the backtest."""
    
    lazy_df = pl.scan_parquet(file_path)

    # 1. Run the base backtest pipeline with costs
    backtest_df = (
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
            pl.when(pl.col("ema_fast") > pl.col("ema_slow")).then(1).otherwise(0).alias("position")
        ])
        .with_columns([
            (pl.col("position") - pl.col("position").shift(1)).fill_null(0).abs().alias("trade_executed")
        ])
        .with_columns([
            (pl.col("position").shift(1).fill_null(0) * pl.col("asset_return"))
            .sub(pl.col("trade_executed") * (0.0005 + 0.0001))
            .alias("net_strategy_return")
        ])
        .with_columns([
            (100000.0 * (1.0 + pl.col("net_strategy_return")).cum_prod()).alias("portfolio_value_net")
        ])
        .collect()
    )

    # 2. Extract series for numerical metric calculations via NumPy
    returns = backtest_df.select("net_strategy_return").to_numpy().flatten()
    portfolio_values = backtest_df.select("portfolio_value_net").to_numpy().flatten()
    
    total_days = len(returns)
    years = total_days / 252.0

    # 3. Compute Metrics
    # Total Return & CAGR
    total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
    cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1.0 / years) - 1.0

    # Annualized Volatility (Standard Deviation of daily returns * sqrt(252))
    annualized_vol = np.std(returns) * np.sqrt(252)

    # Sharpe Ratio (Assuming 0% risk-free rate)
    sharpe_ratio = (cagr / annualized_vol) if annualized_vol != 0 else 0.0

    # Maximum Drawdown Calculation
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - peak) / peak
    max_drawdown = np.min(drawdown)

    return {
        "Starting Capital": 100000.0,
        "Ending Portfolio Value": portfolio_values[-1],
        "Total Return (%)": total_return * 100,
        "CAGR (%)": cagr * 100,
        "Annualized Volatility (%)": annualized_vol * 100,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown (%)": max_drawdown * 100,
    }


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first!")

    metrics = compute_performance_metrics(PARQUET_PATH)

    print("📊 Quantitative Performance Report (Net of Fees & Slippage)")
    print("=" * 60)
    for key, value in metrics.items():
        if "Value" in key or "Capital" in key:
            print(f"{key:<30} : ${value:,.2f}")
        elif "(%)" in key:
            print(f"{key:<30} : {value:.2f}%")
        else:
            print(f"{key:<30} : {value:.4f}")
    print("=" * 60)