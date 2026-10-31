from pathlib import Path
import yfinance as yf
import polars as pl
from engine.backtester import VectorizedBacktester

# Define pipeline file paths
DATA_DIR = Path("./data")
PARQUET_PATH = DATA_DIR / "aapl_daily.parquet"


def run_pipeline():
    """
    Orchestrates the complete quantitative analytics and backtesting pipeline:
    1. Ingests and cleans fresh market data, saving to Apache Parquet.
    2. Initializes the vectorized backtesting engine.
    3. Executes the trading strategy and computes risk metrics.
    """
    print("=" * 60)
    print("🚀 STARTING FINANCIAL TIME-SERIES PIPELINE ORCHESTRATION")
    print("=" * 60)

    # --- STEP 1: Data Ingestion & Cleaning ---
    print("\n[Step 1/3] Fetching and cleaning market data...")
    DATA_DIR.mkdir(exist_ok=True)
    
    ticker = yf.Ticker("AAPL")
    df_raw = ticker.history(period="5y", interval="1d").reset_index()
    
    df_clean = (
        pl.from_pandas(df_raw)
        .rename({
            "Date": "timestamp", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume"
        })
        .select(["timestamp", "open", "high", "low", "close", "volume"])
        .with_columns([
            pl.col("timestamp").dt.cast_time_unit("ms").dt.replace_time_zone("UTC"),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Int64),
        ])
        .filter((pl.col("volume") > 0) & pl.col("close").is_not_null())
        .sort("timestamp")
    )
    
    df_clean.write_parquet(PARQUET_PATH, compression="snappy")
    print(f"✅ Data successfully ingested and saved to {PARQUET_PATH}")

    # --- STEP 2: Strategy Backtest Execution ---
    print("\n[Step 2/3] Executing vectorized backtest strategy...")
    engine = VectorizedBacktester(
        parquet_path=PARQUET_PATH,
        initial_capital=100000.0,
        fee_rate=0.0005,      # 5 bps fee
        slippage_rate=0.0001  # 1 bp slippage
    )
    
    # Run dual EMA crossover (50/200 span)
    df_results = engine.run_strategy(fast_span=50, slow_span=200)
    print("✅ Backtest executed successfully across all trading periods.")

    # --- STEP 3: Metric Calculation & Reporting ---
    print("\n[Step 3/3] Computing institutional risk and performance metrics...")
    metrics = engine.compute_metrics(df_results)

    print("\n" + "=" * 40)
    print("📊 FINAL BACKTEST PERFORMANCE REPORT")
    print("=" * 40)
    for key, val in metrics.items():
        if " (%)" in key or "Return" in key or "CAGR" in key or "Volatility" in key or "Drawdown" in key:
            print(f"• {key}: {val:.2f}%")
        elif "Ratio" in key:
            print(f"• {key}: {val:.4f}")
        else:
            print(f"• {key}: ${val:,.2f}")
    print("=" * 40)
    print("✨ Pipeline execution completed successfully!")


if __name__ == "__main__":
    run_pipeline()