from pathlib import Path
import polars as pl

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def generate_ma_signals(file_path: Path, fast_span: int = 50, slow_span: int = 200) -> pl.DataFrame:
    """Generates dual moving average crossover trading signals using Polars expressions."""
    lazy_df = pl.scan_parquet(file_path)

    signal_plan = (
        lazy_df
        .sort("timestamp")
        .with_columns([
            # 1. Calculate EMAs
            pl.col("close").ewm_mean(span=fast_span, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=slow_span, adjust=False).alias("ema_slow"),
        ])
        .with_columns([
            # 2. Raw Position State: 1 when Fast EMA > Slow EMA, else -1
            pl.when(pl.col("ema_fast") > pl.col("ema_slow"))
            .then(1)
            .otherwise(-1)
            .alias("position")
        ])
        .with_columns([
            # 3. Detect Crossover Triggers by shifting position state
            # If position changes from -1 to 1, trigger is 2 (Buy). If 1 to -1, trigger is -2 (Sell).
            (pl.col("position") - pl.col("position").shift(1))
            .fill_null(0)
            .alias("trade_trigger")
        ])
    )

    return signal_plan.collect()


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first!")

    df_signals = generate_ma_signals(PARQUET_PATH)

    print("✅ Successfully generated vectorized trading signals.")
    print("\nSample Crossover Triggers (Filtering for active trade triggers):")
    print(
        df_signals
        .filter(pl.col("trade_trigger") != 0)
        .select(["timestamp", "close", "ema_fast", "ema_slow", "position", "trade_trigger"])
        .head(5)
    )