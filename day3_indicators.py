from pathlib import Path
import polars as pl

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def compute_technical_indicators(file_path: Path) -> pl.DataFrame:
    """Computes EMAs and Bollinger Bands using pure Polars expressions."""
    lazy_df = pl.scan_parquet(file_path)

    indicator_plan = (
        lazy_df
        .sort("timestamp")
        .with_columns([
            # 1. 50-day and 200-day Exponential Moving Averages (EMA)
            pl.col("close").ewm_mean(span=50, adjust=False).alias("ema_50"),
            pl.col("close").ewm_mean(span=200, adjust=False).alias("ema_200"),

            # 2. 20-day Rolling Mean (Middle Bollinger Band)
            pl.col("close").rolling_mean(window_size=20).alias("bb_middle"),

            # 3. 20-day Rolling Standard Deviation
            pl.col("close").rolling_std(window_size=20).alias("bb_std"),
        ])
        .with_columns([
            # 4. Upper and Lower Bollinger Bands (2 standard deviations)
            (pl.col("bb_middle") + (pl.col("bb_std") * 2)).alias("bb_upper"),
            (pl.col("bb_middle") - (pl.col("bb_std") * 2)).alias("bb_lower"),
        ])
        # Drop temporary standard deviation column if no longer needed
        .drop("bb_std")
    )

    return indicator_plan.collect()


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first!")

    df_indicators = compute_technical_indicators(PARQUET_PATH)

    print("✅ Successfully computed technical indicators.")
    print("\nSample Indicator Data (Rows 19 to 25 to see rolling calculations):")
    print(
        df_indicators.select([
            "timestamp",
            "close",
            "ema_50",
            "ema_200",
            "bb_upper",
            "bb_lower",
        ])
        .slice(19, 7)  # View rows where rolling windows have fully populated data
    )