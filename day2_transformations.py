from pathlib import Path
import polars as pl

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def compute_daily_transformations(file_path: Path) -> pl.DataFrame:
    """Uses Polars Lazy API to calculate daily financial metrics without full memory load."""
    
    # 1. Create a LazyFrame query plan (deferred execution)
    lazy_df = pl.scan_parquet(file_path)

    # 2. Build computation plan using Expressions
    transformation_plan = (
        lazy_df
        .sort("timestamp")
        .with_columns([
            # Daily Simple Return: (Close_t - Close_{t-1}) / Close_{t-1}
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))
            .alias("simple_return"),

            # Daily Log Return: ln(Close_t / Close_{t-1})
            (pl.col("close") / pl.col("close").shift(1))
            .log()
            .alias("log_return"),

            # Absolute Intraday Volatility Spread: High - Low
            (pl.col("high") - pl.col("low"))
            .alias("intraday_range"),

            # Relative Intraday Volatility Spread: (High - Low) / Open
            ((pl.col("high") - pl.col("low")) / pl.col("open"))
            .alias("intraday_volatility_pct"),
        ])
        # Fill the first row's null values (resulting from shift(1)) with 0.0
        .with_columns([
            pl.col("simple_return").fill_null(0.0),
            pl.col("log_return").fill_null(0.0),
        ])
    )

    # 3. Collect/Execute the lazy plan in optimized C++ engine
    result_df: pl.DataFrame = transformation_plan.collect()
    return result_df


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first to create the Parquet file!")

    df_metrics = compute_daily_transformations(PARQUET_PATH)

    print("✅ Successfully executed vectorized transformations.")
    print("\nCalculated Columns Summary:")
    print(
        df_metrics.select([
            "timestamp",
            "close",
            "simple_return",
            "log_return",
            "intraday_volatility_pct",
        ]).head(10)
    )