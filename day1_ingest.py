from pathlib import Path
import polars as pl
import yfinance as yf

# Define data paths
DATA_DIR = Path("./data")
DATA_DIR.mkdir(exist_ok=True)
PARQUET_PATH = DATA_DIR / "aapl_daily.parquet"


def fetch_and_clean_ohlcv(symbol: str = "AAPL", period: str = "5y") -> pl.DataFrame:
    """Fetch ticker data via yfinance and process it into a clean Polars DataFrame."""
    print(f"Fetching {symbol} data for period: {period}...")
    
    # 1. Pull data via yfinance (returns a pandas DataFrame)
    ticker = yf.Ticker(symbol)
    df_pd = ticker.history(period=period, interval="1d").reset_index()

    # 2. Convert directly from Pandas to Polars DataFrame
    df_raw = pl.from_pandas(df_pd)

    # 3. Clean schema and cast timestamps
    df_clean = (
        df_raw
        .rename({
            "Date": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
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

    return df_clean


if __name__ == "__main__":
    df_aapl = fetch_and_clean_ohlcv("AAPL", "5y")
    df_aapl.write_parquet(PARQUET_PATH, compression="snappy")
    
    print(f"✅ Saved clean data to {PARQUET_PATH}")
    print(f"Schema:\n{df_aapl.schema}")
    print(f"Sample data (First 5 rows):\n{df_aapl.head(5)}")