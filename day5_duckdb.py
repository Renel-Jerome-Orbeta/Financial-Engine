from pathlib import Path
import duckdb

PARQUET_PATH = Path("./data/aapl_daily.parquet")


def run_duckdb_analytics(file_path: Path) -> None:
    """Uses DuckDB to run fast SQL queries directly over the Parquet file."""
    
    # Connect to an in-memory DuckDB database
    con = duckdb.connect(database=":memory:")

    print("--- 1. Top 5 Single-Day Percentage Gainers ---")
    query_top_gainers = f"""
        SELECT 
            timestamp,
            close,
            ROUND(((close - LAG(close, 1) OVER (ORDER BY timestamp)) / LAG(close, 1) OVER (ORDER BY timestamp)) * 100, 2) AS pct_gain
        FROM read_parquet('{file_path}')
        ORDER BY pct_gain DESC
        LIMIT 5;
    """
    df_gainers = con.execute(query_top_gainers).pl()
    print(df_gainers)

    print("\n--- 2. Yearly Average Closing Price & Volume Summary ---")
    query_yearly = f"""
        SELECT 
            EXTRACT(YEAR FROM timestamp) AS year,
            ROUND(AVG(close), 2) AS avg_close,
            ROUND(AVG(volume), 0) AS avg_volume,
            COUNT(*) as trading_days
        FROM read_parquet('{file_path}')
        GROUP BY year
        ORDER BY year DESC;
    """
    df_yearly = con.execute(query_yearly).pl()
    print(df_yearly)

    con.close()


if __name__ == "__main__":
    if not PARQUET_PATH.exists():
        raise FileNotFoundError("Run day1_ingest.py first!")

    run_duckdb_analytics(PARQUET_PATH)
    print("\n✅ Week 1 Completed Successfully! You've mastered Polars, Parquet, and DuckDB analytics.")