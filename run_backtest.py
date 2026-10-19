from pathlib import Path
from engine.backtester import VectorizedBacktester

if __name__ == "__main__":
    parquet_file = Path("./data/aapl_daily.parquet")
    
    # Initialize engine
    engine = VectorizedBacktester(parquet_path=parquet_file)
    
    # Run backtest with custom parameters
    df_results = engine.run_strategy(fast_span=50, slow_span=200)
    metrics = engine.compute_metrics(df_results)

    print("🚀 Modular Backtest Execution Complete!")
    print("=" * 60)
    for key, value in metrics.items():
        if "Value" in key or "Capital" in key:
            print(f"{key:<30} : ${value:,.2f}")
        elif "(%)" in key:
            print(f"{key:<30} : {value:.2f}%")
        else:
            print(f"{key:<30} : {value:.4f}")
    print("=" * 60)