from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import json
from engine.backtester import VectorizedBacktester

app = FastAPI(
    title="Financial Time-Series Analytics & Backtesting Engine",
    version="1.0.0",
    description="Production-grade quantitative finance API powered by Polars, FastAPI, and Caching."
)

PARQUET_PATH = Path("./data/aapl_daily.parquet")

# Initialize Redis connection (tries local Redis, falls back gracefully if unavailable)
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
except Exception:
    redis_client = None
    REDIS_AVAILABLE = False


class BacktestRequest(BaseModel):
    fast_span: int = 50
    slow_span: int = 200
    initial_capital: float = 100000.0


@app.get("/")
def health_check():
    """Health check endpoint to verify API and caching status."""
    return {
        "status": "online", 
        "engine": "Polars Vectorized Backtester",
        "redis_caching": REDIS_AVAILABLE
    }


@app.post("/backtest")
def run_backtest_api(request: BacktestRequest):
    """Triggers a backtest simulation, utilizing cache for repeated queries."""
    if not PARQUET_PATH.exists():
        raise HTTPException(
            status_code=404, 
            detail="Parquet data file not found. Run ingestion first."
        )
    
    # Generate a unique cache key based on request parameters
    cache_key = f"backtest:{request.fast_span}:{request.slow_span}:{request.initial_capital}"

    # 1. Check Cache First
    if REDIS_AVAILABLE:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return {
                "source": "cache",
                "data": json.loads(cached_result)
            }

    try:
        # 2. Compute if not cached
        engine = VectorizedBacktester(
            parquet_path=PARQUET_PATH,
            initial_capital=request.initial_capital
        )
        df_results = engine.run_strategy(
            fast_span=request.fast_span, 
            slow_span=request.slow_span
        )
        metrics = engine.compute_metrics(df_results)
        
        response_data = {
            "symbol": "AAPL",
            "parameters": {
                "fast_span": request.fast_span,
                "slow_span": request.slow_span,
            },
            "metrics": metrics
        }

        # 3. Store result in Cache (expires in 1 hour / 3600 seconds)
        if REDIS_AVAILABLE:
            redis_client.setex(cache_key, 3600, json.dumps(response_data))

        return {
            "source": "computation_engine",
            "data": response_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))