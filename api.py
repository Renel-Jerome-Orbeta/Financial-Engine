from pathlib import Path
import json
import redis
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from engine.backtester import VectorizedBacktester

# Initialize FastAPI application with descriptive metadata
app = FastAPI(
    title="Financial Time-Series Analytics & Backtesting API",
    description="High-performance quantitative backtesting API powered by Polars, DuckDB, and FastAPI.",
    version="1.0.0"
)

# Define path to the ingested historical market data Parquet file
PARQUET_PATH = Path("./data/aapl_daily.parquet")

# Setup Redis connection with graceful fallback handling if the container/service is offline
try:
    redis_client = redis.Redis(host="localhost", port=6379, db=0, socket_timeout=2)
    redis_client.ping()  # Test connection
    REDIS_AVAILABLE = True
except (redis.ConnectionError, redis.TimeoutError):
    redis_client = None
    REDIS_AVAILABLE = False


class BacktestRequest(BaseModel):
    """
    Pydantic data model defining and validating incoming request payloads 
    for the backtest endpoint.
    """
    fast_span: int = Field(default=50, ge=5, le=200, description="Span for fast EMA")
    slow_span: int = Field(default=200, ge=20, le=500, description="Span for slow EMA")
    initial_capital: float = Field(default=100000.0, gt=0, description="Starting capital in USD")
    fee_bps: float = Field(default=5.0, ge=0, description="Transaction fee in basis points")
    slippage_bps: float = Field(default=1.0, ge=0, description="Slippage in basis points")


@app.get("/")
def health_check():
    """
    Health check endpoint to verify that the API service is online, 
    check data file availability, and report Redis caching status.
    """
    return {
        "status": "online",
        "engine": "Polars Vectorized Backtester",
        "data_loaded": PARQUET_PATH.exists(),
        "redis_caching": REDIS_AVAILABLE
    }


@app.post("/backtest")
def run_backtest(payload: BacktestRequest):
    """
    Executes a quantitative backtest based on user-supplied parameters. 
    Checks Redis cache first for lightning-fast responses on repeated queries; 
    otherwise, runs the backtest computation engine and caches the result.
    """
    if not PARQUET_PATH.exists():
        raise HTTPException(
            status_code=404, 
            detail="Market data file not found. Please run data ingestion first."
        )

    # Ensure fast span is strictly less than slow span
    if payload.fast_span >= payload.slow_span:
        raise HTTPException(
            status_code=400,
            detail="fast_span must be strictly less than slow_span."
        )

    # Construct a unique cache key based on the request parameters
    cache_key = f"backtest:{payload.fast_span}:{payload.slow_span}:{payload.initial_capital}:{payload.fee_bps}:{payload.slippage_bps}"

    # 1. Check Redis cache if available
    if REDIS_AVAILABLE:
        cached_result = redis_client.get(cache_key)
        if cached_result:
            return {
                "source": "cache",
                "data": json.loads(cached_result)
            }

    # 2. Initialize backtester engine with payload parameters
    engine = VectorizedBacktester(
        parquet_path=PARQUET_PATH,
        initial_capital=payload.initial_capital,
        fee_rate=payload.fee_bps / 10000.0,
        slippage_rate=payload.slippage_bps / 10000.0
    )

    # 3. Run strategy calculation and compute performance metrics
    df_results = engine.run_strategy(fast_span=payload.fast_span, slow_span=payload.slow_span)
    metrics = engine.compute_metrics(df_results)

    response_data = {
        "symbol": "AAPL",
        "parameters": payload.dict(),
        "metrics": metrics
    }

    # 4. Store result in Redis cache with an expiration time (e.g., 1 hour / 3600 seconds)
    if REDIS_AVAILABLE:
        redis_client.setex(cache_key, 3600, json.dumps(response_data))

    return {
        "source": "computation_engine",
        "data": response_data
    }