# Financial Time-Series Analytics & Backtesting Engine

A production-grade, high-performance quantitative finance analytics and backtesting engine built with **Python**, **Polars**, **FastAPI**, **Docker**, **Redis**, and **Streamlit**. 

Designed with modular architecture and vectorized computation principles to simulate trading strategies efficiently at scale.

---

## Key Features

- **Vectorized Data Engine:** Built on **Polars** and **DuckDB** for blazing-fast columnar time-series transformations and SQL querying over Apache Parquet files.
- **Quantitative Backtesting Pipeline:** Simulates dual moving average (EMA) crossover strategies incorporating realistic friction modeling (**5 bps transaction fees** and **1 bp slippage**).
- **Institutional Risk Metrics:** Computes institutional-grade metrics including CAGR, Annualized Volatility, Sharpe Ratio, and Maximum Drawdown.
- **Containerized FastAPI Backend:** Fully containerized via **Docker**, featuring Redis caching fallback logic for lightning-fast sub-millisecond response times on repeated queries.
- **Interactive Web Dashboard:** A **Streamlit** frontend allowing real-time parameter tuning and dynamic equity curve visualization against a benchmark.
- **Automated Test Suite:** Comprehensive unit testing coverage using `pytest` for both backend endpoints and quantitative calculations.

---

## Tech Stack

- **Data Processing & Analytics:** Polars, DuckDB, PyArrow
- **Backend API & Caching:** FastAPI, Uvicorn, Redis
- **Containerization & DevOps:** Docker, Docker Compose
- **Testing:** Pytest, HTTPX
- **Frontend & Visualization:** Streamlit, Plotly

---

## Project Architecture

```text
financial-engine/
│
├── data/
│   └── aapl_daily.parquet         # Ingested OHLCV market data
│
├── engine/
│   ├── __init__.py
│   └── backtester.py              # Core vectorized backtesting engine & metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_engine.py             # Unit tests for quantitative calculations
│   └── test_api.py                # Unit tests for FastAPI endpoints
│
├── api.py                         # FastAPI service with Redis caching
├── app.py                         # Interactive Streamlit dashboard
├── Dockerfile                     # Container definition
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation