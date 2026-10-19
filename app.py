from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import polars as pl
from engine.backtester import VectorizedBacktester

# Page Configuration
st.set_page_config(
    page_title="Financial Time-Series & Backtesting Engine",
    page_icon="📈",
    layout="wide"
)

PARQUET_PATH = Path("./data/aapl_daily.parquet")

st.title("📈 Quantitative Backtesting Engine")
st.markdown(
    "Perform high-performance vectorized backtesting on daily OHLCV time-series data using **Polars**."
)

if not PARQUET_PATH.exists():
    st.error("Parquet data file not found in `./data/aapl_daily.parquet`. Please run ingestion first.")
    st.stop()

# Sidebar - Parameter Controls
st.sidebar.header("Strategy Parameters")
initial_capital = st.sidebar.number_input(
    "Initial Capital ($)", min_value=1000.0, max_value=1000000.0, value=100000.0, step=5000.0
)
fast_span = st.sidebar.slider("Fast EMA Span", min_value=5, max_value=100, value=50)
slow_span = st.sidebar.slider("Slow EMA Span", min_value=20, max_value=300, value=200)

fee_bps = st.sidebar.slider("Transaction Fee (bps)", min_value=0.0, max_value=20.0, value=5.0)
slippage_bps = st.sidebar.slider("Slippage (bps)", min_value=0.0, max_value=10.0, value=1.0)

if fast_span >= slow_span:
    st.sidebar.error("Fast EMA Span must be strictly smaller than Slow EMA Span!")
    st.stop()

# Run Engine Backtest
engine = VectorizedBacktester(
    parquet_path=PARQUET_PATH,
    initial_capital=initial_capital,
    fee_rate=fee_bps / 10000.0,
    slippage_rate=slippage_bps / 10000.0
)

df_results = engine.run_strategy(fast_span=fast_span, slow_span=slow_span)
metrics = engine.compute_metrics(df_results)

# Display Key Metrics
st.subheader("📊 Institutional Risk & Performance Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Final Value", f"${metrics['Ending Portfolio Value']:,.2f}")
col2.metric("Total Return", f"{metrics['Total Return (%)']:.2f}%")
col3.metric("CAGR", f"{metrics['CAGR (%)']:.2f}%")
col4.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
col5.metric("Max Drawdown", f"{metrics['Maximum Drawdown (%)']:.2f}%")

st.divider()

# Convert Polars DataFrame to Pandas for Plotly chart rendering
df_pd = df_results.to_pandas()

# Plot Equity Curve vs Benchmark
st.subheader("📉 Portfolio Equity Curve")

fig_equity = go.Figure()

# Portfolio Net Value
fig_equity.add_trace(
    go.Scatter(
        x=df_pd["timestamp"],
        y=df_pd["portfolio_value_net"],
        mode="lines",
        name="Strategy (Net of Costs)",
        line=dict(color="#00CC96", width=2)
    )
)

# Buy & Hold Benchmark
benchmark_value = (df_pd["close"] / df_pd["close"].iloc[0]) * initial_capital
fig_equity.add_trace(
    go.Scatter(
        x=df_pd["timestamp"],
        y=benchmark_value,
        mode="lines",
        name="AAPL Buy & Hold Benchmark",
        line=dict(color="#636EFA", width=1.5, dash="dash")
    )
)

fig_equity.update_layout(
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=20, b=40),
    template="plotly_white",
    height=450
)

st.plotly_chart(fig_equity, use_container_width=True)

# Price & Moving Averages Chart
st.subheader("🕯️ Price Chart & Crossover Signals")

fig_price = go.Figure()
fig_price.add_trace(
    go.Scatter(x=df_pd["timestamp"], y=df_pd["close"], mode="lines", name="AAPL Close", line=dict(color="#B6E880", width=1))
)
fig_price.add_trace(
    go.Scatter(x=df_pd["timestamp"], y=df_pd["ema_fast"], mode="lines", name=f"EMA {fast_span}", line=dict(color="#FF6692", width=1.5))
)
fig_price.add_trace(
    go.Scatter(x=df_pd["timestamp"], y=df_pd["ema_slow"], mode="lines", name=f"EMA {slow_span}", line=dict(color="#FFA15A", width=1.5))
)

fig_price.update_layout(
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=20, b=40),
    template="plotly_white",
    height=400
)

st.plotly_chart(fig_price, use_container_width=True)