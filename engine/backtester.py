from pathlib import Path
import numpy as np
import polars as pl


class VectorizedBacktester:
    """
    A high-performance vectorized backtesting engine built using Polars.
    Simulates dual Exponential Moving Average (EMA) crossover trading strategies
    with realistic transaction cost and slippage modeling.
    """

    def __init__(
        self,
        parquet_path: Path,
        initial_capital: float = 100000.0,
        fee_rate: float = 0.0005,      # 5 basis points (0.05%) per trade
        slippage_rate: float = 0.0001  # 1 basis point (0.01%) price impact
    ):
        """
        Initializes the backtester with dataset path, starting capital, and cost parameters.

        Parameters:
        -----------
        parquet_path : Path
            Path to the historical OHLCV Apache Parquet dataset.
        initial_capital : float
            Starting cash portfolio value in USD (default: $100,000).
        fee_rate : float
            Proportional transaction fee rate charged on trades.
        slippage_rate : float
            Proportional execution slippage rate modeling market friction.
        """
        self.parquet_path = parquet_path
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

        # Load the Parquet dataset into memory as a Polars DataFrame
        self.df = pl.read_parquet(self.parquet_path)

    def run_strategy(self, fast_span: int = 50, slow_span: int = 200) -> pl.DataFrame:
        """
        Executes the full vectorized pipeline: computes EMAs, generates crossover signals,
        models position changes, accounts for fees/slippage, and calculates strategy returns.

        Parameters:
        -----------
        fast_span : int
            Span parameter for the fast Exponential Moving Average (default: 50).
        slow_span : int
            Span parameter for the slow Exponential Moving Average (default: 200).

        Returns:
        --------
        pl.DataFrame
            A DataFrame containing timestamps, prices, indicators, positions, and net equity curves.
        """
        # 1. Compute technical indicators (Exponential Moving Averages) using Polars expression API
        df_strat = self.df.with_columns([
            pl.col("close").ewm_mean(span=fast_span, adjust=False).alias("ema_fast"),
            pl.col("close").ewm_mean(span=slow_span, adjust=False).alias("ema_slow"),
        ])

        # 2. Generate raw trading signals: 1 when fast EMA > slow EMA (Bullish), else 0 (Bearish/Cash)
        df_strat = df_strat.with_columns(
            pl.when(pl.col("ema_fast") > pl.col("ema_slow"))
            .then(1)
            .otherwise(0)
            .alias("target_position")
        )

        # 3. Shift target position by 1 period to avoid lookahead bias (execute trades on the next bar open/close)
        df_strat = df_strat.with_columns(
            pl.col("target_position").shift(1).fill_null(0).alias("position")
        )

        # 4. Detect trade executions where position changes from the previous period
        df_strat = df_strat.with_columns(
            (pl.col("position") != pl.col("position").shift(1).fill_null(0))
            .cast(pl.Int8)
            .alias("trade_executed")
        )

        # 5. Calculate raw asset returns based on closing prices
        df_strat = df_strat.with_columns([
            pl.col("close").pct_change().fill_null(0.0).alias("market_return")
        ])

        # 6. Calculate gross strategy returns before friction costs
        df_strat = df_strat.with_columns([
            (pl.col("position") * pl.col("market_return")).alias("gross_strategy_return")
        ])

        # 7. Model transaction friction: apply fees and slippage whenever a trade is executed
        df_strat = df_strat.with_columns([
            (
                pl.col("trade_executed") * (self.fee_rate + self.slippage_rate)
            ).alias("transaction_cost")
        ])

        # 8. Compute net strategy returns and compound them into a cumulative portfolio value curve
        df_strat = df_strat.with_columns([
            (pl.col("gross_strategy_return") - pl.col("transaction_cost")).alias("net_strategy_return")
        ])

        df_strat = df_strat.with_columns([
            (
                self.initial_capital * (1 + pl.col("net_strategy_return")).cum_prod()
            ).alias("portfolio_value_net")
        ])

        return df_strat

    def compute_metrics(self, df_results: pl.DataFrame) -> dict:
        """
        Calculates key institutional performance and risk metrics from backtest results.

        Parameters:
        -----------
        df_results : pl.DataFrame
            The processed DataFrame returned by `run_strategy()`.

        Returns:
        --------
        dict
            A dictionary containing institutional metrics (CAGR, Sharpe, Drawdown, Volatility, etc.).
        """
        # Convert necessary columns to NumPy arrays for fast metric computation
        portfolio_vals = df_results["portfolio_value_net"].to_numpy()
        net_returns = df_results["net_strategy_return"].to_numpy()

        # Extract beginning and ending values
        start_val = portfolio_vals[0]
        end_val = portfolio_vals[-1]
        
        # Calculate total return percentage
        total_return = (end_val - start_val) / start_val

        # Approximate number of trading years (assuming 252 trading days per year)
        n_days = len(portfolio_vals)
        n_years = n_days / 252.0 if n_days > 0 else 1.0

        # Compound Annual Growth Rate (CAGR)
        cagr = (end_val / start_val) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        # Annualized Volatility (standard deviation of daily returns scaled by sqrt(252))
        annual_volatility = np.std(net_returns) * np.sqrt(252)

        # Sharpe Ratio (assuming a 0% risk-free rate for simplicity)
        sharpe_ratio = (cagr / annual_volatility) if annual_volatility > 0 else 0.0

        # Maximum Drawdown calculation
        rolling_max = np.maximum.accumulate(portfolio_vals)
        drawdowns = (portfolio_vals - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)

        return {
            "Starting Capital": float(start_val),
            "Ending Portfolio Value": float(end_val),
            "Total Return (%)": float(total_return * 100),
            "CAGR (%)": float(cagr * 100),
            "Annualized Volatility (%)": float(annual_volatility * 100),
            "Sharpe Ratio": float(sharpe_ratio),
            "Maximum Drawdown (%)": float(max_drawdown * 100)
        }