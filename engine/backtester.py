from pathlib import Path
import numpy as np
import polars as pl


class VectorizedBacktester:
    """Production-grade vectorized backtesting engine using Polars."""

    def __init__(
        self,
        parquet_path: Path,
        initial_capital: float = 100000.0,
        fee_rate: float = 0.0005,
        slippage_rate: float = 0.0001,
    ):
        self.parquet_path = parquet_path
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage_rate = slippage_rate

    def run_strategy(self, fast_span: int = 50, slow_span: int = 200) -> pl.DataFrame:
        """Executes the dual moving average strategy simulation."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found at {self.parquet_path}")

        lazy_df = pl.scan_parquet(self.parquet_path)

        backtest_plan = (
            lazy_df
            .sort("timestamp")
            .with_columns([
                pl.col("close").ewm_mean(span=fast_span, adjust=False).alias("ema_fast"),
                pl.col("close").ewm_mean(span=slow_span, adjust=False).alias("ema_slow"),
                ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1))
                .fill_null(0.0)
                .alias("asset_return"),
            ])
            .with_columns([
                pl.when(pl.col("ema_fast") > pl.col("ema_slow"))
                .then(1)
                .otherwise(0)
                .alias("position")
            ])
            .with_columns([
                (pl.col("position") - pl.col("position").shift(1))
                .fill_null(0)
                .abs()
                .alias("trade_executed")
            ])
            .with_columns([
                (pl.col("position").shift(1).fill_null(0) * pl.col("asset_return"))
                .sub(pl.col("trade_executed") * (self.fee_rate + self.slippage_rate))
                .alias("net_strategy_return")
            ])
            .with_columns([
                (self.initial_capital * (1.0 + pl.col("net_strategy_return")).cum_prod())
                .alias("portfolio_value_net")
            ])
        )

        return backtest_plan.collect()

    def compute_metrics(self, df_backtest: pl.DataFrame) -> dict:
        """Computes institutional performance metrics from backtest results."""
        returns = df_backtest.select("net_strategy_return").to_numpy().flatten()
        portfolio_values = df_backtest.select("portfolio_value_net").to_numpy().flatten()

        total_days = len(returns)
        years = total_days / 252.0

        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
        cagr = (portfolio_values[-1] / portfolio_values[0]) ** (1.0 / years) - 1.0 if years > 0 else 0.0
        annualized_vol = np.std(returns) * np.sqrt(252)
        sharpe_ratio = (cagr / annualized_vol) if annualized_vol != 0 else 0.0

        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        return {
            "Starting Capital": self.initial_capital,
            "Ending Portfolio Value": portfolio_values[-1],
            "Total Return (%)": total_return * 100,
            "CAGR (%)": cagr * 100,
            "Annualized Volatility (%)": annualized_vol * 100,
            "Sharpe Ratio": sharpe_ratio,
            "Maximum Drawdown (%)": max_drawdown * 100,
        }