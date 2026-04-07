from __future__ import annotations

import os
from pathlib import Path

# Ensure matplotlib cache is writable in constrained environments.
cache_dir = Path.cwd() / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))

import gymnasium as gym
import gym_anytrading
import numpy as np
import pandas as pd
import yfinance as yf
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import DQN


class DataFetchError(RuntimeError):
    pass


class OHLCVStocksEnv(StocksEnv):
    """StocksEnv variant that uses richer technical features.

    Features per timestep:
    - OHLCV (raw)
    - close_diff (Close_t - Close_{t-1})
    - pct_return (Close_t / Close_{t-1} - 1)
    - log_return (log(Close_t) - log(Close_{t-1}))
    - range_frac ((High-Low)/Close)
    - SMA/EMA ratios (SMA_n/Close - 1, EMA_n/Close - 1)
    - RSI (scaled to 0..1)
    - rolling volatility (std of log returns)
    - volume z-score (rolling)
    """

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        up = delta.clip(lower=0.0)
        down = (-delta).clip(lower=0.0)
        roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
        roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
        rs = roll_up / roll_down.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
        close = d["Close"].astype("float64")

        close_diff = close.diff().fillna(0.0)
        pct_return = close.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0)
        log_return = np.log(close.replace(0.0, np.nan)).diff().replace([np.inf, -np.inf], 0.0).fillna(0.0)

        range_frac = ((d["High"] - d["Low"]) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        sma_5 = (close.rolling(5, min_periods=1).mean() / close.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        sma_10 = (close.rolling(10, min_periods=1).mean() / close.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        ema_10 = (close.ewm(span=10, adjust=False).mean() / close.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        rsi_14 = (OHLCVStocksEnv._rsi(close, 14) / 100.0).clip(0.0, 1.0)

        vol_10 = log_return.rolling(10, min_periods=1).std().fillna(0.0)

        vol_roll_mean = d["Volume"].rolling(20, min_periods=1).mean()
        vol_roll_std = d["Volume"].rolling(20, min_periods=1).std().replace(0.0, np.nan)
        volume_z = ((d["Volume"] - vol_roll_mean) / vol_roll_std).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        feats = pd.DataFrame(
            {
                "Open": d["Open"],
                "High": d["High"],
                "Low": d["Low"],
                "Close": d["Close"],
                "Volume": d["Volume"],
                "close_diff": close_diff,
                "pct_return": pct_return,
                "log_return": log_return,
                "range_frac": range_frac,
                "sma_5_ratio": sma_5,
                "sma_10_ratio": sma_10,
                "ema_10_ratio": ema_10,
                "rsi_14": rsi_14,
                "vol_10": vol_10,
                "volume_z": volume_z,
            },
            index=d.index,
        )
        return feats.fillna(0.0)

    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]

        raw = self.df.iloc[start:end].copy()
        feats = self._build_features(raw)

        if feats.empty:
            raise ValueError("No rows available for the selected frame/window.")

        prices = raw["Close"].to_numpy(dtype=np.float32)
        signal_features = feats.to_numpy(dtype=np.float32)

        return prices, signal_features


def get_price_data(
    ticker: str = "NVDA",
    start: str = "2022-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Return OHLCV dataframe.

    Raises DataFetchError if data cannot be fetched or is empty.
    """

    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
    except Exception:
        df = pd.DataFrame()

    # yfinance may return MultiIndex columns depending on settings/version.
    # gym-anytrading expects single-level OHLCV columns (at least "Close").
    if isinstance(df, pd.DataFrame) and isinstance(df.columns, pd.MultiIndex) and len(df.columns) > 0:
        # Common formats:
        # - columns: (field, ticker) e.g. ("Close", "NVDA")
        # - columns: (ticker, field) e.g. ("NVDA", "Close")
        lvl0 = df.columns.get_level_values(0)
        lvl1 = df.columns.get_level_values(1)
        if "Close" in set(lvl0):
            chosen = next(iter(lvl1))
            df = df.xs(chosen, level=1, axis=1)
        elif "Close" in set(lvl1):
            chosen = next(iter(lvl0))
            df = df.xs(chosen, level=0, axis=1)

    if df is None or df.empty:
        raise DataFetchError(
            f"Could not fetch data for ticker={ticker!r}. "
            "yfinance returned no rows (check internet/proxy access to Yahoo Finance)."
        )

    return df


def make_env(df: pd.DataFrame, window_size: int = 10, frame_end: int = 500):
    frame_end = min(frame_end, len(df) - 1)
    if frame_end <= window_size:
        raise ValueError(f"Not enough data points: len(df)={len(df)}")
    return OHLCVStocksEnv(df=df, frame_bound=(window_size, frame_end), window_size=window_size)

def main():
    try:
        df = get_price_data()
    except DataFetchError as e:
        raise SystemExit(str(e))
    env = make_env(df)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        target_update_interval=500,
        exploration_fraction=0.1,
        device="auto",
    )

    print("Training started...")
    model.learn(total_timesteps=20000)
    print("Training complete.")

    print("\nTesting the trained agent...")
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    import matplotlib

    if os.environ.get("DISPLAY", "") == "":
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 6))
    env.unwrapped.render_all()
    out_path = Path.cwd() / "trades.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()