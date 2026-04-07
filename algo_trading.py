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
from stable_baselines3 import DQN


def get_price_data(
    ticker: str = "NVDA",
    start: str = "2022-01-01",
    end: str = "2026-01-01",
) -> pd.DataFrame:
    """Return OHLCV dataframe.

    If Yahoo download fails/returns empty (e.g., proxy/403), fall back to a
    deterministic synthetic random-walk series so the RL pipeline still runs.
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
        n = 800
        rng = np.random.default_rng(7)
        steps = rng.normal(loc=0.0005, scale=0.02, size=n)
        close = 100.0 * np.exp(np.cumsum(steps))
        idx = pd.date_range(start=start, periods=n, freq="B")
        df = pd.DataFrame(
            {
                "Open": close,
                "High": close * (1.0 + rng.uniform(0.0, 0.01, size=n)),
                "Low": close * (1.0 - rng.uniform(0.0, 0.01, size=n)),
                "Close": close,
                "Volume": rng.integers(1_000_000, 5_000_000, size=n),
            },
            index=idx,
        )

    return df


def make_env(df: pd.DataFrame, window_size: int = 10, frame_end: int = 500):
    frame_end = min(frame_end, len(df) - 1)
    if frame_end <= window_size:
        raise ValueError(f"Not enough data points: len(df)={len(df)}")
    return gym.make("stocks-v0", df=df, frame_bound=(window_size, frame_end), window_size=window_size)

def main():
    df = get_price_data()
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