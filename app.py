from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from flask import Flask, render_template, request

from algo_trading import DataFetchError, get_price_data, make_env

from stable_baselines3 import DQN


@dataclass(frozen=True)
class Recommendation:
    ticker: str
    action: str  # "BUY" or "DON'T BUY"
    confidence_note: str


app = Flask(__name__)

# Keep a tiny in-memory cache so repeated requests are faster.
_model_cache: dict[str, DQN] = {}


def _train_or_get_model(ticker: str, timesteps: int = 3000) -> DQN:
    if ticker in _model_cache:
        return _model_cache[ticker]

    df = get_price_data(ticker=ticker)
    env = make_env(df)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        target_update_interval=500,
        exploration_fraction=0.1,
        device="auto",
    )
    model.learn(total_timesteps=timesteps)
    _model_cache[ticker] = model
    return model


def recommend(ticker: str) -> Recommendation:
    ticker = ticker.strip().upper()
    df = get_price_data(ticker=ticker)

    env = make_env(df)
    model = _train_or_get_model(ticker)

    obs, info = env.reset()
    done = False
    last_action: Optional[int] = None
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        last_action = int(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # gym-anytrading convention: 0 = Sell/Short, 1 = Buy/Long
    action_text = "BUY" if last_action == 1 else "DON'T BUY"

    note = (
        "Model is trained quickly on recent history; treat as a demo signal, not financial advice."
    )

    return Recommendation(
        ticker=ticker,
        action=action_text,
        confidence_note=note,
    )


@app.get("/")
def index():
    return render_template("index.html", result=None)


@app.post("/predict")
def predict():
    ticker = request.form.get("ticker", "").strip()
    if not ticker:
        return render_template(
            "index.html",
            result={
                "error": "Please enter a ticker symbol (e.g. NVDA).",
            },
        )

    try:
        rec = recommend(ticker)
        return render_template(
            "index.html",
            result={
                "ticker": rec.ticker,
                "action": rec.action,
                "note": rec.confidence_note,
            },
        )
    except DataFetchError as e:
        return render_template(
            "index.html",
            result={
                "error": str(e),
            },
        )
    except Exception as e:
        return render_template(
            "index.html",
            result={
                "error": f"{type(e).__name__}: {e}",
            },
        )


if __name__ == "__main__":
    # Local-only "desktop-style" app: run on localhost and open in your browser.
    app.run(host="127.0.0.1", port=5000, debug=True)

