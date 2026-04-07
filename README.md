# Algo Trading (RL demo)

Small algo-trading demo using `gym-anytrading` + `stable-baselines3` (DQN), packaged with a simple local Flask app so you can type a ticker and get a **BUY / DON'T BUY** signal.

## What’s in here

- **`algo_trading.py`**: downloads (or synthesizes) OHLCV data, trains a DQN agent, evaluates it, and saves a plot to `trades.png`.
- **`algo_trading.py`**: downloads OHLCV data, trains a DQN agent, evaluates it, and saves a plot to `trades.png`.
- **`app.py`** + **`templates/index.html`**: a local web UI (desktop-style) where you enter a ticker and get a recommendation.
- **`pyproject.toml` / `uv.lock`**: dependency management via `uv`.

## Setup (uv)

Create the virtual environment and install dependencies:

```bash
uv venv .venv
uv sync
```

Activate the environment:

```bash
source .venv/bin/activate
```

## Run the training script

```bash
python algo_trading.py
```

Output:

- **`trades.png`**: plot showing buy/sell points from the run.

## Run the Flask app (local “desktop-style”)

Start the server:

```bash
python app.py
```

Open:

- `http://127.0.0.1:5000`

Enter a ticker (e.g. `NVDA`, `AAPL`, `TSLA`) and submit to get a **BUY / DON'T BUY** result.

## Notes / gotchas

- **Yahoo / `yfinance` access**: if your network blocks Yahoo (common in some environments), `yfinance` downloads can fail (e.g. 403). In that case this project **will not fabricate data**—it will **show an error** telling you the data could not be fetched.
- **Plotting**: in non-GUI environments the code uses a non-interactive Matplotlib backend and saves images instead of calling `plt.show()`.

## Add / update dependencies

Add a dependency:

```bash
uv add <package>
```

Sync after pulling changes:

```bash
uv sync
```

## Disclaimer

This project is for learning/demo purposes only and is **not** financial advice.

