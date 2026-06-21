# Stock Predictor AI

An AI-powered stock price forecasting app built with Streamlit, a custom LSTM neural network (pure NumPy), and yfinance.

## Features

- **LSTM Neural Network** — 2-layer stacked LSTM built from scratch with NumPy (no TensorFlow/PyTorch dependency)
- **12 Technical Indicators** — RSI, MACD, Bollinger Bands, ATR, OBV, moving averages (MA20/50/200), Stochastics
- **Price Forecast with Confidence Bands** — widening uncertainty bands over the forecast horizon
- **Interactive Charts** — Candlestick + indicator overlay, actual vs predicted, training loss curve
- **Signal Summary** — RSI sentiment, MACD crossover, Bollinger %B, Golden/Death Cross
- **Live Data** — fetches any ticker via yfinance (NYSE, NASDAQ, crypto with -USD suffix)

## Project Structure

```
stock_predictor/
├── app.py           # Streamlit UI — all pages and charts
├── model.py         # LSTM neural network + training + forecasting
├── indicators.py    # Technical indicator computation
├── utils.py         # Formatting helpers + yfinance company info
├── style.css        # Custom CSS for Streamlit
├── requirements.txt # Python dependencies
└── README.md
```

## Deploy to Streamlit Cloud (free)

1. Push this folder to a public GitHub repository
2. Go to https://share.streamlit.io and sign in
3. Click "New app" → select your repo → set `app.py` as the main file
4. Click "Deploy" — Streamlit Cloud installs requirements.txt automatically

The app will be live at `https://<your-app>.streamlit.app` within a minute.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Enter a ticker symbol (AAPL, TSLA, NVDA, BTC-USD, etc.)
2. Choose training data period (1–5 years)
3. Set forecast horizon (7–60 days)
4. Adjust lookback window and training epochs
5. Click **Run Prediction**

## How the Model Works

The LSTM reads sequences of `lookback` days (default 60) of 12 features and predicts the next day's closing price. This is repeated autoregressively to produce the multi-day forecast. Confidence bands are derived from the standard deviation of residuals on recent test data, widened proportionally with forecast distance.

**Important note**: Stock price forecasting is inherently uncertain. This tool is for educational and research purposes. Past patterns do not guarantee future results.

## Supported Tickers

Any symbol available on Yahoo Finance:
- US stocks: `AAPL`, `TSLA`, `MSFT`, `NVDA`, `GOOGL`, `AMZN`
- Indian stocks: `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- Crypto: `BTC-USD`, `ETH-USD`
- ETFs: `SPY`, `QQQ`, `VTI`
