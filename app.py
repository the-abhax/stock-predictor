import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

from model import StockPredictor
from indicators import TechnicalIndicators
from utils import format_currency, format_percent, get_company_info

st.set_page_config(
    page_title="Stock Predictor AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Stock Predictor AI")
    st.markdown("Neural network-powered price forecasting using LSTM + technical analysis.")
    st.divider()

    ticker = st.text_input("Ticker Symbol", value="AAPL", placeholder="e.g. TSLA, NVDA, MSFT").upper().strip()

    period_map = {
        "1 Year": "1y",
        "2 Years": "2y",
        "3 Years": "3y",
        "5 Years": "5y",
    }
    period_label = st.selectbox("Training Data Period", list(period_map.keys()), index=1)
    period = period_map[period_label]

    forecast_days = st.slider("Forecast Horizon (days)", min_value=7, max_value=60, value=30, step=1)

    st.divider()
    st.markdown("**Model Settings**")
    lookback = st.slider("Lookback Window (days)", min_value=30, max_value=120, value=60, step=5)
    epochs = st.slider("Training Epochs", min_value=20, max_value=150, value=50, step=10)

    run = st.button("Run Prediction", type="primary", use_container_width=True)

# ── Main Layout ───────────────────────────────────────────────────────────────
st.markdown("# Stock Predictor AI")
st.markdown("LSTM neural network with technical indicators — fetch live data, train, and forecast.")

if not run:
    st.info("Configure settings in the sidebar and click **Run Prediction** to begin.")
    st.stop()

# ── Data Fetching ─────────────────────────────────────────────────────────────
with st.spinner(f"Fetching data for {ticker}..."):
    try:
        raw = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if raw.empty:
            st.error(f"No data found for ticker **{ticker}**. Check the symbol and try again.")
            st.stop()

        # Flatten multi-index columns if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        info = get_company_info(ticker)

    except Exception as e:
        st.error(f"Data fetch error: {e}")
        st.stop()

company_name = info.get("longName", ticker)
current_price = float(raw["Close"].iloc[-1])
prev_price = float(raw["Close"].iloc[-2])
price_change = current_price - prev_price
price_change_pct = price_change / prev_price * 100

# ── Technical Indicators ──────────────────────────────────────────────────────
ti = TechnicalIndicators(raw)
df = ti.compute_all()

# ── Model Training ────────────────────────────────────────────────────────────
progress_bar = st.progress(0, text="Preparing data...")
predictor = StockPredictor(lookback=lookback)

with st.spinner("Training LSTM model..."):
    progress_bar.progress(10, text="Scaling features...")
    X, y, scaler = predictor.prepare_data(df)

    progress_bar.progress(25, text="Building neural network...")
    predictor.build_model(X.shape[1:])

    history = predictor.train(X, y, epochs=epochs, progress_bar=progress_bar)

    progress_bar.progress(85, text="Generating forecast...")
    forecast_prices, conf_lower, conf_upper = predictor.forecast(df, scaler, days=forecast_days)

    progress_bar.progress(100, text="Done.")
    progress_bar.empty()

# ── Evaluate ──────────────────────────────────────────────────────────────────
train_preds, test_preds, train_actual, test_actual, mae, mape, rmse = predictor.evaluate(df, scaler)

# ── Header Metrics ────────────────────────────────────────────────────────────
st.markdown(f"### {company_name} ({ticker})")

col1, col2, col3, col4, col5 = st.columns(5)
delta_color = "normal"

with col1:
    st.metric("Current Price", format_currency(current_price),
              delta=f"{format_currency(price_change)} ({format_percent(price_change_pct)})")
with col2:
    st.metric("Forecast (end)", format_currency(forecast_prices[-1]))
with col3:
    forecast_return = (forecast_prices[-1] - current_price) / current_price * 100
    direction = "UP" if forecast_return > 0 else "DOWN"
    st.metric("Expected Return", format_percent(forecast_return), delta=direction)
with col4:
    st.metric("Model MAE", format_currency(mae))
with col5:
    st.metric("MAPE", format_percent(mape))

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Price Forecast", "Technical Analysis", "Model Performance", "Raw Data"])

# ────────────────────────────── TAB 1: Forecast ──────────────────────────────
with tab1:
    hist_dates = df.index[-180:]
    hist_prices = df["Close"].iloc[-180:].values

    last_date = df.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_prices,
        mode="lines", name="Historical",
        line=dict(color="#4F86F7", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([[last_date], forecast_dates]),
        y=np.concatenate([[current_price], forecast_prices]),
        mode="lines", name="Forecast",
        line=dict(color="#F97316", width=2.5, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(conf_upper) + list(conf_lower[::-1]),
        fill="toself",
        fillcolor="rgba(249,115,22,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band",
        hoverinfo="skip"
    ))

    fig.update_layout(
        title=f"{ticker} — {forecast_days}-Day Price Forecast",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=480,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Forecast Summary**")
        forecast_table = pd.DataFrame({
            "Date": forecast_dates[::max(1, forecast_days // 10)],
            "Predicted Price": [format_currency(p) for p in forecast_prices[::max(1, forecast_days // 10)]],
            "Lower Bound": [format_currency(p) for p in conf_lower[::max(1, forecast_days // 10)]],
            "Upper Bound": [format_currency(p) for p in conf_upper[::max(1, forecast_days // 10)]],
        })
        st.dataframe(forecast_table, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("**Signal Summary**")
        rsi_val = float(df["RSI"].iloc[-1])
        macd_val = float(df["MACD"].iloc[-1])
        signal_val = float(df["Signal"].iloc[-1])
        bb_pos = (float(df["Close"].iloc[-1]) - float(df["BB_Lower"].iloc[-1])) / \
                 (float(df["BB_Upper"].iloc[-1]) - float(df["BB_Lower"].iloc[-1])) * 100

        signals = {
            "RSI (14)": (rsi_val, "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"),
            "MACD": (macd_val, "Bullish" if macd_val > signal_val else "Bearish"),
            "Bollinger %B": (bb_pos, "Upper band" if bb_pos > 80 else "Lower band" if bb_pos < 20 else "Mid-range"),
            "50/200 MA": (None, "Golden Cross" if float(df["MA50"].iloc[-1]) > float(df["MA200"].iloc[-1]) else "Death Cross"),
        }
        for name, (val, label) in signals.items():
            color = "green" if label in ("Bullish", "Golden Cross", "Neutral", "Mid-range") else \
                    "red" if label in ("Bearish", "Death Cross", "Overbought") else "orange"
            val_str = f"{val:.2f}" if val is not None else ""
            st.markdown(f"**{name}**: {val_str} — :{color}[{label}]")

# ────────────────────────────── TAB 2: Technical Analysis ────────────────────
with tab2:
    plot_df = df.iloc[-252:]  # last year

    fig2 = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=("Price + Bollinger Bands + MAs", "Volume", "RSI (14)", "MACD"),
        row_heights=[0.45, 0.18, 0.18, 0.19]
    )

    # Price + BBands + MAs
    fig2.add_trace(go.Candlestick(
        x=plot_df.index, open=plot_df["Open"], high=plot_df["High"],
        low=plot_df["Low"], close=plot_df["Close"], name="OHLC",
        increasing_line_color="#22c55e", decreasing_line_color="#ef4444"
    ), row=1, col=1)

    for ma, color in [("MA20", "#a78bfa"), ("MA50", "#60a5fa"), ("MA200", "#f97316")]:
        fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df[ma], name=ma,
                                   line=dict(color=color, width=1.2)), row=1, col=1)

    fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Upper"], name="BB Upper",
                               line=dict(color="gray", width=1, dash="dash"), showlegend=False), row=1, col=1)
    fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["BB_Lower"], name="BB Lower",
                               line=dict(color="gray", width=1, dash="dash"),
                               fill="tonexty", fillcolor="rgba(100,100,100,0.07)", showlegend=False), row=1, col=1)

    # Volume
    colors = ["#22c55e" if c >= o else "#ef4444"
              for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig2.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], name="Volume",
                           marker_color=colors, showlegend=False), row=2, col=1)

    # RSI
    fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI"], name="RSI",
                               line=dict(color="#818cf8", width=1.5)), row=3, col=1)
    fig2.add_hline(y=70, line_dash="dot", line_color="red", opacity=0.5, row=3, col=1)
    fig2.add_hline(y=30, line_dash="dot", line_color="green", opacity=0.5, row=3, col=1)

    # MACD
    macd_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in plot_df["MACD_Hist"]]
    fig2.add_trace(go.Bar(x=plot_df.index, y=plot_df["MACD_Hist"], name="Histogram",
                           marker_color=macd_colors, showlegend=False), row=4, col=1)
    fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["MACD"], name="MACD",
                               line=dict(color="#60a5fa", width=1.5)), row=4, col=1)
    fig2.add_trace(go.Scatter(x=plot_df.index, y=plot_df["Signal"], name="Signal",
                               line=dict(color="#f97316", width=1.5)), row=4, col=1)

    fig2.update_layout(
        height=820,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ────────────────────────────── TAB 3: Model Performance ─────────────────────
with tab3:
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Absolute Error", format_currency(mae))
    col2.metric("MAPE", format_percent(mape))
    col3.metric("RMSE", format_currency(rmse))

    st.divider()

    # Actual vs Predicted
    test_index = df.index[len(df) - len(test_actual):]

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=test_index, y=test_actual, name="Actual",
                               line=dict(color="#4F86F7", width=2)))
    fig3.add_trace(go.Scatter(x=test_index, y=test_preds, name="Predicted",
                               line=dict(color="#F97316", width=2, dash="dot")))
    fig3.update_layout(
        title="Actual vs Predicted (Test Set)",
        xaxis_title="Date", yaxis_title="Price (USD)",
        height=380, template="plotly_white", hovermode="x unified"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Training loss
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(y=history["loss"], name="Train Loss",
                               line=dict(color="#4F86F7", width=2)))
    fig4.add_trace(go.Scatter(y=history["val_loss"], name="Val Loss",
                               line=dict(color="#F97316", width=2, dash="dot")))
    fig4.update_layout(
        title="Training Loss Curve",
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        height=320, template="plotly_white"
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("**About the Model**")
    st.markdown("""
- **Architecture**: LSTM (Long Short-Term Memory) neural network with 2 stacked layers
- **Features**: Close price, Volume, RSI, MACD, Bollinger Bands, moving averages (MA20/50/200), ATR, OBV
- **Scaler**: MinMaxScaler on all features independently
- **Train/Test Split**: 80% / 20%
- **Loss**: Mean Squared Error
- **Optimizer**: Adam with learning rate 0.001
- **Regularization**: Dropout (0.2) after each LSTM layer
    """)

# ────────────────────────────── TAB 4: Raw Data ──────────────────────────────
with tab4:
    st.markdown(f"**{len(df)} trading days of data** — most recent 100 rows shown below.")
    display_cols = ["Open", "High", "Low", "Close", "Volume", "MA20", "MA50", "RSI", "MACD"]
    st.dataframe(df[display_cols].tail(100).sort_index(ascending=False).round(2),
                 use_container_width=True)

    csv = df.to_csv().encode("utf-8")
    st.download_button("Download full dataset as CSV", csv,
                       file_name=f"{ticker}_data.csv", mime="text/csv")
