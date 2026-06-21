import pandas as pd
import numpy as np


class TechnicalIndicators:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def compute_all(self) -> pd.DataFrame:
        df = self.df

        # Moving Averages
        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()
        df["MA200"] = df["Close"].rolling(200).mean()

        # Bollinger Bands (20-day, 2 std)
        df["BB_Mid"] = df["MA20"]
        std20 = df["Close"].rolling(20).std()
        df["BB_Upper"] = df["BB_Mid"] + 2 * std20
        df["BB_Lower"] = df["BB_Mid"] - 2 * std20

        # RSI (14-day)
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))

        # MACD (12/26/9 EMA)
        ema12 = df["Close"].ewm(span=12, adjust=False).mean()
        ema26 = df["Close"].ewm(span=26, adjust=False).mean()
        df["MACD"] = ema12 - ema26
        df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
        df["MACD_Hist"] = df["MACD"] - df["Signal"]

        # ATR (14-day Average True Range)
        high_low = df["High"] - df["Low"]
        high_close = (df["High"] - df["Close"].shift()).abs()
        low_close = (df["Low"] - df["Close"].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["ATR"] = true_range.rolling(14).mean()

        # OBV (On-Balance Volume), normalised to price scale
        obv = [0]
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
                obv.append(obv[-1] + df["Volume"].iloc[i])
            elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
                obv.append(obv[-1] - df["Volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["OBV"] = obv
        # Normalize OBV to [0, 1] over rolling 252-day window so it doesn't dominate features
        obv_min = df["OBV"].rolling(252, min_periods=50).min()
        obv_max = df["OBV"].rolling(252, min_periods=50).max()
        df["OBV_norm"] = (df["OBV"] - obv_min) / (obv_max - obv_min + 1e-9)

        # Stochastic %K / %D (14-day)
        low14 = df["Low"].rolling(14).min()
        high14 = df["High"].rolling(14).max()
        df["Stoch_K"] = 100 * (df["Close"] - low14) / (high14 - low14 + 1e-9)
        df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

        df = df.dropna()
        return df
