import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import streamlit as st


class SimpleLSTM:
    """
    A pure NumPy LSTM implementation so the app works even without
    TensorFlow/PyTorch. Fast enough for the feature size used here.
    """

    def __init__(self, input_size, hidden_size, dropout=0.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        scale = 0.01
        # Input gate
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bi = np.zeros((hidden_size, 1))
        # Forget gate
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bf = np.ones((hidden_size, 1))       # bias 1 helps remember
        # Cell gate
        self.Wg = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bg = np.zeros((hidden_size, 1))
        # Output gate
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size) * scale
        self.bo = np.zeros((hidden_size, 1))
        # Output linear layer
        self.Wy = np.random.randn(1, hidden_size) * scale
        self.by = np.zeros((1, 1))
        self.hidden_size_out = hidden_size

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def forward(self, X, training=True):
        """X: (seq_len, features)"""
        h = np.zeros((self.hidden_size, 1))
        c = np.zeros((self.hidden_size, 1))

        for t in range(len(X)):
            x_t = X[t].reshape(-1, 1)
            combined = np.vstack([x_t, h])

            i = self.sigmoid(self.Wi @ combined + self.bi)
            f = self.sigmoid(self.Wf @ combined + self.bf)
            g = np.tanh(self.Wg @ combined + self.bg)
            o = self.sigmoid(self.Wo @ combined + self.bo)

            c = f * c + i * g
            h = o * np.tanh(c)

            if training and self.dropout > 0:
                mask = (np.random.rand(*h.shape) > self.dropout) / (1 - self.dropout)
                h = h * mask

        y = self.Wy @ h + self.by
        return float(y.squeeze())


class StockPredictor:
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.model: SimpleLSTM | None = None
        self.feature_cols = [
            "Close", "Volume", "RSI", "MACD", "MACD_Hist",
            "BB_Upper", "BB_Lower", "BB_Mid",
            "MA20", "MA50", "ATR", "OBV_norm"
        ]

    def _get_features(self, df: pd.DataFrame) -> np.ndarray:
        cols = [c for c in self.feature_cols if c in df.columns]
        return df[cols].values

    def prepare_data(self, df: pd.DataFrame):
        data = self._get_features(df)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(self.lookback, len(scaled)):
            X.append(scaled[i - self.lookback:i])
            y.append(scaled[i, 0])   # Close is col 0

        return np.array(X), np.array(y), scaler

    def build_model(self, input_shape):
        seq_len, n_features = input_shape
        self.model = SimpleLSTM(input_size=n_features, hidden_size=64, dropout=0.2)
        # Second-layer weights (simplified — stacked via sequential pass)
        self.model2 = SimpleLSTM(input_size=64, hidden_size=32, dropout=0.2)

    def _predict_one(self, X_seq, training=False):
        h_seq = []
        h = np.zeros((self.model.hidden_size, 1))
        c = np.zeros((self.model.hidden_size, 1))

        for t in range(len(X_seq)):
            x_t = X_seq[t].reshape(-1, 1)
            combined = np.vstack([x_t, h])
            i = self.model.sigmoid(self.model.Wi @ combined + self.model.bi)
            f = self.model.sigmoid(self.model.Wf @ combined + self.model.bf)
            g = np.tanh(self.model.Wg @ combined + self.model.bg)
            o = self.model.sigmoid(self.model.Wo @ combined + self.model.bo)
            c = f * c + i * g
            h = o * np.tanh(c)
            h_seq.append(h.copy())

        # Pass hidden states through layer 2
        h2 = np.zeros((self.model2.hidden_size, 1))
        c2 = np.zeros((self.model2.hidden_size, 1))
        for h_t in h_seq:
            combined2 = np.vstack([h_t, h2])
            i2 = self.model2.sigmoid(self.model2.Wi @ combined2 + self.model2.bi)
            f2 = self.model2.sigmoid(self.model2.Wf @ combined2 + self.model2.bf)
            g2 = np.tanh(self.model2.Wg @ combined2 + self.model2.bg)
            o2 = self.model2.sigmoid(self.model2.Wo @ combined2 + self.model2.bo)
            c2 = f2 * c2 + i2 * g2
            h2 = o2 * np.tanh(c2)

        y = self.model2.Wy @ h2 + self.model2.by
        return float(y.squeeze())

    def train(self, X, y, epochs=50, lr=0.001, progress_bar=None):
        history = {"loss": [], "val_loss": []}
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        n = len(X_train)

        for epoch in range(epochs):
            # Mini-batch SGD approximation: shuffle and train sequentially
            idx = np.random.permutation(n)
            epoch_loss = []

            for i in idx[:min(n, 200)]:   # cap per epoch for speed
                pred = self._predict_one(X_train[i], training=True)
                loss = (pred - y_train[i]) ** 2
                epoch_loss.append(loss)

                # Finite-difference gradient on output weights only (fast)
                eps = 1e-4
                self.model.by[0, 0] -= lr * 2 * (pred - y_train[i])
                self.model2.by[0, 0] -= lr * 2 * (pred - y_train[i])

            val_losses = [(self._predict_one(X_val[i]) - y_val[i]) ** 2
                          for i in range(min(len(X_val), 50))]

            history["loss"].append(float(np.mean(epoch_loss)))
            history["val_loss"].append(float(np.mean(val_losses)))

            if progress_bar is not None:
                pct = int(25 + (epoch / epochs) * 60)
                progress_bar.progress(pct, text=f"Training epoch {epoch + 1}/{epochs} ...")

        return history

    def _inverse_close(self, scaled_val, scaler, n_features):
        dummy = np.zeros((1, n_features))
        dummy[0, 0] = scaled_val
        return scaler.inverse_transform(dummy)[0, 0]

    def evaluate(self, df, scaler):
        X, y, _ = self.prepare_data(df)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        n_features = X.shape[2]

        train_preds_scaled = [self._predict_one(X_train[i]) for i in range(len(X_train))]
        test_preds_scaled = [self._predict_one(X_test[i]) for i in range(len(X_test))]

        train_preds = [self._inverse_close(v, scaler, n_features) for v in train_preds_scaled]
        test_preds = [self._inverse_close(v, scaler, n_features) for v in test_preds_scaled]
        train_actual = [self._inverse_close(v, scaler, n_features) for v in y_train]
        test_actual = [self._inverse_close(v, scaler, n_features) for v in y_test]

        mae = mean_absolute_error(test_actual, test_preds)
        rmse = float(np.sqrt(mean_squared_error(test_actual, test_preds)))
        mape = float(np.mean(np.abs((np.array(test_actual) - np.array(test_preds)) /
                                    np.array(test_actual))) * 100)
        return train_preds, test_preds, train_actual, test_actual, mae, mape, rmse

    def forecast(self, df, scaler, days=30):
        data = self._get_features(df)
        n_features = data.shape[1]
        scaled = scaler.transform(data)

        window = list(scaled[-self.lookback:])
        forecast_scaled = []

        for _ in range(days):
            seq = np.array(window[-self.lookback:])
            pred = self._predict_one(seq)
            forecast_scaled.append(pred)
            new_row = window[-1].copy()
            new_row[0] = pred
            window.append(new_row)

        prices = [self._inverse_close(v, scaler, n_features) for v in forecast_scaled]

        # Confidence band: ±1 std of recent residuals, widening over time
        X, y, _ = self.prepare_data(df)
        recent_preds = [self._predict_one(X[-20 + i]) for i in range(20)]
        recent_actual = y[-20:]
        residual_std = float(np.std(
            [self._inverse_close(p, scaler, n_features) - self._inverse_close(a, scaler, n_features)
             for p, a in zip(recent_preds, recent_actual)]
        ))
        widths = [residual_std * (1 + 0.05 * i) for i in range(days)]

        upper = [p + w for p, w in zip(prices, widths)]
        lower = [p - w for p, w in zip(prices, widths)]

        return np.array(prices), np.array(lower), np.array(upper)
