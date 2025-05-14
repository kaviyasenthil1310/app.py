import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Set page config
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# Sidebar inputs
st.sidebar.header("Input Parameters")
ticker = st.sidebar.text_input("Stock Ticker (e.g. AAPL)", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
predict_button = st.sidebar.button("Predict")

if predict_button:
    st.subheader(f"Stock Data for {ticker}")
    df = yf.download(ticker, start=start_date, end=end_date)
    
    if df.empty:
        st.error("No data returned. Check ticker or date range.")
    else:
        data = df[['Close']].dropna()
        st.line_chart(data)

        # Preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        def create_sequences(dataset, time_step=60):
            X, y = [], []
            for i in range(time_step, len(dataset)):
                X.append(dataset[i - time_step:i, 0])
                y.append(dataset[i, 0])
            return np.array(X), np.array(y)

        X_train, y_train = create_sequences(train_data)
        X_test, y_test = create_sequences(test_data)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 1)),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

        # Predict
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Visualize
        train = data[:train_size]
        valid = data[train_size:].copy()
        valid = valid.iloc[60:].copy()
        valid['Predictions'] = predictions

        st.subheader("📊 Real vs Predicted Prices")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(train['Close'], label="Train")
        ax.plot(valid[['Close']], label="Real")
        ax.plot(valid[['Predictions']], label="Predictions")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price USD ($)")
        ax.set_title("LSTM Model - Real vs Predicted")
        ax.legend()
        st.pyplot(fig)
