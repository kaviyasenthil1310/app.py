import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction with LSTM")

# Sidebar input
st.sidebar.header("Model Settings")
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
predict_button = st.sidebar.button("Start Prediction")

def create_sequences(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

if predict_button:
    df = yf.download(ticker, start=start_date, end=end_date)

    if df.empty or 'Close' not in df:
        st.error("Error: No data found. Check ticker or date range.")
    else:
        data = df[['Close']].dropna()
        st.subheader(f"Stock Closing Prices for {ticker}")
        st.line_chart(data)

        # Scaling
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Train/test split
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Sequences
        time_step = 60
        if len(train_data) <= time_step or len(test_data) <= time_step:
            st.warning("Not enough data for training. Try a wider date range.")
        else:
            X_train, y_train = create_sequences(train_data, time_step)
            X_test, y_test = create_sequences(test_data, time_step)

            # Reshape
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Model
            model = Sequential()
            model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
            model.add(LSTM(50, return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')

            model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)

            predictions = model.predict(X_test)
            predictions = scaler.inverse_transform(predictions)

            # Prepare plotting
            valid = data[train_size:].copy()
            valid = valid.iloc[time_step:]
            valid['Predictions'] = predictions

            train = data[:train_size]

            st.subheader("📊 Real vs Predicted Prices")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(train['Close'], label='Train')
            ax.plot(valid['Close'], label='Real')
            ax.plot(valid['Predictions'], label='Predicted')
            ax.set_title(f"LSTM Model - {ticker}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            st.pyplot(fig)
