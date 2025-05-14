import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Streamlit UI setup
st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction using LSTM")

# Sidebar input
st.sidebar.header("Enter Stock Info")
ticker = st.sidebar.text_input("Stock Ticker", value="AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
predict_btn = st.sidebar.button("Predict")

# Sequence creation
def create_sequences(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

if predict_btn:
    try:
        # Step 1: Load Data
        df = yf.download(ticker, start=start_date, end=end_date)
        if df.empty:
            st.error("⚠️ No data found. Check the stock ticker or date range.")
        else:
            data = df[['Close']].dropna()
            st.subheader(f"📊 Historical Closing Prices for {ticker}")
            st.line_chart(data)

            # Step 2: Scale
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data)

            # Step 3: Train/Test split
            train_size = int(len(scaled_data) * 0.8)
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]

            if len(train_data) < 60 or len(test_data) < 60:
                st.error("Not enough data to build model. Use a longer date range.")
            else:
                # Step 4: Create sequences
                X_train, y_train = create_sequences(train_data, 60)
                X_test, y_test = create_sequences(test_data, 60)

                X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

                # Step 5: Build LSTM model
                model = Sequential()
                model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
                model.add(LSTM(50, return_sequences=False))
                model.add(Dense(25))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')

                # Step 6: Train
                model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

                # Step 7: Predict
                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)

                # Step 8: Plot
                train = data[:train_size]
                valid = data[train_size:].copy()
                valid = valid.iloc[60:]
                valid['Predictions'] = predictions

                st.subheader("📉 Real vs Predicted Prices")
                fig, ax = plt.subplots(figsize=(14, 6))
                ax.plot(train['Close'], label='Train')
                ax.plot(valid['Close'], label='Real')
                ax.plot(valid['Predictions'], label='Predicted')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                st.pyplot(fig)

    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
