import streamlit as st
import yfinance as yf
import pandas as pd

# Set Streamlit title
st.title("Stock Price Viewer")

# Choose ticker
ticker = st.text_input("Enter stock ticker", value="AAPL")

# Download stock data
data = yf.download(ticker, start="2023-01-01", end="2024-01-01", group_by='ticker')

# If downloading multiple tickers, data will have MultiIndex columns.
# This handles both single and multi-ticker formats.

# Check if columns are a MultiIndex
if isinstance(data.columns, pd.MultiIndex):
    try:
        close_data = data[(ticker, 'Close')]
        close_data.name = f'{ticker} Close'
    except KeyError:
        st.error(f"Could not find Close data for {ticker}. Check ticker symbol.")
        st.stop()
else:
    close_data = data['Close']
    close_data.name = f'{ticker} Close'

# Display line chart
st.line_chart(close_data)
