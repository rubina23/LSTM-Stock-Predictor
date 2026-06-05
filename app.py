import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout

# Main website settings
st.set_page_config(page_title="সহজ Stock Predictor", layout="wide")
st.title('📈 Simple Stock Market Price Predictor App')
st.write('This app uses AI (LSTM model) to analyze historical data and predict future stock prices!')

# Sidebar - User input area
st.sidebar.header('Provide Your Preferences')
ticker = st.sidebar.text_input('Company Ticker (e.g., AAPL, GOOGL, TSLA)', 'AAPL')
start_date = st.sidebar.date_input('Start Date', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('End Date', pd.to_datetime('2023-01-01'))

# 1. Download Data
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

st.subheader(f'**{ticker}** Historical Price Graph')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Actual Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
st.pyplot(fig1)

st.write("---")

# 2. Start Prediction
if st.button('Predict Future Price'):
    with st.spinner('AI model is calculating... Please wait...'):
        
        # Prepare Data (Scaling)
        dataset = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        
        training_data_len = int(np.ceil(len(dataset) * .8))
        test_data = scaled_data[training_data_len - 60: , :]
        
        x_test = []
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
            
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # 3. Create the empty model architecture (with new Input layer)
        model = Sequential()
        model.add(Input(shape=(x_test.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # 4. Load the pre-trained Weights
        model.load_weights('model.weights.h5')

        # Generate Predictions
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Prepare graph to show results
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        st.subheader('রResult: Actual Price vs Predicted Price')
        fig2 = plt.figure(figsize=(16, 8))
        plt.plot(train['Close'], color='blue', label='Historical Price')
        plt.plot(valid['Close'], color='green', label='Actual Price (Current)')
        plt.plot(valid['Predictions'], color='red', label='Predicted Price')
        plt.legend()
        st.pyplot(fig2)
        
        st.success('✅ Prediction generated successfully!')
