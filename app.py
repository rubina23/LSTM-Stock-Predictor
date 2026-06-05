import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title('📈 Stock Market Price Predictor (Pro Version)')
st.write('এই অ্যাপটি একটি আগে থেকে ট্রেইন করা (Pre-trained) LSTM মডেল ব্যবহার করে চোখের পলকে প্রেডিকশন করতে পারে!')

st.sidebar.header('ইউজার ইনপুট')
ticker = st.sidebar.text_input('কোম্পানির কোড (যেমন: AAPL, GOOGL, MSFT)', 'AAPL')
start_date = st.sidebar.date_input('শুরুর তারিখ', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('শেষের তারিখ', pd.to_datetime('2023-01-01'))

# ডেটা ডাউনলোড
data = yf.download(ticker, start=start_date, end=end_date)

st.subheader(f'**{ticker}**-এর ঐতিহাসিক ডেটা')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
st.pyplot(fig1)

st.write("---")

if st.button('Predict Price'):
    with st.spinner('মডেল প্রেডিক্ট করছে...'):
        
        # ১. ডেটা প্রসেসিং
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

        # ২. সেভ করা মডেল লোড করা (ম্যাজিক লাইন!)
        model = load_model('stock_model.h5')

        # ৩. প্রেডিকশন বের করা
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # ৪. গ্রাফ তৈরি
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        st.subheader('Prediction Result (AI vs Actual)')
        fig2 = plt.figure(figsize=(16, 8))
        plt.plot(train['Close'], color='blue', label='Train Data (Past)')
        plt.plot(valid['Close'], color='green', label='Actual Price')
        plt.plot(valid['Predictions'], color='red', label='Predicted Price (AI)')
        plt.legend()
        st.pyplot(fig2)
        
        st.success('✅ মাত্র কয়েক সেকেন্ডেই প্রেডিকশন সফলভাবে সম্পন্ন হয়েছে!')
