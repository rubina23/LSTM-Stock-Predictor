import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout

# ওয়েবসাইটের মূল সেটিং
st.set_page_config(page_title="সহজ Stock Predictor", layout="wide")
st.title('📈 শেয়ার বাজারের দাম অনুমান করার সহজ অ্যাপ')
st.write('এই অ্যাপটি AI (LSTM মডেল) ব্যবহার করে অতীত ডেটা বিশ্লেষণ করে এবং চোখের পলকে শেয়ারের আগামীকালের দাম ধারণা করতে পারে!')

# সাইডবার - ইউজারের ইনপুট নেওয়ার জায়গা
st.sidebar.header('আপনার পছন্দমতো তথ্য দিন')
ticker = st.sidebar.text_input('কোম্পানির কোড (যেমন: AAPL, GOOGL)', 'AAPL')
start_date = st.sidebar.date_input('শুরুর তারিখ', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('শেষের তারিখ', pd.to_datetime('2023-01-01'))

# ১. ডেটা ডাউনলোড করা
data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

st.subheader(f'**{ticker}** কোম্পানির পুরোনো দামের গ্রাফ')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='আসল দাম', color='blue')
plt.xlabel('তারিখ')
plt.ylabel('দাম ($)')
plt.legend()
st.pyplot(fig1)

st.write("---")

# ২. মূল ম্যাজিক (প্রেডিকশন শুরু)
if st.button('ভবিষ্যতের দাম দেখুন'):
    with st.spinner('AI মডেল হিসাব করছে... একটু অপেক্ষা করুন...'):
        
        # ডেটা রেডি করা (Scaling)
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

        # ৩. মডেলের খালি কাঠামো তৈরি করা (নতুন Input লেয়ারসহ)
        model = Sequential()
        model.add(Input(shape=(x_test.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        # ৪. ম্যাজিক: আগে থেকে ট্রেইন করা শুধু ওজন (Weights) লোড করা
        model.load_weights('model.weights.h5')

        # প্রেডিকশন বা অনুমান বের করা
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # রেজাল্ট দেখানোর জন্য গ্রাফ সাজানো
        train = data[:training_data_len]
        valid = data[training_data_len:].copy()
        valid['Predictions'] = predictions

        st.subheader('রেজাল্ট: আসল দাম বনাম AI এর প্রেডিক্ট করা দাম')
        fig2 = plt.figure(figsize=(16, 8))
        plt.plot(train['Close'], color='blue', label='পুরোনো দাম')
        plt.plot(valid['Close'], color='green', label='আসল দাম (বর্তমান)')
        plt.plot(valid['Predictions'], color='red', label='AI এর অনুমান করা দাম')
        plt.legend()
        st.pyplot(fig2)
        
        st.success('✅ ম্যাজিক শেষ! খুব সহজেই প্রেডিকশন বের করা হয়েছে।')
