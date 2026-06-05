import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ওয়েবসাইটের টাইটেল এবং লেআউট সেটিং
st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title('📈 Stock Market Price Predictor (LSTM)')
st.write('এটি একটি ডিপ লার্নিং (LSTM) প্রজেক্ট, যা অতীত ডেটা বিশ্লেষণ করে শেয়ার বাজারের ভবিষ্যৎ দাম প্রেডিক্ট করে।')

# সাইডবার (ইউজারের ইনপুট নেওয়ার জন্য)
st.sidebar.header('ইউজার ইনপুট')
ticker = st.sidebar.text_input('কোম্পানির কোড (যেমন: AAPL, GOOGL, MSFT)', 'AAPL')
start_date = st.sidebar.date_input('শুরুর তারিখ', pd.to_datetime('2015-01-01'))
end_date = st.sidebar.date_input('শেষের তারিখ', pd.to_datetime('2025-01-01'))

# ১. ডেটা ডাউনলোড এবং প্রদর্শন
st.subheader(f'**{ticker}**-এর ঐতিহাসিক ডেটা')
data = yf.download(ticker, start=start_date, end=end_date)
st.write(data.tail())

# সাধারণ গ্রাফ আঁকা
st.subheader('Closing Price vs Time')
fig1 = plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.xlabel('Date')
plt.ylabel('Close Price USD ($)')
plt.legend()
st.pyplot(fig1)

st.write("---")
st.markdown("### 🤖 AI প্রেডিকশন")
st.write("LSTM মডেল তৈরি এবং প্রেডিকশনের জন্য নিচের বাটনে ক্লিক করুন:")

# বাটন ক্লিক করলে মডেল ট্রেনিং এবং প্রেডিকশন শুরু হবে
if st.button('Predict Price'):
    with st.spinner('ডেটা প্রসেসিং এবং AI মডেল ট্রেনিং হচ্ছে... (এতে ১-২ মিনিট সময় লাগতে পারে)'):
        
        # ২. ডেটা প্রসেসিং
        dataset = data['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        training_data_len = int(np.ceil(len(dataset) * .8))
        train_data = scaled_data[0:int(training_data_len), :]

        x_train, y_train = [], []
        for i in range(60, len(train_data)):
            x_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])
        
        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # ৩. LSTM মডেল তৈরি
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # ৪. মডেল ট্রেনিং (ওয়েবসাইটে দ্রুত দেখানোর জন্য epochs=3 দেওয়া হয়েছে)
        model.fit(x_train, y_train, batch_size=32, epochs=3, verbose=0)

        # ৫. টেস্টিং ডেটা প্রস্তুত করা
        test_data = scaled_data[training_data_len - 60: , :]
        x_test = []
        y_test = dataset[training_data_len:, :]
        
        for i in range(60, len(test_data)):
            x_test.append(test_data[i-60:i, 0])
        
        x_test = np.array(x_test)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

        # ৬. প্রেডিকশন বের করা
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # ৭. প্রেডিকশনের গ্রাফ তৈরি করা
        train = data[:training_data_len]
        valid = data[training_data_len:].copy() # Warning এড়াতে copy() ব্যবহার করা হয়েছে
        valid['Predictions'] = predictions

        st.subheader('Prediction Result')
        fig2 = plt.figure(figsize=(16, 8))
        plt.title(f'LSTM Model Prediction for {ticker}')
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Close Price USD ($)', fontsize=14)
        
        plt.plot(train['Close'], color='blue', label='Train Data (Past)')
        plt.plot(valid['Close'], color='green', label='Actual Price')
        plt.plot(valid['Predictions'], color='red', label='Predicted Price (AI)')
        plt.legend(loc='lower right')
        st.pyplot(fig2)
        
        st.success('✅ প্রেডিকশন সফলভাবে সম্পন্ন হয়েছে!')
