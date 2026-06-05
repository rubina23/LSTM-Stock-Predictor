import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title('📈 Stock Market Predictor App')
st.write('এটি একটি সাধারণ ডেমো। এখানে আপনি যেকোনো কোম্পানির শেয়ারের ডেটা দেখতে পারবেন।')

# ইউজারের কাছ থেকে কোম্পানির কোড নেওয়া
ticker = st.text_input('Enter Stock Ticker (e.g., AAPL, GOOGL, TSLA)', 'AAPL')

# ডেটা ডাউনলোড
data = yf.download(ticker, start='2020-01-01', end='2023-01-01')

st.subheader(f'{ticker} Historical Data')
st.write(data.tail())

# গ্রাফ দেখানো
st.subheader('Closing Price vs Time')
fig = plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Close Price')
plt.legend()
st.pyplot(fig)

st.success('পরবর্তী ধাপে এখানে LSTM মডেলের প্রেডিকশন যুক্ত করা হবে!')
