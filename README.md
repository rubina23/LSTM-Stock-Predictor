# 📈 LSTM Stock Price Predictor

## About The Project
This project uses a Deep Learning model (Long Short-Term Memory - LSTM) to predict the future closing prices of stocks (like Apple, Google, etc.) based on past 60-day historical data.

## 🛠️ Built With
* **Python**
* **TensorFlow / Keras** (For the LSTM Neural Network)
* **yfinance** (For fetching real-time stock data)
* **Pandas & NumPy** (For data processing)
* **Matplotlib** (For data visualization)

## 🚀 How It Works
1. Fetches historical stock data using the `yfinance` API.
2. Scales the data between 0 and 1 using `MinMaxScaler`.
3. Trains a multi-layer LSTM model with Dropout to prevent overfitting.
4. Predicts the next day's price and plots a comparison graph between Actual and Predicted prices.

## 📊 Result
<!-- *(এখানে আপনার Colab-এ আসা সুন্দর গ্রাফটির একটি স্ক্রিনশট গিটহাবে আপলোড করে লিংকটি বসিয়ে দেবেন)* -->
