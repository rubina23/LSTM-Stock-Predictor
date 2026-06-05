# 📈 LSTM Stock Market Predictor

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-red)

## 💡 About The Project
This project is an interactive web application that predicts future stock prices using a pre-trained **Long Short-Term Memory (LSTM)** Neural Network. It fetches real-time historical stock data, processes it, and provides a visual comparison between actual stock prices and predicted prices. 

To ensure blazing-fast performance on the web server, the model was trained beforehand using Google Colab, and only the highly optimized **model weights** are loaded during inference to prevent memory crashes and reduce prediction time to just a few seconds.

🔗 **Live Demo:** [Click Here to View the App](https://lstm-stock-predictor-d.streamlit.app/)

## ✨ Key Features
* **Real-time Data Fetching:** Automatically downloads historical stock data using the `yfinance` API.
* **Interactive UI:** Built with Streamlit, allowing users to select any company ticker (e.g., AAPL, GOOGL, TSLA) and custom date ranges.
* **Fast Inference:** Utilizes `.weights.h5` to load pre-trained LSTM weights instead of training on the fly, avoiding server timeouts.
* **Data Visualization:** Compares actual closing prices with AI-predicted prices using detailed Matplotlib charts.

## 🛠️ Tech Stack
* **Language:** Python
* **Deep Learning Framework:** TensorFlow / Keras (LSTM)
* **Web Framework:** Streamlit
* **Data Processing:** Pandas, NumPy, Scikit-learn (MinMaxScaler)
* **Data Visualization:** Matplotlib
* **API:** Yahoo Finance (`yfinance`)

## 📂 Project Structure
* `app.py`: The main Streamlit application script containing the UI and inference logic.
* `model.weights.h5`: The pre-trained weights of the LSTM model.
* `requirements.txt`: List of dependencies required to run the app in the cloud environment.
* `README.md`: Project documentation.

## 🚀 How to Run Locally

If you want to test this project on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/your-username/LSTM-Stock-Predictor.git](https://github.com/your-username/LSTM-Stock-Predictor.git)
cd LSTM-Stock-Predictor

```
#### 2. Install dependencies:
```
pip install -r requirements.txt

```
#### 3. Run the Streamlit app:
```
streamlit run app.py

```
## 📊 Result & Visuals
The model successfully identifies stock price patterns and trends. The final output is an interactive graph where:

- Blue Line: Historical Training Data
- Green Line: Actual Current Price
- Red Line: AI Predicted Price

Designed and Developed for demonstrating Time-Series Data Forecasting using Deep Learning.
