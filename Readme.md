# 🚀 StockVision AI: Next-Gen Stock Price Predictor

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)  
[![Keras](https://img.shields.io/badge/Keras-v2.12-orange)](https://keras.io/)  
[![Streamlit](https://img.shields.io/badge/Streamlit-v1.28-green)](https://streamlit.io/)  
[![yFinance](https://img.shields.io/badge/yFinance-latest-purple)](https://pypi.org/project/yfinance/)

---

## 🔥 Project Overview
**StockVision AI** is an AI-powered stock price prediction system that forecasts market trends **30 days ahead**.  
It leverages **LSTM deep learning**, real-time stock data, and moving average visualizations to provide actionable insights for traders and enthusiasts.  

**Key Features:**  
- Predicts stock prices 30 days into the future using historical data.  
- Visualizes **100-day & 200-day Moving Averages** against closing prices.  
- Interactive **Streamlit interface** for live stock monitoring.  
- Handles **any stock symbol globally** via Yahoo Finance API.  
- Scalable LSTM model with multiple layers and dropout for better accuracy.  

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** Keras, TensorFlow, Numpy, Pandas, Matplotlib, Scikit-learn  
- **APIs:** Yahoo Finance (`yfinance`)  
- **Frontend:** Streamlit for interactive visualization  
- **Data Scaling:** MinMaxScaler for LSTM inputs  

---

## 📈 How It Works
1. Fetches **historical stock data** via `yfinance`.  
2. Computes **100-day and 200-day moving averages**.  
3. Trains a **multi-layer LSTM model** on historical closing prices.  
4. Predicts **next 30 days prices**.  
5. Visualizes historical vs predicted prices in a **dynamic Streamlit dashboard**.  

---

## 🛠 Installation & Usage

1. **Clone the repo:**  
```bash
git clone https://github.com/aaryan7174/StockPricePrediction.git
cd StockPricePrediction
```

## Install dependencies:

pip install -r requirements.txt


## Run the app:

streamlit run app.py

## 🌟 Highlights

Multi-layer LSTM with dropout regularization for robust prediction.

Real-time live stock info & analytics from Yahoo Finance.

Easy-to-use interactive web interface.

Ready for deployment or further research into financial AI forecasting.

## 📌 Author

Aaryan Rana – AI Engineer & Data Enthusiast
