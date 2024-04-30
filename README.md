**Stock Market Predictor**
A web application built with Python and Streamlit for predicting stock prices and displaying live stock information.

**Overview**
This project utilizes machine learning techniques to predict future stock prices based on historical data. It also provides real-time information about a given stock, including previous close, open, bid, ask, day's range, volume, and more.

**Features**
Live Stock Information: Fetches real-time data about a given stock symbol, including various metrics such as previous close, bid, ask, volume, etc.
Price Prediction: Utilizes a machine learning model to predict future stock prices for the next 30 days.
Interactive Visualization: Displays historical stock data along with moving averages and visualizes the predicted prices against the actual prices.

Requirements
Python 3.x
Libraries: numpy, pandas, yfinance, keras, streamlit, matplotlib

Installation
Clone the repository:
git clone https://github.com/your-username/stock-market-predictor.git

Install the required dependencies:
pip install -r requirements.txt


Run the Streamlit app:
streamlit run app.py

Enter the stock symbol in the text input field.
Select the desired timeframe for data analysis.
Explore live stock information and predicted prices.
Screenshots


Credits
Streamlit - For building interactive web applications with Python.
Yahoo Finance - For providing access to financial data.
