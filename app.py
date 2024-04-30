import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import datetime

def get_live_stock_info(stock):
    try:
        stock_info_data = {
            "Previous Close": info.get('previousClose', 'N/A'),
            "Open": info.get('open', 'N/A'),
            "Bid": info.get('bid', 'N/A'),
            "Ask": info.get('ask', 'N/A'),
            "Day's Range": f"{info.get('dayLow', 'N/A')} - {info.get('dayHigh', 'N/A')}",
            "52 Week Range": f"{info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}",
            "Volume": info.get('volume', 'N/A'),
            "Avg. Volume": info.get('averageVolume', 'N/A'),
            "Beta (5Y Monthly)": info.get('beta', 'N/A'),
            "PE Ratio (TTM)": info.get('trailingPE', 'N/A'),
            "EPS (TTM)": info.get('trailingEps', 'N/A'),
            "Ex-Dividend Date": info.get('exDividendDate', 'N/A')
        }

        return stock_info_data

    except Exception as e:
        st.error(f"Error fetching stock information: {e}")
        return None
css_styles = """
<style>
    .hr-line {
        margin-top: 0; 
        margin-bottom: 0; 
        border: none;
        height: 1px;
        background-color: grey; 
        opacity: 0.6;
    }
    .info-container {
        width: 100%;
        margin-bottom: 10px;
    }
    .info-item {
        display: inline-block;
        width: calc(33.3% - 10px); 
        margin-right: 10px;
        padding-bottom: 5px;
        border-bottom: 1px solid black;
        box-sizing: border-box; 
        vertical-align: top; 
        font-size: 14px;
        opacity: 0.8;
    }
</style>
"""

model = load_model(r'C:\Users\hp\College\ML\Project Stock pred\Stock Predictions Model.keras')

st.header('Stock Market Predictor')

# Text input for stock symbol
stock = st.text_input('Enter Stock Symbol','SBIN.NS')
end_date = datetime.datetime.now()
st.markdown("<hr class='hr-line'>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# "country":"India"
stock = stock.upper()


# heading

ticker = yf.Ticker(stock)
info = ticker.info
st.header(f'{info["longName"]} ({info["symbol"]})')
st.markdown("<hr class='hr-line'>", unsafe_allow_html=True)

try:
    hist = ticker.history(period='1d')
    live_closing_price = hist['Close'][0]

    previous_close = info.get('previousClose', None)
    if previous_close is not None:
        price_change = live_closing_price - previous_close
        price_change_pct = ((live_closing_price - previous_close) / previous_close) * 100
    else:
        price_change = 0.0
        price_change_pct = 0.0

    closing_price_html = f"<span style='font-size: 36px;'><strong>₹{live_closing_price:.2f}</strong></span>"
    price_change_html = f"<span style='color: {'red' if price_change < 0 else 'green'}; font-size: 24px;'>₹{-price_change:.2f} ({price_change_pct:.2f}%)</span>"

    st.markdown(f"{closing_price_html} {price_change_html}", unsafe_allow_html=True)
except Exception as e:
    st.error(f'Error: {e}')

# additional info
stock_info = get_live_stock_info(stock)

st.write(css_styles, unsafe_allow_html=True)
html_content = []

for i, (key, value) in enumerate(stock_info.items()):
    if i % 3 == 0:
        html_content.append('<div class="info-container">')
    html_content.append(f'<div class="info-item"><strong>{key}:</strong> {value}</div>')
    if i % 3 == 2 or i == len(stock_info) - 1:
        html_content.append('</div>')
formatted_html = ''.join(html_content)
st.write(formatted_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# time frame selector
timeframe_option = st.selectbox('Select Timeframe', ['5 Years', '1 Week', '1 Month', '1 Year', '2 Years', '10 Years', 'All'])

if timeframe_option == '1 Week':
    start_date = end_date - pd.Timedelta(days=7)
elif timeframe_option == '1 Month':
    start_date = end_date - pd.Timedelta(days=30)
elif timeframe_option == '1 Year':
    start_date = end_date - pd.Timedelta(days=365)
elif timeframe_option == '2 Years':
    start_date = end_date - pd.Timedelta(days=365*2)
elif timeframe_option == '5 Years':
    start_date = end_date - pd.Timedelta(days=365*5)
elif timeframe_option == '10 Years':
    start_date = end_date - pd.Timedelta(days=365*10)

data = yf.download(stock, start_date, end_date)
st.write(f'Stock Data for the Past {timeframe_option}')

# dividing into test and train
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))


# MA100 vs MA200 graph
st.markdown("<br>", unsafe_allow_html=True)
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8,6))
plt.plot(ma_100_days, 'r', label ='MA100')
plt.plot(ma_200_days, 'b', label ='MA200')
plt.plot(data.Close, 'g', label ='Closing price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.title('Price vs MA100 vs MA200')
plt.show()
st.pyplot(fig3)

x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])
x,y = np.array(x), np.array(y)
predict = model.predict(x)


# predicting next 30 days
predictions = []
last_100_days = data_test_scale[-100:]

for _ in range(30):
    x_pred = np.array(last_100_days[-100:]).reshape(1, 100, 1)
    y_pred = model.predict(x_pred)
    predictions.append(y_pred[0, 0])
    last_100_days = np.append(last_100_days, y_pred)

predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

st.subheader('Original Price vs Predicted Price for the Next 30 Days')
fig5 = plt.figure(figsize=(10, 8))
plt.plot(data_test.index[-len(data_test) + 100:], data_test['Close'].values[-len(data_test) + 100:], label='Original Price')
plt.plot(np.arange(len(data_test), len(data_test) + 30), predicted_prices, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig5)

