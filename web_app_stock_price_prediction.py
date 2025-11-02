# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from keras.models import load_model
# import yfinance as yf
# from datetime import datetime

# st.title("Stock price predictor App")

# stock=st.text_input("Enter the Stock ID", "GOOG")

# end = datetime.now()
# start = datetime(end.year -20, end.month, end.day)

# google_data= yf.download(stock,start,end)

# model = load_model("Latest_stock_price_model.keras")
# st.subheader("Stock Data")
# st.write(google_data)

# splitting_len = int(len(google_data)*0.7)
# x_test = pd.DataFrame(google_data['Close'][splitting_len:])

# def plot_graph(figsize,values,full_data, extra_data=0,extra_dataset =None):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(values,'Orange')
#     plt.plot(full_data['Close'],'b')
#     if extra_data:
#         plt.plot(extra_dataset)
#     return fig

# # st.subheader('Original Close price and MA for 250 days ')
# # google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
# # st.pyplot(plot_graph((15,6),google_data['MA_for_250_days'],google_data,0))

# # st.subheader('Original Close price and MA for 200 days ')
# # google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
# # st.pyplot(plot_graph((15,6),google_data['MA_for_200_days'],google_data,0))

# # st.subheader('Original Close price and MA for 100 days ')
# # google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
# # st.pyplot(plot_graph((15,6),google_data['MA_for_100_days'],google_data,0))

# # st.subheader('Original Close price and MA for 100 days and MA for 250 days ')
# # st.pyplot(plot_graph((15,6),google_data['MA_for_100_days'],google_data,1,google_data['MA_FOR_250_days']))



# st.subheader('Original Close price and MA for 250 days')
# st.pyplot(plot_graph((15,6), google_data['MA_for_250_days'], google_data))

# st.subheader('Original Close price and MA for 200 days')
# st.pyplot(plot_graph((15,6), google_data['MA_for_200_days'], google_data))

# st.subheader('Original Close price and MA for 100 days')
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data))

# st.subheader('Original Close price and MA for 100 and 250 days')
# st.pyplot(plot_graph((15,6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))




# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler(feature_range=(0,1))
# scaled_data=scaler.fit_transform(x_test[['Close']])

# x_data =[]
# y_data=[]

# for i in range(100,len(scaled_data)):
#     x_data.append(scaled_data[i-100:i])
#     y_data.append(scaled_data[i])
    
# x_data, y_data = np.array(x_data), np.array(y_data)

# predictions = model.predict(x_data)

# inv_pre = scaler.inverse_transform(predictions)
# inv_y_test =scaler.inverse_transform(y_data)


# ploting_data = pd.DataFrame(
#     {
#         'original_test_data': inv_y_test.reshape(-1),
#         'prediction':inv_pre.reshape(-1)
#     },
#     index = google_data.index[splitting_len+100:]
# )
# st.subheader("original valuse vs prediction values")
# st.write(ploting_data)

# st.subheader('Original Close price vs Predicted Close price')
# fig = plt.figure(figsize=(15,6))
# plt.plot(google_data['Close'][:splitting_len+100], label="Data (Not Used)")
# plt.plot(ploting_data['original_test_data'], label="Original Test")
# plt.plot(ploting_data['prediction'], label="Predicted Test")
# plt.legend()
# st.pyplot(fig)
#streamlit run web_app_stock_price_prediction.py  




import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

st.title("Stock Price Predictor App")

# Stock Input
# stock = st.text_input("Enter the Stock ID (e.g., GOOG, MSFT, ^NSEI, JPM, XRP-USD):", "GOOG")
stock_options = {
    "Google (GOOG)": "GOOG",
    "Microsoft (MSFT)": "MSFT",
    "Nifty 50 (^NSEI)": "^NSEI",
    "JPMorgan Chase (JPM)": "JPM",
    "XRP/USD (Crypto)": "XRP-USD",
    "Other (Type below)": ""
}


# Stock selection
selected_stock_name = st.selectbox("Choose a Stock or Crypto:", list(stock_options.keys()))
stock = stock_options[selected_stock_name]

if stock == "":
    stock = st.text_input("Enter a custom stock/crypto symbol (e.g., AAPL, BTC-USD)", "")

if not stock:
    st.stop()

# Date range
end = datetime.now()
start = datetime(end.year -20, end.month, end.day)

# Download data
# google_data = yf.download(stock, start=start, end=end)
# google_data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in google_data.columns]

# Download data safely
google_data = yf.download(stock, start=start, end=end)

# ✅ FIX 1: Handle missing or empty data
if google_data.empty:
    st.error("❌ No data found for this stock symbol. Please check the symbol and try again.")
    st.stop()

# ✅ FIX 2: Fill missing values (if any)
google_data = google_data.fillna(method='ffill').dropna()

# ✅ Keep this line (you had it before)
google_data.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in google_data.columns]



# Find close column
close_col = None
for col in google_data.columns:
    if "Close" in col:
        close_col = col
        break

if close_col is None:
    st.error("❌ 'Close' column not found in the dataset!")
    st.stop()

# Load trained model
model = load_model("Latest_stock_price_model.keras")

st.subheader("🔍 Recent Stock Data")
st.write(google_data.tail())

# Plot function
def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values, 'orange', label="Moving Average")
    plt.plot(full_data[close_col], 'blue', label="Original")
    if extra_data:
        plt.plot(extra_dataset, 'green', linestyle='--', label="Comparison")
    plt.legend()
    return fig

# Moving Averages
google_data['MA_for_250_days'] = google_data[close_col].rolling(250).mean()
google_data['MA_for_200_days'] = google_data[close_col].rolling(200).mean()
google_data['MA_for_100_days'] = google_data[close_col].rolling(100).mean()

# MA Plots
st.subheader("📊 MA for 250 Days")
st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

st.subheader("📊 MA for 200 Days")
st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

st.subheader("📊 MA for 100 Days")
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

st.subheader("📊 MA 100 vs MA 250 Days")
st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

# Test Data
split_len = int(len(google_data) * 0.7)
x_test = pd.DataFrame(google_data[close_col][split_len:])

# Scale
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(x_test.values.reshape(-1, 1))

# Sequence
x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])
x_data, y_data = np.array(x_data), np.array(y_data)

# Predictions
predictions = model.predict(x_data)
inv_pre = scaler.inverse_transform(predictions)
inv_y = scaler.inverse_transform(y_data)

# DataFrame for plotting
plot_df = pd.DataFrame({
    'original_test_data': inv_y.reshape(-1),
    'prediction': inv_pre.reshape(-1)
}, index=google_data.index[split_len + 100:])

st.subheader("📉 Original vs Predicted Close Price")
st.write(plot_df)

fig = plt.figure(figsize=(15, 6))
plt.plot(pd.concat([google_data[close_col][:split_len + 100], plot_df['original_test_data']], axis=0), label="Original")
plt.plot(pd.concat([google_data[close_col][:split_len + 100], plot_df['prediction']], axis=0), label="Predicted")
plt.legend()
st.pyplot(fig)

# ✅ ✅ ✅ FUTURE PREDICTION SECTION
st.subheader("🚀 Future Prediction for Next 365 Days")

# Last 100 input
last_100 = scaled_data[-100:]
input_seq = last_100.copy()
future_predictions = []

for _ in range(365):
    x_input = np.array(input_seq[-100:]).reshape(1, 100, 1)
    pred = model.predict(x_input, verbose=0)
    input_seq = np.append(input_seq, pred, axis=0)
    future_predictions.append(pred[0][0])

# Inverse scale
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create future dates
last_date = google_data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, 366)]

# Future DataFrame
future_df = pd.DataFrame({'Future_Predicted_Close': future_prices.flatten()}, index=future_dates)
st.write(future_df.head())

# Plot future prediction
fig2 = plt.figure(figsize=(15, 6))
plt.plot(google_data[close_col], label="Historical")
plt.plot(future_df['Future_Predicted_Close'], label="Future Prediction", color='orange', linestyle='--')
plt.xlabel("Date")
plt.ylabel("Price")
plt.title(f"{stock} - 1 Year Future Price Forecast")
plt.legend()
st.pyplot(fig2)




