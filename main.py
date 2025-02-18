import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


ticker = "AAPL"
data = yf.download(ticker, start="2015-01-01", end="2023-01-01")
data.head()
# data_info = data.info()
# data_describe = data.describe()

data["MA20"] = data["Close"].rolling(window=20).mean()


features = data[["Open", "High", "Low", "MA20"]].dropna()
target = data.loc[features.index, "Close"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

# Neural Networks (LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))


def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length : i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


seq_length = 60
X_lstm, y_lstm = create_sequences(scaled_data, seq_length)

X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))

train_size = int(len(X_lstm) * 0.8)
X_train_lstm, X_test_lstm = X_lstm[:train_size], X_lstm[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(seq_length, 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

lstm_model.compile(loss="mean_squared_error", optimizer="adam")
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=20, batch_size=32)

lstm_predictions = lstm_model.predict(X_test_lstm)

# Model Evaluation

mse_lr = mean_squared_error(y_test, lr_predictions)
rmse_lr = np.sqrt(mse_lr)
print("Linear Regression RMSE: ", rmse_lr)

mse_rf = mean_squared_error(y_test, rf_predictions)
rmse_rf = np.sqrt(mse_rf)
print("Random Forest RMSE: ", rmse_rf)

mse_lstm = mean_squared_error(y_test_lstm, lstm_predictions)
rmse_lstm = np.sqrt(mse_lstm)
print("LSTM RMSE: ", rmse_lstm)


# Plot actual vs. predicted prices for Linear Regression.
plt.figure(figsize=(10, 10))
plt.plot(y_test.values, label="Actual Price")
plt.plot(lr_predictions, label="Predicted Price")
plt.title("Stock Price Prediction using Linear Regression")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


# # Plot actual vs. predicted prices for Random Forest Regression.
# plt.figure(figsize=(10, 10))
# plt.plot(y_test.values, label="Actual Price")
# plt.plot(rf_predictions, label="Predicted Price")
# plt.title("Stock Price Prediction using Linear Regression")
# plt.xlabel("Time")
# plt.ylabel("Stock Price")
# plt.legend()
# plt.show()


# # Plot actual vs. predicted prices for LSTM.
# plt.figure(figsize=(10, 10))
# plt.plot(y_test.values, label="Actual Price")
# plt.plot(lstm_predictions, label="Predicted Price")
# plt.title("Stock Price Prediction using Linear Regression")
# plt.xlabel("Time")
# plt.ylabel("Stock Price")
# plt.legend()
# plt.show()
