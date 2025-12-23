!pip install yfinance 
import numpy as np 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
import yfinance as yf 
 
# Load stock data (Apple: AAPL) 
data = yf.download("AAPL", start="2020-01-01", end="2023-01-01") 
prices = data["Close"].values.reshape(-1,1) 
 
# Normalize data 
scaler = MinMaxScaler(feature_range=(0,1)) 
prices_scaled = scaler.fit_transform(prices) 
 
# Create sequences for RNN 
 
 
X, y = [], [] 
for i in range(60, len(prices_scaled)): 
    X.append(prices_scaled[i-60:i]) 
    y.append(prices_scaled[i]) 
X, y = np.array(X), np.array(y) 
 
# Split dataset 
split = int(0.8 * len(X)) 
X_train, X_test = X[:split], X[split:] 
y_train, y_test = y[:split], y[split:] 
 
# LSTM Model 
model = tf.keras.Sequential([ 
    tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(60,1)), 
    tf.keras.layers.LSTM(50), 
    tf.keras.layers.Dense(1) 
]) 
model.compile(optimizer='adam', loss='mse') 
 
# Train model 
model.fit(X_train, y_train, epochs=5, batch_size=32) 
 
# Predict 
pred = model.predict(X_test) 
pred = scaler.inverse_transform(pred) 
real = scaler.inverse_transform(y_test) 
 
# Plot graph 
plt.figure(figsize=(10,4)) 
plt.plot(real, label="Real Price") 
plt.plot(pred, label="Predicted") 
plt.title("Stock Price Prediction using LSTM") 
plt.xlabel("Time") 
plt.ylabel("Price (USD)") 
plt.legend() 
plt.show() 
