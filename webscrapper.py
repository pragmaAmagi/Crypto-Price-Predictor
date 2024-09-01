 #This will generate prediction plots for Bitcoin and Ethereum, saved as PNG files in the project directory.

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime, timedelta
import os
import tensorflow as tf

# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and will be used")
    except RuntimeError as e:
        print(f"Error setting up GPU: {e}")
else:
    print("No GPU available, using CPU")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

def get_crypto_data(crypto):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)  # Get data for the last year
    
    url = f"https://api.coingecko.com/api/v3/coins/{crypto}/market_chart/range"
    params = {
        'vs_currency': 'usd',
        'from': int(start_date.timestamp()),
        'to': int(end_date.timestamp())
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Could not retrieve data for {crypto}. Status code: {response.status_code}")
        return None
    
    data = response.json()
    
    df = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df = df.drop('Timestamp', axis=1)
    
    return df[['Date', 'Price']]

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Price'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

def build_model(look_back):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def predict_crypto_price(crypto):
    # Get data
    data = get_crypto_data(crypto)
    if data is None:
        print(f"Could not retrieve data for {crypto}")
        return
    
    # Prepare data
    X, y, scaler = prepare_data(data)
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_scores = []
    predictions = []
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Build and train model
        model = build_model(X.shape[1])
        model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=0)
        
        # Make predictions
        test_predict = model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        test_predict = scaler.inverse_transform(test_predict)
        y_test = scaler.inverse_transform([y_test])
        
        # Calculate RMSE
        rmse = np.sqrt(mean_squared_error(y_test[0], test_predict[:,0]))
        cv_scores.append(rmse)
        
        predictions.extend(test_predict[:,0])
    
    print(f"Cross-validation RMSE scores for {crypto}: {cv_scores}")
    print(f"Mean RMSE: {np.mean(cv_scores)}")
    print(f"Standard deviation of RMSE: {np.std(cv_scores)}")
    
    # Plot results
    plt.figure(figsize=(16,8))
    plt.plot(data['Date'][-len(predictions):], data['Price'][-len(predictions):], label='Actual Price')
    plt.plot(data['Date'][-len(predictions):], predictions, label='Predicted Price')
    plt.title(f'{crypto.capitalize()} Price Prediction with Time Series Cross-Validation')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig(f'{crypto}_prediction.png')
    print(f"Plot saved as {crypto}_prediction.png")

if __name__ == "__main__":
    predict_crypto_price('bitcoin')
    predict_crypto_price('ethereum')