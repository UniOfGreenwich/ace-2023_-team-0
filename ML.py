from array import array
from ast import mod
from calendar import EPOCH
from pickletools import optimize
from pyexpat import features
import plotly.express as px

from statistics import mode
from tabnanny import verbose
from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout




file_path='data.csv'

BTC_data= pd.read_csv(file_path,index_col='Date', parse_dates=['Date'], dayfirst=True) # Recongize first column as a date
BTC_data= BTC_data.sort_index() 
BTC_data= BTC_data[['2a. high (GBP)','5. volume',]]
BTC_data.columns= ['Price','Volume']
print(BTC_data.head())

#plotting prices
#plt.figure(figsize=(10, 6))  # Set the figure size for better readability
#plt.plot(BTC_data.index, BTC_data['Price'], label='High Price', color='blue')  # Plot the 'Price' column
#plt.title('BTC High Price Over Time')  # Set the title of the plot
#plt.xlabel('Date')  # Set the x-axis label
#plt.ylabel('High Price in GBP')  # Set the y-axis label
#plt.legend()  # Show legend
#plt.grid(True)  # Show grid
#plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
#plt.tight_layout()  # Adjust the layout to make room for the rotated x-axis labels
#plt.show()  # Display the plot


def technical_indicators(dataset):
    # Create 7 and 21 days Moving Average
    dataset['ma7'] = dataset['Price'].rolling(window=7).mean()
    dataset['ma21'] = dataset['Price'].rolling(window=21).mean()
    
    # Create MACD
    dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
    dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
    dataset['MACD'] = dataset['12ema']-dataset['26ema']

    # Create Bollinger Bands
    dataset['20sd'] = dataset['Price'].rolling(window = 21).std()
    dataset['upper_band'] = dataset['ma21'] + (dataset['20sd']*2)
    dataset['lower_band'] = dataset['ma21'] - (dataset['20sd']*2)
    
    # Create Exponential moving average
    dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    
    # Create Momentum
    dataset['momentum'] = dataset['Price']-1
    dataset['log_momentum'] = np.log(dataset['momentum'])

    # Calculate RSI
    delta = dataset['Price'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()

    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    dataset['RSI'] = RSI

     # Calculate SMA (Simple Moving Average)
    dataset['SMA30'] = dataset['Price'].rolling(window=30).mean()  # 30-day SMA
    dataset['SMA21'] = dataset['Price'].rolling(window=21).mean()  # 21-day SMA, if needed
    return dataset


 


BTC_data= technical_indicators(BTC_data)
BTC_data = BTC_data.dropna()
print(BTC_data.head())




print(BTC_data)




# Separate the features
prices = BTC_data[['Price']].values
volumes = BTC_data[['Volume']].values
EMA = BTC_data[['ema']].values
SMA= BTC_data[['SMA30']].values
RSI=BTC_data[['RSI']].values
MACD=BTC_data[['MACD']].values

# Scale each feature independently
scaler_price = MinMaxScaler()
scaler_volume = MinMaxScaler()
scaler_EMA = MinMaxScaler()
scaler_SMA= MinMaxScaler()
scaler_RSI= MinMaxScaler()
scaler_MACD= MinMaxScaler()

prices_scaled = scaler_price.fit_transform(prices)
volumes_scaled = scaler_volume.fit_transform(volumes)
EMA_scaled = scaler_EMA.fit_transform(EMA)
SMA_scaled = scaler_SMA.fit_transform(SMA)
RSI_scaled= scaler_RSI.fit_transform(RSI)
MACD_scaled=scaler_MACD.fit_transform(MACD)


# Concatenate the scaled features
scaled_features = np.concatenate([prices_scaled, volumes_scaled, EMA_scaled,SMA_scaled,RSI_scaled,MACD_scaled], axis=1)


data_size = len(scaled_features) 

print('Data size is ', data_size)

# Define the data size and calculate split indices for an 80%-20% split

train_size = int(0.8 * data_size)
test_size = data_size - train_size

# Split the data
train_data = scaled_features[:train_size]
test_data = scaled_features[train_size:]





print("Data size is ", data_size)
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)
# Scale the data to be between 0 and 1
scaler = MinMaxScaler()
train_data =  scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)


# Assuming train_data has been correctly scaled and split previously...

time_step = 30
num_features = 6

# Initialize X_train and Y_train as empty lists
X_train, Y_train = [], []
print("train_data shape before loop:", train_data.shape)


for i in range(time_step, train_size):
    # Here, we get the past `time_step` records up to the current record, i
    X_train.append(train_data[i - time_step:i, :])
    Y_train.append(train_data[i, 0])

# Now, convert the lists to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)


X_train = X_train.reshape(X_train.shape[0], time_step, -1)

# Verify the shape of X_train before reshaping
print(f'Before reshaping, X_train.shape = {X_train.shape}')

# The first dimension of X_train should be (len(train_data) - time_step)
# The second dimension should be time_step
# The third dimension should be num_features
# Only reshape if X_train.shape is as expected
if X_train.shape == ((len(train_data) - time_step), time_step, num_features):
    X_train = X_train.reshape((X_train.shape[0], time_step, num_features))
else:
    print(f'Cannot reshape X_train of shape {X_train.shape} to (samples, {time_step}, {num_features}).')
    # Handle the incorrect shape, perhaps with an error message or a correction

# Continue with the model definition and training...
print('train_data shape:', train_data.shape)
print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)


dim_exit=1
na=50 

model = Sequential()
model.add(LSTM(units=na, input_shape=(time_step, num_features)))
model.add(Dropout(0.2))  # Dropout 20% of the nodes of the previous layer during training
model.add(Dense(1))
model.add(Dense(units=dim_exit))
model.compile(optimizer='adam',loss='mse')
model.fit(X_train,Y_train,epochs=50,batch_size=32)


X_test = []
y_length = len(test_data)

for i in range(time_step,  y_length):
    X_test.append(test_data[i-time_step:i, :])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],  time_step, num_features))

# Making predictions
predic = model.predict(X_test)

# Inverse transforming the predictions to get them back to the original scale
price_predictions_scaled = scaler_price.inverse_transform(predic)

# Assuming BTC_data index contains the dates and you have followed the previous steps to create predictions_df
dates = BTC_data.index

# Extract dates for test data
test_dates = dates[train_size + time_step:] 
prediction_length = len(test_dates)
# Adjust to account for how the test data is generated

# Ensure the predictions match the number of test dates
adjusted_predic = price_predictions_scaled[:prediction_length]

# Now create the DataFrame with the adjusted predictions
predictions_df = pd.DataFrame(data=adjusted_predic, index=test_dates, columns=["Price Prediction"])

# Extract actual high prices for the corresponding dates from the original DataFrame
# Ensure this uses the same index range as your predictions
actual_high_prices = BTC_data['Price'][train_size + time_step: train_size + time_step + prediction_length]

# Create a DataFrame for actual high prices to ensure alignment in plotting
actual_high_prices_df = pd.DataFrame(data=actual_high_prices.values, index=test_dates, columns=["Actual High Price"])

# Plotting both actual and predicted prices
plt.figure(figsize=(14, 7))
plt.plot(predictions_df.index, predictions_df["Price Prediction"], color="blue", label="Predicted High Price")
plt.plot(actual_high_prices_df.index, actual_high_prices_df["Actual High Price"], color="red", label="Actual High Price")
plt.title("Bitcoin Predicted vs Actual High Prices")
plt.xlabel("Date")
plt.ylabel("High Price (GBP)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout() 
plt.show()


# Predict for the next 30 days
# Prepare the input for the first prediction (the last 60 days from test_data)
last_60_days = test_data[-time_step:]
current_batch = last_60_days.reshape((1, time_step, num_features))

# To store the predictions
future_predictions = []

# Predict the next 30 days
for i in range(30):  # 30 days
    # Predict the next day
    next_day_prediction = model.predict(current_batch)[0]
    
    # Append the prediction to the list
    future_predictions.append(next_day_prediction)
    last_features = current_batch[0, -1, 1:]
    
    # Update the batch to include the new prediction and drop the oldest day
   
    next_day_input = np.hstack([next_day_prediction, last_features]) # Reshape to match the number of features
    next_day_input = next_day_input.reshape((1, 1, num_features))
    current_batch = np.concatenate([current_batch[:, 1:, :], next_day_input], axis=1)
    print(f"current_batch shape: {current_batch.shape}")
    print(f"next_day_input shape: {next_day_input.shape}")


# Inverse transform to get the predictions back to the original scale
future_predictions_scaled = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Prepare dates for plotting the predictions
last_date = BTC_data.index[-1]
start_date = last_date + pd.Timedelta(days=1)  # Start from the day after the last known date
prediction_dates = pd.date_range(start=start_date, periods=30)  # Now correctly creates 30 future dates

# Plotting
plt.figure(figsize=(15,7))
plt.plot(prediction_dates, future_predictions_scaled, color='green', linestyle='--', label='Future Predicted Price')
plt.title('Predicted Future Prices for the Next 30 Days')
plt.xlabel('Date')
plt.ylabel('Price(GBP)')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.show()
