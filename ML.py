from array import array
from ast import mod
from calendar import EPOCH
from pickletools import optimize

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
from keras.layers import Dense, LSTM, Input


file_path='data.csv'

BTC_data= pd.read_csv(file_path,index_col='Date', parse_dates=['Date'], dayfirst=True) # Recongize first column as a date
BTC_data= BTC_data.sort_index() 
BTC_data= BTC_data[['2a. high (GBP)','5. volume']]
BTC_data.columns= ['Price','Volumen']
print(BTC_data.head())


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
    return dataset

 


BTC_data= technical_indicators(BTC_data)
BTC_data = BTC_data.dropna()
print(BTC_data.head())




print(BTC_data)




prices = BTC_data['Price'].to_numpy()



data_size = len(prices)  

# Calculate split indices
train_size = int(0.7 * data_size)  # 70% for training
validation_size = data_size - train_size  # 10% for validation
# The test size is implicitly the remaining 20%

# Split the data
train_data = prices[:train_size]

test_data = prices[validation_size:]

# Scale the data to be between 0 and 1
scaler = MinMaxScaler()
train_data =  scaler.fit_transform(train_data.reshape(-1, 1))
#validation_data= validation_data.reshape(-1,1)
#test_data = scaler.transform(test_data)
test_data = scaler.transform(test_data.reshape(-1, 1))


time_step = 60
X_train=[]
Y_train=[]
m=len(train_data)
print(train_data.shape)  # Debugging: Check shape of train_data

print(len(X_train), len(Y_train))

for i in range(time_step,m):
    X_train.append(train_data[i-time_step:i,0])
    Y_train.append(train_data[i,0])
X_train,Y_train =np.array(X_train),np.array(Y_train)

print(train_data.shape)  # Debugging: Check shape of train_data

print(len(X_train), len(Y_train))

X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

dim_input= (X_train.shape[1],1)
dim_exit=1
na=50 

model = Sequential()
model.add(LSTM(units=na, input_shape=dim_input))
model.add(Dense(units=dim_exit))
model.compile(optimizer='adam',loss='mse')
model.fit(X_train,Y_train,epochs=200,batch_size=64)


X_test = []
for i in range(time_step, m):
    X_test.append(test_data[i-time_step:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Making predictions
predic = model.predict(X_test)

# Inverse transforming the predictions to get them back to the original scale
predic = scaler.inverse_transform(predic)

#print(predic)

dates_for_plotting = BTC_data.index[-len(predic):]

actual_prices_scaled = test_data[time_step:]
actual_prices = scaler.inverse_transform(actual_prices_scaled)
actual_prices = actual_prices[:len(predic)]



dates = range(len(predic))

# Plotting
plt.figure(figsize=(15,7))
predic_flattened = predic.flatten()
plt.plot(dates_for_plotting,actual_prices, color='blue', label='Actual Price')
plt.plot(dates_for_plotting,predic, color='red', linestyle='--', label='Predicted Price')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
plt.gcf().autofmt_xdate()

plt.title('Actual vs Predicted Prices')
plt.xlabel('Date')
plt.ylabel('Price(GBP)')
plt.legend()
plt.show()


# Predict Next Day

# Prepare the input for the first prediction (the last 60 days from test_data)
last_60_days = test_data[-60:]
current_batch = last_60_days.reshape((1, time_step, 1))

# To store the predictions
future_predictions = []

# Predict the next 30 days
for i in range(30):  # 30 days
    # Predict the next day
    next_day_prediction = model.predict(current_batch)[0]
    
    # Append the prediction to the list
    future_predictions.append(next_day_prediction)
    
    # Update the batch to include the new prediction and drop the oldest day
    current_batch = np.append(current_batch[:,1:,:],[[next_day_prediction]], axis=1)

# Inverse transform to get the predictions back to the original scale
future_predictions_scaled = scaler.inverse_transform(future_predictions)

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