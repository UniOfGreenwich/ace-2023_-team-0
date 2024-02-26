from pandas_datareader import data
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

file_path='data.csv'
#All this stuff were committed for don't call the API
#url = 'https://www.alphavantage.co/query?function=DIGITAL_CURRENCY_DAILY&symbol=BTC&market=GBP&apikey=2HQRY5N3I5MPMIDO'
#r = requests.get(url)
#data = r.json()

#time_series = data.get('Time Series (Digital Currency Daily)')

#converting csv to dataframe
#df = pd.DataFrame(time_series).T

# Save the DataFrame to a CSV file
#df.to_csv('data.csv', index=True)

#print("Data saved to data.csv")


BTC_data= pd.read_csv(file_path)#,index_col='Unnamed: 0', parse_dates=True) # Recongize first column as a date
#BTC_data.index.freq='D' #Dealing with dayily data
BTC_data= BTC_data.sort_values('Unnamed: 0')

print(BTC_data.head()) # print first 5 rows

plt.figure(figsize = (18,9))
plt.plot(range(BTC_data.shape[0]),(BTC_data['3a. low (GBP)']+BTC_data['2a. high (GBP)'])/2.0)
plt.xticks(range(0,BTC_data.shape[0],75),BTC_data['Unnamed: 0'].loc[::75],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

# Calculate the mid prices from the highest and lowest
high_prices = BTC_data.loc[:, '2a. high (GBP)'].to_numpy()
low_prices = BTC_data.loc[:, '3a. low (GBP)'].to_numpy()
mid_prices = (high_prices + low_prices) / 2.0


data_size = len(mid_prices)  

# Calculate split indices
train_size = int(0.7 * data_size)  # 70% for training
validation_size = int(0.1 * data_size)  # 10% for validation
# The test size is implicitly the remaining 20%

# Split the data
train_data = mid_prices[:train_size]
validation_data = mid_prices[train_size:train_size + validation_size]
test_data = mid_prices[train_size + validation_size:]

# Scale the data to be between 0 and 1
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
validation_data= validation_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)