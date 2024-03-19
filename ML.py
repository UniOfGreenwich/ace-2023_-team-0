from array import array
from ast import mod
from calendar import EPOCH
from pickletools import optimize
from pyexpat import features
from telnetlib import SE
import plotly.express as px
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
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
from keras.layers import Dense, LSTM, GRU
from keras.layers import Dropout


class Bitcoin:
    

     def __init__(self):
        file_path='BTC-USD.csv'

        self.BTC_data= pd.read_csv(file_path,index_col='Date', parse_dates=['Date'], dayfirst=True) # Recongize first column as a date
        self.BTC_data= self.BTC_data.sort_index() 
        self.BTC_data= self.BTC_data[['High','Volume',]]
        self.BTC_data.columns= ['Price','Volume']
        print(self.BTC_data.head())

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


       


 


        self.BTC_data= self.technical_indicators(self.BTC_data)
        self.BTC_data = self.BTC_data.dropna()
        print(self.BTC_data.head())


        print(self.BTC_data)




        # Separate the features
        prices = self.BTC_data[['Price']].values
        volumes = self.BTC_data[['Volume']].values
        EMA = self.BTC_data[['ema']].values
        SMA= self.BTC_data[['SMA30']].values
        RSI=self.BTC_data[['RSI']].values
        MACD=self.BTC_data[['MACD']].values

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


        # Defining time step and number of features.

        time_step = 30
        num_features = 6

        # Initialize X_train and Y_train as empty lists
        X_train, Y_train = [], []
        print("train_data shape before loop:", train_data.shape)

        #Fill up the lists
        for i in range(time_step, train_size):
            
            X_train.append(train_data[i - time_step:i, :])
            Y_train.append(train_data[i, 0])

        #convert the lists to numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)


        X_train = X_train.reshape(X_train.shape[0], time_step, -1)

        # Verify the shape of X_train before reshaping
        print(f'Before reshaping, X_train.shape = {X_train.shape}')

        
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


        X_test = []
        Y_test=[]
        y_length = len(test_data)

        for i in range(time_step,  y_length):
            X_test.append(test_data[i-time_step:i, :])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],  time_step, num_features))
        Y_test = test_data[time_step:, 0]


        dim_exit=1
        na=50 
        batch_size=64
        epochs=100

        #LSTM
        model = Sequential()
        model.add(LSTM(units=na, input_shape=(time_step, num_features)))
        model.add(Dropout(0.2))  # Dropout 20% of the nodes of the previous layer during training
        model.add(Dense(1))
        model.add(Dense(units=dim_exit))
        model.compile(optimizer='adam',loss='mse')
        model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)





        #GRU
        model2= Sequential()
        model2.add(GRU(units=na, input_shape=(time_step, num_features)),)
        model2.add(Dropout(0.2))
        model2.add(Dense(1))
        model2.add(Dense(units=dim_exit))
        model2.compile(optimizer='adam',loss='mse')
        model2.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)


        # Making predictions
        predic = model.predict(X_test)
        predic2 = model2.predict(X_test)

        # Inverse transforming the predictions to get them back to the original scale
        price_predictions_scaled = scaler_price.inverse_transform(predic)
        price_predictions_scaled2 = scaler_price.inverse_transform(predic2)


        # Evaluation metrices RMSE and MAE

        print("Test data RMSE: ", math.sqrt(mean_squared_error(Y_test,predic)))
        print("Test data MSE: ", mean_squared_error(Y_test,predic))
        print("Test data MAE: ", mean_absolute_error(Y_test,predic))

        ## Variance Regression Score
        print("Test data explained variance regression score:", 
              explained_variance_score(Y_test, predic))

        ## R square score for regression
        print("Test data R2 score:", r2_score(Y_test, predic))

        ## Regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
        print("Test data MGD: ", mean_gamma_deviance(Y_test, predic))
        print("----------------------------------------------------------------------")
        print("Test data MPD: ", mean_poisson_deviance(Y_test, predic))


        #  BTC_data index contains the dates 
        dates = self.BTC_data.index

        # Extract dates for test data
        test_dates = dates[train_size + time_step:] 
        prediction_length = len(test_dates)

        # Adjust to account for how the test data is generated

        # Ensure the predictions match the number of test dates
        adjusted_predic = price_predictions_scaled[:prediction_length]
        adjusted_predic2 = price_predictions_scaled2[:prediction_length]

        #Create the DataFrame with the adjusted predictions
        predictions_df = pd.DataFrame(data=adjusted_predic, index=test_dates, columns=["Price Prediction"])

        predictions_df2 = pd.DataFrame(data=adjusted_predic2, index=test_dates, columns=["Price Prediction 2"])


        # Extract actual high prices 
        actual_high_prices = self.BTC_data['Price'][train_size + time_step: train_size + time_step + prediction_length]

        # Create a DataFrame for actual high prices to ensure alignment in plotting
        actual_high_prices_df = pd.DataFrame(data=actual_high_prices.values, index=test_dates, columns=["Actual High Price"])

        # Plotting both actual and predicted prices
        plt.figure(figsize=(14, 7))
        plt.plot(predictions_df.index, predictions_df["Price Prediction"], color="blue", label="Predicted High Price")
        plt.plot(predictions_df2.index, predictions_df2["Price Prediction 2"], color="yellow", label="Predicted 2 High Price")
        plt.plot(actual_high_prices_df.index, actual_high_prices_df["Actual High Price"], color="red", label="Actual High Price")
        plt.title("Bitcoin Predicted vs Actual High Prices")
        plt.xlabel("Date")
        plt.ylabel("High Price (Dollar)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout() 
        plt.show()


         # Number of days to look back to make future predictions
        time_based = 90
        # Prepare the input for the first prediction by taking the last 90 days from test_data
        last_90_days = test_data[-time_based:]
        # Reshape the last 90 days data to fit the model's expected input shape
        current_batch = last_90_days.reshape((1, time_based, num_features))

        # Initialize two lists to store future predictions for each model
        future_predictions = []
        future_predictions2 = []

        # Predict the next 30 days
        for i in range(30):  # Loop for 30 days
            # Predict the next day using the first model
            next_day_prediction = model.predict(current_batch)[0]
            # Predict the next day using the second model
            next_day_prediction2 = model2.predict(current_batch)[0]
            # Add the predictions to their respective lists
            future_predictions.append(next_day_prediction)
            future_predictions2.append(next_day_prediction2)
    
            # Get the features from the last day in the batch (excluding the target feature)
            last_features = current_batch[0, -1, 1:]
    
            # Combine the next day prediction with the last features
            next_day_input = np.hstack([next_day_prediction, last_features])
            # Reshape the combination to match the expected number of features
            next_day_input = next_day_input.reshape((1, 1, num_features))
            # Update the batch to include the new prediction and drop the oldest day
            current_batch = np.concatenate([current_batch[:, 1:, :], next_day_input], axis=1)
    
            # Output the current shape of the batch for debugging purposes
            print(f"current_batch shape: {current_batch.shape}")
            print(f"next_day_input shape: {next_day_input.shape}")
    
            # Repeat the process for the second model's predictions
            next_day_input2 = np.hstack([next_day_prediction2, last_features])
            next_day_input2 = next_day_input2.reshape((1, 1, num_features))
            print(f"next_day_input shape: {next_day_input2.shape}")

        # Reverse the scaling transformation to convert predictions back to their original scale
        future_predictions_scaled = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_predictions_scaled2 = scaler_price.inverse_transform(np.array(future_predictions2).reshape(-1, 1))

        # Get the last date from the dataset
        last_date = self.BTC_data.index[-1]
        # Calculate the start date for the future predictions by adding one day to the last date
        start_date = last_date + pd.Timedelta(days=1)
        # Create a date range for the next 30 days starting from the start_date
        prediction_dates = pd.date_range(start=start_date, periods=30)

        # Plotting the predictions
        # Set the figure size for better visibility
        plt.figure(figsize=(15,7))
        # Plot the scaled future predictions from the first model
        plt.plot(prediction_dates, future_predictions_scaled, color='green', linestyle='--', label='Future Predicted Price')
        # Plot the scaled future predictions from the second model
        plt.plot(prediction_dates, future_predictions_scaled2, color='orange', linestyle='--', label='Future Predicted Price2')
        # Set the title of the plot
        plt.title('Predicted Future Prices for the Next 30 Days')
        # Set the x-axis label
        plt.xlabel('Date')
        # Set the y-axis label
        plt.ylabel('Price(Dollar)')
        # Display the legend to identify the plotted lines
        plt.legend()
        # Rotate the date labels for better readability
        plt.xticks(rotation=45)
        # Display the plot
        plt.show()


     def technical_indicators(self,dataset):
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




class ETH:
     def __init__(self):
        file_path='ETH-USD.csv'
   
        self.ETH_data= pd.read_csv(file_path,index_col='Date', parse_dates=['Date'], dayfirst=True) # Recongize first column as a date
        self.ETH_data= self.ETH_data.sort_index() 
        self.ETH_data= self.ETH_data[['High','Volume',]]
        self.ETH_data.columns= ['Price','Volume']
        print(self.ETH_data.head())

        
        self.ETH_data= self.technical_indicators(self.ETH_data)
        self.ETH_data = self.ETH_data.dropna()
        print(self.ETH_data.head())


        print(self.ETH_data)




        # Separate the features
        prices = self.ETH_data[['Price']].values
        volumes = self.ETH_data[['Volume']].values
        EMA = self.ETH_data[['ema']].values
        SMA= self.ETH_data[['SMA30']].values
        RSI=self.ETH_data[['RSI']].values
        MACD=self.ETH_data[['MACD']].values

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
            
            X_train.append(train_data[i - time_step:i, :])
            Y_train.append(train_data[i, 0])

        #Convert the lists to numpy arrays
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)


        X_train = X_train.reshape(X_train.shape[0], time_step, -1)

        # Verify the shape of X_train before reshaping
        print(f'Before reshaping, X_train.shape = {X_train.shape}')

        
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


        X_test = []
        Y_test=[]
        y_length = len(test_data)

        for i in range(time_step,  y_length):
            X_test.append(test_data[i-time_step:i, :])
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0],  time_step, num_features))
        Y_test = test_data[time_step:, 0]


        dim_exit=1
        na=50 
        batch_size=64
        epochs=100

        #LSTM
        model = Sequential()
        model.add(LSTM(units=na, input_shape=(time_step, num_features)))
        model.add(Dropout(0.2))  # Dropout 20% of the nodes of the previous layer during training
        model.add(Dense(1))
        model.add(Dense(units=dim_exit))
        model.compile(optimizer='adam',loss='mse')
        model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)





        #GRU
        model2= Sequential()
        model2.add(GRU(units=na, input_shape=(time_step, num_features)),)
        model2.add(Dropout(0.2))
        model2.add(Dense(1))
        model2.add(Dense(units=dim_exit))
        model2.compile(optimizer='adam',loss='mse')
        model2.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size)


        # Making predictions
        predic = model.predict(X_test)
        predic2 = model2.predict(X_test)

        # Inverse transforming the predictions to get them back to the original scale
        price_predictions_scaled = scaler_price.inverse_transform(predic)
        price_predictions_scaled2 = scaler_price.inverse_transform(predic2)


        # Evaluation metrices RMSE and MAE

        print("Test data RMSE: ", math.sqrt(mean_squared_error(Y_test,predic)))
        print("Test data MSE: ", mean_squared_error(Y_test,predic))
        print("Test data MAE: ", mean_absolute_error(Y_test,predic))

        ## Variance Regression Score
        print("Test data explained variance regression score:", 
              explained_variance_score(Y_test, predic))

        ## R square score for regression
        print("Test data R2 score:", r2_score(Y_test, predic))

        ## Regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
        print("Test data MGD: ", mean_gamma_deviance(Y_test, predic))
        print("----------------------------------------------------------------------")
        print("Test data MPD: ", mean_poisson_deviance(Y_test, predic))


        
        dates = self.ETH_data.index

         # Extract dates for test data
        test_dates = dates[train_size + time_step:] 
        test_dates = pd.to_datetime(test_dates)
        prediction_length = len(test_dates)
        print(test_dates.dtype)

        # Adjust to account for how the test data is generated

        # Ensure the predictions match the number of test dates
        adjusted_predic = price_predictions_scaled[:prediction_length]
        adjusted_predic2 = price_predictions_scaled2[:prediction_length]

        #Create the DataFrame with the adjusted predictions
        predictions_df = pd.DataFrame(data=adjusted_predic, index=test_dates, columns=["Price Prediction"])
        predictions_df2 = pd.DataFrame(data=adjusted_predic2, index=test_dates, columns=["Price Prediction 2"])


        # Extract actual high prices 
        actual_high_prices = self.ETH_data['Price'][train_size + time_step: train_size + time_step + prediction_length]

        # Create a DataFrame for actual high prices to ensure alignment in plotting
        actual_high_prices_df = pd.DataFrame(data=actual_high_prices.values, index=test_dates, columns=["Actual High Price"])

        # Plotting both actual and predicted prices
        plt.figure(figsize=(14, 7))
        plt.plot(predictions_df.index, predictions_df["Price Prediction"], color="blue", label="Predicted High Price")
        plt.plot(predictions_df2.index, predictions_df2["Price Prediction 2"], color="yellow", label="Predicted 2 High Price")
        plt.plot(actual_high_prices_df.index, actual_high_prices_df["Actual High Price"], color="red", label="Actual High Price")
        plt.title("ETH Predicted vs Actual High Prices")
        plt.xlabel("Date")
        plt.ylabel("High Price (Dollar)")
        ax = plt.gca()  # Get the current Axes instance
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically find the best location for date ticks
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(ax.xaxis.get_major_locator()))  # Format the date ticks

        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout() 
        plt.show()


   

       

        # Number of days to look back to make future predictions
        time_based = 90
        # Prepare the input for the first prediction by taking the last 90 days from test_data
        last_90_days = test_data[-time_based:]
        # Reshape the last 90 days data to fit the model's expected input shape
        current_batch = last_90_days.reshape((1, time_based, num_features))

        # Initialize two lists to store future predictions for each model
        future_predictions = []
        future_predictions2 = []

        # Predict the next 30 days
        for i in range(30):  # 30 days
           # Predict the next day using the first model
            next_day_prediction = model.predict(current_batch)[0]
            # Predict the next day using the second model
            next_day_prediction2 = model2.predict(current_batch)[0]
             # Add the predictions to their respective lists
            future_predictions.append(next_day_prediction)
            future_predictions2.append(next_day_prediction2)

            # Get the features from the last day in the batch (excluding the target feature)
            last_features = current_batch[0, -1, 1:]
    
               # Combine the next day prediction with the last features
        next_day_input = np.hstack([next_day_prediction, last_features])
        # Reshape the combination to match the expected number of features
        next_day_input = next_day_input.reshape((1, 1, num_features))
        # Update the batch to include the new prediction and drop the oldest day
        current_batch = np.concatenate([current_batch[:, 1:, :], next_day_input], axis=1)
    
        # Output the current shape of the batch for debugging purposes
        print(f"current_batch shape: {current_batch.shape}")
        print(f"next_day_input shape: {next_day_input.shape}")
    
        # Repeat the process for the second model's predictions
        next_day_input2 = np.hstack([next_day_prediction2, last_features])
        next_day_input2 = next_day_input2.reshape((1, 1, num_features))
        print(f"next_day_input shape: {next_day_input2.shape}")

         # Reverse the scaling transformation to convert predictions back to their original scale
        future_predictions_scaled = scaler_price.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        future_predictions_scaled2 = scaler_price.inverse_transform(np.array(future_predictions2).reshape(-1, 1))



        
       # Determine the start date for the predictions by adding one day to the last date in the dataset
        last_date = pd.to_datetime(self.ETH_data.index[-1])
        start_date = last_date + pd.Timedelta(days=1)
        # Create a range of dates for the future predictions
        prediction_dates = pd.date_range(start=start_date, periods=30)


        # Plotting
        plt.figure(figsize=(14,7))
        plt.plot(prediction_dates, future_predictions_scaled, color='green', linestyle='--', label='Future Predicted Price')
        plt.plot(prediction_dates, future_predictions_scaled2, color='orange', linestyle='--', label='Future Predicted Price2')
        plt.title('Predicted Future Prices for the Next 30 Days')
        plt.xlabel('Date')
        plt.ylabel('Price(Dollar)')
        plt.legend()
        plt.xticks(rotation=45)  # Rotate dates for better readability
        plt.show()

     def technical_indicators(self,dataset):
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

    

user_input = input("Press A for predict Bitcoin price or press B for predict ETH price ")

if user_input.upper() == 'A':
    selected_class = Bitcoin()
elif user_input.upper() == 'B':
    selected_class = ETH()
else:
    print("Invalid input")

