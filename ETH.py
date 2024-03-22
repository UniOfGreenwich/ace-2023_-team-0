# -*- coding: cp1252 -*-
from array import array
from ast import mod
from pickletools import optimize
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from statistics import mode
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.layers import Dropout
import yfinance as yf
from datetime import datetime, timedelta







class ETH:
     def __init__(self):
        

        ticker_symbol = "ETH-USD"
        start_date = "2017-11-10"
        end_date = self.get_yesterday_date()

        # Fetch historical data
        self.ETH_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        # Format the DataFrame
        self.ETH_data = self.ETH_data[['High', 'Volume']]
        self.ETH_data.columns = ['Price', 'Volume']
        self.ETH_data.index = pd.to_datetime(self.ETH_data.index)
        self.ETH_data = self.ETH_data.sort_index()
        
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

        time_step = 60
        num_features = 6
        split_ratio = {'train': 0.6, 'validation': 0.2, 'test': 0.2}
        

        
        trimmed_size = len(scaled_features) - (len(scaled_features) % time_step)
        trimmed_data = scaled_features[:trimmed_size]

         # Define the data size and calculate split indices for an 80%-20% split

        train_size = int(trimmed_size * split_ratio['train'])
        validation_size = int(trimmed_size * split_ratio['validation'])
        test_size = trimmed_size - train_size - validation_size
        

        train_data = scaled_features[:train_size]
        validation_data = scaled_features[train_size:train_size + validation_size]
        test_data = scaled_features[-test_size:]





        #print("Data size is ", data_size)
        print("train_data: ", train_data.shape)
        print("validation_data: ", validation_data.shape)
        print("test_data: ", test_data.shape)
       


          # Defining time step and number of features.

        
        prediction_length = len(test_data) - time_step

        X_train, Y_train = self.create_dataset(train_data, time_step)


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


        X_validation, Y_validation = self.create_dataset(validation_data, time_step)

        X_validation = np.reshape(X_validation, (X_validation.shape[0],  time_step, num_features))
        Y_validation = test_data[time_step:, 0]

        X_test, Y_test =self.create_dataset(test_data, time_step)

        X_test = np.reshape(X_test, (X_test.shape[0],  time_step, num_features))
        Y_test = test_data[time_step:, 0]



        dim_exit=1
        na=50 
        batch_size=20
        epochs=20

        #LSTM
        LSTM_model = Sequential()
        LSTM_model.add(LSTM(units=na, input_shape=(time_step, num_features)))
        LSTM_model.add(Dropout(0.3)) 
        LSTM_model.add(Dense(1))
        LSTM_model.add(Dense(units=dim_exit))
        LSTM_model.compile(optimizer='adam',loss='mse')
        history=LSTM_model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_validation, Y_validation),verbose=1)





       #GRU
        GRU_model= Sequential()
        GRU_model.add(GRU(units=na, input_shape=(time_step, num_features)),)
        GRU_model.add(Dropout(0.3))
        GRU_model.add(Dense(1))
        GRU_model.add(Dense(units=dim_exit))
        GRU_model.compile(optimizer='adam',loss='mse')
        history2=GRU_model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_validation, Y_validation),verbose=1)

        # Making predictions
        LSTM_predic = LSTM_model.predict(X_test)       
        GRU_predic = GRU_model.predict(X_test)

         # Inverse transforming the predictions to get them back to the original scale
        price_predictions_scaled = scaler_price.inverse_transform(LSTM_predic)
        price_predictions_scaled2 = scaler_price.inverse_transform(GRU_predic)                

        #Evaluation metrics for LSTM
        # Evaluation metrices RMSE and MAE

        print("LSTM Test data RMSE: ", math.sqrt(mean_squared_error(Y_test,LSTM_predic)))
        print("LSTM Test data MSE: ", mean_squared_error(Y_test,LSTM_predic))
        print("LSTM Test data MAE: ", mean_absolute_error(Y_test,LSTM_predic))

        ## Variance Regression Score
        print("LSTM Test data explained variance regression score:", 
              explained_variance_score(Y_test, LSTM_predic))

        ## R square score for regression
        print("LSTM Test data R2 score:", r2_score(Y_test, LSTM_predic))

        ## Regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
        print("LSTM Test data MGD: ", mean_gamma_deviance(Y_test, LSTM_predic))
        print("LSTM Test data MPD: ", mean_poisson_deviance(Y_test, LSTM_predic))

        print("----------------------------------------------------------------------")
        print("----------------------------------------------------------------------")

        # Evaluation metrics for GRU
        # Evaluation metrices RMSE and MAE

        print("GRU Test data RMSE: ", math.sqrt(mean_squared_error(Y_test,GRU_predic)))
        print("GRU Test data MSE: ", mean_squared_error(Y_test,GRU_predic))
        print("GRU Test data MAE: ", mean_absolute_error(Y_test,GRU_predic))

        ## Variance Regression Score
        print("GRU Test data explained variance regression score:", 
              explained_variance_score(Y_test, GRU_predic))

        ## R square score for regression
        print("GRU Test data R2 score:", r2_score(Y_test, GRU_predic))

        ## Regression Loss Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
        print("GRU Test data MGD: ", mean_gamma_deviance(Y_test, GRU_predic))
        print("GRU Test data MPD: ", mean_poisson_deviance(Y_test, GRU_predic))



        
        # Extract dates for test data
       
        test_dates = self.ETH_data.index[-len(test_data):] 
        test_dates = test_dates[-len(LSTM_predic):] 
        test_dates = pd.to_datetime(test_dates)

        
        assert len(LSTM_predic) == len(test_dates), f"Length of predictions: {len(LSTM_predic)}, Length of test dates: {len(test_dates)}"
        

        # Ensure the predictions match the number of test dates
        adjusted_predic = price_predictions_scaled[:prediction_length]
        adjusted_predic2 = price_predictions_scaled2[:prediction_length]

        #print(len(adjusted_predic), len(test_dates)) 

        


        # Extract actual high prices 
        actual_high_prices = self.ETH_data['Price'][-len(test_data):]
        actual_high_prices_trimmed = actual_high_prices[-len(LSTM_predic):]
       

        # Plotting both actual and predicted prices
        plt.figure(figsize=(14, 7))
        plt.plot(test_dates, actual_high_prices_trimmed, color="red", label="Actual High Price")
        plt.plot(test_dates, adjusted_predic, color="blue", label="Predicted High Price by LSTM")
        plt.plot(test_dates, adjusted_predic2, color="yellow", label="Predicted High Price GRU")
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
        LSTM_future_predictions = []
        GRU_future_predictions = []

        # Predict the next 30 days
        for i in range(30):  # 30 days
           # Predict the next day using the first model
            LSTM_next_day_prediction = LSTM_model.predict(current_batch)[0]
            # Predict the next day using the second model
            GRU_next_day_prediction = GRU_model.predict(current_batch)[0]
             # Add the predictions to their respective lists
            LSTM_future_predictions.append(LSTM_next_day_prediction)
            GRU_future_predictions.append(GRU_next_day_prediction)

            # Get the features from the last day in the batch (excluding the target feature)
            last_features = current_batch[0, -1, 1:]
    
            # Combine the next day prediction with the last features
            LSTM_next_day_input = np.hstack([LSTM_next_day_prediction, last_features])
            # Reshape the combination to match the expected number of features
            LSTM_next_day_input = LSTM_next_day_input.reshape((1, 1, num_features))
            # Update the batch to include the new prediction and drop the oldest day
            current_batch = np.concatenate([current_batch[:, 1:, :], LSTM_next_day_input], axis=1)
    
            # Output the current shape of the batch for debugging purposes
            print(f"current_batch shape: {current_batch.shape}")
            print(f"next_day_input shape: {LSTM_next_day_input.shape}")
    
            # Repeat the process for the second model's predictions
            GRU_next_day_input = np.hstack([GRU_next_day_prediction, last_features])
            GRU_next_day_input = GRU_next_day_input.reshape((1, 1, num_features))
            print(f"next_day_input shape: {GRU_next_day_input.shape}")

         # Reverse the scaling transformation to convert predictions back to their original scale
        LSTM_future_predictions_scaled = scaler_price.inverse_transform(np.array(LSTM_future_predictions).reshape(-1, 1))
        GRU_future_predictions_scaled = scaler_price.inverse_transform(np.array(GRU_future_predictions).reshape(-1, 1))



        
       # Determine the start date for the predictions by adding one day to the last date in the dataset
        last_date = pd.to_datetime(self.ETH_data.index[-1])
        start_date = last_date + pd.Timedelta(days=1)
        # Create a range of dates for the future predictions
        prediction_dates = pd.date_range(start=start_date, periods=30)


        # Plotting
        plt.figure(figsize=(14,7))
        plt.plot(prediction_dates, LSTM_future_predictions_scaled, color='green', linestyle='--', label='Future Predicted Price by LSTM')
        plt.plot(prediction_dates, GRU_future_predictions_scaled, color='orange', linestyle='--', label='Future Predicted Price by GRU')
        plt.title('Predicted Future Prices for the Next 30 Days')
        plt.xlabel('Date')
        plt.ylabel('Price(Dollar)')
        plt.legend()
        plt.xticks(rotation=45)  # Rotate dates for better readability
        plt.show()


     def technical_indicators(self,dataset):
        
    
            # Create MACD
            dataset['26ema'] = dataset['Price'].ewm(span=26).mean()
            dataset['12ema'] = dataset['Price'].ewm(span=12).mean()
            dataset['MACD'] = dataset['12ema']-dataset['26ema']

            
    
            # Create Exponential moving average
            dataset['ema'] = dataset['Price'].ewm(com=0.5).mean()
    
           

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

     def create_dataset(self,data, time_step):
                X, Y = [], []
                for i in range(len(data) - time_step):
                 X.append(data[i:(i + time_step), :])
                 Y.append(data[i + time_step, 0]) 
            
                return np.array(X), np.array(Y)

     def get_yesterday_date(self):
        yesterday = datetime.now() - timedelta(days=1)
        return yesterday.strftime('%Y-%m-%d')






           

           