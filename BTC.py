# -*- coding: cp1252 -*-
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.layers import Dropout
import yfinance as yf
from datetime import datetime, timedelta
from keras.callbacks import Callback


class UpdateProgressBar(Callback):
    def __init__(self,root, progress_bar, progress_label, total_epochs):
        super().__init__()
        self.root = root
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Update progress
        percentage = (epoch + 1) / self.total_epochs * 100
        def update_ui():
            self.progress_bar['value'] = percentage
            self.progress_label.config(text=f"Training Progress: {percentage:.2f}%")
        self.root.after(0, update_ui)

class Bitcoin():
    

     def __init__(self,root,progress_bar=None, progress_label=None, callback=None,stop_flag=None):
        self.stop_flag = stop_flag  
        self.root = root
        self.progress_bar = progress_bar
        self.progress_label = progress_label
        ticker_symbol = "BTC-USD"
        start_date = "2014-09-17"
        end_date = self.get_yesterday_date()

        # Fetch historical data
        self.BTC_data = yf.download(ticker_symbol, start=start_date, end=end_date)

        # Format the DataFrame
        self.BTC_data = self.BTC_data[['High', 'Volume']]
        self.BTC_data.columns = ['Price', 'Volume']
        self.BTC_data.index = pd.to_datetime(self.BTC_data.index)
        self.BTC_data = self.BTC_data.sort_index()
        print(self.BTC_data.head())

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
       

        # Scale each feature independently
        scaler_price = MinMaxScaler()
        scaler_volume = MinMaxScaler()
        scaler_EMA = MinMaxScaler()
        scaler_SMA= MinMaxScaler()
        scaler_RSI= MinMaxScaler()
        

        prices_scaled = scaler_price.fit_transform(prices)
        volumes_scaled = scaler_volume.fit_transform(volumes)
        EMA_scaled = scaler_EMA.fit_transform(EMA)
        SMA_scaled = scaler_SMA.fit_transform(SMA)
        RSI_scaled= scaler_RSI.fit_transform(RSI)
        


        # Concatenate the scaled features
        scaled_features = np.concatenate([prices_scaled, volumes_scaled, EMA_scaled,SMA_scaled,RSI_scaled,], axis=1)
        time_step = 30
        num_features = 5
        split_ratio = {'train': 0.7, 'validation': 0.1, 'test': 0.2}
        

        
        trimmed_size = len(scaled_features) - (len(scaled_features) % time_step)
        trimmed_data = scaled_features[:trimmed_size]

   # Define the data size and calculate split indices for an 70%-10%-20% split

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

        Y_train = train_data[time_step:, 0]
        # Continue with the LSTM_model definition and training...
      


        X_validation, Y_validation = self.create_dataset(validation_data, time_step)

        X_validation = np.reshape(X_validation, (X_validation.shape[0],  time_step, num_features))
        Y_validation = validation_data[time_step:, 0]

        X_test, Y_test =self.create_dataset(test_data, time_step)

        X_test = np.reshape(X_test, (X_test.shape[0],  time_step, num_features))
        Y_test = test_data[time_step:, 0]

        print('train_data shape:', train_data.shape)
        print('X_train shape:', X_train.shape)
        print('Y_train shape:', Y_train.shape)

        dim_exit=1
        na=50 
        batch_size=32
        epochs=50

        #LSTM
        LSTM_model = Sequential()
        LSTM_model.add(LSTM(units=na, input_shape=(time_step, num_features)))
        LSTM_model.add(Dense(1))
        LSTM_model.add(Dense(units=dim_exit))
        LSTM_model.compile(optimizer='adam',loss='mse')

         # Create the UpdateProgressBar callback instance
        if self.progress_bar and self.progress_label:
            progress_callback = UpdateProgressBar(root, self.progress_bar, self.progress_label, epochs)
            callbacks = [progress_callback]
        else:
            callbacks = []

        history_LSTM=LSTM_model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_validation, Y_validation),callbacks=callbacks,verbose=1)


        progress_bar['value'] = 0
        progress_label.config(text="Initializing GRU ...")


        #GRU
        GRU_model= Sequential()
        GRU_model.add(GRU(units=na, input_shape=(time_step, num_features)),)
        GRU_model.add(Dense(1))
        GRU_model.add(Dense(units=dim_exit))
        GRU_model.compile(optimizer='adam',loss='mse')

         # Create the UpdateProgressBar callback instance
        if self.progress_bar and self.progress_label:
            progress_callback = UpdateProgressBar(root,self.progress_bar, self.progress_label, epochs)
            callbacks = [progress_callback]
        else:
            callbacks = []

        history_GRU=GRU_model.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,validation_data=(X_validation, Y_validation),callbacks=callbacks,verbose=1)



        # Making predictions
        LSTM_predic = LSTM_model.predict(X_test)
        GRU_predic = GRU_model.predict(X_test)

        # Inverse transforming the predictions to get them back to the original scale
        LSTM_price_predictions_scaled = scaler_price.inverse_transform(LSTM_predic)
        GRU_price_predictions_scaled = scaler_price.inverse_transform(GRU_predic)


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



       
        progress_bar['value'] = 0
        progress_label.config(text="Initializing Future Prediction ...")

        # Extract dates for test data
       
        test_dates = self.BTC_data.index[-len(test_data):] 
        test_dates = test_dates[-len(LSTM_predic):] 
        
        assert len(LSTM_predic) == len(test_dates), f"Length of predictions: {len(LSTM_predic)}, Length of test dates: {len(test_dates)}"
        

        # Ensure the predictions match the number of test dates
        LSTM_adjusted_predic = LSTM_price_predictions_scaled[:prediction_length]
        GRU_adjusted_predic = GRU_price_predictions_scaled[:prediction_length]

        #print(len(adjusted_predic), len(test_dates)) 

        


        # Extract actual high prices 
        actual_high_prices = self.BTC_data['Price'][-len(test_data):]
        actual_high_prices_trimmed = actual_high_prices[-len(LSTM_predic):]
       

        # Plotting both actual and predicted prices
        plt.figure(figsize=(20, 10))
        plt.plot(test_dates, actual_high_prices_trimmed, color="red", label="Actual High Price")
        plt.plot(test_dates, LSTM_adjusted_predic, color="blue", label="Predicted High Price by LSTM")
        plt.plot(test_dates, GRU_adjusted_predic, color="yellow", label="Predicted High Price by GRU")
        plt.title("Bitcoin Predicted vs Actual High Prices")
        plt.xlabel("Date")
        plt.ylabel("High Price (Dollar)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.get_current_fig_manager().window.state('zoomed')
        plt.show()

         # Number of days to look back to make future predictions
        time_based = 90
        # Prepare the input for the first prediction by taking the last 90 days from test_data
        last_90_days = test_data[-time_based:]
        # Reshape the last 90 days data to fit the LSTM_model's expected input shape
        current_batch = last_90_days.reshape((1, time_based, num_features))

        # Initialize two lists to store future predictions for each LSTM_model
        LSTM_future_predictions = []
        GRU_future_predictions = []

        # Predict the next 30 days
        for i in range(30):  # Loop for 30 days
            # Predict the next day using the first LSTM_model
            LSTM_next_day_prediction = LSTM_model.predict(current_batch)[0]
            # Predict the next day using the second LSTM_model
            GRU_next_day_prediction = GRU_model.predict(current_batch)[0]
            # Add the predictions to their respective lists
            LSTM_future_predictions.append(LSTM_next_day_prediction)
            GRU_future_predictions.append(GRU_next_day_prediction)
    
            # Get the features from the last day in the batch (excluding the target feature)
            last_features = current_batch[0, -1, 1:]
    
            # Combine the next day prediction with the last features
            next_day_input = np.hstack([LSTM_next_day_prediction, last_features])
            # Reshape the combination to match the expected number of features
            next_day_input = next_day_input.reshape((1, 1, num_features))
            # Update the batch to include the new prediction and drop the oldest day
            current_batch = np.concatenate([current_batch[:, 1:, :], next_day_input], axis=1)
    
            # Output the current shape of the batch for debugging purposes
            print(f"current_batch shape: {current_batch.shape}")
            print(f"next_day_input shape: {next_day_input.shape}")
    
            # Repeat the process for  GRU model prediction
            GRU_next_day_input = np.hstack([GRU_next_day_prediction, last_features])
            GRU_next_day_input = GRU_next_day_input.reshape((1, 1, num_features))
            print(f"next_day_input shape: {GRU_next_day_input.shape}")


        # Reverse the scaling transformation to convert predictions back to their original scale
        LSTM_future_predictions_scaled = scaler_price.inverse_transform(np.array(LSTM_future_predictions).reshape(-1, 1))
        GRU_future_predictions_scaled = scaler_price.inverse_transform(np.array(GRU_future_predictions).reshape(-1, 1))

        # Get the last date from the dataset
        last_date = self.BTC_data.index[-1]
        # Calculate the start date for the future predictions by adding one day to the last date
        start_date = last_date + pd.Timedelta(days=1)
        # Create a date range for the next 30 days starting from the start_date
        prediction_dates = pd.date_range(start=start_date, periods=30)

        # Plotting the predictions
        # Set the figure size for better visibility
        plt.figure(figsize=(15,7))
        # Plot the scaled future predictions from the first LSTM_model
        plt.plot(prediction_dates, LSTM_future_predictions_scaled, color='green', linestyle='--', label='Future Predicted Price by LSTM')
        # Plot the scaled future predictions from the second LSTM_model
        plt.plot(prediction_dates, GRU_future_predictions_scaled, color='orange', linestyle='--', label='Future Predicted Price by GRU')
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
        plt.get_current_fig_manager().window.state('zoomed')
        # Display the plot
        plt.show()


     def technical_indicators(self,dataset):
        
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

           

      