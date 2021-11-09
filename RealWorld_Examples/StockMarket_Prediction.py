# using Recurrent Neural Network, Long Short Term Memory(LSTM) to predict the closing stock price.

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
keras = tf.keras

plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('TSLA', data_source='yahoo', start = '2020-01-01', end='2021-10-25')
print(df)

# Get the number of rows & columns in the dataset
print(df.shape)

# Visualize the closing price history
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()

# Create a new datafram with only the close column
data = df.filter(['Close'])
print(data)
# Convert the dataframe to a numpy array
dataset = data.values
# Get the number of rows to train the model on
# Train 80% of the data.
training_data_len = math.ceil(len(dataset) * 0.8)       # math.ceil is to round up
print(training_data_len)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)     # Now the dataset will be scaled between 0 and 1
print(scaled_data)

# Create the training dataset
# Create the scaled training dataset
train_data = scaled_data[0:training_data_len, :]        # Get from index 0 to the training len
print(train_data)
# Split the data into x_train and y_train datasets
x_train = []        # Independent variable
y_train = []        # Dependent variable


# Predicting with past 60 days data
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])       # x_train will contain past 60 days closing data
    y_train.append(train_data[i, 0])            # y_train will contain the 61st day's closing data.

    # Check result
    if i<= 60:
        print(x_train)
        print(y_train)
        print()

# Convert the x_train and y_train to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to meet LSTM's requirement (3D required)
print(x_train.shape)                        #(1908, 60) Rows & Cols
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))           # reshape number of samples, number of times, number of features
                                                                                 # x_train.shape[0] = 1908, x_train_shape[1] = 60, feature is 1 because we have one output

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))       # Create first 50 layers, return_sequences is 'True' since there will be another LSTM model.
                                                                                    # Input_shape is (number of times, number of features)
model.add(LSTM(50, return_sequences=False))                                         # return_sequence is 'False' since there will be no more LSTM model.
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')                          # Loss function is to determine how well the model did
model.save("specific.h5")

newModel = tf.keras.models.load_model('specific.h5')                                #Saving Model
# Train the model
newModel.fit(x_train, y_train, batch_size=1, epochs=2)

# Create the testing data set
# Create a new array containing scaled values
test_data = scaled_data[training_data_len - 60:, :]

# Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]     #from non-trained data set to the end. Remaining 20%

for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert the data to a numpy array
x_test = np.array(x_test)

# Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get the models predicted price values
predictions = newModel.predict(x_test)
predictions = scaler.inverse_transform(predictions)     # Un-scaling value
print(predictions)

# Evaluate model. Get the root mean squared error (RMSE)
rmse = np.sqrt( np.mean(predictions - y_test )**2 )

# Plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
# Visualize the data
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc = 'lower right')
plt.show()

# Show the valid and predicted prices
print(valid)

#Get the quote
tesla_quote = web.DataReader('TSLA', data_source='yahoo', start = '2020-01-01', end = '2021-10-25')

# Create a new dataframe
new_df = tesla_quote.filter(['Close'])
# Get the last 60 days closing price values and convert the dataframe to array
last_60_days = new_df[-60:].values
#Scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)
# Create an empty list
X_text = []
# Append the past 60 days
X_text.append(last_60_days_scaled)
# Convert the X_test data set to a numpy array
X_text = np.array(X_text)
# Reshape the data
X_text = np.reshape(X_text, (X_text.shape[0], X_text.shape[1], 1))
# Get the predicted scaled price
pred_price = newModel.predict(X_text)
#undo the scaling
pred_price = scaler.inverse_transform(pred_price)
print("Predicted price for 10/26/21 : ", pred_price)

#Get the quote
tesla_quote2 = web.DataReader('TSLA', data_source='yahoo', start = '2021-10-19', end = '2021-10-19')
print("Actual Price for 10/19/21", tesla_quote2['Close'])
