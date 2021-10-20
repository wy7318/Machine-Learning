# using Recurrent Neural Network, Long Short Term Memory(LSTM) to predict the closing stock price.

import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Get the stock quote
df = web.DataReader('TSLA', data_source='yahoo', start = '2012-01-01', end='2021-10-10')
print(df)
