import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make numpy printouts easier to read
np.set_printoptions(precision=3, suppress=True)

import  tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']

dataset = pd.read_csv(url, names=column_names, na_values='?',
                      comment='\t', sep=' ', skipinitialspace=True)

# Print last 5 rows
print(dataset.tail())
'''
      MPG  Cylinders  Displacement  ...  Acceleration  Model Year  Origin
393  27.0          4         140.0  ...          15.6          82       1
394  44.0          4          97.0  ...          24.6          82       2
395  32.0          4         135.0  ...          11.6          82       1
396  28.0          4         120.0  ...          18.6          82       1
397  31.0          4         119.0  ...          19.4          82       1
'''

# Clean Data
dataset = dataset.dropna()

# Convert categorical 'Origin' data into one-hot data
origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1
dataset['Europe'] = (origin == 2)*1
dataset['Japan'] = (origin == 3)*1

print(dataset.tail())

'''
      MPG  Cylinders  Displacement  Horsepower  ...  Model Year  USA  Europe  Japan  ===> Origin column turned into three different columns
393  27.0          4         140.0        86.0  ...          82    1       0      0
394  44.0          4          97.0        52.0  ...          82    0       1      0
395  32.0          4         135.0        84.0  ...          82    1       0      0
396  28.0          4         120.0        79.0  ...          82    1       0      0
397  31.0          4         119.0        82.0  ...          82    1       0      0

'''

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)  # use 80% for training
test_dataset = dataset.drop(train_dataset.index)          # Remaining from train_dataset

print(dataset.shape, train_dataset.shape, test_dataset.shape)
train_dataset.describe().transpose()                        # Provide count, mean, std, etc

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

# Function to plot one of columns data from train_features dataset
def plot(feature, x=None, y=None):
    plt.figure(figsize=(10, 8))
    plt.scatter(train_features[feature], train_labels, label='Data')        # Data vs MPG
    if x is not None and y is not None:
        plt.plot(x, y, color='k', label = 'Predictions')
    plt.xlabel(feature)
    plt.ylabel('MPG')
    plt.legend()
    plt.show()

plot('Horsepower')

# Normalize
print(train_dataset.describe().transpose()[['mean', 'std']])

# Normalization
normalizer = preprocessing.Normalization()
# Adapt to the data
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

# When the layer is called it returns the input data, with each feature independently normalized
# (input-mean)/stddev
first = np.array(train_features[:1])
print('First example:', first)
print('Normalized:', normalizer(first).numpy())

# feature = 'Displacement'
# single_feature = np.array(train_features[feature])
# print(single_feature.shape, train_features.shape)
#
# single_feature_normalizer = preprocessing.Normalization()
#
# normalizer.adapt(single_feature)


# Regression
# 1. Normalize the input horsepower
# 2. Apply a linear transformation (y=m*x+b) to produce 1 output using layer.Dense
feature = 'Horsepower'
single_feature = np.array(train_features[feature])
print(single_feature.shape, train_features.shape)

# Normalization
single_feature_normalizer = layers.Normalization(input_shape=[1,], axis=None)   # Define input shape
# Adapt to the data
single_feature_normalizer.adapt(single_feature) ## Error

# Sequential model
single_feature_model = keras.models.Sequential([
    single_feature_normalizer,
    layers.Dense(units=1)   # Linear Model that applies y=m*x+b
])
# single_feature_model.build()
print(single_feature_model.summary())

# loss and optimizer
loss = keras.losses.MeanAbsoluteError() # MeanSquaredError can be used too
optim = keras.optimizers.Adam(lr=0.1)

single_feature_model.compile(optimizer=optim, loss = loss)

history = single_feature_model.fit(
    train_features[feature], train_labels,
    epochs = 100,
    verbose=1,
    # Calculate validation results on 20% of the training data
    validation_split = 0.2)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.show()
plot_loss(history)

single_feature_model.evaluate(
    test_features[feature],
    test_labels, verbose=1
)

# predict and plot
range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)

plot(feature, x, y)

# DNN
dnn_model = keras.Sequential([
    single_feature_normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

dnn_model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(0.001))
print(dnn_model.summary())

dnn_model.fit(
    train_features[feature], train_labels,
    validation_split=0.2,
    verbose =1, epochs=100
)
dnn_model.evaluate(test_features[feature], test_labels, verbose=1)

# Predict and plot
x = tf.linspace(range_min, range_max, 200)
y = dnn_model.predict(x)
print("prediction", y)
plot(feature, x, y)


# Multiple inputs
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss=loss
)

linear_model.fit(
    train_features, train_labels,
    epochs=100,
    verbose=1,
    validation_split=0.2
)

linear_model.evaluate(
    test_features, test_labels, verbose=1
)
