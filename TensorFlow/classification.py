from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']

train_path = tf.keras.utils.get_file(
    "iris_training.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path = tf.keras.utils.get_file(
    "iris_test.csv", "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")


train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0) #header = 0 means row 0 is the header
test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

print(train.head())

train_y = train.pop('Species')
test_y = test.pop('Species')
print(train.head())

print(train.shape) #120 entries, with 4 columns

def input_fn(features, labels, training = True, batch_size=256):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    if training:
        dataset = dataset.shuffle(1000).repeat()
    return dataset.batch(batch_size)

my_feature_columns = []
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
print(my_feature_columns)

#Defined Model with DNNClassifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30, 10],      #Two hidden layers of 30 and 10 nodes respectively
    n_classes=3                 #The model must choose between 3 classes
)

'''
lambda is a function itself. Whatever comes after lambda is the function.
x = lambda: print("hi")
x() <---- this will print hi
'''
#Training Model
classifier.train(
    input_fn=lambda: input_fn(train, train_y, training=True),
    steps=5000
)

#Now Evaluating the data with trained data
eval_result = classifier.evaluate(input_fn=lambda: input_fn(test, test_y, training=False))
print('Test set accuracy : {accuracy:0.3f}\n'.format(**eval_result))

def input_fn(features, batch_size=256):
    return  tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
predict = {}

print("Please type numeric values as prompted.")
for feature in features:
    valid = True
    while valid:
        val = input(feature + ": ")
        if not val.isdigit(): valid = False

    predict[feature] = [float(val)] # ask each feature's length

predictions = classifier.predict(input_fn = lambda: input_fn(predict))
for pred_dict in predictions:
    class_id = pred_dict['class_ids'][0]
    probability = pred_dict['probabilities'][class_id]

    print('Prediction is "{}" O{:.1f}%'.format(
        SPECIES[class_id], 100 * probability
    ))

''' result
Please type numeric values as prompted.
SepalLength: 2.3
SepalWidth: 2.5
PetalLength: 6.3
PetalWidth: 3.4
Prediction is "Virginica" O71.9%
'''
