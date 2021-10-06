from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc


dftrain = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/train.csv') #training data
dfeval = pd.read_csv('http://storage.googleapis.com/tf-datasets/titanic/eval.csv') #training data

print(dftrain.head()) #Returns summary of the dataset

y_train = dftrain.pop('survived') #pick survived column only
y_eval = dfeval.pop('survived')
print(y_train)

print(dftrain["age"]) #Calling specific column
print(dftrain.loc[0], y_train.loc[0]) #.loc[0] : Calling first row of the dataset
                                      #dftrain.loc[0], y_train.loc[0] provides whoever the row[0] survived or not based on 'y_train_ information

print(dftrain.describe()) # provides summary, such as min, mean, std etc

print(dftrain.age.hist(bins=20))                        #Provides histogram of age
print(dftrain.sex.value_counts().plot(kind='barh'))     #Provides survival rate by sex

#Categorical col : Column with non numeric columns
Categorical_columns = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
Numeric_Columns = ['age', 'fare']

feature_columns = []
for feature_name in Categorical_columns:
    vocabulary = dftrain[feature_name].unique() #Find unique values from each category
    # print(vocabulary)
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in Numeric_Columns:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)
