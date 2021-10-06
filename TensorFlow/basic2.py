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

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function():
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #create tf.data.Dataset object with data and its label
        if shuffle:
            ds = ds.shuffle(1000) #randomizing order of data
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

linear_est.train(train_input_fn) #Train
result = linear_est.evaluate(eval_input_fn) #Get model metrics/stats by testing on testing data

print(result['accuracy']) #result variable is simly a dict of stats about model
print(result)

result = list(linear_est.predict(eval_input_fn)) #Making list of predictions with configured linear model
print('Information of person #3\n', dfeval.loc[2]) #printing third person's information in the dataset
print('Probability of person #3 survival based on our linear model : ',result[2]['probabilities'][1]) #result of our third training input ([3]), check probabilities(['probabilities']), probability of not survival ([0] or survival as [1])
print('Survival of person #3 (1: Survived, 0 : Not Survived):',y_eval.loc[3]) #Actual fact whether this person is survived or not

'''Result
Information of person #3
 sex                        female
age                            58
n_siblings_spouses              0
parch                           0
fare                        26.55
class                       First
deck                            C
embark_town           Southampton
alone                           y
Name: 2, dtype: object
Probability of person #3 survival based on our linear model :  0.85323733
Survival of person #3 (1: Survived, 0 : Not Survived): 1
'''
