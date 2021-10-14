'''Natural Language Processing
: Field in computing that deals with communication between natural (human) languages and computer languages
i.e. Spellcheck, auto-complete, voice assistance

1) Recurrent Neural Networks (RNN)
- Sentiment Analysis
- Character Generation

Usually used to process text data. Has a loop, it start to develop analyzation of the data by recurring same layer.
They maintain an internal memory/state of the input that was already processed. Also, RNN contain a loop and process one piece of input at a time


a) Simple RNN Layer
:Access & Stores only one previous output

output_1   output_2    ..... output_n
   ^   --|      ^               ^
-------  |   -------          -------
Layer    |->  Layer    ....    Layer
-------      -------          -------
  ^            ^                ^
Input_1      input_2          input_n
(I)           (am)           (Matt)



b) LSTM (Long-Short Term Memory)
:Can Access to any previous stage's output

'''

'''
Bag of words
> Convert every word into integer. Then analyze sentence and keep the track of frequency of word in a sentence, save its frequencies.
> We may loose the actual meaning of sentence since same amount of word could make different sentences depends on the location in the sentence.
'''
vocab = {} #maps word to integer representing it
word_encoding = 1
def bag_of_words(text):
    global word_encoding

    words = text.lower().split(" ") #create a list of all of the words in the text
    bag = {}                        #Stores all of the encodings and their frequency

    for word in words:
        if word in vocab:
            encoding = vocab[word]  #get encoding from vocab
        else:
            vocab[word] = word_encoding
            encoding = word_encoding
            word_encoding += 1

        if encoding in bag:
            bag[encoding] += 1
        else:
            bag[encoding] = 1
    return bag

text = "this is a test to see if t his test will work is is teat a a"
bag = bag_of_words(text)
print(bag)
print(vocab)

'''
Word Embeddings
- Classify or translate every word into vector

Recurrent Neural
'''
'''
Word Embeddings
- Classify or translate every word into vector

Recurrent Neural
'''

'''Sentiment Analysis
- Movie Review Dataset
'''

from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_label), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

train_data[0]   #Looking at one review

#More Preprocessing
'''
Since each review has different length, we need to unified the length of review to pass into neural network
If review > 250 words, trim off extra
If review < 250 words, add the necessary amount of 0's to make it 250 words
'''
train_data = sequence.pad_sequences(train_data, MAXLEN)     #Perform matching length of the review for train data
test_data = sequence.pad_sequences(test_data, MAXLEN)       #Perform matching length of the review for test data

#Creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation = "sigmoid")    #"sigmoid" sequeeze our value between 0 and 1. Like <0.5 review becomes bad and opposite for good review
])

model.summary()
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 32)          2834688   
_________________________________________________________________
lstm (LSTM)                  (None, 32)                8320      
_________________________________________________________________
dense (Dense)                (None, 1)                 33        
=================================================================
Total params: 2,843,041
Trainable params: 2,843,041
Non-trainable params: 0
_________________________________________________________________
'''

#Training
model.compile(loss = "binary_crossentropy", optimizer="rmsprop", metrics=['acc'])
history = model.fit(train_data, train_label, epochs=10, validation_split=0.2)

results = model.evaluate(test_data, test_labels)
print(results)
