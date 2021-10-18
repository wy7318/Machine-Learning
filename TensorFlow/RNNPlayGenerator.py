#RNN Play Generator
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import numpy as np
import os

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt') #Get file and save as shakespeare.txt


#### If want to upload our own file. Make sure it is txt file
# from google.colab import files
# path_to_file = list(files.upload().keys())[0]

#Read, then decode for py2 compat
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#length of text is the number of characters in it
print('Length of text : {} characters'.format(len(text)))


#Take a look at the first 250 characters in text
print(text[:250])

#Encoding
vocab = sorted(set(text))
#Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}       #Convert every character (Including \n, !, $, etc) into numbers
idx2char = np.array(vocab)                          #Make index to array of every character

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

#Look at how part of text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

#Now encoded number to characters
def int_to_text(ints):
    #Check if passing variant is numpy array or not, make as numpy array if it isn't
    try:
        ints = ints.numpy()
    except:
        pass
    return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))


'''Creating Training Example
This training examples will use a seq_length sequence as input and a 
seq_length sequence as the output where that sequence is the original sequence shifted one letter to the right. 
i.e.
input : Hello | Output : ello 
'''

seq_length = 100 #length of the sequence for training example
examples_per_epoch = len(text)//(seq_length+1)      #to make training sample's seq length's as 100.
                                                    #Divide by 101 will always chunk the text into 100 length
# Creating training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)       #Taking above char_dataset and make batch of 101, drop the remainder

def split_input_target(chunk):  #for the example : hello
    input_text = chunk[:-1]     #hell
    target_text = chunk[1:]     #ello
    return input_text, target_text  #hell, ello

dataset = sequences.map(split_input_target)  #Use map to apply the above function to every entry

for x,y in dataset.take(2):
    print("\n\nEXAMPLE\n")
    print("INPUT")
    print(int_to_text(x))
    print("\nOUTPUT")
    print(int_to_text(y))

'''Result
EXAMPLE

INPUT
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You

OUTPUT
irst Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You 


EXAMPLE

INPUT
are all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you 

OUTPUT
re all resolved rather to die than to famish?

All:
Resolved. resolved.

First Citizen:
First, you k
'''

BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)     #vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it does not attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements

BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

#Building the Model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(rnn_units,
                             return_sequences = True,
                             stateful=True,
                             recurrent_initializer = 'glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

#Creating Loss Function
for input_example_batch, target_example_batch in data.take(1):
    example_batch_predictions = model(input_example_batch)      # Ask model for a prediction on first batch of training data
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")   # print out the output shape

# We can see that the prediction is an array of 64 arrays, one for each entry in the batch
print(len(example_batch_predictions))
print(example_batch_predictions)

#Examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
#notice this is a 2d array of length 100, where each interior array is the prediction for the next

# and finally, prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probability of each character occuring next

# If we want to determine the predicted character we need to sample the output distribution (Pick a value based on probabilities)
sampled_indices = tf.random.categorical(pred, num_samples=1)

#now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars     # and this is what the model predicted for training sequence 1

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#Compile Model
model.compile(optimizer = 'adam', loss=loss)

#Creating Checkpoints
#Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
#Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_prefix,
    save_weights_only=True
)

#Training
history = model.fit(data, epochs = 40, callbacks = [checkpoint_callback])

#Loading the Model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))

#Generating Text
def generate_text(model, start_string):
    #Evaluation Step (generating text using the learned model

    #Number of characters to generate
    num_generate = 800

    #Converting our start string  to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    #Empty string to store our results
    text_generated = []

    #Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text
    #Experiment to find the best setting
    temperature = 1.0

    #Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        #remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        #using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        #We pass the predicted character as the next input to the model
        #along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])
    return (start_string + ''.join(text_generated))

inp = input("Type a starting string: ")
print(generate_text(model, inp))
