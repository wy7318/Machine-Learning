'''Pretrained Models : Using generalized model that already exist
Fine Tuning : We pass our own trained  images'''

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split = ['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str   #creates a function object that we can use to get labels

#display 2 images from the dataset
for image, label in raw_train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))


#Data Processing
IMG_SIZE = 160 #Al images will be resized to 160*160
               #It's better to make size smaller than bigger

def format_example(image, label):
    """
    returns an image that is reshaped to IMA_SIZE
    """

    image = tf.cast(image, tf.float32)  #Convert ebery image to float32
    image = (image/127.5) - 1           #Half of 255 - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)               #
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

for image, label in train.take(2):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))

# Check image size
for img, label in raw_train.take(2):
    print("Original Shape: ", img.shape)

for img, label in train.take(2):
    print("New Shape : ", img.shape)
