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

#Shuffle and batch the images
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Check image size
for img, label in raw_train.take(2):
    print("Original Shape: ", img.shape)

for img, label in train.take(2):
    print("New Shape : ", img.shape)

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

#Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,   #Important. This model is trained on 1.4m images and has 1000different classes but not including them
                                               weights='imagenet')

base_model.summary()

for image, _ in train_batches.take(1):
    pass

feature_batch = base_model(image)
print(feature_batch.shape)

#Freezing the base : disabling the training property of a layer. Simply, we won't make any changes to the weights of any layers tht are frozen during training
base_model.trainable = False
base_model.summary() #Trainable param = 0

#Adding classifier
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer
])
model.summary()
''' Result
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
mobilenetv2_1.00_160 (Functi (None, 5, 5, 1280)        2257984   
_________________________________________________________________
global_average_pooling2d (Gl (None, 1280)              0         
_________________________________________________________________
dense (Dense)                (None, 1)                 1281      
=================================================================
Total params: 2,259,265
Trainable params: 1,281
Non-trainable params: 2,257,984
_________________________________________________________________
'''

#Training the Model
base_learning_rate = 0.0001 #How much am I allowed to modify the weights and bias on this network
model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

#We can evaluate the model now to see how it does before training it on our new images
initial_epochs = 3
validation_steps=20
loss0, accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

#Now we can train it on our images
history = model.fit(train_batches,
                    epochs = initial_epochs,
                    validation_data=validation_batches)

acc = history.history['accuracy']
print(acc)

model.save("dogs_vs_cats.h5")                               #Save trained Model. 'h5' is the saving model extension specifically for keras
new_model=tf.keras.models.load_model('dogs_vs_cats.h5')     #Load trained Model
