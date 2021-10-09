'''Neural Networks
1) Densely Connected Neural Network
Every node in different layers(input, hidden, output) is connected to each other.

Connection : weight
Bias : Some constant numeric value that's connected to the next layer. When bias is connected to another layer, weight is typically 1.

layer's node value = [sum of (Previous connected Node Value)*(weight)] + (Bias value)*(Weight =1)

2) Activation Functions
Function that's applied to the weighed sum of a neuron (node). To prevent output neuron to be out of range.
- Rectified Linear Unit : Make any x value less than 0 to 0. Any value that's positive, keep its original value
Eliminate any negative value

- Tanh (Hyperbolic Tangent) : Squish our value between -1 to 1

- Sigmoid : Squish our values between 0 to 1. Theta(z) = 1/(1+e^(-z))

3) Loss Function
How far away our output is from expected output. Determine if network is good or not depending on the error. Then it will revise the network (its weight and nodes) by going reversely.
- Mean Squared Error
- Mean Absolute Error
- Hinge Loss

4) Optimizer
Function that implements the backpropagation algorithm.
'''

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

fashin_mnist = keras.datasets.fashion_mnist # load dataset
(train_images, train_labels), (test_images, test_labels) = fashin_mnist.load_data() #split into testing and training

print(test_images.shape)

train_images[0,23,23]
print(train_labels[:10])

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(True)
plt.show()

#Data Preprocessing
train_images = train_images / 255.0         #Preprocess to make the value between 0 and 1 for training image
test_images = test_images / 255.0           #Preprocess to make the value between 0 and 1 for test image

#Building Model
model = keras.Sequential([                                  #Sequential : data goes from left to right sequentially
    keras.layers.Flatten(input_shape = (28, 28)),           #input layer (1)
                                                            #Flatten function make (28, 28) flatten to 784 pixels
    keras.layers.Dense(128, activation='relu'),             #hidden layer (2)
                                                            #Dense : Densley connected neurons. 128 neurons.
                                                            #Choose Activation function 'Rectified Linear Unit'
    keras.layers.Dense(10, activation='softmax')            #output layer (3)
                                                            #10 neurons because we have 10 classes
])

#Compile Model
model.compile(optimizer = 'adam',
              loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# #Train Model
# model.fit(train_images, train_labels, epochs=8) #fit : training data
#                                                 #epochs : hyperparameter
#
# #Evaluating the Model
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
# print('Test accuracy:', test_acc)
#
# '''
# training accuracy: 0.9044
# Test accuracy: 0.8766999840736389
#
# training accuracy came out high but actual test accuracy came out lower than that. This is problem of 'overfit'.
# We could adjust epochs number (lower) and see if both training and test accuracy become closer.
# '''
#
# #Make Prediction
# predictions = model.predict(test_images)
# print(class_names[np.argmax(predictions[0])])
# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(True)
# plt.show()


#Guessing function
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]

    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.binary)
    plt.title("Expected: " + label)
    plt.xlabel("Guess: " + guess)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def get_number():
    while True:
        num = input("Pick a number:")
        if num.isdigit():
            num = int(num)
            if 0<= num <= 1000:
                return int(num)
        else:
            print("Try again")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)

