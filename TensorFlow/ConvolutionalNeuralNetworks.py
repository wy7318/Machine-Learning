'''Convolutional Neural Networks.
Deep Computer Vision
1) Image Data
3 dimensions : Image Height, Image Width, Color Channels
Dense Neural Networks 
- Needs similar images to distinguish, in terms of position and size. Cannot apply local pattern.(i.e.Whole Image)
- Analyze input on a global scale and recognize patterns in specific areas

Convolutional Neural Networks
- Look at specific parts of the image and learn the pattern (i.e. Eyes, Ears and more).
Then pass each component to dense neural network to distinguish
- Scan through the entire input a little at a time and learn local patterns
- Main Properties : input size, # of filters, # of sample filters.

Filters : It's what's going to be trained. Create feature map with dot product.

Padding : Adding additional columns to each side of images. To make every pixels to be in center

Stride : Moving sample size. i.e. Stride of 1 means moving pixel by 1

Pooling : Take feature map and create another map with Min, Max, or Average (With result of dot product from feature map)
Typically do 2x2 pooling (or with sample size) and stride of 2.
'''
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

#Load and Split dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

#Normalize pixel values to be betweeb 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_name = ['airplane', 'automobile', 'bird', 'cat', 'deer',
              'dog', 'frog', 'horse', 'ship', 'truck']

#Take one image
IMG_INDEX = 1

plt.imshow(train_images[IMG_INDEX],cmap = plt.cm.binary)
plt.xlabel(class_name[train_labels[IMG_INDEX][0]])
plt.show()


'''CNN Architecture'''
model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32, 32, 3))) #32 : Amount of filters
                                                                                #(3,3) : Sample Size (filter size)
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928     
=================================================================
Total params: 56,320
Trainable params: 56,320
Non-trainable params: 0
_________________________________________________________________
'''
#With above information, now we are inputing those to dense layers

model.add(layers.Flatten()) # Flatten above (4,4,64)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10)) #10 is amount of classes we have for this.

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 15, 15, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 4, 4, 64)          36928     
_________________________________________________________________
flatten (Flatten)            (None, 1024)              0         
_________________________________________________________________
dense (Dense)                (None, 64)                65600     
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650       
=================================================================
Total params: 122,570
Trainable params: 122,570
Non-trainable params: 0
_________________________________________________________________

'''

model.compile(optimizer='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=8,
                    validation_data=(test_images, test_labels))

#Evaluating the Model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print(test_acc)

'''Working with small Dataset
- Data Augmentation
'''

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

#Creates a data generator object that transforms images
datagen = ImageDataGenerator( #Allow augment our images
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

#Pick an image to transform
test_img = train_images[14] #Pick 1 arbitrary image to train
img = image.img_to_array(test_img) #Convert image to numpy array
img = img.reshape((1,) + img.shape) #reshape image

i = 0

for batch in datagen.flow(img, save_prefix='test', save_format='jpeg'):     #this loops runs forever until we break, saving images to current directory
    plt.figure(i)
    plot = plt.imshow(image.img_to_array(batch[0]))
    i += 1
    if i >4:        # show 4 images
        break

plt.show()

