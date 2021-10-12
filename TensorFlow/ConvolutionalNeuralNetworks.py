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
