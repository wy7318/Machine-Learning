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
'''
