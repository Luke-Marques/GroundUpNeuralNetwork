# GROUND UP NEURAL NETWORK
import numpy as np


class GroundUpNeuralNetwork:

    # constructor dunder method
    def __init__(self, x, y):
        self.input = x  # input layer of network
        self.weights1 = np.random.rand(self.input.shape[1], 5)  # rand num matrix of weights between input and layer 1
        self.weights2 = np.random.rand(5, 1)  # rand num matrix of weights between 1st and 2nd/output layer
        self.y = y  # desired output
        self.output = np.zeros(y.shape)  # output is zero filled matrix with shape of y

    # feedforward method
    def feedfoward(self):
        self.layer1 = np.sigmoid(np.dot(self.input, self.weights1))  # computes layer 1 activation (assuming b = 0)
        self.output = np.sigmoid(np.dot(self.layer1, self.weights2))  # computes layer 2 activation (assuming b = 0)

    # backpropagation method
    # def backprop(self):
