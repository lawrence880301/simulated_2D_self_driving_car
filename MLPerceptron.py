import numpy as np
import math
import random

#
# Activate functions
#

def __sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
def __dsigmoid(x):
    s = __sigmoid(x)
    return s * (1.0 - s)

Sigmoid = (__sigmoid, __dsigmoid)

#
# Layers
#

class BaseLayer:
    def __init__(self, inputs, outputs):
        self.inputs, self.outputs = inputs, outputs
    def forward(self, x): raise NotImplementedError
    def backword(self, dA, learning_rate): raise NotImplementedError

class FullyConnectedLayer(BaseLayer):
    def __init__(self, inputs, outputs, act):
        super().__init__(inputs, outputs)
        self.w = np.random.rand(outputs, inputs) - 0.5
        self.b = np.random.rand(outputs, 1) - 0.5
        self.act, self.dact = act
    def forward(self, x):
        self.x = x
        self.z = self.w.dot(x) + self.b
        self.a = self.act(self.z)
        return self.a
    def backword(self, dA, learning_rate):
        dB = dA * self.dact(self.z)
        dW = dB.dot(self.x.T)
        dX = dB.T.dot(self.w).T
        self.w -= dW * learning_rate
        self.b -= dB * learning_rate
        return dX
        

class SigmoidLayer(FullyConnectedLayer):
    def __init__(self, inputs, outputs):
        super().__init__(inputs, outputs, act=Sigmoid)

class InputLayer(BaseLayer):
    def __init__(self, inputs):
        super().__init__(inputs, inputs)
    def forward(self, x):
        assert self.inputs == x.shape[0]
        return x.reshape((x.shape[0], 1))
    def backword(self, dA, learning_rate):
        return dA




#
# NeuralNetwork
#

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def update(self, dA, learning_rate):
        for layer in self.layers[::-1]:
            dA = layer.backword(dA, learning_rate)
    def fit(self, data, learning_rate=0.01, threshold=1e-5, epochs=100000):
        errors = []
        for iter in range(epochs):
            x, y = random.choice(data)
            h = self.predict(x)
            d = h - y.reshape((y.shape[0], 1))
            errors.append((0.5 * d ** 2).sum())
            if len(errors) > len(data): del errors[0]
            error = np.array(errors).mean()
            if math.isfinite(error):
                self.update(d, learning_rate)
                if iter % 1000 == 0:
                    print('iteration', iter, 'error', error)
                if error <= threshold:
                    break
