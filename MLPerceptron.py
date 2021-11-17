import numpy as np
from matplotlib import pyplot as plt  

class Layer:
    def __init__(self, n_input, n_neuron):
        
        self.weights = np.random.rand(n_input + 1, n_neuron)
        self.bias = np.negative(np.ones(n_neuron))
        
    def Net_input(self, x):
        self.net_input = np.dot(x, self.weights[:-1]) +np.dot(self.bias, self.weights[:-1])
        return self.net_input
    
    def activation(self, x):
        self.output = 1 / (1 + np.exp(-self.Net_input(x)))
        return self.output
    
    
    def activation_drv(self, s):
        return s*(1-s)

class MultilayerPerceptron:
    
    def __init__(self, n_layer, n_neuron, n_input, n_output):
        
        self.layers = []
        
        self.layers.append(Layer(n_input, n_neuron))
        [self.layers.append(Layer(n_neuron, n_neuron)) for _ in range(1, n_layer-1)]
        self.layers.append(Layer(n_neuron, n_output))
    
    def feed_forward(self, x):
        
        for layer in self.layers:
            x = layer.activation(x)
                        
        return x
    def back_propagation(self, x, y, l_rate):
        
        o_i = self.feed_forward(x)
        
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            
            if layer != self.layers[-1]:
                layer.delta = np.dot(layer.activation_drv(layer.output),
                                     np.dot(self.layers[i+1].weights, self.layers[i+1].delta))
               
            else:
                layer.error = y - o_i
                layer.delta = layer.error * layer.activation_drv(o_i)
                
        
        for i, layer in enumerate(self.layers):
            layer = self.layers[i]
            output_i = np.atleast_2d(x if i == 0 else self.layers[i - 1].output)
            layer.weights[:-1] = layer.delta * output_i.T * l_rate + layer.weights[:-1] 
            layer.weights[-1] = layer.delta * (-1) * l_rate + layer.weights[-1] 
    def train(self, x, y, l_rate, momentum, n_iter):
        
        costs =[]
        
        for i in range(n_iter):
            for xi, yi in zip(x, y):
                self.back_propagation(xi, yi, l_rate, momentum)
            cost = np.sum((y-self.feed_forward(x))**2) / 2.0
            costs.append(cost)
            
        return costs
    
    def predict(self, x):
        outputs = (self.feed_forward(x)).tolist()
 
        return outputs.index(max(outputs))    