import numpy as np
import random
from metaparams import LEARNING_RATE, PIXELS, NUM_LAYERS, TRAINING, LAYER_SIZES, BATCH_SIZE

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

class Image:
    def __init__(self, pixels):
        #easier for processing
        self.label = int(pixels[0])
        self.pixels = np.array(list(int(p) for p in pixels[1:]))
class Layer:
    def __init__ (self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights= np.random.uniform(-1, 1, (output_size, input_size))
        self.biases = np.random.uniform(-1, 1, output_size)
        self.mods_weights = np.zeros((output_size, input_size))
        self.mods_biases = np.zeros(output_size) #stored changes

        # Empty array, to be filled during backprop
    #0 maps 0 to 0, 1 maps 1 to 0, 783 maps 783 to 0, 784 maps 0 to 1, etc.
class Neuron_Layer:
    def __init__(self, size):
        self.size = size

        self.values = np.zeros(self.size)
        self.derivative = np.zeros(self.size)
        self.activation = sigmoid(self.values)

    def softmax(self):
        self.probs = np.exp(self.values - np.max(self.values))
        self.probs /= np.sum(self.probs)
        return np.argmax(self.probs)
    def cost(self, real):
        return -np.log(self.probs[real])
    def gen_error(self, real):
        self.res = [self.probs[i] - (1 if i == real else 0) for i in range(self.size)]
        return self.res


class Neural_Network:
    def __init__(self):
        self.Neuron_Layers = [Neuron_Layer(PIXELS)]
        self.Param_Layers=[Layer(PIXELS, LAYER_SIZES[0])]
        for i in range(1, NUM_LAYERS):
            self.Param_Layers.append(Layer(LAYER_SIZES[i-1], LAYER_SIZES[i]))
        for size in LAYER_SIZES:
            self.Neuron_Layers.append(Neuron_Layer(size))
    def forward(self, input_pixels):

      self.Neuron_Layers[0].activation = input_pixels/255.0 # Normalize pixels for better performance

      for i in range(NUM_LAYERS):
        prev_layer_activations = self.Neuron_Layers[i].activation
        current_params = self.Param_Layers[i]
        next_neuron_layer = self.Neuron_Layers[i+1]

        # Correct calculation: z = W * a + b
        z = np.dot(current_params.weights, prev_layer_activations) + current_params.biases

        # Store the raw values and the activated values in the next layer
        next_neuron_layer.values = z
        next_neuron_layer.activation = sigmoid(z)

    def backprop(self, layer_num=NUM_LAYERS, error_signal=None):
        if layer_num == 0:
            return
        # Compute gradients for the current layer
        if(layer_num==NUM_LAYERS):
            self.Neuron_Layers[layer_num].derivative = error_signal
        else:
            #oops :) forgot to imprlmeent signmoid between layer
            self.Neuron_Layers[layer_num].derivative = dsigmoid(self.Neuron_Layers[layer_num].values) * (self.Param_Layers[layer_num].weights.T @ self.Neuron_Layers[layer_num+1].derivative)
        self.weight_derivative= np.outer(self.Neuron_Layers[layer_num].derivative, self.Neuron_Layers[layer_num-1].activation.T)
        self.bias_derivative = self.Neuron_Layers[layer_num].derivative
        # Update gradients
        self.Param_Layers[layer_num-1].mods_weights += self.weight_derivative
        self.Param_Layers[layer_num-1].mods_biases += self.bias_derivative
        self.backprop(layer_num-1)

    def update_params(self):
        for layer in self.Param_Layers:
            layer.weights -= LEARNING_RATE * layer.mods_weights
            layer.biases -= LEARNING_RATE * layer.mods_biases
            # Reset the modifications after updating
            layer.mods_weights = np.zeros_like(layer.mods_weights)
            layer.mods_biases = np.zeros_like(layer.mods_biases)




    
        

        

