import csv
from objects import Image, Neuron_Layer, Layer, Neural_Network
import numpy as np

#metaparams
  # Input layer, two hidden layers, output layer
from metaparams import LEARNING_RATE, PIXELS, NUM_LAYERS, TRAINING, LAYER_SIZES, BATCH_SIZE, EPOCHS
#modify generally the

test_data, train_data = np.array([]), np.array([])
# Load the MNIST dataset from CSV files
with open('MNIST_CSV/mnist_test.csv', newline='') as test_file:
    test_reader = csv.reader(test_file)
    test_data = list(test_reader)

with open('MNIST_CSV/mnist_train.csv', newline='') as train_file:
    train_reader = csv.reader(train_file)
    train_data = list(train_reader)



for i in test_data:
    i=Image(i)
for i in train_data:
    i=Image(i)

# Create nodes for the input layer
input_nodes = np.array([Neuron_Layer(i) for i in range(784)])
def gen_error(real, predicted):
    return [predicted[i] - (1 if i == real else 0) for i in range(len(predicted))]
num_batches = len(train_data) // BATCH_SIZE

# Create layers with specified sizes
network = Neural_Network(input_nodes)
epoch_accuracy, epoch_loss, batch_accuracy, batch_loss = [], [], [], []
for i in range(EPOCHS):
    print(f"Epoch {i+1}/{EPOCHS} - Training...")
    epoch_right, epoch_closs = 0.0, 0.0
    for j in range(0, len(train_data)):
        if j % BATCH_SIZE == 0:
            batch_accuracy[-1]/= BATCH_SIZE
            batch_loss[-1]/= BATCH_SIZE
            batch_accuracy.append(0.0), batch_loss.append(0.0)
            if j > 0 and j%1000 ==0:
                network.update_params()
                print(f"Batch {j//BATCH_SIZE} - Accuracy: {batch_accuracy[-1]}, Loss: {batch_loss[-1]}")

            
        network.Neuron_Layers[0] = train_data[j].pixels
        network.forward()
        fin_layer=network.Neuron_Layers[-1]
        predicted = np.argmax(fin_layer.softmax())
        loss = fin_layer.cost(train_data[j].label)
        # update stats
        batch_accuracy[-1] += 1 if predicted == train_data[j].label else 0
        batch_loss[-1] += loss
        error_signal = fin_layer.gen_error(train_data[j].label)
        network.backprop(error_signal)
    epoch_accuracy.append(np.mean(batch_accuracy[-num_batches:]))
    epoch_loss.append(np.mean(batch_loss[-num_batches:]))
    print(f"Epoch {i+1} Complete - Accuracy: {epoch_accuracy[-1]}, Loss: {epoch_loss[-1]}")

