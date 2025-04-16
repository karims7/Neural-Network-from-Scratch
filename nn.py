import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")

# print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # avoid shuffling 

train_data = data[1000:m].T
train_label = train_data[0]
train_pixels = train_data[1:n] / 255
_,m_train = train_pixels.shape

valid_data = data[0:1000].T
valid_label = valid_data[0]
valid_pixels = valid_data[1:n]




def init_params():
    W1 = np.random.rand(10,784) - 0.5
    b1 = np.random.rand(10,1) - 0.5
    W2 = np.random.rand(10,10) - 0.5
    b2 = np.random.rand(10,1) - 0.5
    return W1, b1, W2, b2

def forward_prop(W1, b1, W2, b2, pixels):

    Z1 = W1.dot(pixels) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

    

def ReLU(Z): 
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / sum(np.exp(Z))


# Turn labels into vectors:
def one_hot(label):
    one_hot_label = np.zeros((label.size, label.max() + 1))
    one_hot_label[np.arange(label.size), label] = 1
    one_hot_label = one_hot_label.T
    return one_hot_label

def ReLU_derivative(Z):
    return Z > 0

# Calculating how much weights and biases of output and hidden layer contributed to the error.
# Return all error gradients to fix weights and biases:
def back_prop(Z1, A1, Z2, A2, W1, W2, pixels, label):

    m = label.size
    one_hot_label = one_hot(label)

    dZ2 = A2 - one_hot_label
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_derivative(Z1)
    dW1 = 1 / m * dZ1.dot(pixels.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2

# Update weights and biases: 
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * dW2

    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2

def accuracy(predictions, label):
    print(predictions, label)
    return np.sum(predictions == label) / label.size

def prediction(A2):
    return np.argmax(A2, 0)

def gradient_descent(pixels, label, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, pixels)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W1, W2, pixels, label)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if (i % 10 == 0):
            print("Iteration: ", i)
            print("Accuracy:", accuracy(prediction(A2), label))

    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(train_pixels, train_label, 500, 0.10)

def make_predictions(pixels, W1, b1, W2, b2):
    _,_,_, A2 = forward_prop(W1, b1, W2, b2, pixels)
    return prediction(A2)

def test_prediction(index, W1, b1, W2, b2):
    current_image = train_pixels[:, index, None]
    prediction = make_predictions(train_pixels[:, index, None], W1, b1, W2, b2)

    print("Prediction: ", prediction)
    print("Label: ", train_label[index])

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

dev_predictions = make_predictions(valid_pixels, W1, b1, W2, b2)
accuracy(dev_predictions, valid_label)
