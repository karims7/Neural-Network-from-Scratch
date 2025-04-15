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
train_pixels = train_data[1:n]

valid_data = data[0:1000].T
valid_label = valid_data[0]
valid_pixels = valid_data[1:n]

def init_params():
    W1 = np.random.rand(10,784)
    b1 = np.random.rand(10,1)
    W2 = np.random.rand(10,10)
    b2 = np.random.rand(10,1)
    return W1, b1, W2, b2

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)

def ReLU(Z): 
    return np.maximum(0, Z)

def one_hot(label):
    one_hot_label = np.zeros((label.size, label.max() + 1))
    one_hot_label[np.arange(label.size), label] = 1
    one_hot_label = one_hot_label.T
    return one_hot_label

def derivative_ReLU(Z):
    return Z > 0

def back_prop(Z1, A1, Z2, A2, W2, pixels, label):
    m = label.size
    one_hot_label = one_hot(label)

    dW2 = 1 / m * dZ2.dot(A1.T)
    dW1 = 1 / m * dZ1.dot(pixels.T)

    dZ1 = W2.T.dot(dZ2) * derivative_ReLU(Z1)
    dZ2 = A2 - one_hot_label

    db1 = 1 / m * np.sum(dZ1, 2)
    db2 = 1 / m * np.sum(dZ2, 2)
    
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, DW2, db2, alpha):
    W1 = W1 - alpha * dW1
    W2 = W2 - alpha * DW2

    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2

    return W1, b1, W2, b2
