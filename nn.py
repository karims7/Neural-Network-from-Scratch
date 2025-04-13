import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv("train.csv")

print(data.head())

data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # avoid shuffling 

valid_data = data[0:1000].T
Y_dev = valid_data[0]
X_dev = valid_data[1:n]

train_data = data[1000:m].T
Y_train = train_data[0]
X_train = train_data[1:n]