import numpy as np
import pandas as pd


data = pd.read_csv("dataset/data.csv")
data = np.array(data)

m, n = data.shape
np.random.shuffle(data)

split = int(0.8 * m)

test_data = data[split:].T
test_Y = test_data[0]
test_X = test_data[1:] / 255

train_data = data[:split].T
train_Y = train_data[0]
train_X = train_data[1:] / 255
M = train_Y.size
