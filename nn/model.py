import numpy as np
import os


def initParam():
    w1 = np.random.randn(64, 784) * np.sqrt(2 / 784)
    b1 = np.zeros((64, 1))
    w2 = np.random.randn(10, 64) * np.sqrt(2 / 64)
    b2 = np.zeros((10, 1))

    return w1, b1, w2, b2

def ReLU(z):
    return np.maximum(z, 0)

def softMax(z):
    expz = np.exp(z - np.max(z, axis=0, keepdims=True))
    
    return expz / np.sum(expz, axis=0, keepdims=True)

def forwProp(w1, b1, w2, b2, x):
    z1 = w1.dot(x) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softMax(z2)
    
    return z1, a1, z2, a2

def oneHot(y):
    encoded_y = np.zeros((10, y.size))
    encoded_y[y, np.arange(y.size)] = 1
    
    return encoded_y

def derReLU(z):
    return z > 0

def backProp(w1, z1, a1, w2, z2, a2, x, y, m):
    encoded_y = oneHot(y)
    
    dz2 = a2 - encoded_y 
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    
    dz1 = w2.T.dot(dz2) * derReLU(z1)
    dw1 = 1 / m * dz1.dot(x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)
    
    return dw1, db1, dw2, db2

def updParam(w1, dw1, b1, db1, w2, dw2, b2, db2, lr):
    w1 -= lr * dw1
    b1 -= lr * db1
    w2 -= lr * dw2
    b2 -= lr * db2
    
    return w1, b1, w2, b2

def saveParam(w1, b1, w2, b2):
    np.savez("dataset/parameters.npz", w1=w1, b1=b1, w2=w2, b2=b2)

def loadParam():
    if os.path.exists("dataset/parameters.npz"):
        data = np.load("dataset/parameters.npz")
        
        return data['w1'], data['b1'], data['w2'], data['b2']
    
    return None
