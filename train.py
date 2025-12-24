import numpy as np
from model import initParam, forwProp, backProp, updParam, loadParam, saveParam
from data import M


def pred(a):
    return np.argmax(a, axis=0)

def acc(preds, y):
    return np.sum(preds == y) / y.size

def gradDesc(x, y, steps, lr):
    param = loadParam()
    if param:
        w1, b1, w2, b2 = param
    else:
        w1, b1, w2, b2 = initParam()
    
    for i in range(steps):
        z1, a1, z2, a2 = forwProp(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = backProp(w1, z1, a1, w2, z2, a2, x, y, M)
        w1, b1, w2, b2 = updParam(w1, dw1, b1, db1, w2, dw2, b2, db2, lr)
        
        if i % 50 == 0:
            saveParam(w1, b1, w2, b2)
            
            preds = pred(a2)
            accuracy = acc(preds, y)
            
            print(f'Step {i}, Accuracy: {(accuracy * 100):.2f}%')
    
    return w1, b1, w2, b2
