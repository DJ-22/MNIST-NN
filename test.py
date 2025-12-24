import numpy as np
import matplotlib.pyplot as plt
from model import forwProp
from train import pred, acc


def testNN(w1, b1, w2, b2, x, y):
    _, _, _, a2 = forwProp(w1, b1, w2, b2, x)
    preds = pred(a2)
    accuracy = acc(preds, y)
    
    print(f'Test Accuracy: {(accuracy * 100):.2f}%\n')

def plotAcc(w1, b1, w2, b2, x, y):
    _, _, _, a2 = forwProp(w1, b1, w2, b2, x)
    preds = pred(a2)
    
    acc_val = np.zeros(10)
    tot = np.zeros(10)
    
    for i in range(y.size):
        label = y[i]
        tot[label] += 1
        
        if preds[i] == label:
            acc_val[label] += 1
    
    acc_dig = acc_val / tot * 100
    
    print("Accuracy per Digit:")
    for i in range(10):
        print(f"Digit {i}: {acc_dig[i]:.2f}%")
    
    plt.figure()
    plt.bar(range(10), acc_dig)
    plt.xticks(range(10))
    plt.ylim(0, 100)
    plt.xlabel("Digit")
    plt.ylabel("Accuracy")
    plt.title("Accuracy per Digit")
    plt.show()
