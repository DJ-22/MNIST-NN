from data import train_X, train_Y
from train import gradDesc


if __name__ == '__main__':
    STEPS = 5000
    LR = 0.01
    
    W1, B1, W2, B2 = gradDesc(train_X, train_Y, STEPS, LR)
    print("Training Complete!")
