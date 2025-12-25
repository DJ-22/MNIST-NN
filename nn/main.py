from data import train_X, train_Y, test_X, test_Y
from model import loadParam
from train import gradDesc
from test import testNN, plotAcc
from ui import DigitUI


if __name__ == '__main__':
    print("1: Train the Neural Network")
    print("2: Test the Neural Network")
    print("3: Draw a Digit")
    choice = input("\nEnter your choice (1/2/3): ")
    
    if choice == '1':
        STEPS = int(input("Enter number of training steps: "))
        LR = float(input("Enter learning rate: "))
    
        W1, B1, W2, B2 = gradDesc(train_X, train_Y, STEPS, LR)
        print("Training Completed.\n\n\n")
    elif choice == '2':
        param = loadParam()
        if param:
            W1, B1, W2, B2 = param
        else:
            print("No trained parameters found. Please train the model first.")
            exit()
        
        print("\nEvaluating on Test Set:\n")
        testNN(W1, B1, W2, B2, test_X, test_Y)
        plotAcc(W1, B1, W2, B2, test_X, test_Y)
    elif choice == '3':
        DigitUI()
    else:
        print("Invalid choice. Please enter 1 or 2.")
