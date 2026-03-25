import numpy as np

# Simple Data : AND Boolean function
X = np.array([[0,0],[0,1],[1,0],[1,1]]) #TODO you can try other boolean functions
y = np.array([[1],[1],[1],[0]])

# Weight intitialization
W = np.random.randn(2,1)
b = np.zeros((1,))

# Activation function (sigmoid)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Simple training
lr = 0.1#TODO try different values here and see the difference
nb_epoch = 1000#TODO try different values here and see the difference

#Too few epochs (e.g., nb_epoch = 10): The model hasn't had enough time to look at the data and adjust its weights. The output will likely be a mess of incorrect fractions.
#A tiny learning rate (e.g., lr = 0.0001): The network takes microscopic steps toward the solution. Even with 1000 epochs, it won't be enough time to reach the correct weights.
#A massive learning rate (e.g., lr = 10): The network takes steps that are too large, overshooting the optimal solution and causing the predictions to bounce around wildly (or even break the math).

#The XOR Limitation: A simple, single-layer network like the one in exercice_4_1.py is entirely linear. It tries to draw a single, straight line on a graph to separate the 0s from the 1s.
#The Geometry: If you plot XOR on a 2D grid, the 1s are in opposite corners, and the 0s are in the other opposite corners. It is geometrically impossible to draw just one straight line to separate them.

for epoch in range(nb_epoch):
    z = X @ W + b
    y_pred = sigmoid(z)
    
    # error
    error = y_pred - y
    loss = np.mean(error**2)
    
    # back-propagation
    dW = X.T @ error / len(X)
    db = np.mean(error)
    
    W -= lr * dW
    b -= lr * db

print("Final Prediction :", np.round(sigmoid(X @ W + b)))