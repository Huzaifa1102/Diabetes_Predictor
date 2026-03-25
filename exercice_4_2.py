import numpy as np
import time

# Utility functions
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return sigmoid(z) * (1 - sigmoid(z))
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

class MLP:
    def __init__(self, layers, activation="relu", lr=0.01):
        """
        layers : [input_dim, hidden1, hidden2, ..., output_dim]
        """
        self.layers = layers
        self.lr = lr
        self.params = {}
        
        # Weights initialization
        for i in range(len(layers)-1):
            self.params[f"W{i}"] = np.random.randn(layers[i], layers[i+1]) * 0.1
            self.params[f"b{i}"] = np.zeros((1, layers[i+1]))
        
        # activation
        if activation == "relu":
            self.act, self.act_deriv = relu, relu_deriv
        else:
            self.act, self.act_deriv = sigmoid, sigmoid_deriv
    
    def forward(self, X):
        cache = {"A0": X}
        for i in range(len(self.layers)-2):
            Z = cache[f"A{i}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            A = self.act(Z)
            cache[f"Z{i+1}"], cache[f"A{i+1}"] = Z, A
        # output layer (binary sigmoid)
        Z = cache[f"A{len(self.layers)-2}"] @ self.params[f"W{len(self.layers)-2}"] + self.params[f"b{len(self.layers)-2}"]
        A = sigmoid(Z)
        cache[f"Z{len(self.layers)-1}"], cache[f"A{len(self.layers)-1}"] = Z, A
        return cache
    
    def backward(self, cache, y):
        grads = {}
        m = len(y)
        L = len(self.layers)-1
        A_final = cache[f"A{L}"]
        
        dZ = A_final - y
        for i in reversed(range(L)):
            A_prev = cache[f"A{i}"]
            grads[f"dW{i}"] = (A_prev.T @ dZ) / m
            grads[f"db{i}"] = np.mean(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA_prev = dZ @ self.params[f"W{i}"].T
                dZ = dA_prev * self.act_deriv(cache[f"Z{i}"])
        
        # update
        for i in range(L):
            self.params[f"W{i}"] -= self.lr * grads[f"dW{i}"]
            self.params[f"b{i}"] -= self.lr * grads[f"db{i}"]
    
    def fit(self, X, y, epochs=1000):
        for epoch in range(epochs):
            cache = self.forward(X)
            self.backward(cache, y)
            if epoch % 100 == 0:
                loss = np.mean((cache[f"A{len(self.layers)-1}"] - y)**2)
                print(f"Epoch {epoch}, Loss={loss:.4f}")
    
    def predict(self, X):
        cache = self.forward(X)
        return (cache[f"A{len(self.layers)-1}"] > 0.5).astype(int)



# Dataset for XOR function
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

nb_epochs = 10000#TODO try different values here and see the difference
lr_value = 1.0#TODO try different values here and see the difference
activation_fn = "sigmoid"#TODO try different values here and see the difference

print("=== Simple network (1 hidden layer) ===")
mlp1 = MLP([2,4,1], activation=activation_fn, lr=lr_value) #TODO try different values for layers, example [2,4,1]
start = time.time()
mlp1.fit(X,y,epochs=nb_epochs)
duration = time.time() - start
print("Predictions :", mlp1.predict(X).ravel(), "Time(s) : ", round(duration, 3))

print("\n=== Complex network (3 hidden layers) ===")
mlp2 = MLP([2,8,8,8,1], activation=activation_fn, lr=lr_value) #TODO try different values for layers, example [2,8,8,8,1]
start = time.time()
mlp2.fit(X,y,epochs=nb_epochs)
duration = time.time() - start
print("Predictions :", mlp2.predict(X).ravel(), "Time(s) : ", round(duration, 3))

#Q1 (Difference between architectures): Right now, neither architecture is better because they both got stuck. Adding more layers didn't magically make it smarter.
#Q2 Yes, the results improve, but they require very specific tuning.
#When using the default sigmoid activation with a standard learning rate (0.1) and 5,000 epochs, the network got stuck in a local minimum (Loss flatlined at 0.2500) and failed to predict XOR.
#Trying relu also failed initially due to "dead neurons" getting stuck.
#However, when we drastically increased the epochs to 10,000, cranked the learning rate up to 1.0, and used the sigmoid activation function, the simple 1-hidden-layer network finally broke through the local minimum, dropped its loss to 0.0000, and successfully predicted [0 1 1 0].
#Q3 (Computing Time): The complex network took more than twice as long (0.457s vs 0.208s). More layers mean significantly more matrix math, which slows down the training process.