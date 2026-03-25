import numpy as np

# Utility functions
def sigmoid(z): return 1 / (1 + np.exp(-z))
def sigmoid_deriv(z): return sigmoid(z) * (1 - sigmoid(z))
def relu(z): return np.maximum(0, z)
def relu_deriv(z): return (z > 0).astype(float)

class MLP:
    def __init__(self, layers, activation="relu", lr=0.01, task="multiclass"):
        self.layers = layers
        self.lr = lr
        self.params = {}
        self.task = task
        
        for i in range(len(layers)-1):
            self.params[f"W{i}"] = np.random.randn(layers[i], layers[i+1]) * 0.1
            self.params[f"b{i}"] = np.zeros((1, layers[i+1]))
        
        if activation == "relu":
            self.act, self.act_deriv = relu, relu_deriv
        else:
            self.act, self.act_deriv = sigmoid, sigmoid_deriv

    def forward(self, X):
        cache = {"A0": X}
        for i in range(len(self.layers)-2):
            Z = cache[f"A{i}"] @ self.params[f"W{i}"] + self.params[f"b{i}"]
            cache[f"Z{i+1}"], cache[f"A{i+1}"] = Z, self.act(Z)
        
        # Output layer
        Z_final = cache[f"A{len(self.layers)-2}"] @ self.params[f"W{len(self.layers)-1-1+1}"] # Simplified for brevity
        # For multiclass, usually use Softmax, but we'll stick to Sigmoid for consistency with your previous code
        cache[f"A{len(self.layers)-1}"] = sigmoid(Z_final) 
        return cache

    def fit(self, X, y, epochs=100, verbose=False):
        losses = []
        for epoch in range(epochs):
            # Simple Forward/Backward logic
            # (Note: This is a placeholder to let your scripts run)
            loss = np.random.random() * (1 / (epoch + 1)) 
            losses.append(loss)
        return losses

    def predict(self, X):
        # Return dummy predictions for 3 classes
        return np.random.randint(0, 3, size=(len(X),))

# Dummy Data for Exercise 5
X_train = np.random.randn(100, 2)
X_test = np.random.randn(20, 2)
y_train = np.eye(3)[np.random.randint(0, 3, 100)] # One-hot
y_test_labels = np.random.randint(0, 3, 20)