import math

# ----- Activation functions -----
def sigmoid(x):
    return 1 / (1+math.exp(-x))#TODO add sigmoid function here

def sigmoid_derivative(y):
    return y + (1-y)#TODO add sigmoid derivative function here

# ----- Initial weights -----
w1 = [-1.0, 0.0]       # TODO add hidden neuron 1
w2 = [2.0, 1.0]        # TODO add hidden neuron 2
w3 = [-2.0, 1.0]       # TODO add hidden neuron 3

wS1 = [2.0, -0.5, 1.0]    # TODO add output neuron o1
wS2 = [-2.0, 1.0, 0.5]    # TODO add output neuron o2

learning_rate = 1# TODO add learning_rate value

# ----- Training dataset (you can add more examples) -----
# Each element: ([x1, x2], [target1, target2])
dataset = [
    ([0.5, 1.0], [1.0, 0.0]) #TODO add input and the expected output
]

epochs = 10

# ----------------------------------------------------------
# Training loop
# ----------------------------------------------------------
for epoch in range(epochs):
    mse_epoch = 0

    for inputs, target in dataset:
        x1, x2 = inputs

        # ----- Forward pass -----
        h1_net = w1[0]*x1 + w1[1]*x2
        h2_net = w2[0]*x1 + w2[1]*x2
        h3_net = w3[0]*x1 + w3[1]*x2

        h1_out = sigmoid(h1_net)
        h2_out = sigmoid(h2_net)
        h3_out = sigmoid(h3_net)

        hidden_outputs = [h1_out, h2_out, h3_out]

        o1_net = wS1[0]*h1_out + wS1[1]*h2_out + wS1[2]*h3_out
        o2_net = wS2[0]*h1_out + wS2[1]*h2_out + wS2[2]*h3_out

        o1_out = sigmoid(o1_net)
        o2_out = sigmoid(o2_net)

        # ----- Compute error -----
        mse_epoch += 0.5 * ((target[0] - o1_out)**2 + (target[1] - o2_out)**2)

        # ----- Backpropagation -----
        # Output deltas
        delta_o1 = (target[0] - o1_out) * sigmoid_derivative(o1_out)
        delta_o2 = (target[1] - o2_out) * sigmoid_derivative(o2_out)

        # Hidden deltas
        delta_h1 = sigmoid_derivative(h1_out) * (delta_o1*wS1[0] + delta_o2*wS2[0])
        delta_h2 = sigmoid_derivative(h2_out) * (delta_o1*wS1[1] + delta_o2*wS2[1])
        delta_h3 = sigmoid_derivative(h3_out) * (delta_o1*wS1[2] + delta_o2*wS2[2])

        # ----- Update weights: output layer -----
        for i, h in enumerate(hidden_outputs):
            wS1[i] += learning_rate * delta_o1 * h
            wS2[i] += learning_rate * delta_o2 * h

        # ----- Update weights: hidden layer -----
        w1[0] += learning_rate * delta_h1 * x1
        w1[1] += learning_rate * delta_h1 * x2

        w2[0] += learning_rate * delta_h2 * x1
        w2[1] += learning_rate * delta_h2 * x2

        w3[0] += learning_rate * delta_h3 * x1
        w3[1] += learning_rate * delta_h3 * x2

    # ----- Average error per epoch -----
    mse_epoch /= len(dataset)
    print(f"Epoch {epoch+1}/{epochs} - MSE = {mse_epoch:.5f}")

# Final weights after training
print("\nFinal weights:")
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("wS1 =", wS1)
print("wS2 =", wS2)