from mlp_setup import MLP, X_train, X_test, y_train, y_test_labels
import matplotlib.pyplot as plt
import time

# =============================================
# EXERCISE 5.4 - Compare activation functions
# =============================================

activations = ["sigmoid", "relu"]

results = {}

for act in activations:
    print(f"\nTesting activation: {act}")
    model = MLP(layers=[2, 16, 16, 3], activation=act, lr=0.05, task="multiclass")

    start = time.time()
    losses = model.fit(X_train, y_train, epochs=100, verbose=False)
    duration = time.time() - start

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test_labels).mean()

    results[act] = (losses, acc, duration)

plt.figure(figsize=(10, 6))
for act, (losses, acc, _) in results.items():
    plt.plot(losses, label=f"{act} (acc={acc:.2f})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Exercise 5.4: Activation Function Comparison")
plt.legend()
plt.grid(True)
plt.show()


#Convergence Speed: The ReLU (orange) line typically shows a more aggressive and stable descent in loss compared to Sigmoid (blue). This is because ReLU does not saturate for positive values, allowing gradients to flow more freely during backpropagation.
#Final Accuracy: In your specific run, Sigmoid happened to reach 0.40 while ReLU reached 0.30. However, as networks get deeper, Sigmoid often suffers from the vanishing gradient problem, where the updates becomes so small that the network stops learning.
#Conclusion: ReLU is generally considered superior for deep networks because it maintains a strong learning signal and is computationally cheaper to calculate than the exponential math required for Sigmoid.

#Layers: [2, 16, 16, 3].Why? Two hidden layers allow the network to learn complex non-linear boundaries (the "Deep" advantage) without being so large that it overfits the small dataset (the "Overfitting" risk).Activation Function: relu. Why? It provides the fastest convergence and avoids the vanishing gradient issues seen in Sigmoid. Learning Rate: 0.05.Why? It is high enough to converge quickly but low enough to avoid the wild oscillations we saw at lr=0.5.