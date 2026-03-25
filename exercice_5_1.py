from mlp_setup import MLP, X_train, X_test, y_train, y_test_labels
import matplotlib.pyplot as plt
import time

# =============================================
# EXERCISE 5.1 - Detect Overfitting
# Test multiple architectures
# =============================================

# =============================================
# EXERCISE 5.1 - Detect Overfitting
# =============================================

architectures = [
    [2, 4, 3],          # Simple: Risk of underfitting
    [2, 16, 3],         # Balanced: Likely optimal
    [2, 512, 256, 3],   # Complex: High risk of overfitting
]

results = {}

for arch in architectures:
    print(f"\nTesting architecture: {arch}")
    model = MLP(layers=arch, activation="relu", lr=0.05, task="multiclass")

    start = time.time()
    losses = model.fit(X_train, y_train, epochs=100, verbose=False)
    duration = time.time() - start

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test_labels).mean()

    results[str(arch)] = (losses, acc, duration)

plt.figure(figsize=(10, 6))
for arch, (losses, acc, _) in results.items():
    plt.plot(losses, label=f"{arch} (acc={acc:.2f})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Exercise 5.1: Overfitting Analysis")
plt.legend()
plt.grid(True)
plt.show()

#Training Loss Comparison: All architectures successfully drove the loss down close to zero over 100 epochs. You can see the green line (the most complex model) often sitting lower or dropping more sharply in the early stages because it has the most "brain power" to fit the data.
#Test Set Accuracy: * [2, 4, 3]: 0.40
#[2, 16, 3]: 0.30
#[2, 512, 256, 3]: 0.50
#Generalization explanation: Usually, a very large network (like your green line) would overfit, meaning it gets 100% accuracy on training but fails on testing. In this specific run, it actually performed the best, but that is because the "dataset" is very small and random. In a real-world scenario, the middle architecture ([2, 16, 3]) usually generalizes best because it doesn't have enough neurons to "memorize" the noise.
#Conclusion: There is a clear trade-off. Increasing network complexity can improve accuracy, but it increases the risk of overfitting and significantly slows down training speed (more neurons = more math).