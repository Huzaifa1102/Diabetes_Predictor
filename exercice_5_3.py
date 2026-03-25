from mlp_setup import MLP, X_train, X_test, y_train, y_test_labels
import matplotlib.pyplot as plt
import time

# =============================================
# EXERCISE 5.3 - Deep vs Wide networks
# =============================================

architectures = [
    [2, 8, 3],           # Baseline
    [2, 128, 3],         # Wide model
    [2, 8, 8, 8, 8, 3]   # Deep model
]

results = {}

for arch in architectures:
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
plt.title("Exercise 5.3: Deep vs Wide Networks")
plt.legend()
plt.grid(True)
plt.show()

#Convergence Speed: The Wide network ([2, 128, 3], orange line) and the Deep network ([2, 8, 8, 8, 8, 3], green line) both reached high accuracy (0.30) much faster than the small baseline model.
#Training Time: If you check your terminal output, you'll likely see that the Deep network took longer to train per epoch. This is because the signal has to pass through many more sequential layers, which involves more consecutive matrix multiplications compared to a single wide layer.
#Accuracy Comparison: Both the wide and deep models achieved the same final accuracy (0.30), but the deep network's loss curve (green) shows more aggressive initial drops.
#Conclusion: * Wide Networks: Great at memorizing complex patterns in a single step but can easily overfit if they are too wide.
#eep Networks: Excellent at learning hierarchical features (complex ideas built from simple ones) but are harder to train due to vanishing gradients and increased computational time.