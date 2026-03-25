from mlp_setup import MLP, X_train, X_test, y_train, y_test_labels
import matplotlib.pyplot as plt
import time

# =============================================
# EXERCISE 5.2 - Explore different learning rates
# =============================================

learning_rates = [0.001, 0.01, 0.1, 0.5]

results = {}

for lr in learning_rates:
    print(f"\nTesting learning rate: {lr}")
    model = MLP(layers=[2, 16, 16, 3], activation="relu", lr=lr, task="multiclass")

    start = time.time()
    losses = model.fit(X_train, y_train, epochs=100, verbose=False)
    duration = time.time() - start

    y_pred = model.predict(X_test)
    acc = (y_pred == y_test_labels).mean()

    results[lr] = (losses, acc, duration)

plt.figure(figsize=(10, 6))
for lr, (losses, acc, _) in results.items():
    plt.plot(losses, label=f"lr={lr} (acc={acc:.2f})")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Exercise 5.2: Learning Rate Comparison")
plt.legend()
plt.grid(True)
plt.show()

#Convergence Speed: The high learning rate ($lr=0.5$, red line) drops the loss the fastest in the first few epochs, but it is also the most jagged.Divergence and Oscillation: Look at the beginning of the green ($lr=0.1$) and red ($lr=0.5$) lines. They have much larger "spikes". This is oscillation, where the learning rate is so high the model overshoots the minimum and has to bounce back.Slow Convergence: The blue line ($lr=0.001$) is much smoother but would take many more epochs to reach the same level of precision as the others.Stability vs. Speed Conclusion: There is a fundamental compromise. A high $lr$ reaches the goal faster but risks never settling on the best solution (instability). A low $lr$ is very stable but computationally expensive because it requires significantly more time to converge.