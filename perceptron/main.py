import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 1])          # True labels
y_signed = 2 * y - 1                # Convert to {-1, +1} for loss function

# Initialize and train model
model = Perceptron(learning_rate=0.1, n_epochs=20)
model.fit(X, y)


# Predict and print
preds = model.predict(X)
print("Predictions:", preds)
print("Weights:", model.weights)
print("Bias:", model.bias)


# Compute final perceptron loss
loss = model.compute_perceptron_loss(X, y_signed)
print("Perceptron loss:", loss)



# Plot convergence (mistakes per epoch)
plt.plot(model.mistake_history, marker='o')
plt.title("Mistakes per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Number of Mistakes")
plt.grid(True)
plt.show()