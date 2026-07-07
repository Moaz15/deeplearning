# import numpy as np

# class Perceptron:
#     def __init__(self, learning_rate=0.01, n_epochs=1000):
#         self.lr = learning_rate
#         self.epochs = n_epochs
#         self.weights = None
#         self.bias = None
#         self.mistake_history = []

#     def step_activation(self, z):
#         # Fully vectorized activation: works on scalars, 1D, or 2D arrays
#         return np.where(z >= 0, 1, 0)
    
#     def fit(self, X, y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0.0

#         for epoch in range(self.epochs):
#             mistakes = 0
#             for i in range(n_samples):
#                 z = np.dot(X[i], self.weights) + self.bias
#                 y_pred = self.step_activation(z)
#                 error = y[i] - y_pred

#                 if error != 0:
#                     self.weights += self.lr * error * X[i]
#                     self.bias += self.lr * error
#                     mistakes += 1

#             self.mistake_history.append(mistakes)

#             if mistakes == 0:
#                 print(f"Converged at epoch {epoch+1}")
#                 break
        
#     def predict(self, X):
#         # Clean, vectorized prediction without list comprehensions
#         z = np.dot(X, self.weights) + self.bias
#         return self.step_activation(z)
    
#     def compute_perceptron_loss(self, X, y):
#         """
#         Computes Perceptron Loss. Maps binary y (0/1) to signed y (-1/1)
#         internally so the mathematical formulation stays correct.
#         """
#         z = np.dot(X, self.weights) + self.bias
#         # Convert 0/1 labels to -1/1 labels safely
#         y_signed = np.where(y == 0, -1, 1)
#         margins = -y_signed * z
#         return np.sum(np.maximum(0, margins))




            

                
                    







