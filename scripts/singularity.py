import numpy as np

# Load the .npz file
data = np.load("data/processed/train_data_log1p.npz")

# To see what arrays are inside
print(data.files)

# Example: Access an array named 'X'
X = data['X']

import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = np.load("data/processed/train_data_log1p.npz")
X = data['X']

# Compute singular values
_, S, _ = np.linalg.svd(X, full_matrices=False)

# Plot the singular values
plt.plot(S)
plt.xlabel("Component Index")
plt.ylabel("Singular Value")
plt.title("Singular Values of X")
plt.grid(True)
plt.tight_layout()
plt.show()

