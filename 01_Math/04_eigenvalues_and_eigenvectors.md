# Eigenvalues and Eigenvectors

> **Eigenvalues and eigenvectors reveal the fundamental directions and scaling factors of linear transformations.**

---

## What are Eigenvalues and Eigenvectors?

Given a square matrix $`A`$, an **eigenvector** $`\mathbf{v}`$ and **eigenvalue** $`\lambda`$ satisfy:

```math
A\mathbf{v} = \lambda\mathbf{v}
```

- $`\mathbf{v}`$ is a nonzero vector that only gets scaled (not rotated) by $`A`$.
- $`\lambda`$ is the scaling factor.

**Intuition:**
- An eigenvector is a direction that is unchanged by the transformation $`A`$ (except for stretching or shrinking).
- The eigenvalue tells you how much the eigenvector is stretched or shrunk.
- If $`\lambda > 1`$, the vector is stretched; if $`0 < \lambda < 1`$, it is shrunk; if $`\lambda < 0`$, it is flipped and scaled.

**Example:**
If $`A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}`$, then $`\mathbf{v}_1 = [1, 0]`$ is an eigenvector with $`\lambda_1 = 2`$, and $`\mathbf{v}_2 = [0, 1]`$ is an eigenvector with $`\lambda_2 = 3`$.

---

## Why are They Important?

- They describe the "axes" along which a transformation acts by simple scaling.
- Used in dimensionality reduction (PCA), stability analysis, and more.
- Reveal the "natural" coordinate system for a matrix transformation.
- In deep learning, they help us understand the structure of weight matrices, covariance matrices, and more.

---

## Eigendecomposition

If $`A`$ has $`n`$ linearly independent eigenvectors, it can be decomposed as:

```math
A = Q\Lambda Q^{-1}
```

- $`Q`$ is the matrix of eigenvectors (columns)
- $`\Lambda`$ is a diagonal matrix of eigenvalues

**Intuition:**
- This decomposition expresses $`A`$ as a change of basis (by $`Q`$), scaling (by $`\Lambda`$), and change of basis back (by $`Q^{-1}`$).
- Not all matrices are diagonalizable, but symmetric matrices always are.

---

## Geometric Interpretation

- **Eigenvectors** point in directions that are unchanged by the transformation (except for scaling).
- **Eigenvalues** tell how much those directions are stretched or shrunk.
- In 2D, you can visualize a matrix as transforming the plane: eigenvectors are the special directions that only get longer or shorter, not rotated.

**Visualization:**
- Imagine a rubber sheet being stretched: the eigenvectors are the lines that stay straight, and the eigenvalues tell you how much they stretch.

---

## Python Implementation

Let's compute eigenvalues and eigenvectors for a symmetric matrix, and visualize the effect:

```python
import numpy as np
import matplotlib.pyplot as plt

A = np.array([[4, -2], [-2, 4]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)

# Visualize the effect of A on the eigenvectors
origin = np.zeros(2)
plt.figure(figsize=(6, 6))
for i in range(2):
    v = eigenvectors[:, i]
    plt.quiver(*origin, *v, angles='xy', scale_units='xy', scale=1, color='b', label=f'eigenvector {i+1}')
    plt.quiver(*origin, *(A @ v), angles='xy', scale_units='xy', scale=1, color='r', linestyle='dashed', label=f'A @ eigenvector {i+1}')
plt.xlim(-3, 5)
plt.ylim(-3, 5)
plt.grid(True)
plt.legend()
plt.title('Eigenvectors and their transformation by A')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Verify eigendecomposition
Q = eigenvectors
Lambda = np.diag(eigenvalues)
A_reconstructed = Q @ Lambda @ np.linalg.inv(Q)
print("\nReconstructed A:")
print(A_reconstructed)

# Verify eigenvalue equation for first eigenvector
v1 = eigenvectors[:, 0]
lambda1 = eigenvalues[0]
print("\nA @ v1:", A @ v1)
print("lambda1 * v1:", lambda1 * v1)
```

---

## Application: Principal Component Analysis (PCA)

PCA uses eigenvectors and eigenvalues of the data covariance matrix to find the directions of maximum variance (principal components).

**Intuition:**
- The first principal component is the direction of greatest variance in the data.
- The eigenvalues tell you how much variance is explained by each component.

```python
from sklearn.decomposition import PCA
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=100, n_features=3, centers=3, random_state=42)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Original data shape:", X.shape)
print("PCA data shape:", X_pca.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
```

---

## Why Eigenvalues and Eigenvectors Matter in Deep Learning

- **PCA**: Used for dimensionality reduction and data visualization.
- **Stability**: Analyzing the stability of optimization algorithms.
- **Understanding transformations**: Reveal the "natural" axes of a transformation.
- **Spectral methods**: Used in graph neural networks, clustering, and more.

**Summary:**
Mastering eigenvalues and eigenvectors gives you powerful tools for understanding data and models! They are essential for many advanced techniques in deep learning. 