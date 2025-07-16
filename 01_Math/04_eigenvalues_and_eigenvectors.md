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

---

## Why are They Important?

- They describe the "axes" along which a transformation acts by simple scaling.
- Used in dimensionality reduction (PCA), stability analysis, and more.

---

## Eigendecomposition

If $`A`$ has $`n`$ linearly independent eigenvectors, it can be decomposed as:

```math
A = Q\Lambda Q^{-1}
```

- $`Q`$ is the matrix of eigenvectors (columns)
- $`\Lambda`$ is a diagonal matrix of eigenvalues

---

## Geometric Interpretation

- **Eigenvectors** point in directions that are unchanged by the transformation (except for scaling).
- **Eigenvalues** tell how much those directions are stretched or shrunk.

---

## Python Implementation

Let's compute eigenvalues and eigenvectors for a symmetric matrix:

```python
import numpy as np

A = np.array([[4, -2], [-2, 4]])
eigenvalues, eigenvectors = np.linalg.eig(A)

print("Matrix A:")
print(A)
print("\nEigenvalues:")
print(eigenvalues)
print("\nEigenvectors (columns):")
print(eigenvectors)

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

Mastering eigenvalues and eigenvectors gives you powerful tools for understanding data and models! 