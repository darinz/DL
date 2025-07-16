# Matrix Decompositions

> **Matrix decompositions break complex matrices into simpler, structured pieces for analysis and computation.**

---

## What is a Matrix Decomposition?

A **matrix decomposition** expresses a matrix as a product of simpler matrices. This is useful for solving equations, understanding structure, and efficient computation.

---

## LU Decomposition

Decomposes a square matrix $`A`$ into:

```math
A = LU
```

- $`L`$ is lower triangular (all entries above the diagonal are zero)
- $`U`$ is upper triangular (all entries below the diagonal are zero)

---

## QR Decomposition

Decomposes a matrix $`A`$ into:

```math
A = QR
```

- $`Q`$ is orthogonal ($`Q^T Q = I`$)
- $`R`$ is upper triangular

---

## Singular Value Decomposition (SVD)

Decomposes any $`m \times n`$ matrix $`A`$ into:

```math
A = U\Sigma V^T
```

- $`U`$ and $`V`$ are orthogonal matrices
- $`\Sigma`$ is a diagonal matrix of singular values

SVD is fundamental for data compression, noise reduction, and more.

---

## Python Implementation: SVD

```python
import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
U, S, Vt = np.linalg.svd(A)

print("Matrix A:")
print(A)
print("\nU matrix:")
print(U)
print("\nSingular values:")
print(S)
print("\nV^T matrix:")
print(Vt)

# Reconstruct A
Sigma = np.zeros_like(A, dtype=float)
Sigma[:len(S), :len(S)] = np.diag(S)
A_reconstructed = U @ Sigma @ Vt
print("\nReconstructed A:")
print(A_reconstructed)
```

---

## Why Matrix Decompositions Matter in Deep Learning

- **Solving systems**: Used for efficient solutions to linear systems.
- **Understanding structure**: Reveal important properties (e.g., rank, null space).
- **Dimensionality reduction**: SVD is the basis for PCA and other techniques.

Matrix decompositions are essential tools for both theory and practice in deep learning! 