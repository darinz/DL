# Vector Spaces and Subspaces

> **Vector spaces provide the setting for all of linear algebra, and subspaces are their building blocks.**

---

## What is a Vector Space?

A **vector space** is a set of vectors that is closed under vector addition and scalar multiplication. This means:
- If $`\mathbf{u}`$ and $`\mathbf{v}`$ are in the space, so is $`\mathbf{u} + \mathbf{v}`$.
- If $`\mathbf{v}`$ is in the space and $`c`$ is any scalar, then $`c\mathbf{v}`$ is also in the space.

Formally, a vector space $`V`$ over $`\mathbb{R}`$ (the real numbers) satisfies:
- **Additive identity:** There is a zero vector $`\mathbf{0}`$ such that $`\mathbf{v} + \mathbf{0} = \mathbf{v}`$ for all $`\mathbf{v}`$.
- **Additive inverse:** For every $`\mathbf{v}`$, there is $`-\mathbf{v}`$ such that $`\mathbf{v} + (-\mathbf{v}) = \mathbf{0}`$.
- **Distributive, associative, and commutative properties** (see any linear algebra text for full list).

---

## Subspaces

A **subspace** is a subset of a vector space that is itself a vector space (with the same operations).

**Example:** The set of all vectors in $`\mathbb{R}^3`$ where the third component is zero forms a subspace (the $`xy`$-plane).

---

## Linear Independence

A set of vectors $`\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n`$ is **linearly independent** if:

```math
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0}
```

implies $`c_1 = c_2 = \cdots = c_n = 0`$.

If not, the vectors are **linearly dependent** (at least one can be written as a combination of the others).

---

## Basis and Dimension

A **basis** of a vector space is a set of linearly independent vectors that span the space (every vector in the space can be written as a combination of basis vectors).

The **dimension** of a vector space is the number of vectors in any basis.

---

## Python Implementation: Checking Linear Independence

Let's check if three vectors in $`\mathbb{R}^3`$ are linearly independent:

```python
import numpy as np

v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([1, 1, 0])

# Stack vectors as columns
V = np.column_stack([v1, v2, v3])

# The rank tells us the number of linearly independent columns
rank = np.linalg.matrix_rank(V)
print("Rank of V:", rank)
print("Number of vectors:", V.shape[1])

if rank == V.shape[1]:
    print("Vectors are linearly independent")
else:
    print("Vectors are linearly dependent")
```

---

## Why Vector Spaces and Subspaces Matter in Deep Learning

- **Feature spaces**: Data is represented as vectors in high-dimensional spaces.
- **Hidden layers**: Each layer in a neural network transforms data into a new subspace.
- **Understanding capacity**: The dimension of a space relates to the expressive power of a model.

Grasping vector spaces and subspaces is key to understanding the structure and power of deep learning models! 