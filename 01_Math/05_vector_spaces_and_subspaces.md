# Vector Spaces and Subspaces

> **Vector spaces provide the setting for all of linear algebra, and subspaces are their building blocks.**

---

## What is a Vector Space?

A **vector space** is a set of vectors that is closed under vector addition and scalar multiplication. This means:
- If $\mathbf{u}$ and $\mathbf{v}$ are in the space, so is $\mathbf{u} + \mathbf{v}$.
- If $\mathbf{v}$ is in the space and $c$ is any scalar, then $c\mathbf{v}$ is also in the space.

Formally, a vector space $V$ over $\mathbb{R}$ (the real numbers) satisfies:
- **Additive identity:** There is a zero vector $\mathbf{0}$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ for all $\mathbf{v}$.
- **Additive inverse:** For every $\mathbf{v}$, there is $-\mathbf{v}$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$.
- **Distributive, associative, and commutative properties** (see any linear algebra text for full list).

**Step-by-step:**
- Check if the set contains the zero vector.
- Check if adding any two vectors in the set stays in the set.
- Check if multiplying any vector by a scalar stays in the set.

**Intuition:**
- A vector space is like a "playground" where you can add vectors and scale them, and you never leave the playground.
- Examples: $\mathbb{R}^2$ (the plane), $\mathbb{R}^3$ (3D space), the set of all $n$-dimensional real vectors, the set of all polynomials, the set of all $m \times n$ matrices.
- Non-examples: The set of all vectors with positive entries (not closed under scalar multiplication by negative numbers).

> **Tip:** Most data in deep learning lives in high-dimensional vector spaces!

---

## Subspaces

A **subspace** is a subset of a vector space that is itself a vector space (with the same operations).

**How to check if $W$ is a subspace of $V$:**
1. $\mathbf{0} \in W$
2. If $\mathbf{u}, \mathbf{v} \in W$, then $\mathbf{u} + \mathbf{v} \in W$
3. If $\mathbf{v} \in W$ and $c \in \mathbb{R}$, then $c\mathbf{v} \in W$

**Step-by-step:**
- Check if the zero vector is in $W$.
- Check if $W$ is closed under addition and scalar multiplication.

**Example:** The set of all vectors in $\mathbb{R}^3$ where the third component is zero forms a subspace (the $xy$-plane).

**Visualization:**
- In $\mathbb{R}^3$, subspaces are lines through the origin, planes through the origin, or the whole space.

> **Pitfall:** A subspace must contain the origin and be closed under all vector space operations.

---

## Linear Independence

A set of vectors $\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n$ is **linearly independent** if:

```math
c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_n\mathbf{v}_n = \mathbf{0}
```

implies $c_1 = c_2 = \cdots = c_n = 0$.

If not, the vectors are **linearly dependent** (at least one can be written as a combination of the others).

**Step-by-step:**
- Set up the equation above.
- If the only solution is all $c_i = 0$, the vectors are independent.
- If there is a nonzero solution, they are dependent.

**Intuition:**
- Linearly independent vectors "point in new directions"—none can be built from the others.
- If you can write one as a combination of the others, they are dependent.

**Example:**
- $[1, 0, 0], [0, 1, 0], [0, 0, 1]$ in $\mathbb{R}^3$ are independent.
- $[1, 2], [2, 4]$ in $\mathbb{R}^2$ are dependent (second is a multiple of the first).

> **Tip:** The maximum number of linearly independent vectors in a space is its dimension.

---

## Basis and Dimension

A **basis** of a vector space is a set of linearly independent vectors that span the space (every vector in the space can be written as a combination of basis vectors).

The **dimension** of a vector space is the number of vectors in any basis.

**Step-by-step:**
- Find a set of independent vectors that can build any vector in the space.
- Count them: that's the dimension.

**Example:**
- The standard basis for $\mathbb{R}^3$ is $[1,0,0], [0,1,0], [0,0,1]$.
- Any three independent vectors in $\mathbb{R}^3$ form a basis.
- The dimension of $\mathbb{R}^3$ is 3.

> **Pitfall:** A basis must be both independent and span the space.

---

## Python Implementation: Checking Linear Independence

Let's check if three vectors in $\mathbb{R}^3$ are linearly independent:

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

**Code Annotations:**
- `np.column_stack` stacks vectors as columns of a matrix.
- `np.linalg.matrix_rank` computes the rank (number of independent columns).
- If the rank equals the number of vectors, they are independent.

**Visualization:**
- In 2D, plot two vectors. If they are not multiples of each other, they are independent.
- In 3D, three vectors are independent if they do not all lie in the same plane.

> **Tip:** Try changing the vectors to see when they become dependent!

---

## Why Vector Spaces and Subspaces Matter in Deep Learning

- **Feature spaces**: Data is represented as vectors in high-dimensional spaces.
- **Hidden layers**: Each layer in a neural network transforms data into a new subspace.
- **Understanding capacity**: The dimension of a space relates to the expressive power of a model.
- **Embeddings**: Word embeddings, image embeddings, etc., live in vector spaces.

> **Summary:** Grasping vector spaces and subspaces is key to understanding the structure and power of deep learning models! They provide the language for all data, features, and learned representations. 