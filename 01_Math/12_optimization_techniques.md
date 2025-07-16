# Optimization Techniques

> **Optimization is the process of finding the best parameters for a model, and is at the heart of training deep learning systems.**

---

## 1. What is Optimization?

In deep learning, **optimization** means finding the set of parameters $`\theta`$ that minimize a loss function $`L(\theta)`$.
- The loss function measures how well the model predicts the data.
- The optimization algorithm updates $`\theta`$ to reduce $`L`$.

---

## 2. Gradient Descent

The most fundamental optimization algorithm in deep learning is **gradient descent**:

```math
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
```

Where:
- $`\theta_t`$ are the parameters at step $`t`$
- $`\alpha`$ is the learning rate (step size)
- $`\nabla L(\theta_t)`$ is the gradient of the loss function with respect to $`\theta`$

### Intuition
- The gradient $`\nabla L(\theta)`$ points in the direction of steepest increase of $`L`$.
- We move in the **opposite** direction to minimize $`L`$.
- The learning rate $`\alpha`$ controls how big each step is.

### Example
Let $`L(\theta) = (\theta - 3)^2`$.
- $`\nabla L(\theta) = 2(\theta - 3)`$
- Update: $`\theta_{t+1} = \theta_t - 2\alpha (\theta_t - 3)`$
- $`\theta`$ converges to 3.

---

## 3. Stochastic Gradient Descent (SGD)

**SGD** uses a random subset (mini-batch) of the data at each step:

```math
\theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t, \mathcal{B}_t)
```

Where $`\mathcal{B}_t`$ is a mini-batch at step $`t`$.

### Why Use SGD?
- Faster updates (especially for large datasets)
- Introduces noise, which can help escape local minima
- More scalable for deep learning

---

## 4. Momentum

**Momentum** helps accelerate gradient descent by adding a fraction of the previous update:

```math
v_{t+1} = \beta v_t + (1-\beta)\nabla L(\theta_t)
```
```math
\theta_{t+1} = \theta_t - \alpha v_{t+1}
```

Where:
- $`v_t`$ is the velocity (running average of gradients)
- $`\beta`$ is the momentum coefficient (e.g., 0.9)

### Intuition
- Momentum smooths the updates, helping to avoid oscillations.
- It can speed up convergence, especially in ravines or on noisy surfaces.

---

## 5. Learning Rate Strategies

- **Constant learning rate**: Simple, but may not work for all problems.
- **Learning rate decay**: Reduce $`\alpha`$ over time.
- **Adaptive methods**: Algorithms like Adam, RMSProp adjust the learning rate for each parameter.

---

## 6. Python Implementation: Optimization Algorithms

Let's compare basic gradient descent and momentum on the Rosenbrock function (a classic test for optimization algorithms).

```python
import numpy as np
import matplotlib.pyplot as plt

def rosenbrock_function(x, y):
    """Rosenbrock function for testing optimization"""
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(x, y):
    """Gradient of Rosenbrock function"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

def gradient_descent(f, grad_f, initial_point, learning_rate=0.001, max_iter=1000):
    """Basic gradient descent"""
    point = np.array(initial_point)
    trajectory = [point.copy()]
    for i in range(max_iter):
        gradient = grad_f(*point)
        point = point - learning_rate * gradient
        trajectory.append(point.copy())
        if np.linalg.norm(gradient) < 1e-6:
            break
    return np.array(trajectory)

def momentum_gradient_descent(f, grad_f, initial_point, learning_rate=0.001, momentum=0.9, max_iter=1000):
    """Gradient descent with momentum"""
    point = np.array(initial_point)
    velocity = np.zeros_like(point)
    trajectory = [point.copy()]
    for i in range(max_iter):
        gradient = grad_f(*point)
        velocity = momentum * velocity + (1 - momentum) * gradient
        point = point - learning_rate * velocity
        trajectory.append(point.copy())
        if np.linalg.norm(gradient) < 1e-6:
            break
    return np.array(trajectory)

# Test optimization algorithms
initial_point = [-1.5, -1.5]
# Run both algorithms
gd_trajectory = gradient_descent(rosenbrock_function, rosenbrock_gradient, initial_point, learning_rate=0.0001)
momentum_trajectory = momentum_gradient_descent(rosenbrock_function, rosenbrock_gradient, initial_point, learning_rate=0.0001)

# Visualize optimization trajectories
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z = rosenbrock_function(X, Y)

plt.figure(figsize=(12, 5))
# Contour plot with trajectories
plt.subplot(1, 2, 1)
plt.contour(X, Y, Z, levels=20)
plt.plot(gd_trajectory[:, 0], gd_trajectory[:, 1], 'r-', label='Gradient Descent', linewidth=2)
plt.plot(momentum_trajectory[:, 0], momentum_trajectory[:, 1], 'b-', label='Momentum', linewidth=2)
plt.plot(initial_point[0], initial_point[1], 'go', markersize=10, label='Start')
plt.plot(1, 1, 'k*', markersize=15, label='Optimum (1,1)')
plt.title('Optimization Trajectories')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# Function values over iterations
plt.subplot(1, 2, 2)
gd_values = [rosenbrock_function(x, y) for x, y in gd_trajectory]
momentum_values = [rosenbrock_function(x, y) for x, y in momentum_trajectory]
plt.plot(gd_values, 'r-', label='Gradient Descent', linewidth=2)
plt.plot(momentum_values, 'b-', label='Momentum', linewidth=2)
plt.title('Function Values Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('f(x,y)')
plt.legend()
plt.grid(True)
plt.yscale('log')
plt.tight_layout()
plt.show()
print(f"Gradient Descent converged in {len(gd_trajectory)} iterations")
print(f"Momentum converged in {len(momentum_trajectory)} iterations")
```

**Code Annotations:**
- The Rosenbrock function is a common test for optimization algorithms (it has a narrow, curved valley).
- `gradient_descent` and `momentum_gradient_descent` implement the two algorithms.
- The contour plot shows the optimization paths.
- The function value plot shows convergence speed.

---

## 7. Why Optimization Techniques Matter in Deep Learning

- **Training:** Optimization algorithms are used to minimize loss functions and train models.
- **Convergence:** Good optimization methods speed up convergence and improve results.
- **Advanced methods:** Many variants (Adam, RMSProp, etc.) build on these basics.
- **Generalization:** Proper optimization can help models generalize better to new data.

### Example: Adam Optimizer (Overview)

Adam combines momentum and adaptive learning rates:
- Maintains running averages of gradients and squared gradients.
- Adapts learning rate for each parameter.
- Widely used in deep learning.

---

## 8. Summary

- Optimization is central to deep learning.
- Gradient descent and its variants are the workhorses of model training.
- Understanding these methods helps you tune, debug, and improve deep learning models.

**Further Reading:**
- [Gradient Descent (Wikipedia)](https://en.wikipedia.org/wiki/Gradient_descent)
- [Optimization for Deep Learning (Stanford CS231n)](http://cs231n.stanford.edu/)
- [Adam Optimizer (arXiv)](https://arxiv.org/abs/1412.6980) 