# Optimization Algorithms in Deep Learning

Optimization algorithms are the engines that drive neural network training by updating model parameters to minimize the loss function. The choice of optimizer significantly impacts training speed, convergence, and final model performance.

## Table of Contents

1. [Introduction](#introduction)
2. [Gradient Descent Fundamentals](#gradient-descent-fundamentals)
3. [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
4. [Momentum-Based Methods](#momentum-based-methods)
5. [Adaptive Learning Rate Methods](#adaptive-learning-rate-methods)
6. [Advanced Optimizers](#advanced-optimizers)
7. [Implementation in Python](#implementation-in-python)
8. [Optimizer Selection](#optimizer-selection)
9. [Practical Considerations](#practical-considerations)

---

## Introduction

### What is Optimization?

Optimization in deep learning involves finding the optimal set of parameters $\theta$ that minimize the loss function $L(\theta)$:

```math
\theta^* = \arg\min_{\theta} L(\theta)
```

### The Optimization Problem

Given a neural network with parameters $\theta$ and loss function $L(\theta)$, we need to find:

```math
\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)
```

Where:
- $\theta_t$ are the parameters at step $t$
- $\eta_t$ is the learning rate at step $t$
- $\nabla L(\theta_t)$ is the gradient of the loss function

### Key Challenges

1. **Local Minima**: Getting stuck in suboptimal solutions
2. **Saddle Points**: Flat regions with zero gradients
3. **Ill-Conditioning**: Different parameters requiring different learning rates
4. **Noise**: Stochastic gradients introduce variance

---

## Gradient Descent Fundamentals

### Basic Gradient Descent

**Update Rule:**
```math
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
```

**Properties:**
- Simple and intuitive
- Guaranteed convergence for convex functions
- Can be slow for ill-conditioned problems
- Sensitive to learning rate choice

### Batch vs. Stochastic vs. Mini-batch

#### Batch Gradient Descent
```math
\nabla L(\theta) = \frac{1}{n} \sum_{i=1}^{n} \nabla L_i(\theta)
```
- Uses entire dataset
- Computationally expensive
- Stable gradients
- Can get stuck in local minima

#### Stochastic Gradient Descent (SGD)
```math
\nabla L(\theta) = \nabla L_i(\theta) \text{ for random } i
```
- Uses single sample
- Fast updates
- Noisy gradients
- Can escape local minima

#### Mini-batch Gradient Descent
```math
\nabla L(\theta) = \frac{1}{m} \sum_{i \in B} \nabla L_i(\theta)
```
- Uses subset of data
- Balance between speed and stability
- Most commonly used in practice

---

## Stochastic Gradient Descent (SGD)

### Basic SGD

**Update Rule:**
```math
\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)
```

**Properties:**
- Simple and widely used
- Good generalization
- Can be slow to converge
- Sensitive to learning rate

### SGD with Learning Rate Decay

**Update Rule:**
```math
\eta_t = \eta_0 \cdot \text{decay}(t)
\theta_{t+1} = \theta_t - \eta_t \nabla L(\theta_t)
```

**Common Decay Functions:**

1. **Step Decay:**
```math
\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}
```

2. **Exponential Decay:**
```math
\eta_t = \eta_0 \cdot e^{-\lambda t}
```

3. **Cosine Annealing:**
```math
\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(\frac{t}{T}\pi))
```

---

## Momentum-Based Methods

### SGD with Momentum

**Update Rule:**
```math
\begin{align}
v_{t+1} &= \beta v_t + \nabla L(\theta_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{align}
```

**Properties:**
- Accelerates convergence
- Helps escape local minima
- Reduces oscillation
- $\beta$ typically 0.9

### Nesterov Momentum

**Update Rule:**
```math
\begin{align}
v_{t+1} &= \beta v_t + \nabla L(\theta_t - \beta v_t) \\
\theta_{t+1} &= \theta_t - \eta v_{t+1}
\end{align}
```

**Properties:**
- Better theoretical convergence
- "Look-ahead" gradient
- Often outperforms standard momentum

---

## Adaptive Learning Rate Methods

### AdaGrad

**Update Rule:**
```math
\begin{align}
G_t &= G_{t-1} + (\nabla L(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)
\end{align}
```

**Properties:**
- Adapts learning rate per parameter
- Good for sparse data
- Learning rate can become too small
- Accumulates squared gradients

### RMSprop

**Update Rule:**
```math
\begin{align}
G_t &= \beta G_{t-1} + (1-\beta)(\nabla L(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \nabla L(\theta_t)
\end{align}
```

**Properties:**
- Exponential moving average of squared gradients
- Prevents learning rate from becoming too small
- $\beta$ typically 0.9
- Good for non-convex optimization

### Adam

**Update Rule:**
```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}
```

**Properties:**
- Combines momentum and adaptive learning rates
- Bias correction for early iterations
- $\beta_1$ typically 0.9, $\beta_2$ typically 0.999
- Most popular optimizer in practice

### AdamW

**Update Rule:**
```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t - \eta \lambda \theta_t
\end{align}
```

**Properties:**
- Decoupled weight decay
- Better generalization than Adam
- $\lambda$ is weight decay parameter

---

## Advanced Optimizers

### AdaBelief

**Update Rule:**
```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t) \\
s_t &= \beta_2 s_{t-1} + (1-\beta_2) (\nabla L(\theta_t) - m_t)^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{s}_t &= \frac{s_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{s}_t} + \epsilon} \hat{m}_t
\end{align}
```

**Properties:**
- Adapts to gradient variance
- Better convergence than Adam
- More robust to noise

### RAdam

**Update Rule:**
```math
\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\theta_t))^2 \\
\rho_t &= \rho_{\infty} - \frac{2t\beta_2^t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t \text{ if } \rho_t > 4 \\
\theta_{t+1} &= \theta_t - \eta \hat{m}_t \text{ otherwise}
\end{align}
```

**Properties:**
- Rectifies Adam's variance
- Better early training stability
- Automatic warmup

---

## Implementation in Python

### Basic Optimizer Framework

```python
import numpy as np

class Optimizer:
    """Base class for optimizers"""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.t = 0
    
    def step(self, params, grads):
        """Update parameters"""
        raise NotImplementedError
    
    def zero_grad(self):
        """Reset gradients"""
        pass

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01, momentum=0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocity = None
    
    def step(self, params, grads):
        """SGD update"""
        if self.velocity is None:
            self.velocity = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            self.velocity[i] = self.momentum * self.velocity[i] + grad
            param -= self.learning_rate * self.velocity[i]
        
        self.t += 1

class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
    
    def step(self, params, grads):
        """Adam update"""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Example usage
if __name__ == "__main__":
    # Simple test
    params = [np.random.randn(2, 2), np.random.randn(2, 1)]
    grads = [np.random.randn(2, 2), np.random.randn(2, 1)]
    
    # Test SGD
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    sgd.step(params, grads)
    
    # Test Adam
    adam = Adam(learning_rate=0.001)
    adam.step(params, grads)
```

### Advanced Optimizer Implementations

```python
class RMSprop(Optimizer):
    def __init__(self, learning_rate=0.001, beta=0.9, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.v = None
    
    def step(self, params, grads):
        """RMSprop update"""
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Update moving average of squared gradients
            self.v[i] = self.beta * self.v[i] + (1 - self.beta) * (grad ** 2)
            
            # Update parameters
            param -= self.learning_rate * grad / (np.sqrt(self.v[i]) + self.epsilon)

class AdaGrad(Optimizer):
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
        self.G = None
    
    def step(self, params, grads):
        """AdaGrad update"""
        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Accumulate squared gradients
            self.G[i] += grad ** 2
            
            # Update parameters
            param -= self.learning_rate * grad / (np.sqrt(self.G[i]) + self.epsilon)

class AdamW(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                 epsilon=1e-8, weight_decay=0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = None
        self.v = None
    
    def step(self, params, grads):
        """AdamW update"""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(params, grads)):
            # Add weight decay to gradient
            grad += self.weight_decay * param
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second raw moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

# Learning rate schedulers
class LearningRateScheduler:
    def __init__(self, optimizer, initial_lr):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.t = 0
    
    def step(self):
        """Update learning rate"""
        raise NotImplementedError

class StepLR(LearningRateScheduler):
    def __init__(self, optimizer, initial_lr, step_size, gamma=0.1):
        super().__init__(optimizer, initial_lr)
        self.step_size = step_size
        self.gamma = gamma
    
    def step(self):
        """Step decay"""
        if self.t % self.step_size == 0:
            self.optimizer.learning_rate *= self.gamma
        self.t += 1

class CosineAnnealingLR(LearningRateScheduler):
    def __init__(self, optimizer, initial_lr, T_max, eta_min=0):
        super().__init__(optimizer, initial_lr)
        self.T_max = T_max
        self.eta_min = eta_min
    
    def step(self):
        """Cosine annealing"""
        self.optimizer.learning_rate = self.eta_min + \
            (self.initial_lr - self.eta_min) * \
            (1 + np.cos(np.pi * self.t / self.T_max)) / 2
        self.t += 1
```

### Complete Training Loop

```python
class NeuralNetworkTrainer:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, x, y):
        """Single training step"""
        # Forward pass
        y_pred = self.model.forward(x)
        
        # Compute loss
        loss = self.loss_fn.compute(y, y_pred)
        
        # Backward pass
        grads = self.model.backward(x, y)
        
        # Update parameters
        self.optimizer.step(self.model.parameters(), grads)
        
        return loss
    
    def train(self, train_loader, epochs, scheduler=None):
        """Training loop"""
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for x, y in train_loader:
                loss = self.train_step(x, y)
                epoch_loss += loss
                num_batches += 1
                
                if scheduler:
                    scheduler.step()
            
            avg_loss = epoch_loss / num_batches
            losses.append(avg_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
        
        return losses

# Example training
if __name__ == "__main__":
    # Create model, optimizer, and trainer
    model = SimpleNeuralNetwork([2, 4, 1])
    optimizer = Adam(learning_rate=0.001)
    loss_fn = MeanSquaredError()
    trainer = NeuralNetworkTrainer(model, optimizer, loss_fn)
    
    # Create scheduler
    scheduler = CosineAnnealingLR(optimizer, 0.001, T_max=1000)
    
    # Training data
    X = np.random.randn(2, 100)
    y = np.random.randn(1, 100)
    
    # Train
    losses = trainer.train([(X, y)], epochs=100, scheduler=scheduler)
```

---

## Optimizer Selection

### Comparison Table

| Optimizer | Pros | Cons | Best For |
|-----------|------|------|----------|
| SGD | Simple, good generalization | Slow convergence, sensitive to LR | Fine-tuning, convex problems |
| SGD + Momentum | Faster convergence, escapes local minima | Additional hyperparameter | General purpose |
| Adam | Fast convergence, adaptive LR | May generalize worse | Most deep learning tasks |
| AdamW | Better generalization than Adam | Additional hyperparameter | When weight decay is important |
| RMSprop | Good for non-convex problems | Learning rate can become small | RNNs, non-convex optimization |
| AdaGrad | Good for sparse data | Learning rate becomes too small | Sparse data, convex problems |

### Selection Guidelines

1. **Start with Adam**: Good default choice for most problems
2. **Use SGD for fine-tuning**: Often better generalization
3. **Consider AdamW**: When weight decay is important
4. **Use RMSprop**: For RNNs and non-convex problems
5. **Try AdaGrad**: For sparse data

---

## Practical Considerations

### Hyperparameter Tuning

```python
def hyperparameter_search(model_class, train_data, val_data, 
                         learning_rates=[0.001, 0.01, 0.1],
                         optimizers=['adam', 'sgd', 'rmsprop']):
    """Grid search for best optimizer and learning rate"""
    best_score = float('inf')
    best_params = {}
    
    for lr in learning_rates:
        for opt_name in optimizers:
            # Create optimizer
            if opt_name == 'adam':
                optimizer = Adam(learning_rate=lr)
            elif opt_name == 'sgd':
                optimizer = SGD(learning_rate=lr)
            elif opt_name == 'rmsprop':
                optimizer = RMSprop(learning_rate=lr)
            
            # Train model
            model = model_class()
            trainer = NeuralNetworkTrainer(model, optimizer, MeanSquaredError())
            losses = trainer.train(train_data, epochs=50)
            
            # Evaluate on validation set
            val_score = evaluate_model(model, val_data)
            
            if val_score < best_score:
                best_score = val_score
                best_params = {'lr': lr, 'optimizer': opt_name}
    
    return best_params, best_score
```

### Gradient Clipping

```python
def clip_gradients(gradients, max_norm=1.0):
    """Clip gradients to prevent exploding gradients"""
    total_norm = np.sqrt(sum(np.sum(g**2) for g in gradients))
    clip_coef = max_norm / (total_norm + 1e-6)
    
    if clip_coef < 1:
        return [g * clip_coef for g in gradients]
    
    return gradients

# Usage in optimizer
class ClippedAdam(Adam):
    def __init__(self, learning_rate=0.001, max_norm=1.0, **kwargs):
        super().__init__(learning_rate, **kwargs)
        self.max_norm = max_norm
    
    def step(self, params, grads):
        # Clip gradients
        grads = clip_gradients(grads, self.max_norm)
        
        # Standard Adam update
        super().step(params, grads)
```

### Learning Rate Warmup

```python
class WarmupLR(LearningRateScheduler):
    def __init__(self, optimizer, initial_lr, warmup_steps):
        super().__init__(optimizer, initial_lr)
        self.warmup_steps = warmup_steps
    
    def step(self):
        """Linear warmup"""
        if self.t < self.warmup_steps:
            self.optimizer.learning_rate = self.initial_lr * (self.t + 1) / self.warmup_steps
        self.t += 1
```

---

## Summary

Optimization algorithms are crucial for neural network training:

1. **SGD**: Simple baseline with good generalization
2. **Momentum**: Accelerates convergence and escapes local minima
3. **Adaptive Methods**: Automatically adjust learning rates per parameter
4. **Adam**: Most popular choice for deep learning
5. **Advanced Methods**: Address specific issues like variance and bias

Key considerations:
- **Problem Type**: Different optimizers work better for different problems
- **Hyperparameters**: Learning rate, momentum, and decay parameters
- **Scheduling**: Learning rate scheduling can improve convergence
- **Stability**: Gradient clipping and warmup for training stability

The choice of optimizer significantly impacts training efficiency and final model performance. 