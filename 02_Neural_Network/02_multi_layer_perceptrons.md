# Multi-layer Perceptrons (MLPs): Feedforward Neural Networks

A comprehensive guide to understanding Multi-layer Perceptrons, the foundation of modern deep learning architectures.

> **Learning Objective**: Understand the architecture, mathematical foundations, training algorithms, and practical implementation of MLPs.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture Overview](#architecture-overview)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Universal Approximation Theorem](#universal-approximation-theorem)
5. [Backpropagation Algorithm](#backpropagation-algorithm)
6. [Implementation in Python](#implementation-in-python)
7. [Training and Optimization](#training-and-optimization)
8. [Practical Examples](#practical-examples)
9. [Advanced Topics](#advanced-topics)

---

## Introduction

Multi-layer Perceptrons (MLPs) extend the single perceptron by stacking multiple layers of neurons, enabling the learning of complex, non-linear patterns and representations. They form the foundation of modern deep learning and can approximate any continuous function given sufficient capacity.

### What is an MLP?

An MLP is a feedforward neural network that:
- Consists of multiple layers of neurons
- Processes information in a forward direction only
- Uses non-linear activation functions
- Can learn complex, non-linear mappings
- Is trained using backpropagation

### Key Characteristics

- **Feedforward**: Information flows only in the forward direction
- **Fully Connected**: Each neuron connects to all neurons in adjacent layers
- **Non-linear**: Uses activation functions to introduce non-linearity
- **Universal**: Can approximate any continuous function
- **Supervised Learning**: Trained with labeled data

---

## Architecture Overview

### Layer Structure

An MLP consists of three types of layers:

1. **Input Layer**: Receives raw input features
2. **Hidden Layers**: Intermediate layers that learn representations
3. **Output Layer**: Produces final predictions

### Network Architecture

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → ... → Hidden Layer L → Output Layer
```

### Mathematical Representation

For a network with $L$ layers:

```math
\begin{align}
h^{(0)} &= x \quad \text{(Input layer)} \\
h^{(l)} &= f^{(l)}(W^{(l)}h^{(l-1)} + b^{(l)}) \quad \text{(Hidden layers)} \\
y &= h^{(L)} \quad \text{(Output layer)}
\end{align}
```

Where:
- **$h^{(l)}$**: Activations at layer $l$
- **$W^{(l)}$**: Weight matrix for layer $l$
- **$b^{(l)}$**: Bias vector for layer $l$
- **$f^{(l)}$**: Activation function for layer $l$

### Layer Dimensions

For a network with layer sizes $[n_0, n_1, n_2, \ldots, n_L]$:

- **$W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$**: Weight matrix
- **$b^{(l)} \in \mathbb{R}^{n_l}$**: Bias vector
- **$h^{(l)} \in \mathbb{R}^{n_l}$**: Activation vector

---

## Mathematical Foundation

### Forward Propagation

The forward propagation through an MLP can be expressed as:

```math
\begin{align}
z^{(l)} &= W^{(l)}h^{(l-1)} + b^{(l)} \quad \text{(Linear transformation)} \\
h^{(l)} &= f^{(l)}(z^{(l)}) \quad \text{(Non-linear activation)}
\end{align}
```

Where:
- **$z^{(l)}$**: Pre-activation values (before applying activation function)
- **$h^{(l)}$**: Post-activation values (after applying activation function)

### Activation Functions

Common activation functions for hidden layers:

#### ReLU (Rectified Linear Unit)
```math
f(x) = \max(0, x)
```

#### Sigmoid
```math
f(x) = \frac{1}{1 + e^{-x}}
```

#### Tanh (Hyperbolic Tangent)
```math
f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
```

### Loss Function

For regression tasks, commonly use Mean Squared Error (MSE):

```math
L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
```

For classification tasks, commonly use Cross-Entropy Loss:

```math
L(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)
```

---

## Universal Approximation Theorem

### Statement

**Universal Approximation Theorem**: A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of $\mathbb{R}^n$.

### Mathematical Formulation

For any continuous function $f: [0,1]^n \rightarrow \mathbb{R}$ and any $\epsilon > 0$, there exists a feedforward neural network with one hidden layer such that:

```math
|f(x) - \hat{f}(x)| < \epsilon \quad \forall x \in [0,1]^n
```

### Implications

1. **Representational Power**: MLPs can represent any continuous function
2. **Hidden Layer Sufficiency**: One hidden layer is theoretically sufficient
3. **Practical Considerations**: Multiple layers may be more efficient
4. **Training Challenges**: Universal approximation doesn't guarantee easy training

### Proof Sketch

The proof involves:
1. **Density of Step Functions**: Any continuous function can be approximated by step functions
2. **Step Function Representation**: Step functions can be represented by perceptrons
3. **Linear Combination**: Weighted sum of step functions approximates the target function

---

## Backpropagation Algorithm

Backpropagation is the algorithm for efficiently computing gradients in neural networks using the chain rule of calculus.

### Chain Rule

For a composite function $f(g(x))$, the derivative is:

```math
\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}
```

### Gradient Computation

The gradient of the loss with respect to weights is computed using:

```math
\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial z^{(L)}} \cdot \frac{\partial z^{(L)}}{\partial W^{(l)}}
```

### Backpropagation Equations

#### Output Layer Gradients

```math
\begin{align}
\delta^{(L)} &= \frac{\partial L}{\partial h^{(L)}} \odot f'^{(L)}(z^{(L)}) \\
\frac{\partial L}{\partial W^{(L)}} &= \delta^{(L)} \cdot (h^{(L-1)})^T \\
\frac{\partial L}{\partial b^{(L)}} &= \delta^{(L)}
\end{align}
```

#### Hidden Layer Gradients

```math
\begin{align}
\delta^{(l)} &= (W^{(l+1)})^T \delta^{(l+1)} \odot f'^{(l)}(z^{(l)}) \\
\frac{\partial L}{\partial W^{(l)}} &= \delta^{(l)} \cdot (h^{(l-1)})^T \\
\frac{\partial L}{\partial b^{(l)}} &= \delta^{(l)}
\end{align}
```

Where $\odot$ denotes element-wise multiplication.

### Algorithm Steps

1. **Forward Pass**: Compute all activations and pre-activations
2. **Backward Pass**: Compute gradients starting from output layer
3. **Weight Update**: Update weights using gradient descent

---

## Implementation in Python

### Basic MLP Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class MLP:
    def __init__(self, layer_sizes: List[int], activation='relu', learning_rate=0.01):
        """
        Initialize Multi-Layer Perceptron
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden_size, ..., output_size]
            activation: Activation function ('relu', 'sigmoid', 'tanh')
            learning_rate: Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.initialize_parameters()
        
        # Training history
        self.training_loss = []
        self.validation_loss = []
    
    def initialize_parameters(self):
        """Initialize weights and biases using Xavier/Glorot initialization"""
        for i in range(self.num_layers - 1):
            # Xavier initialization
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            scale = np.sqrt(2.0 / (fan_in + fan_out))
            
            w = np.random.randn(fan_out, fan_in) * scale
            b = np.zeros((fan_out, 1))
            
            self.weights.append(w)
            self.biases.append(b)
    
    def activation_function(self, x: np.ndarray, derivative: bool = False) -> np.ndarray:
        """Apply activation function or its derivative"""
        if self.activation == 'relu':
            if derivative:
                return (x > 0).astype(float)
            else:
                return np.maximum(0, x)
        
        elif self.activation == 'sigmoid':
            if derivative:
                sigmoid = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
                return sigmoid * (1 - sigmoid)
            else:
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        
        elif self.activation == 'tanh':
            if derivative:
                return 1 - np.tanh(x) ** 2
            else:
                return np.tanh(x)
        
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")
    
    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Forward propagation
        
        Args:
            X: Input data of shape (n_features, n_samples)
            
        Returns:
            Tuple of (activations, pre_activations)
        """
        activations = [X]
        pre_activations = []
        
        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            pre_activations.append(z)
            
            # Non-linear activation
            if i == self.num_layers - 2:  # Output layer
                # For regression, use linear activation
                a = z
            else:  # Hidden layers
                a = self.activation_function(z)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                 pre_activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward propagation
        
        Args:
            X: Input data
            y: Target values
            activations: List of activation values from forward pass
            pre_activations: List of pre-activation values from forward pass
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = X.shape[1]  # Number of samples
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = activations[-1] - y.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Gradient of weights and biases
            weight_gradients[i] = np.dot(delta, activations[i].T) / m
            bias_gradients[i] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Gradient of activations (skip for input layer)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.activation_function(pre_activations[i-1], derivative=True)
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], 
                         bias_gradients: List[np.ndarray]):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute Mean Squared Error loss"""
        return np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, 
            validation_data: Tuple[np.ndarray, np.ndarray] = None, 
            verbose: bool = True) -> dict:
        """
        Train the MLP
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Target values of shape (n_samples,)
            epochs: Number of training epochs
            validation_data: Tuple of (X_val, y_val) for validation
            verbose: Whether to print training progress
            
        Returns:
            Dictionary containing training history
        """
        # Transpose X for matrix operations
        X = X.T
        y = y.reshape(-1, 1)
        
        # Initialize validation data
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.T
            y_val = y_val.reshape(-1, 1)
        
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # Forward pass
            activations, pre_activations = self.forward(X)
            
            # Compute training loss
            train_loss = self.compute_loss(activations[-1], y)
            history['train_loss'].append(train_loss)
            
            # Compute validation loss
            if validation_data is not None:
                val_activations, _ = self.forward(X_val)
                val_loss = self.compute_loss(val_activations[-1], y_val)
                history['val_loss'].append(val_loss)
            
            # Backward pass
            weight_gradients, bias_gradients = self.backward(X, y, activations, pre_activations)
            
            # Update parameters
            self.update_parameters(weight_gradients, bias_gradients)
            
            # Print progress
            if verbose and epoch % 100 == 0:
                if validation_data is not None:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        X = X.T
        activations, _ = self.forward(X)
        return activations[-1].flatten()
    
    def plot_training_history(self, history: dict):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(history['train_loss'], label='Training Loss', color='blue')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss', color='red')
        
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
```

### Enhanced MLP with Regularization

```python
class RegularizedMLP(MLP):
    def __init__(self, layer_sizes: List[int], activation='relu', learning_rate=0.01,
                 dropout_rate=0.2, l2_reg=0.01):
        """
        MLP with regularization techniques
        
        Args:
            dropout_rate: Probability of dropping neurons during training
            l2_reg: L2 regularization coefficient
        """
        super().__init__(layer_sizes, activation, learning_rate)
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.dropout_masks = []
    
    def forward(self, X: np.ndarray, training: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Forward propagation with dropout"""
        activations = [X]
        pre_activations = []
        self.dropout_masks = []
        
        for i in range(self.num_layers - 1):
            # Linear transformation
            z = np.dot(self.weights[i], activations[i]) + self.biases[i]
            pre_activations.append(z)
            
            # Non-linear activation
            if i == self.num_layers - 2:  # Output layer
                a = z
            else:  # Hidden layers
                a = self.activation_function(z)
                
                # Apply dropout during training
                if training and self.dropout_rate > 0:
                    mask = np.random.binomial(1, 1 - self.dropout_rate, size=a.shape) / (1 - self.dropout_rate)
                    a *= mask
                    self.dropout_masks.append(mask)
            
            activations.append(a)
        
        return activations, pre_activations
    
    def backward(self, X: np.ndarray, y: np.ndarray, activations: List[np.ndarray], 
                 pre_activations: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Backward propagation with L2 regularization"""
        m = X.shape[1]
        
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Output layer gradient
        delta = activations[-1] - y.reshape(-1, 1)
        
        # Backpropagate through layers
        for i in range(self.num_layers - 2, -1, -1):
            # Gradient of weights (with L2 regularization)
            weight_gradients[i] = np.dot(delta, activations[i].T) / m + self.l2_reg * self.weights[i]
            bias_gradients[i] = np.sum(delta, axis=1, keepdims=True) / m
            
            # Gradient of activations
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.activation_function(pre_activations[i-1], derivative=True)
                
                # Apply dropout mask during backpropagation
                if self.dropout_masks:
                    delta *= self.dropout_masks[i-1]
        
        return weight_gradients, bias_gradients
    
    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Compute loss with L2 regularization"""
        mse_loss = np.mean((y_pred - y_true.reshape(-1, 1)) ** 2)
        
        # Add L2 regularization term
        l2_loss = 0
        for w in self.weights:
            l2_loss += np.sum(w ** 2)
        l2_loss = self.l2_reg * l2_loss / 2
        
        return mse_loss + l2_loss
```

---

## Training and Optimization

### Gradient Descent Variants

#### Stochastic Gradient Descent (SGD)

```python
def sgd_update(parameters, gradients, learning_rate):
    """Basic SGD update"""
    for param, grad in zip(parameters, gradients):
        param -= learning_rate * grad
```

#### Adam Optimizer

```python
class AdamOptimizer:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Time step
    
    def update(self, parameters, gradients):
        """Adam update rule"""
        if self.m is None:
            self.m = [np.zeros_like(p) for p in parameters]
            self.v = [np.zeros_like(p) for p in parameters]
        
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(parameters, gradients)):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)
            
            # Compute bias-corrected first moment estimate
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            
            # Compute bias-corrected second moment estimate
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Update parameters
            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
```

### Learning Rate Scheduling

```python
class LearningRateScheduler:
    def __init__(self, initial_lr=0.01, decay_rate=0.1, decay_steps=1000):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.step = 0
    
    def get_learning_rate(self):
        """Get current learning rate"""
        return self.initial_lr * (self.decay_rate ** (self.step / self.decay_steps))
    
    def step(self):
        """Increment step counter"""
        self.step += 1
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss):
        """Check if training should stop"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.should_stop = True
        
        return self.should_stop
```

---

## Practical Examples

### Example 1: XOR Problem

```python
def xor_example():
    """Demonstrate MLP solving XOR problem"""
    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    
    # Create MLP with 2 hidden neurons
    mlp = MLP(layer_sizes=[2, 4, 1], activation='relu', learning_rate=0.1)
    
    # Train
    history = mlp.fit(X_xor, y_xor, epochs=1000, verbose=False)
    
    # Test
    predictions = mlp.predict(X_xor)
    
    print("XOR Problem Results:")
    print("Input\t\tTarget\tPrediction\tCorrect?")
    print("-" * 40)
    for i in range(len(X_xor)):
        pred = 1 if predictions[i] > 0.5 else 0
        correct = "✓" if pred == y_xor[i] else "✗"
        print(f"{X_xor[i]}\t\t{y_xor[i]}\t{pred}\t\t{correct}")
    
    # Plot training history
    mlp.plot_training_history(history)

# Run XOR example
xor_example()
```

### Example 2: Function Approximation

```python
def function_approximation_example():
    """Demonstrate MLP approximating a non-linear function"""
    # Generate data
    np.random.seed(42)
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X).flatten() + 0.1 * np.random.randn(100)
    
    # Split into train and validation
    train_idx = np.random.choice(100, 80, replace=False)
    val_idx = np.setdiff1d(np.arange(100), train_idx)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    
    # Create MLP
    mlp = MLP(layer_sizes=[1, 20, 10, 1], activation='tanh', learning_rate=0.01)
    
    # Train
    history = mlp.fit(X_train, y_train, epochs=2000, 
                     validation_data=(X_val, y_val), verbose=False)
    
    # Predict
    y_pred = mlp.predict(X)
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Plot function approximation
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, c='blue', alpha=0.6, label='Training Data')
    plt.scatter(X_val, y_val, c='red', alpha=0.6, label='Validation Data')
    plt.plot(X, y_pred, 'g-', linewidth=2, label='MLP Prediction')
    plt.plot(X, np.sin(X), 'k--', linewidth=2, label='True Function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Function Approximation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training history
    plt.subplot(1, 2, 2)
    mlp.plot_training_history(history)
    
    plt.tight_layout()
    plt.show()

# Run function approximation example
function_approximation_example()
```

### Example 3: Classification Problem

```python
def classification_example():
    """Demonstrate MLP for classification"""
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    
    # Generate moon dataset
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create MLP for classification
    mlp = RegularizedMLP(layer_sizes=[2, 50, 25, 1], activation='relu', 
                        learning_rate=0.01, dropout_rate=0.2, l2_reg=0.01)
    
    # Train
    history = mlp.fit(X_train, y_train, epochs=1000, verbose=False)
    
    # Predict
    y_pred_proba = mlp.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot decision boundary
    plt.figure(figsize=(12, 4))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    mlp.plot_training_history(history)
    
    # Plot decision boundary
    plt.subplot(1, 2, 2)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                        np.arange(y_min, y_max, 0.02))
    
    Z = mlp.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    plt.scatter(X_test[y_test == 0][:, 0], X_test[y_test == 0][:, 1], 
               c='red', alpha=0.7, label='Class 0')
    plt.scatter(X_test[y_test == 1][:, 0], X_test[y_test == 1][:, 1], 
               c='blue', alpha=0.7, label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Run classification example
classification_example()
```

---

## Advanced Topics

### Weight Initialization

```python
def weight_initialization_comparison():
    """Compare different weight initialization strategies"""
    def create_mlp(init_method):
        mlp = MLP(layer_sizes=[1, 10, 1], activation='tanh', learning_rate=0.01)
        
        if init_method == 'xavier':
            # Xavier/Glorot initialization
            for i, w in enumerate(mlp.weights):
                fan_in = mlp.layer_sizes[i]
                fan_out = mlp.layer_sizes[i + 1]
                scale = np.sqrt(2.0 / (fan_in + fan_out))
                mlp.weights[i] = np.random.randn(*w.shape) * scale
        
        elif init_method == 'he':
            # He initialization
            for i, w in enumerate(mlp.weights):
                fan_in = mlp.layer_sizes[i]
                scale = np.sqrt(2.0 / fan_in)
                mlp.weights[i] = np.random.randn(*w.shape) * scale
        
        return mlp
    
    # Generate data
    X = np.linspace(-3, 3, 100).reshape(-1, 1)
    y = np.sin(X).flatten()
    
    # Test different initializations
    methods = ['random', 'xavier', 'he']
    histories = {}
    
    for method in methods:
        mlp = create_mlp(method)
        history = mlp.fit(X, y, epochs=500, verbose=False)
        histories[method] = history
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    for method, history in histories.items():
        plt.plot(history['train_loss'], label=f'{method.capitalize()} Initialization')
    
    plt.title('Weight Initialization Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.show()

# Run weight initialization comparison
weight_initialization_comparison()
```

### Gradient Clipping

```python
def gradient_clipping_example():
    """Demonstrate gradient clipping to prevent exploding gradients"""
    def clip_gradients(gradients, max_norm=1.0):
        """Clip gradients to prevent exploding gradients"""
        total_norm = 0
        for grad in gradients:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)
        
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for grad in gradients:
                grad *= clip_coef
        
        return gradients
    
    # Create MLP with gradient clipping
    mlp = MLP(layer_sizes=[1, 100, 1], activation='tanh', learning_rate=0.1)
    
    # Generate data with high variance
    X = np.linspace(-10, 10, 100).reshape(-1, 1)
    y = np.sin(X).flatten() + 2 * np.random.randn(100)
    
    # Train with gradient clipping
    X = X.T
    y = y.reshape(-1, 1)
    
    for epoch in range(1000):
        # Forward pass
        activations, pre_activations = mlp.forward(X)
        
        # Backward pass
        weight_gradients, bias_gradients = mlp.backward(X, y, activations, pre_activations)
        
        # Clip gradients
        weight_gradients = clip_gradients(weight_gradients, max_norm=1.0)
        bias_gradients = clip_gradients(bias_gradients, max_norm=1.0)
        
        # Update parameters
        mlp.update_parameters(weight_gradients, bias_gradients)
        
        if epoch % 100 == 0:
            loss = mlp.compute_loss(activations[-1], y)
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Run gradient clipping example
gradient_clipping_example()
```

---

## Summary

Multi-layer Perceptrons are powerful neural network architectures that:

1. **Universal Approximation**: Can approximate any continuous function
2. **Non-linear Learning**: Can learn complex, non-linear patterns
3. **Flexible Architecture**: Can be designed for various tasks
4. **Foundation**: Form the basis for modern deep learning

### Key Takeaways

- **Architecture**: Multiple layers with non-linear activations
- **Training**: Backpropagation with gradient descent
- **Regularization**: Dropout, L2 regularization, early stopping
- **Optimization**: Various optimizers and learning rate schedules
- **Applications**: Classification, regression, function approximation

### Next Steps

Understanding MLPs provides the foundation for:
- **Convolutional Neural Networks** (CNNs) for computer vision
- **Recurrent Neural Networks** (RNNs) for sequential data
- **Transformers** for attention-based architectures
- **Modern deep learning frameworks** (PyTorch, TensorFlow)

MLPs demonstrate the power of combining simple computational units into complex, learnable systems that can solve a wide variety of problems. 