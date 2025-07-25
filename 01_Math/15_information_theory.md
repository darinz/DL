# Information Theory for Deep Learning

> **Essential information theory concepts that provide the mathematical foundation for loss functions, model compression, and understanding neural network behavior.**

---

## Table of Contents

1. [Entropy and Information](#entropy-and-information)
2. [Cross-Entropy and KL Divergence](#cross-entropy-and-kl-divergence)
3. [Mutual Information](#mutual-information)
4. [Applications in Deep Learning](#applications-in-deep-learning)

---

## 1. Entropy and Information

### Shannon Entropy

**Entropy** measures the average uncertainty or randomness in a random variable. For a discrete random variable $`X`$ with probability mass function $`p(x)`$:

```math
H(X) = -\sum_{x} p(x) \log p(x)
```

- **Intuition:** Entropy quantifies the unpredictability of a random variable. Higher entropy means more uncertainty.
- **Units:** Bits (if log base 2) or nats (if natural log).
- **Deep Learning Relevance:** Entropy underlies the concept of information content and is foundational for loss functions like cross-entropy.

#### Example
- A fair coin ($`p(H) = p(T) = 0.5`$): $`H(X) = 1`$ bit (maximum uncertainty)
- A biased coin ($`p(H) = 0.9, p(T) = 0.1`$): $`H(X) < 1`$ bit (less uncertainty)

> **Analogy:**
> - Entropy is like the "surprise" you get when you see the outcome. If you always expect heads, but get tails, that's surprising!

### Properties of Entropy

1. **Non-negativity:** $`H(X) \geq 0`$ (entropy can't be negative)
2. **Maximum entropy:** For $`n`$ outcomes, maximum entropy is $`\log n`$ (achieved with uniform distribution)
3. **Additivity:** For independent $`X`$ and $`Y`$: $`H(X,Y) = H(X) + H(Y)`$

> **Tip:**
> - Maximum entropy means maximum uncertainty (e.g., a fair die). Minimum entropy (zero) means no uncertainty (e.g., always the same outcome).

### Conditional Entropy

The entropy of $`X`$ given $`Y`$:
```math
H(X|Y) = -\sum_{x, y} p(x, y) \log p(x|y)
```
- Measures the remaining uncertainty in $`X`$ after knowing $`Y`$.
- **Deep Learning Relevance:** Conditional entropy is related to how much information is left after observing another variable (e.g., label given input).

### Python Implementation: Entropy

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def calculate_entropy(probabilities):
    """Calculate Shannon entropy
    Args:
        probabilities: array-like, probabilities of outcomes (should sum to 1)
    Returns:
        Entropy in bits (log base 2)
    """
    # Remove zero probabilities to avoid log(0) (which is undefined)
    probs = probabilities[probabilities > 0]
    return -np.sum(probs * np.log2(probs))

# Example: Entropy of different distributions
def entropy_examples():
    # Uniform distribution (maximum entropy)
    uniform_probs = np.array([0.25, 0.25, 0.25, 0.25])
    uniform_entropy = calculate_entropy(uniform_probs)
    
    # Skewed distribution (one outcome much more likely)
    skewed_probs = np.array([0.8, 0.1, 0.05, 0.05])
    skewed_entropy = calculate_entropy(skewed_probs)
    
    # Deterministic distribution (minimum entropy)
    deterministic_probs = np.array([1.0, 0.0, 0.0, 0.0])
    deterministic_entropy = calculate_entropy(deterministic_probs)
    
    print("Entropy Examples:")
    print(f"Uniform distribution: {uniform_entropy:.3f} bits (max uncertainty)")
    print(f"Skewed distribution: {skewed_entropy:.3f} bits (less uncertainty)")
    print(f"Deterministic distribution: {deterministic_entropy:.3f} bits (no uncertainty)")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    distributions = [uniform_probs, skewed_probs, deterministic_probs]
    names = ['Uniform', 'Skewed', 'Deterministic']
    entropies = [uniform_entropy, skewed_entropy, deterministic_entropy]
    
    for i, (probs, name, ent) in enumerate(zip(distributions, names, entropies)):
        plt.subplot(1, 3, i+1)
        plt.bar(range(len(probs)), probs, alpha=0.7)
        plt.title(f'{name}\nEntropy: {ent:.3f} bits')
        plt.xlabel('Outcome')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

entropy_examples()
```

**Code Annotations:**
- Calculates and visualizes entropy for different distributions.
- Shows how entropy changes with uniformity and skewness.
- Avoids log(0) by removing zero probabilities (important for numerical stability).
- **Try it:** Change the probabilities to see how entropy changes!

---

## 2. Cross-Entropy and KL Divergence

### Cross-Entropy

**Cross-entropy** measures the average number of bits needed to encode data from distribution $`p`$ using distribution $`q`$:

```math
H(p, q) = -\sum_{x} p(x) \log q(x)
```
- Used as a loss function in classification tasks (e.g., softmax output vs. true labels).
- Lower cross-entropy means $`q`$ is closer to $`p`$ (better model predictions).
- **Deep Learning Relevance:** Cross-entropy is the standard loss for classification because it directly measures how well the predicted probabilities match the true distribution.

> **Analogy:**
> - Cross-entropy is like asking: "How many bits do I need to encode the true data if I use my model's predictions?"

### Kullback-Leibler (KL) Divergence

**KL divergence** measures the difference between two probability distributions:

```math
D_{KL}(p \| q) = \sum_{x} p(x) \log \frac{p(x)}{q(x)}
```
- $`D_{KL}(p \| q) \geq 0`$ (non-negative)
- $`D_{KL}(p \| q) = 0`$ if and only if $`p = q`$
- Not symmetric: $`D_{KL}(p \| q) \neq D_{KL}(q \| p)`$
- **Deep Learning Relevance:** KL divergence is used in variational autoencoders (VAEs), regularization, and to measure how much the model diverges from the true distribution.

> **Pitfall:**
> - KL is not a true distance: $`D_{KL}(p \| q) \neq D_{KL}(q \| p)`$.
> - If $q(x) = 0$ where $p(x) > 0$, KL divergence is infinite (model must assign nonzero probability to all true outcomes).

### Relationship

```math
D_{KL}(p \| q) = H(p, q) - H(p)
```
- KL divergence is the extra entropy (cost) from using $`q`$ instead of $`p`$.

#### Example
- $`p`$ is the true distribution, $`q`$ is the model's prediction.
- KL divergence penalizes when $`q`$ assigns low probability to events that actually occur.

### Python Implementation: Cross-Entropy and KL Divergence

```python
def cross_entropy(p, q):
    """Calculate cross-entropy between distributions p and q
    Args:
        p: true distribution (array-like)
        q: predicted distribution (array-like)
    Returns:
        Cross-entropy in bits
    """
    # Ensure no zero probabilities for numerical stability
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    return -np.sum(p * np.log2(q))

def kl_divergence(p, q):
    """Calculate KL divergence from p to q
    Args:
        p: true distribution (array-like)
        q: predicted distribution (array-like)
    Returns:
        KL divergence in bits
    """
    # Ensure no zero probabilities
    p = np.maximum(p, 1e-10)
    q = np.maximum(q, 1e-10)
    return np.sum(p * np.log2(p / q))

# Example: Cross-entropy and KL divergence
def cross_entropy_examples():
    # True distribution
    p = np.array([0.3, 0.2, 0.5])
    
    # Different predicted distributions
    q1 = np.array([0.3, 0.2, 0.5])  # Perfect prediction
    q2 = np.array([0.4, 0.3, 0.3])  # Good prediction
    q3 = np.array([0.1, 0.1, 0.8])  # Poor prediction
    
    predictions = [q1, q2, q3]
    names = ['Perfect', 'Good', 'Poor']
    
    print("Cross-Entropy and KL Divergence Examples:")
    print(f"True distribution: {p}")
    
    for q, name in zip(predictions, names):
        ce = cross_entropy(p, q)
        kl = kl_divergence(p, q)
        print(f"{name} prediction: {q}")
        print(f"  Cross-entropy: {ce:.3f} bits")
        print(f"  KL divergence: {kl:.3f} bits")
        print()
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    for i, (q, name) in enumerate(zip(predictions, names)):
        plt.subplot(1, 3, i+1)
        x = np.arange(len(p))
        width = 0.35
        
        plt.bar(x - width/2, p, width, label='True', alpha=0.7)
        plt.bar(x + width/2, q, width, label='Predicted', alpha=0.7)
        
        ce = cross_entropy(p, q)
        kl = kl_divergence(p, q)
        plt.title(f'{name}\nCE: {ce:.3f}, KL: {kl:.3f}')
        plt.xlabel('Outcome')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

cross_entropy_examples()
```

**Code Annotations:**
- Calculates and visualizes cross-entropy and KL divergence for different prediction scenarios.
- Shows how loss increases as predictions diverge from the true distribution.
- Ensures numerical stability by avoiding log(0).
- **Try it:** Change the predicted distributions to see how the loss changes!

---

## 3. Mutual Information

### Definition

**Mutual information** measures the amount of information shared between two random variables:

```math
I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
```
- Quantifies how much knowing $`Y`$ reduces uncertainty about $`X`$.
- **Deep Learning Relevance:** Mutual information is used in representation learning, feature selection, and understanding how much information about the input is preserved in the learned representation.

### Properties

1. **Non-negativity:** $`I(X; Y) \geq 0`$
2. **Symmetry:** $`I(X; Y) = I(Y; X)`$
3. **Independence:** $`I(X; Y) = 0`$ if and only if $`X`$ and $`Y`$ are independent

> **Analogy:**
> - Mutual information is like the overlap in a Venn diagram: how much two variables "share" in terms of information.

#### Example
- If $`X`$ and $`Y`$ are perfectly correlated, $`I(X; Y)`$ is maximized.
- If $`X`$ and $`Y`$ are independent, $`I(X; Y) = 0`$.

### Python Implementation: Mutual Information

```python
def mutual_information(p_xy):
    """Calculate mutual information from joint distribution p_xy
    Args:
        p_xy: 2D array, joint probability distribution of X and Y
    Returns:
        Mutual information in bits
    """
    # Calculate marginal distributions
    p_x = np.sum(p_xy, axis=1)
    p_y = np.sum(p_xy, axis=0)
    
    # Calculate entropies
    h_x = calculate_entropy(p_x)
    h_y = calculate_entropy(p_y)
    h_xy = calculate_entropy(p_xy.flatten())
    
    # Mutual information
    mi = h_x + h_y - h_xy
    return mi

# Example: Mutual information between correlated variables
def mutual_information_example():
    # Create correlated data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate correlated Gaussian data
    rho = 0.7  # correlation coefficient
    x = np.random.normal(0, 1, n_samples)
    y = rho * x + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_samples)
    
    # Discretize for mutual information calculation
    x_bins = np.linspace(x.min(), x.max(), 10)
    y_bins = np.linspace(y.min(), y.max(), 10)
    
    # Create joint histogram
    joint_hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
    joint_prob = joint_hist / np.sum(joint_hist)
    
    # Calculate mutual information
    mi = mutual_information(joint_prob)
    
    print(f"Mutual Information Example:")
    print(f"Correlation coefficient: {rho}")
    print(f"Mutual information: {mi:.3f} bits")
    
    # Visualize
    plt.figure(figsize=(12, 4))
    
    # Scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(x, y, alpha=0.6)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Correlated Data (ρ = {rho})')
    plt.grid(True)
    
    # Joint distribution
    plt.subplot(1, 3, 2)
    plt.imshow(joint_prob, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Joint Distribution')
    plt.xlabel('Y bins')
    plt.ylabel('X bins')
    
    # Marginal distributions
    plt.subplot(1, 3, 3)
    p_x = np.sum(joint_prob, axis=1)
    p_y = np.sum(joint_prob, axis=0)
    
    plt.plot(p_x, 'b-', label='P(X)', linewidth=2)
    plt.plot(p_y, 'r-', label='P(Y)', linewidth=2)
    plt.title('Marginal Distributions')
    plt.xlabel('Bin')
    plt.ylabel('Probability')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

mutual_information_example()
```

**Code Annotations:**
- Calculates and visualizes mutual information for correlated variables.
- Shows how mutual information increases with correlation.
- Demonstrates the relationship between joint and marginal distributions.
- **Try it:** Change the correlation coefficient to see how MI changes!

---

## 4. Applications in Deep Learning

Information theory is deeply integrated into deep learning:

### Loss Functions

- **Cross-Entropy Loss:** Standard loss for classification, measures the distance between true and predicted distributions.
- **KL Divergence:** Used in variational autoencoders (VAEs), regularization, and model compression.

### Model Compression and Representation Learning

- **Information Bottleneck Principle:**
  1. Maximize mutual information with the target (retain relevant information)
  2. Minimize mutual information with the input (compress irrelevant information)
- Guides the design of efficient and robust representations.
- **Deep Learning Relevance:** The information bottleneck explains why deep networks can learn compressed, generalizable features.

### Feature Selection

- **Mutual Information:** Used to select features that share the most information with the target variable.
- **Tip:** Features with high mutual information with the target are more useful for prediction.

### Uncertainty Quantification

- **Entropy:** Measures model uncertainty and confidence in predictions.
- **Pitfall:** High entropy means the model is unsure; low entropy means confident predictions (but not necessarily correct!).

### Python Implementation: Deep Learning Applications

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# Cross-entropy loss for classification
def classification_loss_example():
    """Demonstrate cross-entropy loss for classification
    Shows how loss increases as predictions diverge from the true label.
    """
    # True labels (one-hot encoded)
    y_true = np.array([
        [1, 0, 0],  # Class 0
        [0, 1, 0],  # Class 1
        [0, 0, 1],  # Class 2
        [1, 0, 0],  # Class 0
        [0, 1, 0]   # Class 1
    ])
    
    # Different prediction scenarios
    predictions = {
        'Perfect': np.array([
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05]
        ]),
        'Good': np.array([
            [0.7, 0.2, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.7, 0.1]
        ]),
        'Poor': np.array([
            [0.3, 0.4, 0.3],
            [0.4, 0.3, 0.3],
            [0.3, 0.3, 0.4],
            [0.4, 0.3, 0.3],
            [0.3, 0.4, 0.3]
        ])
    }
    
    print("Cross-Entropy Loss Examples:")
    for name, y_pred in predictions.items():
        loss = log_loss(y_true, y_pred)
        print(f"{name} predictions: {loss:.3f}")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        plt.subplot(1, 3, i+1)
        
        # Plot true vs predicted for first class
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.7)
        plt.plot([0, 1], [0, 1], 'r--', linewidth=2)
        
        loss = log_loss(y_true, y_pred)
        plt.title(f'{name}\nLoss: {loss:.3f}')
        plt.xlabel('True Probability')
        plt.ylabel('Predicted Probability')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Information bottleneck example
def information_bottleneck_example():
    """Simple information bottleneck demonstration
    Shows how mutual information can be used to analyze feature relevance and redundancy.
    """
    # Generate data with some redundancy
    np.random.seed(42)
    n_samples = 1000
    
    # Input features
    x1 = np.random.normal(0, 1, n_samples)
    x2 = 0.8 * x1 + 0.2 * np.random.normal(0, 1, n_samples)  # Correlated with x1
    x3 = np.random.normal(0, 1, n_samples)  # Independent
    
    # Target (depends on x1 and x3)
    y = 0.6 * x1 + 0.4 * x3 + 0.1 * np.random.normal(0, 1, n_samples)
    
    # Calculate mutual information
    def estimate_mi(x, y, bins=20):
        """Estimate mutual information between x and y"""
        # Discretize
        x_bins = np.linspace(x.min(), x.max(), bins)
        y_bins = np.linspace(y.min(), y.max(), bins)
        
        joint_hist, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])
        joint_prob = joint_hist / np.sum(joint_hist)
        
        return mutual_information(joint_prob)
    
    # Calculate MI between features and target
    mi_x1_y = estimate_mi(x1, y)
    mi_x2_y = estimate_mi(x2, y)
    mi_x3_y = estimate_mi(x3, y)
    
    # Calculate MI between features
    mi_x1_x2 = estimate_mi(x1, x2)
    mi_x1_x3 = estimate_mi(x1, x3)
    
    print("Information Bottleneck Analysis:")
    print(f"MI(X1, Y): {mi_x1_y:.3f} bits")
    print(f"MI(X2, Y): {mi_x2_y:.3f} bits")
    print(f"MI(X3, Y): {mi_x3_y:.3f} bits")
    print(f"MI(X1, X2): {mi_x1_x2:.3f} bits")
    print(f"MI(X1, X3): {mi_x1_x3:.3f} bits")
    
    # Visualize
    plt.figure(figsize=(15, 5))
    
    # Feature correlations
    plt.subplot(1, 3, 1)
    plt.scatter(x1, x2, alpha=0.6)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(f'X1 vs X2 (MI: {mi_x1_x2:.3f})')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.scatter(x1, x3, alpha=0.6)
    plt.xlabel('X1')
    plt.ylabel('X3')
    plt.title(f'X1 vs X3 (MI: {mi_x1_x3:.3f})')
    plt.grid(True)
    
    # Feature-target relationships
    plt.subplot(1, 3, 3)
    plt.scatter(x1, y, alpha=0.6, label=f'X1 (MI: {mi_x1_y:.3f})')
    plt.scatter(x3, y, alpha=0.6, label=f'X3 (MI: {mi_x3_y:.3f})')
    plt.xlabel('Feature Value')
    plt.ylabel('Target')
    plt.title('Feature-Target Relationships')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Run examples
classification_loss_example()
information_bottleneck_example()
```

**Code Annotations:**
- Calculates and visualizes mutual information for correlated variables.
- Demonstrates cross-entropy loss for classification with different prediction qualities.
- Shows information bottleneck analysis with mutual information between features and targets.
- Visualizes feature redundancy and relevance.
- **Try it:** Change the feature correlations or prediction probabilities to see how the results change!

---

## 5. Summary

Information theory provides fundamental insights for deep learning:

1. **Cross-entropy loss** is the standard loss function for classification
2. **KL divergence** is used in variational autoencoders and model compression
3. **Mutual information** helps understand feature importance and model behavior
4. **Information bottleneck** guides representation learning
5. **Entropy** provides measures of uncertainty and randomness

Key applications include:
- **Loss function design** based on information-theoretic principles
- **Model compression** using information bottleneck
- **Feature selection** using mutual information
- **Uncertainty quantification** using entropy measures
- **Representation learning** guided by information theory

Understanding information theory enables:
- Better loss function design
- More efficient model compression
- Improved feature selection
- Deeper understanding of model behavior

---

## Further Reading

- **"Elements of Information Theory"** by Thomas M. Cover and Joy A. Thomas
- **"Information Theory, Inference, and Learning Algorithms"** by David J.C. MacKay
- **"Deep Learning"** by Ian Goodfellow, Yoshua Bengio, and Aaron Courville 