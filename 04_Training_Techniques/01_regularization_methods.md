# Regularization Methods

> **Key Insight:** Regularization is the secret weapon for building neural networks that generalize well. It helps prevent overfitting and makes your models robust to new, unseen data.

---

## Table of Contents
1. [Dropout](#dropout)
2. [Weight Decay (L2 Regularization)](#weight-decay-l2-regularization)
3. [Early Stopping](#early-stopping)
4. [Data Augmentation](#data-augmentation)
5. [Summary](#summary)

---

## Dropout

Dropout is a regularization technique that randomly deactivates neurons during training to prevent co-adaptation and improve generalization.

### Mathematical Formulation

Dropout can be expressed as:

```math
\text{Dropout}(x, p) = \begin{cases}
\frac{x}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
```

Where:
- $`x`$ is the input activation
- $`p`$ is the dropout probability
- During training, neurons are randomly zeroed with probability $`p`$
- During inference, activations are scaled by $`1-p`$ to maintain expected values

### Intuition

The key insight behind dropout is that by randomly deactivating neurons during training:
1. **Prevents co-adaptation:** Neurons cannot rely on specific combinations of other neurons
2. **Forces redundancy:** The network learns multiple representations for the same features
3. **Improves generalization:** The network becomes more robust to missing inputs

> **Did you know?**
> Dropout can be seen as training an ensemble of subnetworks and averaging their predictions at test time.

### Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropoutLayer:
    """Custom dropout implementation for educational purposes"""
    
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        
    def forward(self, x, training=True):
        if training:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1-self.p, size=x.shape) / (1-self.p)
            return x * self.mask
        else:
            # During inference, just return the input
            return x
    
    def backward(self, grad_output):
        return grad_output * self.mask

# PyTorch implementation
class DropoutExample(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.5):
        super(DropoutExample, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout after activation
        x = self.fc2(x)
        return x

# Example usage
def demonstrate_dropout():
    # Create sample data
    batch_size, input_size, hidden_size, output_size = 32, 784, 256, 10
    x = torch.randn(batch_size, input_size)
    
    # Create model with dropout
    model = DropoutExample(input_size, hidden_size, output_size, dropout_p=0.5)
    
    # Training mode (dropout active)
    model.train()
    output_train = model(x)
    print(f"Training output shape: {output_train.shape}")
    
    # Evaluation mode (dropout inactive)
    model.eval()
    output_eval = model(x)
    print(f"Evaluation output shape: {output_eval.shape}")
    
    # Compare outputs
    print(f"Output difference: {torch.abs(output_train - output_eval).mean():.4f}")

demonstrate_dropout()
```

> **Code Commentary:**
> - Dropout is only active during training (`model.train()`).
> - During evaluation (`model.eval()`), dropout is turned off and activations are scaled.
> - Dropout is typically applied after activation functions in hidden layers.

### Dropout Variants

#### 1. Spatial Dropout

For convolutional networks, spatial dropout drops entire feature maps:

```python
class SpatialDropout2d(nn.Module):
    def __init__(self, p=0.5):
        super(SpatialDropout2d, self).__init__()
        self.p = p
        
    def forward(self, x):
        if self.training:
            # x shape: (batch, channels, height, width)
            batch, channels, height, width = x.shape
            mask = torch.bernoulli(torch.ones(batch, channels, 1, 1) * (1 - self.p))
            mask = mask / (1 - self.p)
            return x * mask
        return x
```

#### 2. Alpha Dropout

Maintains self-normalizing properties for SELU activations:

```python
class AlphaDropout(nn.Module):
    def __init__(self, p=0.5, alpha=-1.7580993408473766):
        super(AlphaDropout, self).__init__()
        self.p = p
        self.alpha = alpha
        
    def forward(self, x):
        if self.training:
            # SELU-specific dropout
            mask = torch.bernoulli(torch.ones_like(x) * (1 - self.p))
            mask = mask / (1 - self.p)
            return mask * x + self.alpha * (1 - mask)
        return x
```

### Best Practices

1. **Dropout Rates:**
   - Input layers: 0.2-0.3
   - Hidden layers: 0.3-0.5
   - Output layers: Usually not applied
2. **Placement:** Apply dropout after activation functions
3. **Combination:** Often used with other regularization techniques

> **Common Pitfall:**
> Using too high a dropout rate can cause underfitting. Start with 0.5 for hidden layers and tune as needed.

---

## Weight Decay (L2 Regularization)

Weight decay adds a penalty term to the loss function to discourage large weights, helping prevent overfitting.

### Mathematical Formulation

The total loss with L2 regularization is:

```math
L_{\text{total}} = L_{\text{original}} + \frac{\lambda}{2} \sum_{i} w_i^2
```

The gradient becomes:

```math
\frac{\partial L_{\text{total}}}{\partial w_i} = \frac{\partial L_{\text{original}}}{\partial w_i} + \lambda w_i
```

And the weight update rule:

```math
w_i \leftarrow w_i - \alpha \left(\frac{\partial L_{\text{original}}}{\partial w_i} + \lambda w_i\right) = (1 - \alpha\lambda)w_i - \alpha\frac{\partial L_{\text{original}}}{\partial w_i}
```

### Intuition

Weight decay works by:
1. **Penalizing large weights:** Large weights increase the regularization term
2. **Encouraging smaller weights:** Smaller weights lead to smoother decision boundaries
3. **Preventing overfitting:** Reduces model complexity

> **Did you know?**
> L2 regularization is mathematically equivalent to placing a Gaussian prior on the weights in a Bayesian framework.

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WeightDecayExample:
    def __init__(self, model, weight_decay=1e-4):
        self.model = model
        self.weight_decay = weight_decay
        
    def compute_l2_penalty(self):
        """Compute L2 penalty manually"""
        l2_penalty = 0.0
        for param in self.model.parameters():
            l2_penalty += torch.sum(param ** 2)
        return 0.5 * self.weight_decay * l2_penalty
    
    def train_step(self, x, y, optimizer, criterion):
        # Forward pass
        outputs = self.model(x)
        
        # Compute original loss
        original_loss = criterion(outputs, y)
        
        # Add L2 penalty
        l2_penalty = self.compute_l2_penalty()
        total_loss = original_loss + l2_penalty
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item()

# Using PyTorch's built-in weight decay
def demonstrate_weight_decay():
    # Create model
    model = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    )
    
    # Optimizer with weight decay
    optimizer_with_decay = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Optimizer without weight decay
    optimizer_no_decay = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    
    print("Weight decay is automatically handled by PyTorch optimizers")
    print("Set weight_decay parameter in optimizer constructor")

# Manual implementation for educational purposes
class ManualWeightDecay:
    def __init__(self, model, lr=0.001, weight_decay=1e-4):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
```

### Weight Decay vs L2 Regularization

While often used interchangeably, there are subtle differences:

1. **Weight Decay**: Directly modifies the weight update rule
2. **L2 Regularization**: Adds penalty to the loss function

For SGD, they are equivalent when $`\lambda = \alpha \cdot \text{weight\_decay}`$

### Hyperparameter Tuning

```python
def grid_search_weight_decay():
    weight_decay_values = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    results = {}
    
    for wd in weight_decay_values:
        # Train model with different weight decay values
        model = create_model()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=wd)
        
        # Train and evaluate
        train_loss, val_loss = train_and_evaluate(model, optimizer)
        results[wd] = {'train_loss': train_loss, 'val_loss': val_loss}
    
    return results
```

---

## Early Stopping

Early stopping is a regularization technique that halts training when the model's performance on a validation set stops improving, preventing overfitting.

### Intuition

- **Key Insight:** Training a neural network for too long can cause it to memorize the training data, leading to poor generalization. Early stopping monitors validation performance and stops training at the optimal point.

### How It Works

1. **Monitor Validation Loss:** After each epoch, evaluate the model on a validation set.
2. **Patience Parameter:** If the validation loss does not improve for a set number of epochs (patience), stop training.
3. **Restore Best Weights:** Optionally, revert to the model weights that achieved the best validation loss.

### Python Implementation

```python
import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = np.inf
        self.counter = 0
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore_best_weights(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
```

> **Try it yourself!**
> Train a model with and without early stopping. Plot the training and validation loss curves. Where does overfitting begin?

### Best Practices

- Use a separate validation set for early stopping.
- Set patience based on how noisy your validation loss is.
- Combine with other regularization methods for best results.

> **Common Pitfall:**
> Using the training set for early stopping can lead to overfitting. Always use a separate validation set.

---

## Data Augmentation

Data augmentation increases the diversity of the training data by applying random transformations, helping the model generalize better.

### Intuition

- **Key Insight:** By exposing the model to many variations of the data, data augmentation simulates a larger dataset and makes the model robust to real-world variations.

### Common Augmentation Techniques

- **Image:** Flipping, rotation, scaling, cropping, color jitter, noise
- **Text:** Synonym replacement, random insertion, back-translation
- **Audio:** Time stretching, pitch shifting, noise injection

### Mathematical Formulation

Let $`x`$ be an input sample and $`T`$ a random transformation:

```math
x' = T(x)
```

The loss is then averaged over all possible transformations:

```math
L_{aug} = \mathbb{E}_{T \sim \mathcal{T}} [L(f(T(x)), y)]
```

Where $`\mathcal{T}`$ is the set of possible transformations.

### Python Implementation (Image Example)

```python
from torchvision import transforms
from PIL import Image

# Define augmentation pipeline
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor()
])

# Apply to an image
img = Image.open('example.jpg')
augmented_img = transform(img)
```

> **Did you know?**
> Data augmentation is especially powerful in computer vision, but is also used in NLP and audio tasks.

### Best Practices

- Use augmentation only on the training set, not validation/test sets.
- Choose augmentations that reflect real-world variations in your data.
- Don't overdo it—too much augmentation can make the task harder for the model.

> **Common Pitfall:**
> Applying augmentation to validation or test data can lead to misleading performance metrics.

---

## Summary

Regularization is essential for building robust, generalizable neural networks. The most common techniques include:

- $`\textbf{Dropout}`$: Randomly deactivates neurons to prevent co-adaptation
- $`\textbf{Weight Decay (L2)}`$: Penalizes large weights to encourage simpler models
- $`\textbf{Early Stopping}`$: Stops training before overfitting occurs
- $`\textbf{Data Augmentation}`$: Increases data diversity for better generalization

> **Key Insight:**
> The best results often come from combining several regularization techniques. Experiment, monitor validation performance, and tune your approach for each problem! 