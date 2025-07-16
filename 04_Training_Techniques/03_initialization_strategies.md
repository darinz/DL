# Initialization Strategies

Proper weight initialization is crucial for training deep neural networks effectively. Poor initialization can lead to vanishing or exploding gradients, slow convergence, or complete training failure. This guide covers the most important initialization strategies with detailed explanations, mathematical formulations, and practical Python implementations.

## Table of Contents

1. [Xavier/Glorot Initialization](#xavierglorot-initialization)
2. [He Initialization](#he-initialization)
3. [Orthogonal Initialization](#orthogonal-initialization)
4. [Pre-trained Weights](#pre-trained-weights)
5. [Advanced Initialization Techniques](#advanced-initialization-techniques)
6. [Practical Guidelines](#practical-guidelines)

## Xavier/Glorot Initialization

Xavier/Glorot initialization is designed for sigmoid and tanh activation functions, maintaining the variance of activations and gradients across layers.

### Mathematical Foundation

The key insight is to maintain the variance of activations and gradients across layers. For a layer with $`n_{\text{in}}`$ inputs and $`n_{\text{out}}`$ outputs:

**Variance of activations:**
```math
\text{Var}(a^{(l)}) = n_{\text{in}}^{(l-1)} \cdot \text{Var}(W^{(l)}) \cdot \text{Var}(a^{(l-1)})
```

**Variance of gradients:**
```math
\text{Var}\left(\frac{\partial L}{\partial a^{(l-1)}}\right) = n_{\text{out}}^{(l)} \cdot \text{Var}(W^{(l)}) \cdot \text{Var}\left(\frac{\partial L}{\partial a^{(l)}}\right)
```

**Optimal variance for weights:**
```math
\text{Var}(W^{(l)}) = \frac{2}{n_{\text{in}}^{(l-1)} + n_{\text{out}}^{(l)}}
```

### Mathematical Formulation

**Normal Distribution:**
```math
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}} + n_{\text{out}}}\right)
```

**Uniform Distribution:**
```math
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)
```

### Python Implementation

```python
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt

class XavierInitializer:
    """Custom Xavier/Glorot initialization implementation"""
    
    @staticmethod
    def normal(tensor, gain=1.0):
        """Xavier normal initialization"""
        fan_in, fan_out = XavierInitializer._calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(2.0 / (fan_in + fan_out))
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    
    @staticmethod
    def uniform(tensor, gain=1.0):
        """Xavier uniform initialization"""
        fan_in, fan_out = XavierInitializer._calculate_fan_in_and_fan_out(tensor)
        bound = gain * np.sqrt(6.0 / (fan_in + fan_out))
        return torch.nn.init.uniform_(tensor, a=-bound, b=bound)
    
    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor):
        """Calculate fan_in and fan_out for a tensor"""
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        
        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        
        return fan_in, fan_out

# Example usage
def demonstrate_xavier_initialization():
    # Create a linear layer
    input_size, output_size = 100, 50
    linear_layer = nn.Linear(input_size, output_size)
    
    # Apply Xavier initialization
    XavierInitializer.normal(linear_layer.weight)
    XavierInitializer.uniform(linear_layer.bias)
    
    # Check the statistics
    weight_mean = linear_layer.weight.mean().item()
    weight_std = linear_layer.weight.std().item()
    expected_std = np.sqrt(2.0 / (input_size + output_size))
    
    print(f"Xavier Normal Initialization:")
    print(f"  Weight mean: {weight_mean:.6f} (expected: 0.0)")
    print(f"  Weight std: {weight_std:.6f} (expected: {expected_std:.6f})")
    
    # Test with uniform initialization
    linear_layer_uniform = nn.Linear(input_size, output_size)
    XavierInitializer.uniform(linear_layer_uniform.weight)
    
    weight_std_uniform = linear_layer_uniform.weight.std().item()
    expected_std_uniform = np.sqrt(6.0 / (input_size + output_size)) / np.sqrt(3)
    
    print(f"\nXavier Uniform Initialization:")
    print(f"  Weight std: {weight_std_uniform:.6f} (expected: {expected_std_uniform:.6f})")

demonstrate_xavier_initialization()
```

### Xavier Initialization in Neural Networks

```python
class XavierNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(XavierNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layer = nn.Linear(prev_size, hidden_size)
            # Apply Xavier initialization
            XavierInitializer.normal(layer.weight)
            XavierInitializer.uniform(layer.bias)
            layers.append(layer)
            layers.append(nn.Tanh())  # Tanh activation
            prev_size = hidden_size
        
        # Output layer
        output_layer = nn.Linear(prev_size, output_size)
        XavierInitializer.normal(output_layer.weight)
        XavierInitializer.uniform(output_layer.bias)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test Xavier initialization with forward pass
def test_xavier_forward_pass():
    input_size, hidden_sizes, output_size = 10, [20, 15], 5
    model = XavierNet(input_size, hidden_sizes, output_size)
    
    # Create input data
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean():.6f}")
    print(f"Output std: {output.std():.6f}")
    
    # Check activations at each layer
    activations = []
    x_current = x
    
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            x_current = layer(x_current)
            activations.append(x_current.detach())
        elif isinstance(layer, nn.Tanh):
            x_current = layer(x_current)
    
    print(f"\nActivation statistics:")
    for i, act in enumerate(activations):
        print(f"  Layer {i+1}: mean={act.mean():.4f}, std={act.std():.4f}")

test_xavier_forward_pass()
```

## He Initialization

He initialization is optimized for ReLU activation functions, accounting for the fact that ReLU zeroes out negative activations.

### Mathematical Foundation

For ReLU activations, approximately half of the activations are zeroed out, so the variance is reduced by a factor of 2:

**Variance of activations with ReLU:**
```math
\text{Var}(a^{(l)}) = \frac{1}{2} \cdot n_{\text{in}}^{(l-1)} \cdot \text{Var}(W^{(l)}) \cdot \text{Var}(a^{(l-1)})
```

**Optimal variance for weights:**
```math
\text{Var}(W^{(l)}) = \frac{2}{n_{\text{in}}^{(l-1)}}
```

### Mathematical Formulation

**Normal Distribution:**
```math
W_{ij} \sim \mathcal{N}\left(0, \frac{2}{n_{\text{in}}}\right)
```

**Uniform Distribution:**
```math
W_{ij} \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}}}}, \sqrt{\frac{6}{n_{\text{in}}}}\right)
```

### Python Implementation

```python
class HeInitializer:
    """Custom He initialization implementation"""
    
    @staticmethod
    def normal(tensor, gain=2.0):
        """He normal initialization"""
        fan_in, _ = HeInitializer._calculate_fan_in_and_fan_out(tensor)
        std = gain * np.sqrt(1.0 / fan_in)
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    
    @staticmethod
    def uniform(tensor, gain=2.0):
        """He uniform initialization"""
        fan_in, _ = HeInitializer._calculate_fan_in_and_fan_out(tensor)
        bound = gain * np.sqrt(3.0 / fan_in)
        return torch.nn.init.uniform_(tensor, a=-bound, b=bound)
    
    @staticmethod
    def _calculate_fan_in_and_fan_out(tensor):
        """Calculate fan_in and fan_out for a tensor"""
        dimensions = tensor.dim()
        if dimensions < 2:
            raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
        
        if dimensions == 2:  # Linear
            fan_in = tensor.size(1)
            fan_out = tensor.size(0)
        else:
            num_input_fmaps = tensor.size(1)
            num_output_fmaps = tensor.size(0)
            receptive_field_size = 1
            if tensor.dim() > 2:
                receptive_field_size = tensor[0][0].numel()
            fan_in = num_input_fmaps * receptive_field_size
            fan_out = num_output_fmaps * receptive_field_size
        
        return fan_in, fan_out

# Example usage
def demonstrate_he_initialization():
    # Create a linear layer
    input_size, output_size = 100, 50
    linear_layer = nn.Linear(input_size, output_size)
    
    # Apply He initialization
    HeInitializer.normal(linear_layer.weight)
    HeInitializer.uniform(linear_layer.bias)
    
    # Check the statistics
    weight_mean = linear_layer.weight.mean().item()
    weight_std = linear_layer.weight.std().item()
    expected_std = np.sqrt(2.0 / input_size)
    
    print(f"He Normal Initialization:")
    print(f"  Weight mean: {weight_mean:.6f} (expected: 0.0)")
    print(f"  Weight std: {weight_std:.6f} (expected: {expected_std:.6f})")
    
    # Test with uniform initialization
    linear_layer_uniform = nn.Linear(input_size, output_size)
    HeInitializer.uniform(linear_layer_uniform.weight)
    
    weight_std_uniform = linear_layer_uniform.weight.std().item()
    expected_std_uniform = np.sqrt(6.0 / input_size) / np.sqrt(3)
    
    print(f"\nHe Uniform Initialization:")
    print(f"  Weight std: {weight_std_uniform:.6f} (expected: {expected_std_uniform:.6f})")

demonstrate_he_initialization()
```

### He Initialization with ReLU Networks

```python
class HeNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(HeNet, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layer = nn.Linear(prev_size, hidden_size)
            # Apply He initialization
            HeInitializer.normal(layer.weight)
            HeInitializer.uniform(layer.bias)
            layers.append(layer)
            layers.append(nn.ReLU())  # ReLU activation
            prev_size = hidden_size
        
        # Output layer
        output_layer = nn.Linear(prev_size, output_size)
        HeInitializer.normal(output_layer.weight)
        HeInitializer.uniform(output_layer.bias)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Test He initialization with ReLU
def test_he_forward_pass():
    input_size, hidden_sizes, output_size = 10, [20, 15], 5
    model = HeNet(input_size, hidden_sizes, output_size)
    
    # Create input data
    batch_size = 32
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean():.6f}")
    print(f"Output std: {output.std():.6f}")
    
    # Check activations at each layer
    activations = []
    x_current = x
    
    for i, layer in enumerate(model.network):
        if isinstance(layer, nn.Linear):
            x_current = layer(x_current)
            activations.append(x_current.detach())
        elif isinstance(layer, nn.ReLU):
            x_current = layer(x_current)
    
    print(f"\nActivation statistics:")
    for i, act in enumerate(activations):
        print(f"  Layer {i+1}: mean={act.mean():.4f}, std={act.std():.4f}")
        print(f"    ReLU sparsity: {(act <= 0).float().mean():.2%}")

test_he_forward_pass()
```

### Comparison: Xavier vs He Initialization

```python
def compare_xavier_vs_he():
    """Compare Xavier and He initialization with different activations"""
    
    input_size, hidden_size, output_size = 100, 50, 10
    batch_size = 32
    
    # Test with Tanh activation
    xavier_tanh = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.Tanh(),
        nn.Linear(hidden_size, output_size)
    )
    
    he_relu = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )
    
    # Apply initializations
    XavierInitializer.normal(xavier_tanh[0].weight)
    XavierInitializer.normal(xavier_tanh[2].weight)
    HeInitializer.normal(he_relu[0].weight)
    HeInitializer.normal(he_relu[2].weight)
    
    # Test with input data
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    xavier_output = xavier_tanh(x)
    he_output = he_relu(x)
    
    print("Xavier + Tanh:")
    print(f"  Output mean: {xavier_output.mean():.4f}")
    print(f"  Output std: {xavier_output.std():.4f}")
    
    print("\nHe + ReLU:")
    print(f"  Output mean: {he_output.mean():.4f}")
    print(f"  Output std: {he_output.std():.4f}")
    
    # Check intermediate activations
    xavier_hidden = xavier_tanh[1](xavier_tanh[0](x))
    he_hidden = he_relu[1](he_relu[0](x))
    
    print(f"\nHidden layer activations:")
    print(f"  Xavier+Tanh: mean={xavier_hidden.mean():.4f}, std={xavier_hidden.std():.4f}")
    print(f"  He+ReLU: mean={he_hidden.mean():.4f}, std={he_hidden.std():.4f}")

compare_xavier_vs_he()
```

## Orthogonal Initialization

Orthogonal initialization initializes weights as orthogonal matrices to preserve gradient flow and prevent vanishing/exploding gradients.

### Mathematical Foundation

Orthogonal matrices have the property that $`W^T W = I`$, which means:
- Eigenvalues have magnitude 1
- Preserves gradient magnitude
- Useful for recurrent networks and transformers

### Mathematical Formulation

```math
W = U \Sigma V^T
```

Where $`U`$ and $`V`$ are orthogonal matrices, and $`\Sigma`$ contains singular values.

### Python Implementation

```python
class OrthogonalInitializer:
    """Custom orthogonal initialization implementation"""
    
    @staticmethod
    def orthogonal(tensor, gain=1.0):
        """Orthogonal initialization"""
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")
        
        rows = tensor.size(0)
        cols = tensor.numel() // rows
        
        flattened = tensor.new(rows, cols).normal_(0, 1)
        
        # Compute QR decomposition
        q, r = torch.qr(flattened)
        
        # Make Q orthogonal
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph.unsqueeze(0)
        
        # Reshape back
        tensor.data = q.view_as(tensor)
        
        # Scale by gain
        tensor.data *= gain
        
        return tensor
    
    @staticmethod
    def orthogonal_rnn(tensor, gain=1.0):
        """Orthogonal initialization for RNN weights"""
        if tensor.ndimension() < 2:
            raise ValueError("Only tensors with 2 or more dimensions are supported")
        
        rows = tensor.size(0)
        cols = tensor.numel() // rows
        
        flattened = tensor.new(rows, cols).normal_(0, 1)
        
        # Compute QR decomposition
        q, r = torch.qr(flattened)
        
        # Make Q orthogonal
        d = torch.diag(r, 0)
        ph = d.sign()
        q *= ph.unsqueeze(0)
        
        # Reshape back
        tensor.data = q.view_as(tensor)
        
        # Scale by gain
        tensor.data *= gain
        
        return tensor

# Example usage
def demonstrate_orthogonal_initialization():
    # Create a weight matrix
    rows, cols = 50, 50
    weight_matrix = torch.empty(rows, cols)
    
    # Apply orthogonal initialization
    OrthogonalInitializer.orthogonal(weight_matrix)
    
    # Check orthogonality
    identity = torch.eye(rows)
    orthogonality_error = torch.norm(weight_matrix @ weight_matrix.T - identity)
    
    print(f"Orthogonal Initialization:")
    print(f"  Matrix shape: {weight_matrix.shape}")
    print(f"  Orthogonality error: {orthogonality_error:.6f}")
    print(f"  Weight mean: {weight_matrix.mean():.6f}")
    print(f"  Weight std: {weight_matrix.std():.6f}")
    
    # Check eigenvalues
    eigenvals = torch.linalg.eigvals(weight_matrix)
    eigenval_magnitudes = torch.abs(eigenvals)
    
    print(f"  Eigenvalue magnitudes - Min: {eigenval_magnitudes.min():.4f}, Max: {eigenval_magnitudes.max():.4f}")

demonstrate_orthogonal_initialization()
```

### Orthogonal Initialization in RNNs

```python
class OrthogonalRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(OrthogonalRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input-to-hidden weights
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        
        # Apply orthogonal initialization
        OrthogonalInitializer.orthogonal(self.weight_ih)
        OrthogonalInitializer.orthogonal(self.weight_hh)
        
        # Initialize biases
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)
    
    def forward(self, x, h0=None):
        # x shape: (seq_len, batch, input_size)
        seq_len, batch_size, _ = x.shape
        
        if h0 is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        h = h0
        
        for t in range(seq_len):
            # RNN step
            h = torch.tanh(
                torch.mm(x[t], self.weight_ih.T) + self.bias_ih +
                torch.mm(h, self.weight_hh.T) + self.bias_hh
            )
            outputs.append(h)
        
        return torch.stack(outputs), h

# Test orthogonal RNN
def test_orthogonal_rnn():
    input_size, hidden_size, seq_len, batch_size = 10, 20, 15, 8
    
    rnn = OrthogonalRNN(input_size, hidden_size)
    x = torch.randn(seq_len, batch_size, input_size)
    
    output, hidden = rnn(x)
    
    print(f"RNN output shape: {output.shape}")
    print(f"Hidden state shape: {hidden.shape}")
    print(f"Output mean: {output.mean():.4f}")
    print(f"Output std: {output.std():.4f}")
    
    # Check gradient flow
    loss = output.sum()
    loss.backward()
    
    print(f"\nGradient statistics:")
    print(f"  weight_ih grad std: {rnn.weight_ih.grad.std():.4f}")
    print(f"  weight_hh grad std: {rnn.weight_hh.grad.std():.4f}")

test_orthogonal_rnn()
```

## Pre-trained Weights

Pre-trained weights initialize networks with weights from models trained on large datasets, enabling transfer learning.

### Transfer Learning Approaches

1. **Feature Extraction**: Freeze pre-trained layers, train only new layers
2. **Fine-tuning**: Update all layers with smaller learning rate
3. **Progressive Unfreezing**: Gradually unfreeze layers during training

### Python Implementation

```python
import torchvision.models as models
from torchvision import transforms

class TransferLearningModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained ResNet
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace final layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
        
        # Initialize new layer
        nn.init.xavier_normal_(self.backbone.fc.weight)
        nn.init.zeros_(self.backbone.fc.bias)
    
    def forward(self, x):
        return self.backbone(x)
    
    def unfreeze_backbone(self, learning_rate_factor=0.1):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True
            param.data *= learning_rate_factor

# Example usage
def demonstrate_transfer_learning():
    # Create model with pre-trained weights
    model = TransferLearningModel(num_classes=10, pretrained=True, freeze_backbone=True)
    
    # Create dummy data
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    
    print(f"Transfer learning model:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Output mean: {output.mean():.4f}")
    print(f"  Output std: {output.std():.4f}")
    
    # Check which parameters are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.1%})")
    
    # Unfreeze backbone for fine-tuning
    model.unfreeze_backbone()
    
    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters after unfreezing: {trainable_params_after:,} / {total_params:,} ({trainable_params_after/total_params:.1%})")

demonstrate_transfer_learning()
```

### Progressive Unfreezing

```python
class ProgressiveUnfreezingTrainer:
    def __init__(self, model, num_epochs_per_stage=5):
        self.model = model
        self.num_epochs_per_stage = num_epochs_per_stage
        self.current_stage = 0
        
    def get_trainable_parameters(self):
        """Get parameters that should be trained at current stage"""
        if self.current_stage == 0:
            # Only train the final layer
            return [p for name, p in self.model.named_parameters() if 'fc' in name]
        elif self.current_stage == 1:
            # Train last few layers
            return [p for name, p in self.model.named_parameters() 
                   if any(layer in name for layer in ['layer4', 'fc'])]
        else:
            # Train all parameters
            return list(self.model.parameters())
    
    def advance_stage(self):
        """Advance to next training stage"""
        self.current_stage += 1
        print(f"Advancing to training stage {self.current_stage}")
        
        # Unfreeze more layers
        if self.current_stage == 1:
            for name, param in self.model.named_parameters():
                if 'layer4' in name:
                    param.requires_grad = True
        elif self.current_stage == 2:
            for param in self.model.parameters():
                param.requires_grad = True

# Example usage
def demonstrate_progressive_unfreezing():
    model = TransferLearningModel(num_classes=10, pretrained=True, freeze_backbone=True)
    trainer = ProgressiveUnfreezingTrainer(model)
    
    # Simulate training stages
    for stage in range(3):
        trainable_params = trainer.get_trainable_parameters()
        num_trainable = sum(p.numel() for p in trainable_params)
        total_params = sum(p.numel() for p in model.parameters())
        
        print(f"Stage {stage}: {num_trainable:,} / {total_params:,} parameters trainable ({num_trainable/total_params:.1%})")
        
        if stage < 2:
            trainer.advance_stage()

demonstrate_progressive_unfreezing()
```

## Advanced Initialization Techniques

### LSUV (Layer-Sequential Unit-Variance) Initialization

```python
class LSUVInitializer:
    """Layer-Sequential Unit-Variance initialization"""
    
    @staticmethod
    def initialize_layer(layer, data, target_std=1.0, max_iter=10, tolerance=1e-3):
        """Initialize a layer to have unit variance output"""
        
        # Forward pass to get output statistics
        with torch.no_grad():
            output = layer(data)
            current_std = output.std()
            
            # Iteratively adjust weights
            for iteration in range(max_iter):
                if abs(current_std - target_std) < tolerance:
                    break
                
                # Adjust weights
                layer.weight.data *= target_std / current_std
                
                # Recompute output
                output = layer(data)
                current_std = output.std()
        
        return current_std

# Example usage
def demonstrate_lsuv():
    # Create a layer
    layer = nn.Linear(100, 50)
    
    # Create input data
    batch_size = 32
    x = torch.randn(batch_size, 100)
    
    # Apply LSUV initialization
    final_std = LSUVInitializer.initialize_layer(layer, x)
    
    print(f"LSUV Initialization:")
    print(f"  Final output std: {final_std:.6f}")
    print(f"  Target std: 1.0")
    print(f"  Weight std: {layer.weight.std():.6f}")

demonstrate_lsuv()
```

### Kaiming Initialization Variants

```python
class KaimingInitializer:
    """Kaiming initialization variants"""
    
    @staticmethod
    def normal(tensor, mode='fan_in', nonlinearity='relu'):
        """Kaiming normal initialization"""
        return nn.init.kaiming_normal_(tensor, mode=mode, nonlinearity=nonlinearity)
    
    @staticmethod
    def uniform(tensor, mode='fan_in', nonlinearity='relu'):
        """Kaiming uniform initialization"""
        return nn.init.kaiming_uniform_(tensor, mode=mode, nonlinearity=nonlinearity)

# Example usage
def demonstrate_kaiming():
    # Create layers
    linear_relu = nn.Linear(100, 50)
    linear_tanh = nn.Linear(100, 50)
    
    # Apply Kaiming initialization
    KaimingInitializer.normal(linear_relu.weight, nonlinearity='relu')
    KaimingInitializer.normal(linear_tanh.weight, nonlinearity='tanh')
    
    print(f"Kaiming Initialization:")
    print(f"  ReLU layer weight std: {linear_relu.weight.std():.6f}")
    print(f"  Tanh layer weight std: {linear_tanh.weight.std():.6f}")

demonstrate_kaiming()
```

## Practical Guidelines

### Choosing Initialization Strategy

```python
def choose_initialization_strategy():
    """Guidelines for choosing initialization strategies"""
    
    guidelines = {
        'Xavier/Glorot': {
            'use_for': ['Sigmoid', 'Tanh'],
            'formula': 'sqrt(2 / (fan_in + fan_out))',
            'advantages': ['Maintains variance across layers', 'Good for sigmoid/tanh'],
            'disadvantages': ['Not optimal for ReLU']
        },
        'He': {
            'use_for': ['ReLU', 'Leaky ReLU', 'PReLU'],
            'formula': 'sqrt(2 / fan_in)',
            'advantages': ['Accounts for ReLU sparsity', 'Good for deep networks'],
            'disadvantages': ['May be too large for shallow networks']
        },
        'Orthogonal': {
            'use_for': ['RNNs', 'LSTMs', 'Transformers'],
            'formula': 'QR decomposition',
            'advantages': ['Preserves gradient magnitude', 'Good for recurrent networks'],
            'disadvantages': ['Computationally expensive', 'Not always necessary']
        },
        'Pre-trained': {
            'use_for': ['Transfer learning', 'Limited data'],
            'formula': 'Load from pre-trained model',
            'advantages': ['Faster convergence', 'Better performance'],
            'disadvantages': ['Requires pre-trained model', 'Domain mismatch']
        }
    }
    
    return guidelines

# Example usage
def demonstrate_guidelines():
    guidelines = choose_initialization_strategy()
    
    for strategy, info in guidelines.items():
        print(f"\n{strategy}:")
        print(f"  Use for: {', '.join(info['use_for'])}")
        print(f"  Formula: {info['formula']}")
        print(f"  Advantages: {', '.join(info['advantages'])}")
        print(f"  Disadvantages: {', '.join(info['disadvantages'])}")

demonstrate_guidelines()
```

### Initialization Comparison

```python
def compare_initializations():
    """Compare different initialization strategies"""
    
    input_size, hidden_size, output_size = 100, 50, 10
    batch_size = 32
    
    # Create models with different initializations
    models = {
        'Xavier': nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        ),
        'He': nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ),
        'Orthogonal': nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )
    }
    
    # Apply initializations
    XavierInitializer.normal(models['Xavier'][0].weight)
    XavierInitializer.normal(models['Xavier'][2].weight)
    
    HeInitializer.normal(models['He'][0].weight)
    HeInitializer.normal(models['He'][2].weight)
    
    OrthogonalInitializer.orthogonal(models['Orthogonal'][0].weight)
    OrthogonalInitializer.orthogonal(models['Orthogonal'][2].weight)
    
    # Test with input data
    x = torch.randn(batch_size, input_size)
    
    results = {}
    for name, model in models.items():
        output = model(x)
        results[name] = {
            'mean': output.mean().item(),
            'std': output.std().item(),
            'max': output.max().item(),
            'min': output.min().item()
        }
    
    print("Initialization Comparison:")
    for name, stats in results.items():
        print(f"\n{name}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std: {stats['std']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

compare_initializations()
```

### Initialization Best Practices

```python
def initialization_best_practices():
    """Best practices for weight initialization"""
    
    practices = {
        'Choose based on activation': {
            'Sigmoid/Tanh': 'Use Xavier/Glorot',
            'ReLU/Leaky ReLU': 'Use He initialization',
            'Linear': 'Use Xavier/Glorot'
        },
        'Consider network depth': {
            'Shallow networks': 'Xavier/He work well',
            'Deep networks': 'He initialization preferred',
            'Very deep networks': 'Consider orthogonal + careful tuning'
        },
        'Handle different layer types': {
            'Convolutional layers': 'Use He initialization',
            'Recurrent layers': 'Use orthogonal initialization',
            'Attention layers': 'Use Xavier or orthogonal'
        },
        'Transfer learning': {
            'Feature extraction': 'Freeze pre-trained, random init for new layers',
            'Fine-tuning': 'Use smaller learning rate for pre-trained layers',
            'Progressive unfreezing': 'Start with frozen, gradually unfreeze'
        }
    }
    
    return practices

# Example usage
def demonstrate_best_practices():
    practices = initialization_best_practices()
    
    for category, recommendations in practices.items():
        print(f"\n{category}:")
        for situation, recommendation in recommendations.items():
            print(f"  {situation}: {recommendation}")

demonstrate_best_practices()
```

## Summary

Weight initialization is a critical component of training deep neural networks:

1. **Xavier/Glorot**: Best for sigmoid and tanh activations, maintains variance across layers
2. **He**: Optimized for ReLU activations, accounts for sparsity
3. **Orthogonal**: Preserves gradient magnitude, useful for recurrent networks
4. **Pre-trained**: Enables transfer learning, faster convergence

Key considerations:
- **Activation function**: Choose initialization based on the activation function
- **Network depth**: Deeper networks may require more careful initialization
- **Architecture**: Different architectures benefit from different strategies
- **Data characteristics**: Consider the scale and distribution of input data

Proper initialization can significantly improve training stability, convergence speed, and final model performance. 