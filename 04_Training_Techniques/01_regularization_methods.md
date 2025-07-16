# Regularization Methods

Regularization techniques are essential for preventing overfitting and improving the generalization ability of neural networks. This guide covers the most important regularization methods with detailed explanations, mathematical formulations, and practical Python implementations.

## Table of Contents

1. [Dropout](#dropout)
2. [Weight Decay (L2 Regularization)](#weight-decay-l2-regularization)
3. [Early Stopping](#early-stopping)
4. [Data Augmentation](#data-augmentation)

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
1. **Prevents co-adaptation**: Neurons cannot rely on specific combinations of other neurons
2. **Forces redundancy**: The network learns multiple representations for the same features
3. **Improves generalization**: The network becomes more robust to missing inputs

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

1. **Dropout Rates**: 
   - Input layers: 0.2-0.3
   - Hidden layers: 0.3-0.5
   - Output layers: Usually not applied

2. **Placement**: Apply dropout after activation functions

3. **Combination**: Often used with other regularization techniques

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
1. **Penalizing large weights**: Large weights increase the regularization term
2. **Encouraging smaller weights**: Smaller weights lead to smoother decision boundaries
3. **Preventing overfitting**: Reduces model complexity

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
        
    def step(self):
        """Manual weight update with L2 regularization"""
        with torch.no_grad():
            for param in self.model.parameters():
                # Apply weight decay
                param.data -= self.lr * self.weight_decay * param.data
                
    def zero_grad(self):
        """Zero gradients"""
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.zero_()

demonstrate_weight_decay()
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

## Early Stopping

Early stopping monitors validation performance and stops training when overfitting begins.

### Mathematical Intuition

Early stopping works by:
1. **Monitoring validation loss**: $`L_{\text{val}}(\theta_t)`$ at epoch $`t`$
2. **Tracking best performance**: $`L_{\text{best}} = \min_{i \leq t} L_{\text{val}}(\theta_i)`$
3. **Stopping criterion**: Stop when $`L_{\text{val}}(\theta_t) > L_{\text{best}} + \epsilon`$ for $`k`$ consecutive epochs

### Python Implementation

```python
import copy
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())

# Training loop with early stopping
def train_with_early_stopping(model, train_loader, val_loader, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    early_stopping = EarlyStopping(patience=10)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}')
        
        # Check early stopping
        if early_stopping(val_losses[-1], model):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

# Visualization of early stopping
def plot_early_stopping_results(train_losses, val_losses):
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss with Early Stopping')
    plt.legend()
    plt.grid(True)
    plt.show()
```

### Advanced Early Stopping Strategies

#### 1. Learning Rate Scheduling with Early Stopping

```python
class EarlyStoppingWithLR:
    def __init__(self, patience=7, min_delta=0, lr_factor=0.5):
        self.patience = patience
        self.min_delta = min_delta
        self.lr_factor = lr_factor
        self.best_loss = None
        self.counter = 0
        
    def __call__(self, val_loss, optimizer):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.counter = 0
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            # Reduce learning rate
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.lr_factor
            self.counter = 0
            print(f"Reducing learning rate to {optimizer.param_groups[0]['lr']}")
            
        return False
```

#### 2. Multiple Metric Early Stopping

```python
class MultiMetricEarlyStopping:
    def __init__(self, patience=7, metrics=['loss'], mode='min'):
        self.patience = patience
        self.metrics = metrics
        self.mode = mode
        self.best_scores = {metric: None for metric in metrics}
        self.counter = 0
        
    def __call__(self, val_metrics, model):
        should_stop = True
        
        for metric in self.metrics:
            if metric not in val_metrics:
                continue
                
            current_score = val_metrics[metric]
            
            if self.best_scores[metric] is None:
                self.best_scores[metric] = current_score
                should_stop = False
            elif self.mode == 'min':
                if current_score < self.best_scores[metric]:
                    self.best_scores[metric] = current_score
                    should_stop = False
            else:  # mode == 'max'
                if current_score > self.best_scores[metric]:
                    self.best_scores[metric] = current_score
                    should_stop = False
        
        if should_stop:
            self.counter += 1
        else:
            self.counter = 0
            
        return self.counter >= self.patience
```

## Data Augmentation

Data augmentation expands training data through transformations to improve generalization.

### Mathematical Foundation

Data augmentation works by:
1. **Expanding the training distribution**: $`P_{\text{aug}}(x) = \int P(x|t) P(t) dt`$
2. **Improving invariance**: Learning transformations that preserve class labels
3. **Regularization effect**: Reducing overfitting through increased data diversity

### Image Data Augmentation

```python
import torchvision.transforms as transforms
from PIL import Image
import cv2
import albumentations as A

class ImageAugmentation:
    def __init__(self):
        # Basic transforms
        self.basic_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Advanced transforms using Albumentations
        self.advanced_transforms = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Flip(p=0.5),
            A.Transpose(p=0.5),
            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
            ], p=0.2),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            A.OneOf([
                A.OpticalDistortion(p=0.3),
                A.GridDistortion(p=0.1),
                A.IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.HueSaturationValue(p=0.3),
        ])
    
    def apply_basic_augmentation(self, image):
        """Apply basic PyTorch transforms"""
        return self.basic_transforms(image)
    
    def apply_advanced_augmentation(self, image):
        """Apply advanced Albumentations transforms"""
        # Convert PIL to numpy for Albumentations
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        augmented = self.advanced_transforms(image=image)
        return augmented['image']

# Custom augmentation techniques
class CustomAugmentation:
    @staticmethod
    def mixup(x1, x2, y1, y2, alpha=0.2):
        """Mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        x = lam * x1 + (1 - lam) * x2
        y = lam * y1 + (1 - lam) * y2
        
        return x, y
    
    @staticmethod
    def cutmix(x1, x2, y1, y2, alpha=1.0):
        """CutMix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x1.shape[0]
        index = torch.randperm(batch_size)
        
        y_a, y_b = y1, y2[index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(x1.size(), lam)
        x1[:, :, bbx1:bbx2, bby1:bby2] = x2[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x1.size()[-1] * x1.size()[-2]))
        y = lam * y_a + (1 - lam) * y_b
        
        return x1, y

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    
    return bbx1, bby1, bbx2, bby2
```

### Text Data Augmentation

```python
import nltk
from nltk.corpus import wordnet
import random

class TextAugmentation:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def synonym_replacement(self, text, n=1):
        """Replace n words with synonyms"""
        words = nltk.word_tokenize(text)
        n = min(n, len(words))
        
        new_words = words.copy()
        random_word_list = list(set([word for word in words if word.isalpha()]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            synonyms = self.get_synonyms(random_word)
            if len(synonyms) >= 2:
                synonym = random.choice(list(synonyms))
                new_words = [synonym if word == random_word else word for word in new_words]
                num_replaced += 1
            if num_replaced >= n:
                break
        
        return ' '.join(new_words)
    
    def get_synonyms(self, word):
        """Get synonyms for a word"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                synonyms.add(synonym)
        if word in synonyms:
            synonyms.remove(word)
        return list(synonyms)
    
    def random_insertion(self, text, n=1):
        """Insert n random words"""
        words = nltk.word_tokenize(text)
        n = min(n, len(words))
        
        new_words = words.copy()
        for _ in range(n):
            add_word = random.choice(words)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, add_word)
        
        return ' '.join(new_words)
    
    def random_deletion(self, text, p=0.1):
        """Randomly delete words with probability p"""
        words = nltk.word_tokenize(text)
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            rand_int = random.randint(0, len(words) - 1)
            return words[rand_int]
        
        return ' '.join(new_words)
    
    def random_swap(self, text, n=1):
        """Randomly swap n pairs of words"""
        words = nltk.word_tokenize(text)
        n = min(n, len(words) // 2)
        
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)

# Example usage
def demonstrate_text_augmentation():
    text = "The quick brown fox jumps over the lazy dog."
    augmenter = TextAugmentation()
    
    print("Original:", text)
    print("Synonym replacement:", augmenter.synonym_replacement(text, n=2))
    print("Random insertion:", augmenter.random_insertion(text, n=2))
    print("Random deletion:", augmenter.random_deletion(text, p=0.2))
    print("Random swap:", augmenter.random_swap(text, n=2))

demonstrate_text_augmentation()
```

### Audio Data Augmentation

```python
import librosa
import numpy as np

class AudioAugmentation:
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def time_stretch(self, audio, rate_range=(0.8, 1.2)):
        """Time stretching augmentation"""
        rate = np.random.uniform(*rate_range)
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, steps_range=(-4, 4)):
        """Pitch shifting augmentation"""
        steps = np.random.uniform(*steps_range)
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=steps)
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add Gaussian noise"""
        noise = np.random.normal(0, 1, len(audio))
        return audio + noise_factor * noise
    
    def time_shift(self, audio, shift_range=(-0.1, 0.1)):
        """Time shifting augmentation"""
        shift = int(np.random.uniform(*shift_range) * self.sample_rate)
        if shift > 0:
            return np.pad(audio, (shift, 0), mode='constant')
        else:
            return audio[-shift:]
    
    def frequency_mask(self, audio, freq_mask_param=10):
        """Frequency masking for spectrograms"""
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Apply frequency masking
        f = np.random.randint(0, freq_mask_param)
        f_zero = np.random.randint(0, mel_spec_db.shape[0] - f)
        mel_spec_db[f_zero:f_zero + f, :] = 0
        
        return librosa.db_to_power(mel_spec_db)

# Example usage
def demonstrate_audio_augmentation():
    # Load sample audio (you would replace this with your audio file)
    # audio, sr = librosa.load('sample_audio.wav')
    
    # For demonstration, create synthetic audio
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(22050 * duration))
    audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    augmenter = AudioAugmentation()
    
    # Apply augmentations
    stretched = augmenter.time_stretch(audio)
    pitched = augmenter.pitch_shift(audio)
    noisy = augmenter.add_noise(audio)
    shifted = augmenter.time_shift(audio)
    
    print("Audio augmentation techniques applied successfully")
    print(f"Original length: {len(audio)}")
    print(f"Stretched length: {len(stretched)}")
    print(f"Pitched length: {len(pitched)}")
    print(f"Noisy length: {len(noisy)}")
    print(f"Shifted length: {len(shifted)}")

demonstrate_audio_augmentation()
```

### Best Practices for Data Augmentation

1. **Domain-Specific Augmentations**: Choose augmentations that preserve semantic meaning
2. **Validation Set**: Don't apply augmentations to validation data
3. **Combination**: Use multiple augmentation techniques together
4. **Hyperparameter Tuning**: Tune augmentation parameters based on validation performance
5. **Consistency**: Apply same augmentations during training and inference for consistency

### Advanced Augmentation Strategies

#### 1. AutoAugment

```python
class AutoAugment:
    def __init__(self):
        self.policies = [
            [('Posterize', 0.4, 8), ('Rotate', 0.6, 9)],
            [('Solarize', 0.6, 5), ('AutoContrast', 0.6, 5)],
            [('Equalize', 0.8, 8), ('Rotate', 0.4, 9)],
            [('Posterize', 0.6, 7), ('Equalize', 0.4, 6)],
            [('Equalize', 0.4, 7), ('Solarize', 0.2, 4)],
        ]
    
    def apply_policy(self, image, policy):
        """Apply a specific augmentation policy"""
        for operation, probability, magnitude in policy:
            if random.random() < probability:
                image = self.apply_operation(image, operation, magnitude)
        return image
    
    def apply_operation(self, image, operation, magnitude):
        """Apply a single augmentation operation"""
        # Implementation would depend on the specific operation
        # This is a simplified version
        return image
```

#### 2. RandAugment

```python
class RandAugment:
    def __init__(self, n=2, m=10):
        self.n = n  # Number of augmentations to apply
        self.m = m  # Magnitude of augmentations
        self.augment_list = [
            'identity', 'auto_contrast', 'equalize', 'rotate', 'solarize',
            'color', 'posterize', 'contrast', 'brightness', 'sharpness',
            'shear_x', 'shear_y', 'translate_x', 'translate_y'
        ]
    
    def __call__(self, image):
        """Apply random augmentations"""
        ops = np.random.choice(self.augment_list, self.n)
        for op in ops:
            image = self.apply_operation(image, op, self.m)
        return image
    
    def apply_operation(self, image, operation, magnitude):
        """Apply a single operation with given magnitude"""
        # Implementation would depend on the specific operation
        return image
```

## Summary

Regularization methods are essential for training robust neural networks:

1. **Dropout**: Prevents co-adaptation by randomly deactivating neurons
2. **Weight Decay**: Penalizes large weights to encourage simpler models
3. **Early Stopping**: Prevents overfitting by monitoring validation performance
4. **Data Augmentation**: Expands training data through transformations

These techniques can be used individually or in combination to achieve better generalization performance. The choice of regularization method depends on the specific problem, data characteristics, and model architecture. 