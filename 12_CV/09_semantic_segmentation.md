# Semantic Segmentation

## 1. Overview

Semantic segmentation assigns a class label to each pixel in an image, providing dense pixel-level understanding of the scene.

**Mathematical Definition:**
```math
S: \mathbb{R}^{H \times W \times C} \rightarrow \{1, 2, ..., K\}^{H \times W}
```

Where $K$ is the number of classes.

## 2. Fully Convolutional Networks (FCN)

### Architecture
**Encoder-Decoder Structure:**
```math
F = \text{Encoder}(I) \in \mathbb{R}^{H/32 \times W/32 \times C}
```
```math
S = \text{Decoder}(F) \in \mathbb{R}^{H \times W \times K}
```

### Skip Connections
**Feature Fusion:**
```math
F_{out} = \text{Conv}(F_{high}) + \text{Upsample}(F_{low})
```

## 3. U-Net Architecture

### Encoder Path
**Contraction:**
```math
F_{i+1} = \text{Down}(F_i) = \text{MaxPool}(\text{ConvBlock}(F_i))
```

### Decoder Path
**Expansion:**
```math
F_{i-1} = \text{Up}(F_i) = \text{ConvBlock}(\text{Concat}(\text{Upsample}(F_i), F_{skip}))
```

### Skip Connections
**Concatenation:**
```math
F_{concat} = [F_{encoder}, F_{decoder}]
```

## 4. DeepLab Family

### DeepLab v1
**Atrous Convolution:**
```math
y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]
```

### DeepLab v2
**Atrous Spatial Pyramid Pooling (ASPP):**
```math
F_{ASPP} = \text{Concat}(F_{rate=6}, F_{rate=12}, F_{rate=18}, F_{rate=24})
```

### DeepLab v3+
**Encoder-Decoder with ASPP:**
```math
F_{encoder} = \text{ASPP}(\text{ResNet}(I))
```
```math
F_{decoder} = \text{Decoder}(F_{encoder}, F_{low})
```

## 5. Loss Functions

### Cross-Entropy Loss
```math
L_{CE} = -\sum_{i=1}^{H \times W} \sum_{c=1}^{K} y_{i,c} \log(\hat{y}_{i,c})
```

### Dice Loss
```math
L_{Dice} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i}{\sum_{i} y_i + \sum_{i} \hat{y}_i}
```

### Focal Loss
```math
L_{Focal} = -\sum_{i} (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i)
```

## 6. Evaluation Metrics

### IoU (Intersection over Union)
```math
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
```

### mIoU (mean IoU)
```math
\text{mIoU} = \frac{1}{K} \sum_{c=1}^{K} \text{IoU}_c
```

### Pixel Accuracy
```math
\text{Accuracy} = \frac{\sum_{i} \mathbb{1}[y_i = \hat{y}_i]}{H \times W}
```

## 7. Python Implementation

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create synthetic dataset
def create_synthetic_segmentation_data(num_samples=100, image_size=(128, 128), num_classes=3):
    images = []
    masks = []
    
    for _ in range(num_samples):
        # Create random image
        image = np.random.rand(*image_size, 3)
        
        # Create segmentation mask
        mask = np.zeros(image_size, dtype=np.uint8)
        
        # Add random shapes
        for _ in range(np.random.randint(2, 6)):
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
            class_id = np.random.randint(1, num_classes)
            
            if shape_type == 'circle':
                center = (np.random.randint(20, image_size[0]-20), 
                         np.random.randint(20, image_size[1]-20))
                radius = np.random.randint(10, 30)
                y, x = np.ogrid[:image_size[0], :image_size[1]]
                circle_mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
                mask[circle_mask] = class_id
                
            elif shape_type == 'rectangle':
                x1, y1 = np.random.randint(10, image_size[0]-40), np.random.randint(10, image_size[1]-40)
                x2, y2 = x1 + np.random.randint(20, 40), y1 + np.random.randint(20, 40)
                mask[y1:y2, x1:x2] = class_id
        
        images.append(image)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# U-Net implementation
class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(96, 32)
        
        self.final = nn.Conv2d(32, num_classes, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(nn.MaxPool2d(2)(enc1))
        enc3 = self.enc3(nn.MaxPool2d(2)(enc2))
        enc4 = self.enc4(nn.MaxPool2d(2)(enc3))
        
        # Decoder with skip connections
        dec4 = self.up4(enc4)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.up3(dec4)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.up2(dec3)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.up1(dec2)
        dec1 = torch.cat([dec1, x], dim=1)
        dec1 = self.dec1(dec1)
        
        return self.final(dec1)

# Training function
def train_segmentation_model(model, train_loader, num_epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {total_loss/len(train_loader):.4f}')
    
    return model

# Evaluation metrics
def calculate_iou(pred_mask, true_mask, num_classes):
    ious = []
    for class_id in range(num_classes):
        pred_binary = (pred_mask == class_id)
        true_binary = (true_mask == class_id)
        
        intersection = np.logical_and(pred_binary, true_binary).sum()
        union = np.logical_or(pred_binary, true_binary).sum()
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    return np.mean(ious)

# Main demonstration
def demonstrate_semantic_segmentation():
    # Create dataset
    images, masks = create_synthetic_segmentation_data(200, (64, 64), 3)
    
    # Convert to PyTorch tensors
    images = torch.FloatTensor(images.transpose(0, 3, 1, 2))
    masks = torch.LongTensor(masks)
    
    # Create data loader
    dataset = torch.utils.data.TensorDataset(images, masks)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Train model
    model = UNet(num_classes=3)
    model = train_segmentation_model(model, train_loader, num_epochs=30)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        test_image = images[0:1]
        test_mask = masks[0:1]
        
        output = model(test_image)
        pred_mask = torch.argmax(output, dim=1)[0].numpy()
        true_mask = test_mask[0].numpy()
        
        iou = calculate_iou(pred_mask, true_mask, 3)
        print(f"Test IoU: {iou:.4f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(images[0].permute(1, 2, 0).numpy())
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask, cmap='tab10')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='tab10')
    axes[2].set_title(f'Prediction (IoU: {iou:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demonstrate_semantic_segmentation()
```

This guide covers semantic segmentation fundamentals with practical implementations of U-Net architecture and evaluation metrics. 