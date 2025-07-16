# Semantic Segmentation

> **Key Insight:** Semantic segmentation is the foundation for pixel-level scene understanding, enabling applications like autonomous driving, medical imaging, and robotics.

## 1. Overview

Semantic segmentation assigns a class label to each pixel in an image, providing dense pixel-level understanding of the scene. Unlike image classification (one label per image) or object detection (bounding boxes), semantic segmentation provides a fine-grained map of what is where in the image.

$`\text{Semantic Segmentation:}`$
```math
S: \mathbb{R}^{H \times W \times C} \rightarrow \{1, 2, ..., K\}^{H \times W}
```
Where $`K`$ is the number of classes, $`H`$ and $`W`$ are the image height and width, and $`C`$ is the number of channels (e.g., 3 for RGB).

> **Did you know?** Semantic segmentation does not distinguish between different instances of the same class (see: instance segmentation).

---

## 2. Fully Convolutional Networks (FCN)

### Architecture

FCNs replace fully connected layers with convolutional layers, allowing input images of arbitrary size and producing output maps of the same spatial dimensions.

$`\text{Encoder-Decoder Structure:}`$
```math
F = \text{Encoder}(I) \in \mathbb{R}^{H/32 \times W/32 \times C}
S = \text{Decoder}(F) \in \mathbb{R}^{H \times W \times K}
```

- **Encoder:** Extracts hierarchical features, reducing spatial resolution.
- **Decoder:** Upsamples features to original resolution, producing per-pixel class scores.

### Skip Connections

Skip connections help recover spatial details lost during downsampling by fusing features from earlier layers.

$`\text{Feature Fusion:}`$
```math
F_{out} = \text{Conv}(F_{high}) + \text{Upsample}(F_{low})
```

> **Common Pitfall:** Without skip connections, segmentation maps may be overly coarse and miss fine details.

---

## 3. U-Net Architecture

U-Net is a popular encoder-decoder architecture with symmetric skip connections, originally designed for biomedical image segmentation.

### Encoder Path (Contracting)
$`\text{Downsampling:}`$
```math
F_{i+1} = \text{Down}(F_i) = \text{MaxPool}(\text{ConvBlock}(F_i))
```

### Decoder Path (Expanding)
$`\text{Upsampling:}`$
```math
F_{i-1} = \text{Up}(F_i) = \text{ConvBlock}(\text{Concat}(\text{Upsample}(F_i), F_{skip}))
```

### Skip Connections
$`\text{Concatenation:}`$
```math
F_{concat} = [F_{encoder}, F_{decoder}]
```

> **Key Insight:** U-Net's skip connections allow precise localization by combining low-level spatial and high-level semantic information.

---

## 4. DeepLab Family

DeepLab models introduce atrous (dilated) convolutions and spatial pyramid pooling for multi-scale context aggregation.

### DeepLab v1: Atrous Convolution
$`\text{Atrous convolution allows larger receptive fields without increasing parameters.}`$
```math
y[i] = \sum_{k} x[i + r \cdot k] \cdot w[k]
```
Where $`r`$ is the dilation rate.

### DeepLab v2: Atrous Spatial Pyramid Pooling (ASPP)
$`\text{Multi-scale feature extraction:}`$
```math
F_{ASPP} = \text{Concat}(F_{rate=6}, F_{rate=12}, F_{rate=18}, F_{rate=24})
```

### DeepLab v3+: Encoder-Decoder with ASPP
```math
F_{encoder} = \text{ASPP}(\text{ResNet}(I))
F_{decoder} = \text{Decoder}(F_{encoder}, F_{low})
```

> **Try it yourself!** Experiment with different dilation rates in atrous convolutions to see their effect on segmentation quality.

---

## 5. Loss Functions

### Cross-Entropy Loss
$`\text{Standard for multi-class segmentation:}`$
```math
L_{CE} = -\sum_{i=1}^{H \times W} \sum_{c=1}^{K} y_{i,c} \log(\hat{y}_{i,c})
```

### Dice Loss
$`\text{Balances class imbalance, especially in medical images:}`$
```math
L_{Dice} = 1 - \frac{2 \sum_{i} y_i \hat{y}_i}{\sum_{i} y_i + \sum_{i} \hat{y}_i}
```

### Focal Loss
$`\text{Focuses on hard-to-classify pixels:}`$
```math
L_{Focal} = -\sum_{i} (1 - \hat{y}_i)^\gamma y_i \log(\hat{y}_i)
```

> **Common Pitfall:** Using only cross-entropy loss can lead to poor performance on imbalanced datasets. Consider combining with Dice or Focal loss.

---

## 6. Evaluation Metrics

### IoU (Intersection over Union)
$`\text{Measures overlap between prediction and ground truth:}`$
```math
\text{IoU} = \frac{|A \cap B|}{|A \cup B|}
```

### mIoU (mean IoU)
$`\text{Average IoU across all classes:}`$
```math
\text{mIoU} = \frac{1}{K} \sum_{c=1}^{K} \text{IoU}_c
```

### Pixel Accuracy
$`\text{Fraction of correctly classified pixels:}`$
```math
\text{Accuracy} = \frac{\sum_{i} \mathbb{1}[y_i = \hat{y}_i]}{H \times W}
```

> **Did you know?** mIoU is the standard metric for segmentation challenges like PASCAL VOC and Cityscapes.

---

## 7. Python Implementation

Below is a simplified U-Net implementation for semantic segmentation, with detailed commentary:

```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Create synthetic dataset
# Key Insight: Synthetic data lets us test segmentation pipelines without real labels.
def create_synthetic_segmentation_data(num_samples=100, image_size=(128, 128), num_classes=3):
    images = []
    masks = []
    for _ in range(num_samples):
        image = np.random.rand(*image_size, 3)  # Random RGB image
        mask = np.zeros(image_size, dtype=np.uint8)
        for _ in range(np.random.randint(2, 6)):
            shape_type = np.random.choice(['circle', 'rectangle', 'triangle'])
            class_id = np.random.randint(1, num_classes)
            if shape_type == 'circle':
                center = (np.random.randint(20, image_size[0]-20), np.random.randint(20, image_size[1]-20))
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

# U-Net implementation with commentary
class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        # Encoder: Downsampling path
        self.enc1 = self.conv_block(3, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        # Decoder: Upsampling path
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
        # Two 3x3 conv layers with ReLU
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
        dec1 = torch.cat([dec1, x], dim=1)  # Not standard, but for demonstration
        dec1 = self.dec1(dec1)
        return self.final(dec1)

# Training function with commentary
# Common Pitfall: Overfitting on small synthetic datasets. Use validation and data augmentation in practice.
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

# Evaluation metrics with commentary
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
# Try it yourself! Change the number of classes or image size and observe the effect on segmentation quality.
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

---

## 8. Summary Table

| Model         | Key Feature                | Typical Use Case         |
|---------------|---------------------------|-------------------------|
| FCN           | Fully convolutional        | General segmentation    |
| U-Net         | Symmetric skip connections| Medical, small datasets |
| DeepLab v3+   | Atrous conv, ASPP         | Large-scale, multi-scale|

---

## 9. Conceptual Connections

- **Instance Segmentation:** Distinguishes object instances, not just classes.
- **Object Detection:** Bounding boxes, not pixel-level.
- **Image Classification:** One label per image.

---

## 10. Actionable Next Steps

- Try training on a real dataset (e.g., Pascal VOC, Cityscapes).
- Experiment with different loss functions and metrics.
- Visualize intermediate feature maps to build intuition.

---

> **Summary:**
> Semantic segmentation is a core computer vision task, requiring both local and global context. Modern architectures (U-Net, DeepLab) and loss functions (Dice, Focal) enable robust pixel-level predictions. Practice, experiment, and visualize to master this topic! 