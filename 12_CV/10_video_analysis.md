# Video Analysis in Computer Vision

## 1. Overview

Video analysis is a branch of computer vision focused on understanding and interpreting visual information from video sequences. Unlike static images, videos provide temporal information, enabling the analysis of motion, actions, and object trajectories over time.

**Key Tasks:**
- Action Recognition
- Video Object Detection
- Video Tracking
- Temporal Modeling

---

## 2. Action Recognition

### 2.1. Problem Definition
Given a video $V$ (a sequence of frames $I_1, I_2, ..., I_T$), the goal is to assign an action label $y$:

```math
f: (I_1, I_2, ..., I_T) \rightarrow y
```

### 2.2. 3D Convolutional Neural Networks (3D CNNs)
3D CNNs extend 2D convolutions to the temporal dimension, capturing spatiotemporal features.

**3D Convolution:**
```math
Y(i, j, k) = \sum_{m} \sum_{n} \sum_{l} X(i+m, j+n, k+l) \cdot W(m, n, l)
```
Where $X$ is the input video volume, $W$ is the 3D kernel.

**Example (PyTorch):**
```python
import torch
import torch.nn as nn

class Simple3DCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.fc = nn.Linear(16*8*8*8, num_classes)
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### 2.3. Two-Stream Networks
Two-stream networks process RGB frames and optical flow separately, then fuse the results.

- **Spatial Stream:** Processes appearance (RGB)
- **Temporal Stream:** Processes motion (optical flow)

**Fusion:**
```math
f_{final} = \alpha f_{spatial} + (1-\alpha) f_{temporal}
```

### 2.4. I3D (Inflated 3D ConvNet)
I3D inflates 2D CNN filters into 3D, leveraging pre-trained 2D models for video.

---

## 3. Video Object Detection

### 3.1. Temporal Consistency
Object detection in videos must ensure consistent predictions across frames.

- **Tubelets:** Sequences of bounding boxes linked across frames.
- **Tracklets:** Short object tracks for linking detections.

### 3.2. Video Tracking
Tracking objects across frames involves associating detections over time.

**Tracking-by-Detection:**
1. Detect objects in each frame
2. Associate detections using motion and appearance cues

**Hungarian Algorithm** is often used for assignment.

**IoU for Association:**
```math
IoU = \frac{Area(B_{t} \cap B_{t+1})}{Area(B_{t} \cup B_{t+1})}
```

**Simple Tracker Example:**
```python
import numpy as np

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou
```

### 3.3. Temporal Modeling
Temporal models (e.g., RNNs, LSTMs, Transformers) capture dependencies across frames.

**RNN for Sequence Modeling:**
```math
h_t = \sigma(W_x x_t + W_h h_{t-1} + b)
```

**Transformer Attention:**
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

---

## 4. Evaluation Metrics

- **Top-1/Top-5 Accuracy** (for action recognition)
- **mAP (mean Average Precision)** (for detection)
- **MOTA/MOTP** (Multiple Object Tracking Accuracy/Precision)
- **IDF1** (ID F1-score for tracking)

---

## 5. Practical Example: Video Action Recognition Pipeline

```python
# Pseudocode for a video action recognition pipeline
import torch
import torchvision
from torchvision import transforms

# Load video frames (as tensor of shape [T, C, H, W])
frames = torch.randn(16, 3, 64, 64)  # Example: 16 frames
frames = frames.unsqueeze(0)  # Add batch dimension [1, 16, 3, 64, 64]
frames = frames.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]

model = Simple3DCNN(num_classes=10)
outputs = model(frames)
print(outputs.shape)  # [1, 10]
```

---

## 6. References
- Karpathy et al., "Large-scale Video Classification with Convolutional Neural Networks," CVPR 2014
- Simonyan & Zisserman, "Two-Stream Convolutional Networks for Action Recognition in Videos," NIPS 2014
- Carreira & Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset," CVPR 2017 (I3D)
- Wu et al., "Tracklet Association by Online Target-Specific Metric Learning and Coherent Dynamics Estimation," TPAMI 2013
- Milan et al., "MOT16: A Benchmark for Multi-Object Tracking," arXiv 2016 