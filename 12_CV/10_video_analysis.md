# Video Analysis in Computer Vision

> **Key Insight:** Video analysis leverages both spatial and temporal information, enabling understanding of motion, actions, and object interactions over time—crucial for applications like surveillance, sports analytics, and autonomous vehicles.

## 1. Overview

Video analysis is a branch of computer vision focused on understanding and interpreting visual information from video sequences. Unlike static images, videos provide temporal information, enabling the analysis of motion, actions, and object trajectories over time.

**Key Tasks:**
- Action Recognition
- Video Object Detection
- Video Tracking
- Temporal Modeling

> **Did you know?** Video data is often orders of magnitude larger than image data, making efficient modeling and storage a key challenge.

---

## 2. Action Recognition

### 2.1. Problem Definition
Given a video $`V`$ (a sequence of frames $`I_1, I_2, ..., I_T`$), the goal is to assign an action label $`y`$:

```math
f: (I_1, I_2, ..., I_T) \rightarrow y
```

- $`I_t`$: The $`t`$-th frame in the video
- $`y`$: Action class (e.g., "jumping", "walking")

### 2.2. 3D Convolutional Neural Networks (3D CNNs)
3D CNNs extend 2D convolutions to the temporal dimension, capturing spatiotemporal features.

$`\text{3D Convolution:}`$
```math
Y(i, j, k) = \sum_{m} \sum_{n} \sum_{l} X(i+m, j+n, k+l) \cdot W(m, n, l)
```
Where $`X`$ is the input video volume, $`W`$ is the 3D kernel.

> **Geometric Intuition:** 3D convolutions slide a 3D kernel across both space and time, learning motion patterns as well as spatial features.

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
> **Code Commentary:**
> - `Conv3d` operates on [batch, channels, time, height, width].
> - Pooling reduces spatial and temporal resolution.
> - The final fully connected layer outputs class scores.

### 2.3. Two-Stream Networks
Two-stream networks process RGB frames and optical flow separately, then fuse the results.

- **Spatial Stream:** Processes appearance (RGB)
- **Temporal Stream:** Processes motion (optical flow)

$`\text{Fusion:}`$
```math
f_{final} = \alpha f_{spatial} + (1-\alpha) f_{temporal}
```

> **Common Pitfall:** Optical flow computation can be noisy and computationally expensive. Pre-compute and cache flows for efficiency.

### 2.4. I3D (Inflated 3D ConvNet)
I3D inflates 2D CNN filters into 3D, leveraging pre-trained 2D models for video. This allows transfer learning from large image datasets.

> **Try it yourself!** Take a pre-trained 2D ResNet and "inflate" its kernels to 3D for video tasks.

---

## 3. Video Object Detection & Tracking

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

$`\text{IoU for Association:}`$
```math
\text{IoU} = \frac{Area(B_{t} \cap B_{t+1})}{Area(B_{t} \cup B_{t+1})}
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
> **Code Commentary:**
> - Computes intersection-over-union (IoU) between two bounding boxes.
> - Used for associating detections across frames.

### 3.3. Temporal Modeling
Temporal models (e.g., RNNs, LSTMs, Transformers) capture dependencies across frames.

$`\text{RNN for Sequence Modeling:}`$
```math
h_t = \sigma(W_x x_t + W_h h_{t-1} + b)
```

$`\text{Transformer Attention:}`$
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```

> **Key Insight:** Transformers can model long-range dependencies in video, outperforming RNNs on many benchmarks.

---

## 4. Evaluation Metrics

- **Top-1/Top-5 Accuracy** (for action recognition)
- **mAP (mean Average Precision)** (for detection)
- **MOTA/MOTP** (Multiple Object Tracking Accuracy/Precision)
- **IDF1** (ID F1-score for tracking)

> **Did you know?** MOTA penalizes false positives, missed targets, and identity switches, making it a comprehensive tracking metric.

---

## 5. Practical Example: Video Action Recognition Pipeline

Below is a pseudocode pipeline for video action recognition using a 3D CNN:

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
> **Try it yourself!** Change the number of frames or input size and observe the effect on model output.

---

## 6. Summary Table

| Task                  | Key Model/Method         | Typical Metric         |
|-----------------------|-------------------------|-----------------------|
| Action Recognition    | 3D CNN, Two-Stream, I3D | Top-1/Top-5 Accuracy  |
| Video Detection       | Tubelets, Tracklets     | mAP                   |
| Video Tracking        | RNN, Transformer        | MOTA, MOTP, IDF1      |

---

## 7. Conceptual Connections

- **Image Classification:** Frame-level, no temporal modeling.
- **Object Detection:** Per-frame, no association across time.
- **Action Recognition:** Requires both spatial and temporal context.

---

## 8. Actionable Next Steps

- Try training a 3D CNN on a small video dataset (e.g., UCF101).
- Experiment with optical flow as input to a two-stream network.
- Visualize feature maps across time to build intuition.

---

> **Summary:**
> Video analysis combines spatial and temporal reasoning. Modern models (3D CNNs, Transformers) and metrics (mAP, MOTA) enable robust understanding of dynamic scenes. Practice, experiment, and visualize to master this topic!

---

## 9. References
- Karpathy et al., "Large-scale Video Classification with Convolutional Neural Networks," CVPR 2014
- Simonyan & Zisserman, "Two-Stream Convolutional Networks for Action Recognition in Videos," NIPS 2014
- Carreira & Zisserman, "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset," CVPR 2017 (I3D)
- Wu et al., "Tracklet Association by Online Target-Specific Metric Learning and Coherent Dynamics Estimation," TPAMI 2013
- Milan et al., "MOT16: A Benchmark for Multi-Object Tracking," arXiv 2016 