# Video Analysis in Computer Vision

> **Key Insight:** Video analysis leverages both spatial and temporal information, enabling understanding of motion, actions, and object interactions over time—crucial for applications like surveillance, sports analytics, and autonomous vehicles.

## 1. Overview

Video analysis is a branch of computer vision focused on understanding and interpreting visual information from video sequences. Unlike static images, videos provide temporal information, enabling the analysis of motion, actions, and object trajectories over time.

> **Explanation:**
> Video analysis is like watching a movie and understanding not just what's in each frame, but how things move and change over time. While image analysis focuses on single moments, video analysis captures the dynamics - who's doing what, how objects move, and how actions unfold. This temporal dimension makes video analysis both more powerful and more challenging than image analysis.

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

> **Explanation:**
> Action recognition is about understanding what people or objects are doing in a video. Instead of just recognizing objects (like "person" or "ball"), action recognition identifies activities (like "running," "throwing," or "dancing"). This requires understanding both the spatial appearance and the temporal motion patterns.

```math
f: (I_1, I_2, ..., I_T) \rightarrow y
```
> **Math Breakdown:**
> - $f$: Action recognition function (the neural network).
> - $(I_1, I_2, ..., I_T)$: Sequence of $T$ video frames.
> - $I_t$: The $t$-th frame in the video.
> - $y$: Action class label (e.g., "jumping", "walking", "cooking").
> - The function maps a temporal sequence to a discrete action class.
> - This is a sequence-to-class classification problem.

- $`I_t`$: The $`t`$-th frame in the video
- $`y`$: Action class (e.g., "jumping", "walking")

### 2.2. 3D Convolutional Neural Networks (3D CNNs)
3D CNNs extend 2D convolutions to the temporal dimension, capturing spatiotemporal features.

> **Explanation:**
> 3D CNNs are like 2D CNNs but with an extra dimension for time. Instead of processing each frame independently, 3D CNNs slide a 3D kernel across both space and time, learning patterns that capture both appearance and motion. This allows them to recognize actions by understanding how visual features change over time.

$`\text{3D Convolution:}`$
```math
Y(i, j, k) = \sum_{m} \sum_{n} \sum_{l} X(i+m, j+n, k+l) \cdot W(m, n, l)
```
> **Math Breakdown:**
> - $X(i+m, j+n, k+l)$: Input video volume at position $(i+m, j+n, k+l)$.
> - $W(m, n, l)$: 3D convolution kernel weight at position $(m, n, l)$.
> - $Y(i, j, k)$: Output feature at position $(i, j, k)$.
> - The triple sum is over the 3D kernel dimensions.
> - This computes a weighted sum over a 3D spatiotemporal neighborhood.
> - The kernel learns to detect spatiotemporal patterns (e.g., motion edges).

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
> **Code Walkthrough:**
> - `Conv3d` operates on [batch, channels, time, height, width] tensors.
> - kernel_size=3 creates a 3×3×3 spatiotemporal kernel.
> - padding=1 maintains spatial and temporal dimensions.
> - `MaxPool3d(2)` reduces all dimensions by half.
> - The final fully connected layer outputs class scores.
> - This is a simple but effective 3D CNN architecture.

> **Code Commentary:**
> - `Conv3d` operates on [batch, channels, time, height, width].
> - Pooling reduces spatial and temporal resolution.
> - The final fully connected layer outputs class scores.

### 2.3. Two-Stream Networks
Two-stream networks process RGB frames and optical flow separately, then fuse the results.

> **Explanation:**
> Two-stream networks recognize that appearance (what things look like) and motion (how things move) are complementary sources of information for action recognition. The spatial stream processes RGB frames to understand appearance, while the temporal stream processes optical flow to understand motion. Combining both streams often leads to better performance than either alone.

- **Spatial Stream:** Processes appearance (RGB)
- **Temporal Stream:** Processes motion (optical flow)

$`\text{Fusion:}`$
```math
f_{final} = \alpha f_{spatial} + (1-\alpha) f_{temporal}
```
> **Math Breakdown:**
> - $f_{spatial}$: Features from the spatial stream (appearance).
> - $f_{temporal}$: Features from the temporal stream (motion).
> - $\alpha$: Weighting parameter (typically 0.5).
> - $f_{final}$: Combined features for final classification.
> - This is a simple weighted average fusion.
> - More sophisticated fusion methods include concatenation and attention.

> **Common Pitfall:** Optical flow computation can be noisy and computationally expensive. Pre-compute and cache flows for efficiency.

### 2.4. I3D (Inflated 3D ConvNet)
I3D inflates 2D CNN filters into 3D, leveraging pre-trained 2D models for video. This allows transfer learning from large image datasets.

> **Explanation:**
> I3D solves the problem of limited video training data by "inflating" pre-trained 2D CNN filters into 3D. Instead of training from scratch, I3D takes a 2D filter and copies it across the temporal dimension, then fine-tunes on video data. This leverages the vast amount of pre-trained 2D models and makes training much more efficient.

> **Try it yourself!** Take a pre-trained 2D ResNet and "inflate" its kernels to 3D for video tasks.

---

## 3. Video Object Detection & Tracking

### 3.1. Temporal Consistency
Object detection in videos must ensure consistent predictions across frames.

> **Explanation:**
> Video object detection is more than just running image object detection on each frame. Objects should be detected consistently across frames - if a car is detected in frame 1, it should still be detected in frame 2, and the bounding box should move smoothly. This requires temporal modeling to ensure consistency.

- **Tubelets:** Sequences of bounding boxes linked across frames.
- **Tracklets:** Short object tracks for linking detections.

### 3.2. Video Tracking
Tracking objects across frames involves associating detections over time.

> **Explanation:**
> Video tracking is about following objects as they move through the video. The challenge is to maintain object identity across frames, even when objects move, change appearance, or get occluded. This is crucial for understanding object behavior and interactions over time.

**Tracking-by-Detection:**
1. Detect objects in each frame
2. Associate detections using motion and appearance cues

**Hungarian Algorithm** is often used for assignment.

$`\text{IoU for Association:}`$
```math
\text{IoU} = \frac{Area(B_{t} \cap B_{t+1})}{Area(B_{t} \cup B_{t+1})}
```
> **Math Breakdown:**
> - $B_t$: Bounding box in frame $t$.
> - $B_{t+1}$: Bounding box in frame $t+1$.
> - $Area(B_t \cap B_{t+1})$: Area of intersection between boxes.
> - $Area(B_t \cup B_{t+1})$: Area of union of boxes.
> - IoU measures how much the boxes overlap.
> - Higher IoU suggests the same object.
> - Used as a similarity metric for association.

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
> **Code Walkthrough:**
> - Computes intersection coordinates: $(xA, yA)$ to $(xB, yB)$.
> - Calculates intersection area: $(xB - xA) \times (yB - yA)$.
> - Calculates individual box areas.
> - Returns IoU: intersection / union.
> - Handles edge case where boxes don't overlap (interArea = 0).
> - This is a fundamental function for object tracking.

> **Code Commentary:**
> - Computes intersection-over-union (IoU) between two bounding boxes.
> - Used for associating detections across frames.

### 3.3. Temporal Modeling
Temporal models (e.g., RNNs, LSTMs, Transformers) capture dependencies across frames.

> **Explanation:**
> Temporal modeling is about understanding how information flows through time. While CNNs are great at spatial features, they don't naturally handle temporal sequences. RNNs, LSTMs, and Transformers are designed to process sequences and can capture long-range temporal dependencies that are crucial for video understanding.

$`\text{RNN for Sequence Modeling:}`$
```math
h_t = \sigma(W_x x_t + W_h h_{t-1} + b)
```
> **Math Breakdown:**
> - $x_t$: Input at time step $t$ (e.g., frame features).
> - $h_{t-1}$: Hidden state from previous time step.
> - $W_x$: Weight matrix for input.
> - $W_h$: Weight matrix for hidden state.
> - $b$: Bias term.
> - $\sigma$: Activation function (e.g., tanh, ReLU).
> - $h_t$: New hidden state.
> - This creates a memory of previous inputs.

$`\text{Transformer Attention:}`$
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
```
> **Math Breakdown:**
> - $Q$: Query matrix (what we're looking for).
> - $K$: Key matrix (what's available).
> - $V$: Value matrix (the actual information).
> - $QK^T$: Computes attention scores between queries and keys.
> - $\sqrt{d_k}$: Scaling factor to prevent large values.
> - $\text{softmax}$: Normalizes attention scores to probabilities.
> - The result is a weighted combination of values.
> - This allows the model to focus on relevant parts of the sequence.

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
> **Code Walkthrough:**
> - Creates random video frames as example data.
> - unsqueeze(0) adds batch dimension.
> - permute reorders dimensions to [batch, channels, time, height, width].
> - This is the standard format for 3D CNNs in PyTorch.
> - The model outputs class probabilities for action recognition.

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