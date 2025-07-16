# Image Processing Basics

> **Key Insight:** Image processing techniques are the building blocks for all computer vision systems, from denoising to feature extraction.

> **Did you know?** The Gaussian filter is inspired by the normal distribution and is used in everything from photography to deep learning!

## 1. Filtering and Enhancement

### Linear Filters

Linear filters operate on the principle of linear superposition and are fundamental to image processing.

#### Gaussian Filter
A smoothing filter that reduces noise while preserving edges.

**1D Gaussian Function:**
```math
G(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}}
```

**2D Gaussian Function:**
```math
G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
```

**Properties:**
- Smoothing effect increases with $\sigma$
- Separable: $G(x, y) = G(x) \cdot G(y)$
- Preserves edges better than uniform averaging

> **Geometric Intuition:** The Gaussian filter blurs an image by averaging pixels with their neighbors, weighted by distance—closer pixels have more influence.

#### Mean Filter
Simple averaging filter that reduces noise but blurs edges.

**Kernel:**
```math
K = \frac{1}{n^2} \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}
```

> **Common Pitfall:** Mean filtering can overly blur important details and edges.

#### Median Filter
Non-linear filter that preserves edges while removing salt-and-pepper noise.

**Operation:**
```math
I'(x, y) = \text{median}\{I(i, j) : (i, j) \in N(x, y)\}
```

Where $N(x, y)$ is the neighborhood around pixel $(x, y)$.

> **Try it yourself!** Add salt-and-pepper noise to an image and compare the results of mean vs. median filtering.

### Non-Linear Filters

#### Bilateral Filter
Preserves edges while smoothing, combining spatial and intensity similarity.

**Bilateral Filter Formula:**
```math
I'(x, y) = \frac{1}{W_p} \sum_{i,j} I(i, j) \cdot w_s(i, j) \cdot w_r(i, j)
```

Where:
- $w_s(i, j) = e^{-\frac{(i-x)^2 + (j-y)^2}{2\sigma_s^2}}$ (spatial weight)
- $w_r(i, j) = e^{-\frac{(I(i,j) - I(x,y))^2}{2\sigma_r^2}}$ (range weight)
- $W_p = \sum_{i,j} w_s(i, j) \cdot w_r(i, j)$ (normalization factor)

> **Key Insight:** The bilateral filter is widely used in computational photography for edge-preserving smoothing.

## 2. Edge Detection

### Gradient-Based Methods

#### Sobel Operator
Computes gradient magnitude and direction using convolution kernels.

**Sobel Kernels:**
```math
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}
```

**Gradient Magnitude:**
```math
|\nabla I| = \sqrt{G_x^2 + G_y^2}
```

**Gradient Direction:**
```math
\theta = \arctan\left(\frac{G_y}{G_x}\right)
```

> **Geometric Intuition:** The Sobel operator highlights regions of rapid intensity change—edges—by computing local gradients.

#### Laplacian Operator
Second-order derivative operator that detects edges at zero crossings.

**Laplacian Kernel:**
```math
\nabla^2 = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}
```

**Laplacian of Gaussian (LoG):**
```math
\text{LoG}(x, y) = \frac{1}{\pi\sigma^4}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right)e^{-\frac{x^2 + y^2}{2\sigma^2}}
```

### Canny Edge Detection

A multi-stage algorithm that produces optimal edge detection.

**Steps:**
1. **Gaussian Smoothing:** Reduce noise
2. **Gradient Computation:** Calculate magnitude and direction
3. **Non-maximum Suppression:** Thin edges
4. **Double Thresholding:** Classify edges as strong/weak
5. **Edge Tracking:** Connect strong edges

**Gradient Magnitude:**
```math
|\nabla I| = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}
```

**Gradient Direction:**
```math
\theta = \arctan\left(\frac{\partial I}{\partial y} / \frac{\partial I}{\partial x}\right)
```

> **Try it yourself!** Apply Canny edge detection to a photo and visualize the detected edges.

## 3. Morphological Operations

### Basic Operations

#### Erosion
Shrinks objects and removes small details.

```math
(A \ominus B)(x, y) = \min\{A(x+i, y+j) : (i, j) \in B\}
```

#### Dilation
Expands objects and fills small holes.

```math
(A \oplus B)(x, y) = \max\{A(x-i, y-j) : (i, j) \in B\}
```

#### Opening
Erosion followed by dilation, removes small objects.

```math
A \circ B = (A \ominus B) \oplus B
```

#### Closing
Dilation followed by erosion, fills small holes.

```math
A \bullet B = (A \oplus B) \ominus B
```

### Structuring Elements

Common structuring elements:

**Square:**
```math
B = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
```

**Cross:**
```math
B = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 0 \end{bmatrix}
```

**Disk:**
```math
B = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 0 \end{bmatrix}
```

> **Key Insight:** Morphological operations are powerful for cleaning up binary images, extracting shapes, and preparing data for object detection.

## 4. Histogram Processing

### Histogram Equalization

Improves image contrast by spreading pixel intensities across the full range.

**Cumulative Distribution Function (CDF):**
```math
cdf(k) = \sum_{i=0}^{k} p(i)
```

**Equalization Transformation:**
```math
T(k) = \text{round}\left(\frac{cdf(k) - cdf_{min}}{(M \times N) - cdf_{min}} \times (L-1)\right)
```

Where:
- $M \times N$ is the image size
- $L$ is the number of intensity levels
- $cdf_{min}$ is the minimum non-zero CDF value

### Contrast Limited Adaptive Histogram Equalization (CLAHE)

Improves local contrast while limiting amplification of noise.

**Clipping Limit:**
```math
\text{clip limit} = \alpha \times \frac{M \times N}{L}
```

Where $\alpha$ is the clipping factor (typically 2-4).

**Local Histogram Equalization:**
```math
T_{local}(k) = \text{round}\left(\frac{cdf_{local}(k) - cdf_{local,min}}{(M_{local} \times N_{local}) - cdf_{local,min}} \times (L-1)\right)
```

> **Did you know?** CLAHE is widely used in medical imaging to enhance local contrast in X-rays and CT scans.

## 5. Noise Reduction Techniques

### Additive White Gaussian Noise (AWGN)

**Model:**
```math
I_{noisy}(x, y) = I_{original}(x, y) + \eta(x, y)
```

Where $\eta(x, y) \sim \mathcal{N}(0, \sigma^2)$.

### Salt-and-Pepper Noise

**Model:**
```math
I_{noisy}(x, y) = \begin{cases}
0 & \text{with probability } p/2 \\
255 & \text{with probability } p/2 \\
I_{original}(x, y) & \text{with probability } 1-p
\end{cases}
```

### Wiener Filter

Optimal linear filter for noise reduction.

**Frequency Domain:**
```math
H(u, v) = \frac{P_f(u, v)}{P_f(u, v) + P_n(u, v)}
```

Where:
- $P_f(u, v)$ is the power spectrum of the original image
- $P_n(u, v)$ is the power spectrum of the noise

> **Common Pitfall:** Over-smoothing with aggressive noise reduction can erase important image details.

## 6. Python Implementation Examples

### Basic Filtering Operations

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d
```
*These code snippets demonstrate basic filtering, edge detection, and noise reduction using OpenCV, NumPy, and SciPy.*

---

> **Key Insight:** Mastering image processing basics is essential for building robust computer vision pipelines and understanding how deep learning models process visual data. 