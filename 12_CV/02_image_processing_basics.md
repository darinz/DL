# Image Processing Basics

> **Key Insight:** Image processing techniques are the building blocks for all computer vision systems, from denoising to feature extraction.

> **Did you know?** The Gaussian filter is inspired by the normal distribution and is used in everything from photography to deep learning!

## 1. Filtering and Enhancement

### Linear Filters

Linear filters operate on the principle of linear superposition and are fundamental to image processing.

> **Explanation:**
> Linear filters are mathematical operations that process each pixel based on its neighbors in a predictable, linear way. They're called "linear" because if you apply the filter to two images and add the results, it's the same as adding the images first and then applying the filter.

#### Gaussian Filter
A smoothing filter that reduces noise while preserving edges.

> **Explanation:**
> The Gaussian filter is like a smart blur that reduces noise while keeping important edges sharp. It works by averaging each pixel with its neighbors, but gives more weight to closer pixels and less weight to distant ones.

**1D Gaussian Function:**
```math
G(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{x^2}{2\sigma^2}}
```
> **Math Breakdown:**
> - $\sigma$: Standard deviation that controls how much blurring occurs (larger = more blur).
> - $e^{-\frac{x^2}{2\sigma^2}}$: Exponential decay that gives more weight to pixels closer to the center.
> - $\frac{1}{\sigma\sqrt{2\pi}}$: Normalization factor to make the total weight sum to 1.

**2D Gaussian Function:**
```math
G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}
```
> **Math Breakdown:**
> - This is the 2D version of the Gaussian function.
> - $x^2 + y^2$ represents the squared distance from the center.
> - The function is circularly symmetric (same value at same distance from center).

**Properties:**
- Smoothing effect increases with $\sigma$
- Separable: $G(x, y) = G(x) \cdot G(y)$
- Preserves edges better than uniform averaging

> **Geometric Intuition:** The Gaussian filter blurs an image by averaging pixels with their neighbors, weighted by distance—closer pixels have more influence.

#### Mean Filter
Simple averaging filter that reduces noise but blurs edges.

> **Explanation:**
> The mean filter simply averages each pixel with its neighbors. It's like taking a small window around each pixel and replacing the pixel with the average of all values in that window.

**Kernel:**
```math
K = \frac{1}{n^2} \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}
```
> **Math Breakdown:**
> - This is an $n \times n$ matrix where all values are 1.
> - $\frac{1}{n^2}$ normalizes the kernel so the sum of all weights equals 1.
> - For a 3×3 kernel, each pixel gets weight $\frac{1}{9}$.

> **Common Pitfall:** Mean filtering can overly blur important details and edges.

#### Median Filter
Non-linear filter that preserves edges while removing salt-and-pepper noise.

> **Explanation:**
> The median filter replaces each pixel with the median value of its neighborhood. Unlike the mean filter, it's not affected by extreme values (outliers), making it excellent for removing salt-and-pepper noise while preserving edges.

**Operation:**
```math
I'(x, y) = \text{median}\{I(i, j) : (i, j) \in N(x, y)\}
```
> **Math Breakdown:**
> - $N(x, y)$: Neighborhood around pixel $(x, y)$ (e.g., 3×3 or 5×5 window).
> - $\text{median}$: Middle value when all pixels in the neighborhood are sorted.
> - This is a non-linear operation because the median of two images is not the same as the median of each image separately.

Where $N(x, y)$ is the neighborhood around pixel $(x, y)$.

> **Try it yourself!** Add salt-and-pepper noise to an image and compare the results of mean vs. median filtering.

### Non-Linear Filters

#### Bilateral Filter
Preserves edges while smoothing, combining spatial and intensity similarity.

> **Explanation:**
> The bilateral filter is like a smart Gaussian filter that considers both spatial distance AND how similar the pixel values are. It smooths areas with similar colors while preserving edges where colors change abruptly.

**Bilateral Filter Formula:**
```math
I'(x, y) = \frac{1}{W_p} \sum_{i,j} I(i, j) \cdot w_s(i, j) \cdot w_r(i, j)
```
> **Math Breakdown:**
> - $I(i, j)$: Pixel value at position $(i, j)$.
> - $w_s(i, j)$: Spatial weight (based on distance).
> - $w_r(i, j)$: Range weight (based on intensity difference).
> - $W_p$: Normalization factor to make weights sum to 1.

Where:
- $w_s(i, j) = e^{-\frac{(i-x)^2 + (j-y)^2}{2\sigma_s^2}}$ (spatial weight)
- $w_r(i, j) = e^{-\frac{(I(i,j) - I(x,y))^2}{2\sigma_r^2}}$ (range weight)
- $W_p = \sum_{i,j} w_s(i, j) \cdot w_r(i, j)$ (normalization factor)

> **Key Insight:** The bilateral filter is widely used in computational photography for edge-preserving smoothing.

## 2. Edge Detection

### Gradient-Based Methods

> **Explanation:**
> Edge detection finds boundaries between different regions in an image. Gradient-based methods look for rapid changes in pixel intensity, which typically occur at edges.

#### Sobel Operator
Computes gradient magnitude and direction using convolution kernels.

> **Explanation:**
> The Sobel operator uses two 3×3 kernels to compute the gradient in the x and y directions. It then combines these to find the magnitude and direction of the gradient at each pixel.

**Sobel Kernels:**
```math
G_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad
G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}
```
> **Math Breakdown:**
> - $G_x$: Detects vertical edges (changes in x-direction).
> - $G_y$: Detects horizontal edges (changes in y-direction).
> - The weights are designed to give more importance to the center row/column.
> - The negative weights on one side and positive on the other create a gradient measurement.

**Gradient Magnitude:**
```math
|\nabla I| = \sqrt{G_x^2 + G_y^2}
```
> **Math Breakdown:**
> - This computes the strength of the edge at each pixel.
> - Uses the Pythagorean theorem to combine x and y gradients.
> - Larger values indicate stronger edges.

**Gradient Direction:**
```math
\theta = \arctan\left(\frac{G_y}{G_x}\right)
```
> **Math Breakdown:**
> - This gives the direction of the edge (perpendicular to the gradient).
> - $\arctan$ converts the ratio of gradients to an angle.
> - The edge runs perpendicular to this direction.

> **Geometric Intuition:** The Sobel operator highlights regions of rapid intensity change—edges—by computing local gradients.

#### Laplacian Operator
Second-order derivative operator that detects edges at zero crossings.

> **Explanation:**
> The Laplacian operator is a second-order derivative that measures how quickly the gradient is changing. It's very sensitive to noise but can detect edges more precisely than first-order methods.

**Laplacian Kernel:**
```math
\nabla^2 = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 4 & -1 \\ 0 & -1 & 0 \end{bmatrix}
```
> **Math Breakdown:**
> - This kernel computes the second derivative in both x and y directions.
> - The center pixel (4) is positive, surrounded by negative weights (-1).
> - This measures how much the center pixel differs from its neighbors.

**Laplacian of Gaussian (LoG):**
```math
\text{LoG}(x, y) = \frac{1}{\pi\sigma^4}\left(1 - \frac{x^2 + y^2}{2\sigma^2}\right)e^{-\frac{x^2 + y^2}{2\sigma^2}}
```
> **Math Breakdown:**
> - This combines Gaussian smoothing with Laplacian edge detection.
> - The Gaussian reduces noise before applying the Laplacian.
> - The result is a more robust edge detector.

### Canny Edge Detection

A multi-stage algorithm that produces optimal edge detection.

> **Explanation:**
> Canny edge detection is a sophisticated algorithm that produces thin, continuous edges. It works in multiple stages: smooth the image, find gradients, thin the edges, and connect them.

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
> **Math Breakdown:**
> - $\frac{\partial I}{\partial x}$: Partial derivative in x-direction (rate of change).
> - $\frac{\partial I}{\partial y}$: Partial derivative in y-direction.
> - The magnitude gives the strength of the edge.

**Gradient Direction:**
```math
\theta = \arctan\left(\frac{\partial I}{\partial y} / \frac{\partial I}{\partial x}\right)
```
> **Math Breakdown:**
> - This gives the direction of the gradient.
> - The edge runs perpendicular to this direction.
> - Used in non-maximum suppression to determine edge orientation.

> **Try it yourself!** Apply Canny edge detection to a photo and visualize the detected edges.

## 3. Morphological Operations

### Basic Operations

> **Explanation:**
> Morphological operations work on binary images (black and white) and are used to clean up noise, connect broken lines, and extract shapes. They're based on set theory and work by sliding a structuring element over the image.

#### Erosion
Shrinks objects and removes small details.

> **Explanation:**
> Erosion makes objects smaller by removing pixels from the boundaries. It's like "eating away" at the edges of white regions. Small objects disappear entirely, and large objects become smaller.

```math
(A \ominus B)(x, y) = \min\{A(x+i, y+j) : (i, j) \in B\}
```
> **Math Breakdown:**
> - $A$: Input binary image (white = 1, black = 0).
> - $B$: Structuring element (small binary pattern).
> - For each position $(x, y)$, look at all pixels covered by the structuring element.
> - Take the minimum value (if any pixel is black, the result is black).

#### Dilation
Expands objects and fills small holes.

> **Explanation:**
> Dilation makes objects larger by adding pixels to the boundaries. It's like "growing" the white regions. Small holes get filled, and objects become larger.

```math
(A \oplus B)(x, y) = \max\{A(x-i, y-j) : (i, j) \in B\}
```
> **Math Breakdown:**
> - Similar to erosion but takes the maximum instead of minimum.
> - If any pixel under the structuring element is white, the result is white.
> - This expands white regions and fills small holes.

#### Opening
Erosion followed by dilation, removes small objects.

> **Explanation:**
> Opening is erosion followed by dilation. It removes small objects and smooths the boundaries of larger objects without changing their size much.

```math
A \circ B = (A \ominus B) \oplus B
```
> **Math Breakdown:**
> - First apply erosion: $(A \ominus B)$ removes small objects and shrinks boundaries.
> - Then apply dilation: $\oplus B$ restores the size of remaining objects.
> - Small objects that were completely removed by erosion don't come back.

#### Closing
Dilation followed by erosion, fills small holes.

> **Explanation:**
> Closing is dilation followed by erosion. It fills small holes and connects nearby objects without changing the overall size much.

```math
A \bullet B = (A \oplus B) \ominus B
```
> **Math Breakdown:**
> - First apply dilation: $(A \oplus B)$ fills holes and connects nearby objects.
> - Then apply erosion: $\ominus B$ shrinks the boundaries back to original size.
> - Small holes that were filled by dilation stay filled.

### Structuring Elements

Common structuring elements:

> **Explanation:**
> The structuring element is like a small template that determines how the morphological operation works. Different shapes produce different effects.

**Square:**
```math
B = \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix}
```
> **Math Breakdown:**
> - 3×3 square that affects all 8 neighbors plus the center.
> - Good for general-purpose morphological operations.
> - Produces isotropic (direction-independent) results.

**Cross:**
```math
B = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 0 \end{bmatrix}
```
> **Math Breakdown:**
> - Only affects the 4 neighbors (up, down, left, right).
> - More selective than the square.
> - Good for preserving diagonal features.

**Disk:**
```math
B = \begin{bmatrix} 0 & 1 & 0 \\ 1 & 1 & 1 \\ 0 & 1 & 0 \end{bmatrix}
```
> **Math Breakdown:**
> - Approximates a circular shape.
> - More isotropic than square or cross.
> - Good for operations that should be rotation-invariant.

> **Key Insight:** Morphological operations are powerful for cleaning up binary images, extracting shapes, and preparing data for object detection.

## 4. Histogram Processing

### Histogram Equalization

Improves image contrast by spreading pixel intensities across the full range.

> **Explanation:**
> Histogram equalization redistributes pixel intensities to make better use of the available range. It stretches the histogram to cover the full range, making dark areas brighter and bright areas darker, improving overall contrast.

**Cumulative Distribution Function (CDF):**
```math
cdf(k) = \sum_{i=0}^{k} p(i)
```
> **Math Breakdown:**
> - $p(i)$: Probability of intensity level $i$ (histogram value divided by total pixels).
> - $cdf(k)$: Cumulative sum up to intensity level $k$.
> - This gives the fraction of pixels with intensity ≤ $k$.

**Equalization Transformation:**
```math
T(k) = \text{round}\left(\frac{cdf(k) - cdf_{min}}{(M \times N) - cdf_{min}} \times (L-1)\right)
```
> **Math Breakdown:**
> - $M \times N$: Total number of pixels in the image.
> - $L$: Number of intensity levels (typically 256 for 8-bit images).
> - $cdf_{min}$: Minimum non-zero CDF value.
> - This maps each intensity level to a new value that spreads the histogram.

Where:
- $M \times N$ is the image size
- $L$ is the number of intensity levels
- $cdf_{min}$ is the minimum non-zero CDF value

### Contrast Limited Adaptive Histogram Equalization (CLAHE)

Improves local contrast while limiting amplification of noise.

> **Explanation:**
> CLAHE is like histogram equalization but applied to small regions of the image. It improves local contrast without over-amplifying noise, making it better for medical imaging and other applications where noise is a concern.

**Clipping Limit:**
```math
\text{clip limit} = \alpha \times \frac{M \times N}{L}
```
> **Math Breakdown:**
> - $\alpha$: Clipping factor (typically 2-4).
> - $M \times N$: Size of the local region.
> - $L$: Number of intensity levels.
> - This limits how much any histogram bin can be enhanced.

Where $\alpha$ is the clipping factor (typically 2-4).

**Local Histogram Equalization:**
```math
T_{local}(k) = \text{round}\left(\frac{cdf_{local}(k) - cdf_{local,min}}{(M_{local} \times N_{local}) - cdf_{local,min}} \times (L-1)\right)
```
> **Math Breakdown:**
> - Similar to global histogram equalization but applied to local regions.
> - $cdf_{local}$: CDF computed from the local region's histogram.
> - $M_{local} \times N_{local}$: Size of the local region.
> - This enhances contrast in each local region independently.

> **Did you know?** CLAHE is widely used in medical imaging to enhance local contrast in X-rays and CT scans.

## 5. Noise Reduction Techniques

### Additive White Gaussian Noise (AWGN)

> **Explanation:**
> AWGN is the most common type of noise in digital images. It adds random values from a normal distribution to each pixel, making the image look grainy or speckled.

**Model:**
```math
I_{noisy}(x, y) = I_{original}(x, y) + \eta(x, y)
```
> **Math Breakdown:**
> - $I_{original}(x, y)$: Original pixel value.
> - $\eta(x, y)$: Noise value at position $(x, y)$.
> - The noise is added to the original signal.

Where $\eta(x, y) \sim \mathcal{N}(0, \sigma^2)$.

### Salt-and-Pepper Noise

> **Explanation:**
> Salt-and-pepper noise randomly sets some pixels to pure white (salt) and others to pure black (pepper). It's common in old photographs and faulty sensors.

**Model:**
```math
I_{noisy}(x, y) = \begin{cases}
0 & \text{with probability } p/2 \\
255 & \text{with probability } p/2 \\
I_{original}(x, y) & \text{with probability } 1-p
\end{cases}
```
> **Math Breakdown:**
> - $p$: Probability of noise (e.g., 0.05 for 5% noise).
> - With probability $p/2$, pixel becomes black (0).
> - With probability $p/2$, pixel becomes white (255).
> - With probability $1-p$, pixel keeps its original value.

### Wiener Filter

Optimal linear filter for noise reduction.

> **Explanation:**
> The Wiener filter is an optimal linear filter that minimizes the mean square error between the original and filtered images. It works best when you know the noise characteristics and the image statistics.

**Wiener Filter Formula:**
```math
H(u, v) = \frac{P_f(u, v)}{P_f(u, v) + P_n(u, v)}
```
> **Math Breakdown:**
> - $H(u, v)$: Frequency response of the Wiener filter.
> - $P_f(u, v)$: Power spectrum of the original signal.
> - $P_n(u, v)$: Power spectrum of the noise.
> - The filter attenuates frequencies where noise is stronger than signal.

**In the Spatial Domain:**
```math
\hat{f}(x, y) = \mu + \frac{\sigma_f^2}{\sigma_f^2 + \sigma_n^2}(g(x, y) - \mu)
```
> **Math Breakdown:**
> - $\mu$: Local mean of the image.
> - $\sigma_f^2$: Local variance of the original signal.
> - $\sigma_n^2$: Variance of the noise.
> - $g(x, y)$: Noisy pixel value.
> - This adapts the filtering based on local image statistics.

> **Key Insight:** The Wiener filter is particularly effective for AWGN and is widely used in image restoration applications. 