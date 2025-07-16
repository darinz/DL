# Fundamental Concepts in Computer Vision

> **Key Insight:** Understanding how images are represented, transformed, and processed is the foundation for all computer vision tasks.

> **Did you know?** The RGB color model was inspired by the way human eyes perceive color using three types of cones!

## 1. Image Representation

### Digital Images
A digital image is a 2D array of pixels, where each pixel represents the intensity or color value at that location.

**Mathematical Representation:**
```math
I(x, y) = \begin{cases}
f(x, y) & \text{for grayscale images} \\
(f_R(x, y), f_G(x, y), f_B(x, y)) & \text{for color images}
\end{cases}
```

Where:
- $I(x, y)$ is the image function
- $(x, y)$ are spatial coordinates
- $f(x, y)$ is the intensity value for grayscale
- $f_R, f_G, f_B$ are red, green, and blue channel values

> **Geometric Intuition:** Think of an image as a grid, where each cell (pixel) holds a value representing brightness or color.

### Image Types

#### Grayscale Images
- Single channel with intensity values
- Typically 8-bit (0-255) or normalized (0-1)
- Mathematical representation: $I(x, y) \in [0, 255]$ or $[0, 1]$

#### Color Images
- Multiple channels (RGB, HSV, LAB, etc.)
- RGB: $I(x, y) = (R(x, y), G(x, y), B(x, y))$
- Each channel: $R(x, y), G(x, y), B(x, y) \in [0, 255]$

#### Multi-channel Images
- Hyperspectral: $I(x, y) = (I_1(x, y), I_2(x, y), ..., I_n(x, y))$
- Medical imaging: CT, MRI with multiple slices

> **Try it yourself!** Load an image with OpenCV or PIL and inspect its shape and channels.

## 2. Color Spaces

### RGB Color Space
The most common color space representing colors as combinations of Red, Green, and Blue.

```math
C_{RGB} = (R, G, B)
```

**Properties:**
- Additive color model
- Device-dependent
- Not perceptually uniform

### HSV Color Space
Hue, Saturation, Value color space that separates color information from intensity.

```math
C_{HSV} = (H, S, V)
```

Where:
- $H \in [0, 360°]$ (Hue - color type)
- $S \in [0, 1]$ (Saturation - color purity)
- $V \in [0, 1]$ (Value - brightness)

**Conversion from RGB to HSV:**
```math
V = \max(R, G, B)
S = \begin{cases}
\frac{V - \min(R, G, B)}{V} & \text{if } V \neq 0 \\
0 & \text{otherwise}
\end{cases}
```

### LAB Color Space
Perceptually uniform color space designed to approximate human vision.

```math
C_{LAB} = (L, a, b)
```

Where:
- $L \in [0, 100]$ (Lightness)
- $a \in [-128, 127]$ (Green-Red axis)
- $b \in [-128, 127]$ (Blue-Yellow axis)

> **Key Insight:** LAB is more perceptually uniform than RGB, making it useful for color-based segmentation and comparison.

## 3. Mathematical Foundations

### Convolution
A fundamental operation in image processing that combines two functions to produce a third function.

**1D Convolution:**
```math
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
```

**2D Convolution (for images):**
```math
(I * K)(x, y) = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} I(i, j) K(x-i, y-j)
```

Where:
- $I$ is the input image
- $K$ is the kernel/filter
- $(x, y)$ are spatial coordinates

> **Geometric Intuition:** Convolution slides a small window (kernel) over the image, combining pixel values to detect patterns like edges or textures.

### Fourier Transform
Transforms an image from spatial domain to frequency domain.

**2D Discrete Fourier Transform:**
```math
F(u, v) = \frac{1}{MN} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}
```

**Inverse 2D DFT:**
```math
f(x, y) = \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) e^{j2\pi(\frac{ux}{M} + \frac{vy}{N})}
```

> **Did you know?** The Fourier transform is used in JPEG compression and many image filtering techniques.

### Sampling and Quantization
- **Sampling**: Converting continuous spatial coordinates to discrete grid
- **Quantization**: Converting continuous intensity values to discrete levels

**Nyquist-Shannon Sampling Theorem:**
```math
f_s > 2f_{max}
```

Where:
- $f_s$ is the sampling frequency
- $f_{max}$ is the highest frequency component

> **Common Pitfall:** If you sample below the Nyquist rate, you get aliasing—spurious patterns that aren't in the original image.

## 4. Geometric Transformations

### Translation
Moving an image by a fixed offset.

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
```

### Rotation
Rotating an image around a point (usually the center).

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```

### Scaling
Resizing an image by factors $s_x$ and $s_y$.

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```

### Affine Transformation
Combines translation, rotation, scaling, and shearing.

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
```

### Homography (Perspective Transformation)
Handles perspective changes and projective transformations.

```math
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
```

> **Try it yourself!** Apply a rotation or affine transformation to an image using OpenCV or PIL. What happens to the image?

## 5. Python Implementation Examples

### Basic Image Operations

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Create a simple test image
def create_test_image(size=(100, 100)):
    """Create a test image with a gradient and circle."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Create gradient
    for i in range(size[0]):
        for j in range(size[1]):
            img[i, j] = [i, j, (i + j) // 2]
    
    # Add a circle
    center = (size[0]//2, size[1]//2)
    radius = min(size) // 4
    cv2.circle(img, center, radius, (255, 255, 255), -1)
    
    return img

# Color space conversions
def demonstrate_color_spaces(image):
    """Demonstrate different color spaces."""
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original RGB')
    axes[0, 1].imshow(hsv)
    axes[0, 1].set_title('HSV')
    axes[1, 0].imshow(lab)
    axes[1, 0].set_title('LAB')
    axes[1, 1].imshow(gray, cmap='gray')
    axes[1, 1].set_title('Grayscale')
    
    plt.tight_layout()
    plt.show()

# Geometric transformations
def demonstrate_transformations(image):
    """Demonstrate various geometric transformations."""
    height, width = image.shape[:2]
    
    # Translation matrix
    translation_matrix = np.float32([[1, 0, 50], [0, 1, 30]])
    translated = cv2.warpAffine(image, translation_matrix, (width, height))
    
    # Rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width//2, height//2), 45, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    # Scaling
    scaled = cv2.resize(image, None, fx=1.5, fy=1.5)
    
    # Affine transformation
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    affine_matrix = cv2.getAffineTransform(pts1, pts2)
    affine_transformed = cv2.warpAffine(image, affine_matrix, (width, height))
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(translated)
    axes[0, 1].set_title('Translated')
    axes[0, 2].imshow(rotated)
    axes[0, 2].set_title('Rotated')
    axes[1, 0].imshow(scaled)
    axes[1, 0].set_title('Scaled')
    axes[1, 1].imshow(affine_transformed)
    axes[1, 1].set_title('Affine Transformed')
    axes[1, 2].axis('off')
```
*These code snippets demonstrate basic image creation, color space conversion, and geometric transformations using OpenCV and matplotlib.*

---

> **Key Insight:** Mastering these fundamental concepts is essential for tackling advanced computer vision tasks like object detection, segmentation, and recognition. 