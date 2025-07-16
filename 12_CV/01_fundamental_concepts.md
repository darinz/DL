# Fundamental Concepts in Computer Vision

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
    
    plt.tight_layout()
    plt.show()

# Convolution implementation
def custom_convolution(image, kernel):
    """Implement 2D convolution manually."""
    # Get dimensions
    img_height, img_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape
    
    # Calculate padding
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image
    padded = cv2.copyMakeBorder(image, pad_height, pad_height, 
                               pad_width, pad_width, cv2.BORDER_REFLECT)
    
    # Initialize output
    output = np.zeros_like(image, dtype=np.float32)
    
    # Apply convolution
    for i in range(img_height):
        for j in range(img_width):
            for c in range(image.shape[2]):
                output[i, j, c] = np.sum(
                    padded[i:i+kernel_height, j:j+kernel_width, c] * kernel
                )
    
    return output

# Fourier Transform demonstration
def demonstrate_fourier_transform(image):
    """Demonstrate Fourier Transform of an image."""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Fourier Transform
    f_transform = np.fft.fft2(gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_shift) + 1)
    
    # Apply inverse Fourier Transform
    f_ishift = np.fft.ifftshift(f_shift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(gray, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(magnitude_spectrum, cmap='gray')
    axes[1].set_title('Magnitude Spectrum')
    axes[2].imshow(img_back, cmap='gray')
    axes[2].set_title('Reconstructed Image')
    
    plt.tight_layout()
    plt.show()

# Main demonstration
if __name__ == "__main__":
    # Create test image
    test_image = create_test_image((200, 200))
    
    # Demonstrate color spaces
    demonstrate_color_spaces(test_image)
    
    # Demonstrate transformations
    demonstrate_transformations(test_image)
    
    # Demonstrate convolution
    kernel = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]) / 9  # 3x3 averaging kernel
    
    convolved = custom_convolution(test_image, kernel)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(convolved.astype(np.uint8))
    plt.title('After Convolution')
    plt.show()
    
    # Demonstrate Fourier Transform
    demonstrate_fourier_transform(test_image)
```

### Advanced Concepts

```python
# Image interpolation methods
def demonstrate_interpolation(image):
    """Demonstrate different interpolation methods."""
    # Resize with different interpolation methods
    methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (name, method) in enumerate(methods.items()):
        resized = cv2.resize(image, (50, 50), interpolation=method)
        enlarged = cv2.resize(resized, (200, 200), interpolation=method)
        
        row, col = i // 2, i % 2
        axes[row, col].imshow(enlarged)
        axes[row, col].set_title(f'{name}')
    
    plt.tight_layout()
    plt.show()

# Histogram analysis
def analyze_image_histogram(image):
    """Analyze image histogram for different color channels."""
    colors = ('b', 'g', 'r')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    
    # Histogram
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        axes[1].plot(hist, color=color)
    
    axes[1].set_title('Color Histogram')
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Frequency')
    axes[1].legend(['Blue', 'Green', 'Red'])
    
    plt.tight_layout()
    plt.show()
```

This comprehensive guide covers the fundamental concepts in computer vision, providing both theoretical understanding and practical implementation. The mathematical foundations are essential for understanding more advanced computer vision techniques, while the Python examples demonstrate how these concepts are applied in practice. 