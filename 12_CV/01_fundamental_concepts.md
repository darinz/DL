# Fundamental Concepts in Computer Vision

Computer vision is built upon several fundamental concepts that form the foundation for understanding how machines process and interpret visual information. This guide covers the essential mathematical and computational principles.

## Table of Contents

1. [Image Representation](#image-representation)
2. [Color Spaces](#color-spaces)
3. [Mathematical Foundations](#mathematical-foundations)
4. [Geometric Transformations](#geometric-transformations)

## Image Representation

### Digital Images as Arrays

Digital images are represented as 2D arrays (matrices) where each element corresponds to a pixel. For grayscale images, each pixel has a single intensity value, while color images have multiple channels.

```python
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Create a simple grayscale image
def create_grayscale_image():
    # 5x5 grayscale image with values 0-255
    image = np.array([
        [0, 50, 100, 150, 200],
        [25, 75, 125, 175, 225],
        [50, 100, 150, 200, 250],
        [75, 125, 175, 225, 255],
        [100, 150, 200, 250, 255]
    ], dtype=np.uint8)
    
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Grayscale Image')
    plt.colorbar()
    
    plt.subplot(1, 2, 2)
    plt.imshow(image, cmap='viridis')
    plt.title('Grayscale Image (Viridis Colormap)')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    return image

# Create a color image
def create_color_image():
    # 3x3 RGB image
    image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],      # Red, Green, Blue
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]], # Yellow, Magenta, Cyan
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]]  # Gray, White, Black
    ], dtype=np.uint8)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.title('Color Image')
    plt.axis('off')
    plt.show()
    
    return image

# Image properties
def analyze_image_properties(image):
    print(f"Image shape: {image.shape}")
    print(f"Data type: {image.dtype}")
    print(f"Value range: {image.min()} - {image.max()}")
    print(f"Memory usage: {image.nbytes} bytes")
    
    if len(image.shape) == 3:
        print(f"Channels: {image.shape[2]}")
        for i in range(image.shape[2]):
            print(f"Channel {i} range: {image[:,:,i].min()} - {image[:,:,i].max()}")

# Example usage
if __name__ == "__main__":
    gray_img = create_grayscale_image()
    color_img = create_color_image()
    
    print("Grayscale image properties:")
    analyze_image_properties(gray_img)
    
    print("\nColor image properties:")
    analyze_image_properties(color_img)
```

### Image Coordinate System

In computer vision, the coordinate system typically uses $(x, y)$ where:
- $x$ represents the column (horizontal position)
- $y$ represents the row (vertical position)
- Origin $(0, 0)$ is at the top-left corner

```python
def demonstrate_coordinate_system():
    # Create a 10x10 image with coordinate markers
    image = np.zeros((10, 10), dtype=np.uint8)
    
    # Mark specific coordinates
    coordinates = [(0, 0), (5, 5), (9, 9), (0, 9), (9, 0)]
    for x, y in coordinates:
        image[y, x] = 255  # Note: y comes first in numpy indexing
    
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')
    plt.title('Coordinate System Demonstration')
    
    # Add coordinate labels
    for x, y in coordinates:
        plt.text(x, y, f'({x},{y})', color='red', fontsize=8, 
                ha='center', va='center')
    
    plt.grid(True, alpha=0.3)
    plt.show()

demonstrate_coordinate_system()
```

## Color Spaces

### RGB Color Space

RGB (Red, Green, Blue) is the most common color space for digital images. Each pixel is represented by three values corresponding to the intensity of red, green, and blue channels.

```python
def rgb_color_space_demo():
    # Create RGB color wheel
    size = 100
    center = size // 2
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Convert to polar coordinates
    r = np.sqrt((x - center)**2 + (y - center)**2)
    theta = np.arctan2(y - center, x - center)
    
    # Normalize radius and angle
    r = np.clip(r / center, 0, 1)
    theta = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Create RGB channels
    red = np.where(r <= 1, 255 * (1 - r), 0)
    green = np.where(r <= 1, 255 * np.sin(theta * np.pi), 0)
    blue = np.where(r <= 1, 255 * np.cos(theta * np.pi), 0)
    
    # Combine channels
    rgb_image = np.stack([red, green, blue], axis=2).astype(np.uint8)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(red, cmap='Reds')
    plt.title('Red Channel')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(green, cmap='Greens')
    plt.title('Green Channel')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(blue, cmap='Blues')
    plt.title('Blue Channel')
    plt.colorbar()
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_image)
    plt.title('RGB Color Wheel')
    plt.axis('off')
    plt.show()

rgb_color_space_demo()
```

### HSV Color Space

HSV (Hue, Saturation, Value) is often more intuitive for color-based image processing tasks.

```python
def rgb_to_hsv(r, g, b):
    """Convert RGB values to HSV"""
    r, g, b = r/255.0, g/255.0, b/255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    diff = cmax - cmin
    
    # Hue calculation
    if diff == 0:
        h = 0
    elif cmax == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif cmax == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    else:
        h = (60 * ((r-g)/diff) + 240) % 360
    
    # Saturation calculation
    s = 0 if cmax == 0 else diff / cmax
    
    # Value calculation
    v = cmax
    
    return h, s, v

def hsv_color_space_demo():
    # Create HSV color wheel
    size = 100
    center = size // 2
    
    # Create coordinate grids
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    
    # Convert to polar coordinates
    r = np.sqrt((x - center)**2 + (y - center)**2)
    theta = np.arctan2(y - center, x - center)
    
    # Normalize radius and angle
    r = np.clip(r / center, 0, 1)
    theta = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
    
    # Create HSV values
    h = theta * 360  # Hue: 0-360 degrees
    s = r            # Saturation: 0-1
    v = 1.0          # Value: 1.0 for full brightness
    
    # Convert HSV to RGB for display
    rgb_image = np.zeros((size, size, 3))
    
    for i in range(size):
        for j in range(size):
            h_val, s_val, v_val = h[i, j], s[i, j], v
            rgb_image[i, j] = hsv_to_rgb(h_val, s_val, v_val)
    
    rgb_image = (rgb_image * 255).astype(np.uint8)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rgb_image)
    plt.title('HSV Color Wheel')
    plt.axis('off')
    plt.show()

def hsv_to_rgb(h, s, v):
    """Convert HSV values to RGB"""
    h = h / 60
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if i == 0:
        return v, t, p
    elif i == 1:
        return q, v, p
    elif i == 2:
        return p, v, t
    elif i == 3:
        return p, q, v
    elif i == 4:
        return t, p, v
    else:
        return v, p, q

hsv_color_space_demo()
```

### Color Space Conversions

```python
import cv2

def color_space_conversions():
    # Load a sample image
    image = np.array([
        [[255, 0, 0], [0, 255, 0], [0, 0, 255]],
        [[255, 255, 0], [255, 0, 255], [0, 255, 255]],
        [[128, 128, 128], [255, 255, 255], [0, 0, 0]]
    ], dtype=np.uint8)
    
    # Convert to different color spaces
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('RGB')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(hsv)
    axes[0, 1].set_title('HSV')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(lab)
    axes[1, 0].set_title('LAB')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(yuv)
    axes[1, 1].set_title('YUV')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print channel information
    print("RGB channels:", image.shape)
    print("HSV channels:", hsv.shape)
    print("LAB channels:", lab.shape)
    print("YUV channels:", yuv.shape)

color_space_conversions()
```

## Mathematical Foundations

### Convolution Operations

Convolution is a fundamental operation in computer vision, used for filtering, feature detection, and image processing.

#### 1D Convolution

The 1D convolution of two functions $f$ and $g$ is defined as:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau$$

For discrete signals:

$$(f * g)[n] = \sum_{m=-\infty}^{\infty} f[m] g[n - m]$$

```python
def convolution_1d_demo():
    # Create simple signals
    signal = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    kernel = np.array([1, 1, 1]) / 3  # Moving average filter
    
    # Manual convolution
    def manual_conv1d(signal, kernel):
        n = len(signal)
        k = len(kernel)
        result = np.zeros(n)
        
        for i in range(n):
            for j in range(k):
                if i - j >= 0 and i - j < n:
                    result[i] += signal[i - j] * kernel[j]
        
        return result
    
    # Using numpy convolution
    conv_numpy = np.convolve(signal, kernel, mode='same')
    conv_manual = manual_conv1d(signal, kernel)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.stem(signal, use_line_collection=True)
    plt.title('Original Signal')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.stem(kernel, use_line_collection=True)
    plt.title('Kernel (Moving Average)')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.stem(conv_numpy, use_line_collection=True, label='Numpy')
    plt.stem(conv_manual, use_line_collection=True, markerfmt='ro', label='Manual')
    plt.title('Convolution Result')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print("Original signal:", signal)
    print("Kernel:", kernel)
    print("Convolution result:", conv_numpy)

convolution_1d_demo()
```

#### 2D Convolution

For 2D images, convolution is defined as:

$$(I * K)(i,j) = \sum_{m} \sum_{n} I(m,n) K(i-m, j-n)$$

```python
def convolution_2d_demo():
    # Create a simple 2D image
    image = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=float)
    
    # Create different kernels
    kernels = {
        'Identity': np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
        'Blur': np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,
        'Edge Detection': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'Sharpen': np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    }
    
    def manual_conv2d(image, kernel):
        """Manual 2D convolution implementation"""
        h, w = image.shape
        kh, kw = kernel.shape
        
        # Pad the image
        pad_h, pad_w = kh // 2, kw // 2
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
        
        result = np.zeros_like(image)
        
        for i in range(h):
            for j in range(w):
                result[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kernel)
        
        return result
    
    # Apply different kernels
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    for idx, (name, kernel) in enumerate(kernels.items()):
        row = (idx + 1) // 3
        col = (idx + 1) % 3
        
        result = manual_conv2d(image, kernel)
        
        axes[row, col].imshow(result, cmap='gray')
        axes[row, col].set_title(f'{name} Kernel')
        axes[row, col].axis('off')
        
        # Add text annotations
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                axes[row, col].text(j, i, f'{result[i, j]:.1f}', 
                                  ha='center', va='center', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Compare with scipy convolution
    from scipy import signal
    
    print("Manual vs Scipy convolution comparison:")
    for name, kernel in kernels.items():
        manual_result = manual_conv2d(image, kernel)
        scipy_result = signal.convolve2d(image, kernel, mode='same')
        
        print(f"{name}: {'Match' if np.allclose(manual_result, scipy_result) else 'Different'}")

convolution_2d_demo()
```

### Fourier Transform

The Fourier Transform decomposes a signal into its frequency components:

$$F(\omega) = \int_{-\infty}^{\infty} f(t) e^{-j\omega t} dt$$

For discrete signals (DFT):

$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

```python
def fourier_transform_demo():
    # Create a signal with multiple frequencies
    t = np.linspace(0, 1, 1000)
    signal = (np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
              0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
              0.3 * np.sin(2 * np.pi * 100 * t))  # 100 Hz component
    
    # Compute FFT
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])
    
    # Compute magnitude spectrum
    magnitude = np.abs(fft_result)
    
    # Plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(t, signal)
    plt.title('Time Domain Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(frequencies[:len(frequencies)//2], magnitude[:len(magnitude)//2])
    plt.title('Frequency Domain (Magnitude Spectrum)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find peak frequencies
    peak_indices = np.argsort(magnitude[:len(magnitude)//2])[-3:]
    peak_frequencies = frequencies[peak_indices]
    
    print("Peak frequencies found:")
    for i, freq in enumerate(peak_frequencies):
        print(f"Peak {i+1}: {abs(freq):.1f} Hz")

fourier_transform_demo()
```

## Geometric Transformations

### Affine Transformations

Affine transformations preserve parallel lines and include translation, rotation, scaling, and shearing.

#### Translation

Translation moves every point by a constant offset:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} x \\ y \end{pmatrix} + \begin{pmatrix} t_x \\ t_y \end{pmatrix}$$

```python
def translation_demo():
    # Create a simple shape
    points = np.array([
        [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]  # Square
    ])
    
    # Translation vector
    tx, ty = 2, 1
    
    # Apply translation
    translated_points = points + np.array([tx, ty])
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original')
    plt.plot(translated_points[:, 0], translated_points[:, 1], 'r-', linewidth=2, label='Translated')
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title(f'Translation by ({tx}, {ty})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()

translation_demo()
```

#### Rotation

Rotation around the origin by angle $\theta$:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

```python
def rotation_demo():
    # Create a simple shape
    points = np.array([
        [0, 0], [2, 0], [2, 1], [0, 1], [0, 0]  # Rectangle
    ])
    
    # Rotation angle (in radians)
    theta = np.pi / 4  # 45 degrees
    
    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # Apply rotation
    rotated_points = points @ rotation_matrix.T
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original')
    plt.plot(rotated_points[:, 0], rotated_points[:, 1], 'r-', linewidth=2, label='Rotated')
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title(f'Rotation by {np.degrees(theta):.0f}°')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()
    
    print(f"Rotation matrix:\n{rotation_matrix}")

rotation_demo()
```

#### Scaling

Scaling changes the size of objects:

$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} s_x & 0 \\ 0 & s_y \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

```python
def scaling_demo():
    # Create a simple shape
    points = np.array([
        [0, 0], [2, 0], [2, 1], [0, 1], [0, 0]  # Rectangle
    ])
    
    # Scaling factors
    sx, sy = 1.5, 2.0
    
    # Scaling matrix
    scaling_matrix = np.array([
        [sx, 0],
        [0, sy]
    ])
    
    # Apply scaling
    scaled_points = points @ scaling_matrix.T
    
    # Plotting
    plt.figure(figsize=(10, 6))
    
    plt.plot(points[:, 0], points[:, 1], 'b-', linewidth=2, label='Original')
    plt.plot(scaled_points[:, 0], scaled_points[:, 1], 'r-', linewidth=2, label='Scaled')
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title(f'Scaling by ({sx}, {sy})')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.show()
    
    print(f"Scaling matrix:\n{scaling_matrix}")

scaling_demo()
```

### Homography Transformations

Homography transformations can handle perspective changes and are represented by a 3×3 matrix:

$$\begin{pmatrix} x' \\ y' \\ w' \end{pmatrix} = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

```python
def homography_demo():
    # Create a simple image (grid pattern)
    size = 100
    image = np.zeros((size, size))
    
    # Create grid pattern
    for i in range(0, size, 10):
        image[i, :] = 1
        image[:, i] = 1
    
    # Define source and destination points for homography
    src_points = np.array([
        [0, 0], [size-1, 0], [size-1, size-1], [0, size-1]
    ], dtype=np.float32)
    
    dst_points = np.array([
        [10, 20], [size-10, 10], [size-20, size-10], [20, size-20]
    ], dtype=np.float32)
    
    # Calculate homography matrix
    homography_matrix = cv2.findHomography(src_points, dst_points)[0]
    
    # Apply homography transformation
    transformed_image = cv2.warpPerspective(image, homography_matrix, (size, size))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title('Transformed Image (Homography)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Homography matrix:")
    print(homography_matrix)

homography_demo()
```

## Summary

This guide covered the fundamental concepts in computer vision:

1. **Image Representation**: Understanding how digital images are stored as arrays
2. **Color Spaces**: RGB, HSV, and other color representations
3. **Mathematical Foundations**: Convolution operations and Fourier transforms
4. **Geometric Transformations**: Affine and homography transformations

These concepts form the building blocks for more advanced computer vision techniques. Understanding these fundamentals is crucial for implementing and understanding complex computer vision algorithms.

### Key Takeaways

- Digital images are 2D arrays of pixel values
- Color spaces provide different ways to represent color information
- Convolution is essential for filtering and feature extraction
- Fourier transforms reveal frequency domain information
- Geometric transformations preserve or modify spatial relationships

### Next Steps

With these fundamentals in place, you can now explore:
- Image processing techniques (filtering, enhancement)
- Feature detection and description
- Object detection and recognition
- Deep learning approaches in computer vision 