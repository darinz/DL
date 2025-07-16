# Image Processing Basics

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

#### Mean Filter
Simple averaging filter that reduces noise but blurs edges.

**Kernel:**
```math
K = \frac{1}{n^2} \begin{bmatrix} 1 & 1 & \cdots & 1 \\ 1 & 1 & \cdots & 1 \\ \vdots & \vdots & \ddots & \vdots \\ 1 & 1 & \cdots & 1 \end{bmatrix}
```

#### Median Filter
Non-linear filter that preserves edges while removing salt-and-pepper noise.

**Operation:**
```math
I'(x, y) = \text{median}\{I(i, j) : (i, j) \in N(x, y)\}
```

Where $N(x, y)$ is the neighborhood around pixel $(x, y)$.

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

## 6. Python Implementation Examples

### Basic Filtering Operations

```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import convolve2d

# Create test image with noise
def create_noisy_image(size=(256, 256)):
    """Create a test image with various noise types."""
    # Create base image with geometric shapes
    image = np.zeros(size, dtype=np.uint8)
    
    # Add a circle
    center = (size[0]//2, size[1]//2)
    radius = min(size) // 4
    cv2.circle(image, center, radius, 128, -1)
    
    # Add a rectangle
    rect_start = (50, 50)
    rect_end = (150, 100)
    cv2.rectangle(image, rect_start, rect_end, 200, -1)
    
    # Add Gaussian noise
    gaussian_noise = np.random.normal(0, 20, size).astype(np.uint8)
    noisy_gaussian = cv2.add(image, gaussian_noise)
    
    # Add salt and pepper noise
    salt_pepper = image.copy()
    noise_pixels = np.random.random(size) < 0.05
    salt_pepper[noise_pixels] = np.random.choice([0, 255], size=noise_pixels.sum())
    
    return image, noisy_gaussian, salt_pepper

# Linear filtering
def demonstrate_linear_filters(image):
    """Demonstrate various linear filtering techniques."""
    # Gaussian filter
    gaussian_filtered = cv2.GaussianBlur(image, (15, 15), 2)
    
    # Mean filter
    mean_filtered = cv2.blur(image, (15, 15))
    
    # Custom Gaussian kernel
    def create_gaussian_kernel(size, sigma):
        """Create a 2D Gaussian kernel."""
        kernel = np.zeros((size, size))
        center = size // 2
        
        for i in range(size):
            for j in range(size):
                x, y = i - center, j - center
                kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        return kernel / np.sum(kernel)
    
    custom_gaussian_kernel = create_gaussian_kernel(15, 2)
    custom_gaussian_filtered = convolve2d(image, custom_gaussian_kernel, mode='same')
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(gaussian_filtered, cmap='gray')
    axes[0, 1].set_title('Gaussian Filter')
    axes[1, 0].imshow(mean_filtered, cmap='gray')
    axes[1, 0].set_title('Mean Filter')
    axes[1, 1].imshow(custom_gaussian_filtered, cmap='gray')
    axes[1, 1].set_title('Custom Gaussian Filter')
    
    plt.tight_layout()
    plt.show()

# Non-linear filtering
def demonstrate_nonlinear_filters(image):
    """Demonstrate non-linear filtering techniques."""
    # Median filter
    median_filtered = cv2.medianBlur(image, 5)
    
    # Bilateral filter
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)
    
    # Non-local means denoising
    nlm_filtered = cv2.fastNlMeansDenoising(image)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(median_filtered, cmap='gray')
    axes[0, 1].set_title('Median Filter')
    axes[1, 0].imshow(bilateral_filtered, cmap='gray')
    axes[1, 0].set_title('Bilateral Filter')
    axes[1, 1].imshow(nlm_filtered, cmap='gray')
    axes[1, 1].set_title('Non-local Means')
    
    plt.tight_layout()
    plt.show()

# Edge detection
def demonstrate_edge_detection(image):
    """Demonstrate various edge detection methods."""
    # Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255)
    
    # Laplacian edge detection
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Canny edge detection
    canny_edges = cv2.Canny(image, 50, 150)
    
    # Custom Sobel implementation
    def custom_sobel(image):
        """Custom Sobel edge detection implementation."""
        # Sobel kernels
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Apply convolution
        grad_x = convolve2d(image, sobel_x_kernel, mode='same')
        grad_y = convolve2d(image, sobel_y_kernel, mode='same')
        
        # Compute magnitude
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        return magnitude
    
    custom_sobel_result = custom_sobel(image)
    custom_sobel_result = np.uint8(custom_sobel_result / custom_sobel_result.max() * 255)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(sobel_magnitude, cmap='gray')
    axes[0, 1].set_title('Sobel Magnitude')
    axes[0, 2].imshow(laplacian, cmap='gray')
    axes[0, 2].set_title('Laplacian')
    axes[1, 0].imshow(canny_edges, cmap='gray')
    axes[1, 0].set_title('Canny Edges')
    axes[1, 1].imshow(custom_sobel_result, cmap='gray')
    axes[1, 1].set_title('Custom Sobel')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Morphological operations
def demonstrate_morphological_operations(image):
    """Demonstrate morphological operations."""
    # Create binary image for morphological operations
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    # Erosion
    eroded = cv2.erode(binary, kernel, iterations=1)
    
    # Dilation
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Opening (erosion followed by dilation)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Closing (dilation followed by erosion)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(binary, cmap='gray')
    axes[0, 1].set_title('Binary Image')
    axes[0, 2].imshow(eroded, cmap='gray')
    axes[0, 2].set_title('Erosion')
    axes[1, 0].imshow(dilated, cmap='gray')
    axes[1, 0].set_title('Dilation')
    axes[1, 1].imshow(opened, cmap='gray')
    axes[1, 1].set_title('Opening')
    axes[1, 2].imshow(closed, cmap='gray')
    axes[1, 2].set_title('Closing')
    
    plt.tight_layout()
    plt.show()

# Histogram processing
def demonstrate_histogram_processing(image):
    """Demonstrate histogram processing techniques."""
    # Histogram equalization
    equalized = cv2.equalizeHist(image)
    
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(image)
    
    # Calculate histograms
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized], [0], None, [256], [0, 256])
    hist_clahe = cv2.calcHist([clahe_result], [0], None, [256], [0, 256])
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Images
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title('Histogram Equalization')
    axes[0, 2].imshow(clahe_result, cmap='gray')
    axes[0, 2].set_title('CLAHE')
    
    # Histograms
    axes[1, 0].plot(hist_original)
    axes[1, 0].set_title('Original Histogram')
    axes[1, 0].set_xlabel('Pixel Intensity')
    axes[1, 0].set_ylabel('Frequency')
    
    axes[1, 1].plot(hist_equalized)
    axes[1, 1].set_title('Equalized Histogram')
    axes[1, 1].set_xlabel('Pixel Intensity')
    axes[1, 1].set_ylabel('Frequency')
    
    axes[1, 2].plot(hist_clahe)
    axes[1, 2].set_title('CLAHE Histogram')
    axes[1, 2].set_xlabel('Pixel Intensity')
    axes[1, 2].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

# Noise reduction comparison
def demonstrate_noise_reduction(original, noisy_gaussian, noisy_salt_pepper):
    """Compare different noise reduction techniques."""
    # Gaussian noise reduction
    gaussian_median = cv2.medianBlur(noisy_gaussian, 5)
    gaussian_bilateral = cv2.bilateralFilter(noisy_gaussian, 9, 75, 75)
    gaussian_gaussian = cv2.GaussianBlur(noisy_gaussian, (5, 5), 1)
    
    # Salt and pepper noise reduction
    sp_median = cv2.medianBlur(noisy_salt_pepper, 5)
    sp_bilateral = cv2.bilateralFilter(noisy_salt_pepper, 9, 75, 75)
    sp_gaussian = cv2.GaussianBlur(noisy_salt_pepper, (5, 5), 1)
    
    # Display results
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Gaussian noise results
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title('Original')
    axes[0, 1].imshow(noisy_gaussian, cmap='gray')
    axes[0, 1].set_title('Gaussian Noise')
    axes[0, 2].imshow(gaussian_median, cmap='gray')
    axes[0, 2].set_title('Median Filter')
    axes[0, 3].imshow(gaussian_bilateral, cmap='gray')
    axes[0, 3].set_title('Bilateral Filter')
    
    # Salt and pepper noise results
    axes[1, 0].imshow(original, cmap='gray')
    axes[1, 0].set_title('Original')
    axes[1, 1].imshow(noisy_salt_pepper, cmap='gray')
    axes[1, 1].set_title('Salt & Pepper Noise')
    axes[1, 2].imshow(sp_median, cmap='gray')
    axes[1, 2].set_title('Median Filter')
    axes[1, 3].imshow(sp_bilateral, cmap='gray')
    axes[1, 3].set_title('Bilateral Filter')
    
    # Comparison histograms
    axes[2, 0].hist(original.ravel(), bins=50, alpha=0.7, label='Original')
    axes[2, 0].hist(noisy_gaussian.ravel(), bins=50, alpha=0.7, label='Gaussian Noise')
    axes[2, 0].set_title('Gaussian Noise Histogram')
    axes[2, 0].legend()
    
    axes[2, 1].hist(original.ravel(), bins=50, alpha=0.7, label='Original')
    axes[2, 1].hist(noisy_salt_pepper.ravel(), bins=50, alpha=0.7, label='Salt & Pepper')
    axes[2, 1].set_title('Salt & Pepper Histogram')
    axes[2, 1].legend()
    
    axes[2, 2].hist(gaussian_median.ravel(), bins=50, alpha=0.7, label='Median Filtered')
    axes[2, 2].set_title('Median Filter Result')
    axes[2, 2].legend()
    
    axes[2, 3].hist(gaussian_bilateral.ravel(), bins=50, alpha=0.7, label='Bilateral Filtered')
    axes[2, 3].set_title('Bilateral Filter Result')
    axes[2, 3].legend()
    
    plt.tight_layout()
    plt.show()

# Main demonstration
if __name__ == "__main__":
    # Create test images
    original, noisy_gaussian, noisy_salt_pepper = create_noisy_image()
    
    # Demonstrate linear filters
    demonstrate_linear_filters(noisy_gaussian)
    
    # Demonstrate non-linear filters
    demonstrate_nonlinear_filters(noisy_gaussian)
    
    # Demonstrate edge detection
    demonstrate_edge_detection(original)
    
    # Demonstrate morphological operations
    demonstrate_morphological_operations(original)
    
    # Demonstrate histogram processing
    demonstrate_histogram_processing(original)
    
    # Demonstrate noise reduction
    demonstrate_noise_reduction(original, noisy_gaussian, noisy_salt_pepper)
```

### Advanced Techniques

```python
# Advanced filtering techniques
def advanced_filtering_demo():
    """Demonstrate advanced filtering techniques."""
    # Create a complex test image
    image = np.zeros((256, 256), dtype=np.uint8)
    
    # Add multiple shapes with different intensities
    cv2.circle(image, (128, 128), 50, 100, -1)
    cv2.rectangle(image, (50, 50), (100, 100), 150, -1)
    cv2.rectangle(image, (180, 180), (220, 220), 200, -1)
    
    # Add texture
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    
    # Wiener filter implementation
    def wiener_filter(image, noise_var=100):
        """Simple Wiener filter implementation."""
        # Convert to frequency domain
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        
        # Create Wiener filter
        rows, cols = image.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create frequency coordinates
        u, v = np.meshgrid(np.arange(cols), np.arange(rows))
        
        # Wiener filter transfer function
        H = 1 / (1 + noise_var / (np.abs(f_shift)**2 + 1e-10))
        
        # Apply filter
        filtered_shift = f_shift * H
        filtered_transform = np.fft.ifftshift(filtered_shift)
        filtered_image = np.real(np.fft.ifft2(filtered_transform))
        
        return np.clip(filtered_image, 0, 255).astype(np.uint8)
    
    # Apply different filters
    gaussian = cv2.GaussianBlur(image, (15, 15), 2)
    bilateral = cv2.bilateralFilter(image, 9, 75, 75)
    wiener = wiener_filter(image, noise_var=100)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 1].imshow(gaussian, cmap='gray')
    axes[0, 1].set_title('Gaussian Filter')
    axes[1, 0].imshow(bilateral, cmap='gray')
    axes[1, 0].set_title('Bilateral Filter')
    axes[1, 1].imshow(wiener, cmap='gray')
    axes[1, 1].set_title('Wiener Filter')
    
    plt.tight_layout()
    plt.show()

# Multi-scale edge detection
def multi_scale_edge_detection(image):
    """Demonstrate multi-scale edge detection."""
    # Apply Gaussian blur at different scales
    scales = [1, 2, 4, 8]
    edge_images = []
    
    for scale in scales:
        # Blur image
        blurred = cv2.GaussianBlur(image, (0, 0), scale)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        edge_images.append(edges)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, (scale, edges) in enumerate(zip(scales, edge_images)):
        row, col = i // 2, i % 2
        axes[row, col].imshow(edges, cmap='gray')
        axes[row, col].set_title(f'Scale σ={scale}')
    
    plt.tight_layout()
    plt.show()

# Advanced morphological operations
def advanced_morphology_demo():
    """Demonstrate advanced morphological operations."""
    # Create a complex binary image
    image = np.zeros((200, 200), dtype=np.uint8)
    
    # Add multiple objects
    cv2.circle(image, (50, 50), 20, 255, -1)
    cv2.circle(image, (150, 50), 15, 255, -1)
    cv2.rectangle(image, (50, 120), (150, 180), 255, -1)
    
    # Add noise
    noise = np.random.random(image.shape) < 0.1
    image[noise] = 255
    
    # Define different structuring elements
    kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    
    # Apply morphological operations
    results = {}
    kernels = {'Rectangle': kernel_rect, 'Ellipse': kernel_ellipse, 'Cross': kernel_cross}
    
    for name, kernel in kernels.items():
        results[f'{name}_erosion'] = cv2.erode(image, kernel)
        results[f'{name}_dilation'] = cv2.dilate(image, kernel)
        results[f'{name}_opening'] = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        results[f'{name}_closing'] = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Display results
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    
    # Original image
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original')
    
    # Display results for each kernel
    for i, (name, kernel) in enumerate(kernels.items()):
        row = i + 1
        axes[row, 0].imshow(results[f'{name}_erosion'], cmap='gray')
        axes[row, 0].set_title(f'{name} Erosion')
        axes[row, 1].imshow(results[f'{name}_dilation'], cmap='gray')
        axes[row, 1].set_title(f'{name} Dilation')
        axes[row, 2].imshow(results[f'{name}_opening'], cmap='gray')
        axes[row, 2].set_title(f'{name} Opening')
        axes[row, 3].imshow(results[f'{name}_closing'], cmap='gray')
        axes[row, 3].set_title(f'{name} Closing')
    
    plt.tight_layout()
    plt.show()
```

This comprehensive guide covers the fundamental image processing techniques used in computer vision. Each section includes detailed mathematical formulations and practical Python implementations, making it suitable for both theoretical understanding and practical application. 