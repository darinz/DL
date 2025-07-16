# Fundamental Concepts in Computer Vision

> **Key Insight:** Understanding how images are represented, transformed, and processed is the foundation for all computer vision tasks.

> **Did you know?** The RGB color model was inspired by the way human eyes perceive color using three types of cones!

## 1. Image Representation

### Digital Images
A digital image is a 2D array of pixels, where each pixel represents the intensity or color value at that location.

> **Explanation:**
> A digital image is like a grid of tiny squares (pixels), where each square has a number that represents how bright or what color it should be. Think of it like a mosaic where each tile has a specific value.

**Mathematical Representation:**
```math
I(x, y) = \begin{cases}
f(x, y) & \text{for grayscale images} \\
(f_R(x, y), f_G(x, y), f_B(x, y)) & \text{for color images}
\end{cases}
```
> **Math Breakdown:**
> - $I(x, y)$: The image function that gives the pixel value at position $(x, y)$.
> - $(x, y)$: Spatial coordinates (like row and column numbers).
> - $f(x, y)$: Intensity value for grayscale images (0 = black, 255 = white).
> - $f_R, f_G, f_B$: Red, green, and blue channel values for color images.

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

> **Explanation:**
> Grayscale images have only one value per pixel, representing how bright that pixel is. 0 is black, 255 is white, and values in between are shades of gray.

#### Color Images
- Multiple channels (RGB, HSV, LAB, etc.)
- RGB: $I(x, y) = (R(x, y), G(x, y), B(x, y))$
- Each channel: $R(x, y), G(x, y), B(x, y) \in [0, 255]$

> **Explanation:**
> Color images use three numbers per pixel (for RGB), each representing the amount of red, green, or blue light. By combining these three values, we can create any color.

#### Multi-channel Images
- Hyperspectral: $I(x, y) = (I_1(x, y), I_2(x, y), ..., I_n(x, y))$
- Medical imaging: CT, MRI with multiple slices

> **Explanation:**
> Some images have more than three channels. Hyperspectral images might have hundreds of channels representing different wavelengths of light, while medical images might have multiple slices or different types of measurements.

> **Try it yourself!** Load an image with OpenCV or PIL and inspect its shape and channels.

## 2. Color Spaces

### RGB Color Space
The most common color space representing colors as combinations of Red, Green, and Blue.

```math
C_{RGB} = (R, G, B)
```
> **Math Breakdown:**
> - $C_{RGB}$: Color in RGB space.
> - $R, G, B$: Red, green, and blue components, each typically in range [0, 255].
> - Each component represents the intensity of that primary color.

**Properties:**
- Additive color model
- Device-dependent
- Not perceptually uniform

> **Explanation:**
> RGB is additive because you add light to create colors. It's device-dependent because the same RGB values might look different on different screens. It's not perceptually uniform because equal changes in RGB values don't correspond to equal changes in how humans perceive color.

### HSV Color Space
Hue, Saturation, Value color space that separates color information from intensity.

```math
C_{HSV} = (H, S, V)
```
> **Math Breakdown:**
> - $H$: Hue (0-360°) - what color it is (red, blue, green, etc.).
> - $S$: Saturation (0-1) - how pure the color is (0 = gray, 1 = pure color).
> - $V$: Value (0-1) - how bright the color is (0 = black, 1 = brightest).

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
> **Math Breakdown:**
> - $V$ is the maximum of the three RGB values, representing brightness.
> - $S$ measures how much the color differs from gray. If all RGB values are equal, the color is gray (saturation = 0).

### LAB Color Space
Perceptually uniform color space designed to approximate human vision.

```math
C_{LAB} = (L, a, b)
```
> **Math Breakdown:**
> - $L$: Lightness (0-100) - how bright the color appears to humans.
> - $a$: Green-Red axis (-128 to 127) - negative values are green, positive are red.
> - $b$: Blue-Yellow axis (-128 to 127) - negative values are blue, positive are yellow.

Where:
- $L \in [0, 100]$ (Lightness)
- $a \in [-128, 127]$ (Green-Red axis)
- $b \in [-128, 127]$ (Blue-Yellow axis)

> **Key Insight:** LAB is more perceptually uniform than RGB, making it useful for color-based segmentation and comparison.

## 3. Mathematical Foundations

### Convolution
A fundamental operation in image processing that combines two functions to produce a third function.

> **Explanation:**
> Convolution is like sliding a small window (called a kernel or filter) over an image and computing a weighted sum of the pixels under the window. This is used to detect patterns like edges, blur images, or enhance features.

**1D Convolution:**
```math
(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau
```
> **Math Breakdown:**
> - $f$: Input function (like an image row).
> - $g$: Kernel function (the filter).
> - $t$: Position where we're computing the result.
> - The integral multiplies $f$ and $g$ at each position and sums the results.

**2D Convolution (for images):**
```math
(I * K)(x, y) = \sum_{i=-\infty}^{\infty} \sum_{j=-\infty}^{\infty} I(i, j) K(x-i, y-j)
```
> **Math Breakdown:**
> - $I$: Input image.
> - $K$: 2D kernel/filter.
> - $(x, y)$: Position in the output image.
> - The double sum computes the weighted average of pixels around position $(x, y)$.

Where:
- $I$ is the input image
- $K$ is the kernel/filter
- $(x, y)$ are spatial coordinates

> **Geometric Intuition:** Convolution slides a small window (kernel) over the image, combining pixel values to detect patterns like edges or textures.

### Fourier Transform
Transforms an image from spatial domain to frequency domain.

> **Explanation:**
> The Fourier transform breaks down an image into its frequency components. Low frequencies represent smooth areas, while high frequencies represent edges and fine details. This is useful for filtering, compression, and understanding image structure.

**2D Discrete Fourier Transform:**
```math
F(u, v) = \frac{1}{MN} \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} f(x, y) e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}
```
> **Math Breakdown:**
> - $F(u, v)$: Frequency domain representation.
> - $f(x, y)$: Spatial domain image.
> - $M, N$: Image dimensions.
> - $e^{-j2\pi(\frac{ux}{M} + \frac{vy}{N})}$: Complex exponential representing different frequencies.

**Inverse 2D DFT:**
```math
f(x, y) = \sum_{u=0}^{M-1} \sum_{v=0}^{N-1} F(u, v) e^{j2\pi(\frac{ux}{M} + \frac{vy}{N})}
```
> **Math Breakdown:**
> This transforms the frequency domain back to the spatial domain, reconstructing the original image.

> **Did you know?** The Fourier transform is used in JPEG compression and many image filtering techniques.

### Sampling and Quantization
- **Sampling**: Converting continuous spatial coordinates to discrete grid
- **Quantization**: Converting continuous intensity values to discrete levels

> **Explanation:**
> Sampling is like taking a photo - you convert the continuous world into a grid of pixels. Quantization is like choosing how many shades of gray to use - you convert continuous brightness values into discrete numbers.

**Nyquist-Shannon Sampling Theorem:**
```math
f_s > 2f_{max}
```
> **Math Breakdown:**
> - $f_s$: Sampling frequency (how many samples per unit distance).
> - $f_{max}$: Highest frequency component in the signal.
> - You need to sample at least twice as fast as the highest frequency to avoid losing information.

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
> **Math Breakdown:**
> - $(x, y)$: Original pixel position.
> - $(x', y')$: New pixel position after translation.
> - $(t_x, t_y)$: Translation offset in x and y directions.

### Rotation
Rotating an image around a point (usually the center).

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```
> **Math Breakdown:**
> - $\theta$: Rotation angle in radians.
> - The 2×2 matrix is the rotation matrix that rotates points around the origin.
> - Positive angles rotate counterclockwise.

### Scaling
Resizing an image by factors $s_x$ and $s_y$.

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix}
```
> **Math Breakdown:**
> - $s_x, s_y$: Scaling factors for x and y directions.
> - Values > 1 make the image larger, values < 1 make it smaller.
> - If $s_x = s_y$, the scaling is uniform (maintains aspect ratio).

### Affine Transformation
Combines translation, rotation, scaling, and shearing.

```math
\begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix}
```
> **Math Breakdown:**
> - The 2×2 matrix handles rotation, scaling, and shearing.
> - The translation vector $(t_x, t_y)$ handles shifting.
> - This is the most general linear transformation in 2D.

### Homography (Perspective Transformation)
Handles perspective changes and projective transformations.

```math
\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}
```
> **Math Breakdown:**
> - This is a 3×3 matrix that handles perspective transformations.
> - The final coordinates are $(x'/w', y'/w')$ (homogeneous coordinates).
> - Can handle perspective changes like viewing a rectangle from an angle.

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
```
> **Code Walkthrough:**
> - Creates a 3-channel image filled with zeros (black).
> - Fills each pixel with RGB values based on its position, creating a gradient.
> - Draws a white circle in the center using OpenCV's circle function.

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
```
> **Code Walkthrough:**
> - Converts the input image to different color spaces using OpenCV.
> - Creates a 2×2 subplot to display the original and converted images.
> - Shows how the same image looks in different color representations.

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
```
> **Code Walkthrough:**
> - Applies different geometric transformations to the input image.
> - Translation moves the image by 50 pixels right and 30 pixels down.
> - Rotation rotates the image 45 degrees around its center.
> - Scaling makes the image 1.5 times larger in both dimensions. 