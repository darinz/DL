# 3D Vision

## 1. Overview

3D vision involves understanding and processing three-dimensional data, including point clouds, meshes, and volumetric representations. This field is crucial for robotics, autonomous vehicles, augmented reality, and computer graphics.

**Mathematical Representation:**
```math
P = \{p_i = (x_i, y_i, z_i) \in \mathbb{R}^3 : i = 1, 2, ..., N\}
```

Where $P$ is a point cloud with $N$ points in 3D space.

## 2. Point Cloud Processing

### PointNet Architecture

PointNet is a deep learning architecture designed for point cloud processing.

#### Architecture
**Input Transformation:**
```math
T_1 = \text{MLP}(P) \in \mathbb{R}^{N \times 64}
```

**Feature Transformation:**
```math
T_2 = \text{MLP}(T_1) \in \mathbb{R}^{N \times 1024}
```

**Global Feature:**
```math
F_{global} = \max_{i} T_2(i) \in \mathbb{R}^{1024}
```

**Classification/Regression:**
```math
Y = \text{MLP}(F_{global}) \in \mathbb{R}^{C}
```

#### Permutation Invariance
**Symmetric Function:**
```math
f(\{x_1, ..., x_n\}) = \gamma \circ g(h(x_1), ..., h(x_n))
```

Where:
- $h$ is a shared MLP
- $g$ is a symmetric function (max pooling)
- $\gamma$ is a final MLP

### PointNet++

PointNet++ extends PointNet with hierarchical feature learning.

#### Hierarchical Sampling
**Farthest Point Sampling (FPS):**
```math
p_{i+1} = \arg\max_{p \in P \setminus \{p_1, ..., p_i\}} \min_{j \leq i} \|p - p_j\|_2
```

#### Grouping
**Ball Query:**
```math
N(p_i, r) = \{p_j : \|p_i - p_j\|_2 \leq r\}
```

**K-Nearest Neighbors:**
```math
N(p_i, k) = \{p_j : j \in \text{top-k}(\|p_i - p_j\|_2)\}
```

#### Feature Aggregation
**Multi-scale Grouping:**
```math
F_i = \text{concat}(F_i^1, F_i^2, ..., F_i^S)
```

Where $F_i^s$ is the feature at scale $s$.

## 3. Voxel-Based Methods

### VoxelNet

VoxelNet converts point clouds to voxels for 3D object detection.

#### Voxelization
**Point to Voxel Assignment:**
```math
v_{ijk} = \{p \in P : \lfloor p_x/v \rfloor = i, \lfloor p_y/v \rfloor = j, \lfloor p_z/v \rfloor = k\}
```

Where $v$ is the voxel size.

#### Voxel Feature Encoding (VFE)
**VFE Layer:**
```math
F_{out} = \max_{p \in v} \text{concat}(p, F_{in}(p))
```

#### Convolutional Middle Layers
**3D Convolution:**
```math
F_{i+1} = \text{Conv3D}(F_i, W_i) + b_i
```

### PointPillars

PointPillars uses pillars (vertical columns) for efficient 3D detection.

#### Pillar Generation
**Pillar Assignment:**
```math
p_{ij} = \{p \in P : \lfloor p_x/d_x \rfloor = i, \lfloor p_y/d_y \rfloor = j\}
```

Where $d_x, d_y$ are pillar dimensions.

#### Pillar Feature Net
**Feature Encoding:**
```math
F_{pillar} = \text{PFN}(p_{ij}) \in \mathbb{R}^{C}
```

#### 2D Convolutional Backbone
**Pseudo-image:**
```math
I_{pseudo} = \text{reshape}(F_{pillars}) \in \mathbb{R}^{H \times W \times C}
```

## 4. Multi-View Geometry

### Epipolar Geometry

#### Fundamental Matrix
**Epipolar Constraint:**
```math
x_2^T F x_1 = 0
```

**Fundamental Matrix:**
```math
F = K_2^{-T} [t]_{\times} R K_1^{-1}
```

Where $[t]_{\times}$ is the skew-symmetric matrix of translation vector.

#### Essential Matrix
**Essential Matrix:**
```math
E = [t]_{\times} R
```

**Relationship:**
```math
F = K_2^{-T} E K_1^{-1}
```

### Structure from Motion (SfM)

#### Triangulation
**Linear Triangulation:**
```math
\begin{bmatrix}
x_1 p_1^3 - p_1^1 \\
y_1 p_1^3 - p_1^2 \\
x_2 p_2^3 - p_2^1 \\
y_2 p_2^3 - p_2^2
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z \\
1
\end{bmatrix} = 0
```

Where $p_i^j$ is the $j$-th row of projection matrix $P_i$.

#### Bundle Adjustment
**Cost Function:**
```math
\min_{P_i, X_j} \sum_{i,j} \|x_{ij} - P_i X_j\|_2^2
```

### SLAM (Simultaneous Localization and Mapping)

#### Visual SLAM
**Feature Matching:**
```math
M_{ij} = \text{match}(f_i, f_j)
```

**Pose Estimation:**
```math
T_{i+1} = \arg\min_T \sum_j \|x_j - \pi(T X_j)\|_2^2
```

Where $\pi$ is the projection function.

#### Loop Closure
**Similarity Score:**
```math
S_{ij} = \text{similarity}(F_i, F_j)
```

## 5. 3D Reconstruction

### Stereo Vision

#### Disparity Computation
**Disparity:**
```math
d = x_l - x_r
```

**Depth:**
```math
Z = \frac{f \cdot B}{d}
```

Where:
- $f$ is focal length
- $B$ is baseline
- $d$ is disparity

#### Stereo Matching
**Cost Function:**
```math
C(x, y, d) = \|I_l(x, y) - I_r(x-d, y)\|
```

**Semi-Global Matching (SGM):**
```math
L_r(p, d) = C(p, d) + \min(L_r(p-r, d), L_r(p-r, d\pm1) + P_1, \min_k L_r(p-r, k) + P_2)
```

### Multi-View Stereo (MVS)

#### PatchMatch Stereo
**Patch Similarity:**
```math
S(p, q) = \sum_{i,j} w(i, j) \|I_1(p + (i,j)) - I_2(q + (i,j))\|_2
```

**Depth Refinement:**
```math
d_{new} = d_{old} + \Delta d
```

#### COLMAP
**Photometric Consistency:**
```math
C(p) = \sum_{i,j} \|I_i(p) - I_j(p)\|_2
```

**Geometric Consistency:**
```math
G(p) = \sum_{i,j} \|D_i(p) - D_j(p)\|_2
```

### Deep Learning for 3D Reconstruction

#### Learning-based MVS
**Cost Volume:**
```math
C(d) = \text{concat}(I_1, I_2(d), I_3(d), ...)
```

**Depth Prediction:**
```math
D = \text{softmax}(\text{Conv3D}(C))
```

#### NeRF (Neural Radiance Fields)
**Volume Rendering:**
```math
C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt
```

**Transmittance:**
```math
T(t) = \exp\left(-\int_{t_n}^t \sigma(r(s)) ds\right)
```

## 6. Python Implementation Examples

### Basic Point Cloud Processing

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

# Create synthetic point cloud
def create_synthetic_point_cloud(num_points=1000):
    """Create synthetic 3D point cloud."""
    # Create a simple 3D shape (cube with noise)
    points = np.random.rand(num_points, 3) * 2 - 1  # Cube from -1 to 1
    
    # Add some structure
    # Create a sphere
    sphere_center = np.array([0.5, 0.5, 0.5])
    sphere_radius = 0.3
    sphere_points = np.random.randn(num_points//2, 3)
    sphere_points = sphere_points / np.linalg.norm(sphere_points, axis=1, keepdims=True) * sphere_radius
    sphere_points += sphere_center
    
    # Create a plane
    plane_points = np.random.rand(num_points//2, 3)
    plane_points[:, 2] = -0.5  # Fixed z-coordinate
    
    # Combine points
    all_points = np.vstack([sphere_points, plane_points])
    
    # Add noise
    noise = np.random.normal(0, 0.05, all_points.shape)
    all_points += noise
    
    return all_points

# Point cloud visualization
def visualize_point_cloud(points, title="Point Cloud"):
    """Visualize 3D point cloud."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap='viridis', s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    plt.show()

# Point cloud downsampling
def downsample_point_cloud(points, target_size=500):
    """Downsample point cloud using random sampling."""
    if len(points) <= target_size:
        return points
    
    indices = np.random.choice(len(points), target_size, replace=False)
    return points[indices]

# Farthest Point Sampling (FPS)
def farthest_point_sampling(points, num_samples):
    """Implement Farthest Point Sampling."""
    if len(points) <= num_samples:
        return points
    
    # Initialize with random point
    sampled_indices = [np.random.randint(len(points))]
    remaining_indices = list(range(len(points)))
    remaining_indices.remove(sampled_indices[0])
    
    for _ in range(num_samples - 1):
        # Calculate distances to sampled points
        distances = cdist(points[remaining_indices], points[sampled_indices])
        min_distances = np.min(distances, axis=1)
        
        # Find farthest point
        farthest_idx = remaining_indices[np.argmax(min_distances)]
        sampled_indices.append(farthest_idx)
        remaining_indices.remove(farthest_idx)
    
    return points[sampled_indices]

# Point cloud clustering
def cluster_point_cloud(points, eps=0.1, min_samples=5):
    """Cluster point cloud using DBSCAN."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    return labels

# Point cloud registration (ICP)
def iterative_closest_point(source, target, max_iterations=50, tolerance=1e-6):
    """Implement Iterative Closest Point algorithm."""
    # Initialize transformation
    transformation = np.eye(4)
    
    for iteration in range(max_iterations):
        # Find closest points
        distances = cdist(source, target)
        correspondences = np.argmin(distances, axis=1)
        
        # Calculate centroids
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target[correspondences], axis=0)
        
        # Center points
        source_centered = source - source_centroid
        target_centered = target[correspondences] - target_centroid
        
        # Calculate rotation matrix
        H = source_centered.T @ target_centered
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure proper rotation matrix
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # Calculate translation
        t = target_centroid - R @ source_centroid
        
        # Update transformation
        current_transformation = np.eye(4)
        current_transformation[:3, :3] = R
        current_transformation[:3, 3] = t
        
        transformation = current_transformation @ transformation
        
        # Transform source points
        source_homogeneous = np.hstack([source, np.ones((len(source), 1))])
        source = (transformation @ source_homogeneous.T).T[:, :3]
        
        # Check convergence
        if np.linalg.norm(t) < tolerance:
            break
    
    return transformation, source

# Voxelization
def voxelize_point_cloud(points, voxel_size=0.1):
    """Convert point cloud to voxel grid."""
    # Calculate voxel indices
    voxel_indices = np.floor(points / voxel_size).astype(int)
    
    # Find unique voxels
    unique_voxels, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    
    # Calculate voxel centers
    voxel_centers = unique_voxels * voxel_size + voxel_size / 2
    
    return voxel_centers, unique_voxels

# Point cloud normal estimation
def estimate_normals(points, k_neighbors=10):
    """Estimate surface normals using PCA."""
    from sklearn.neighbors import NearestNeighbors
    
    # Find k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    normals = np.zeros_like(points)
    
    for i in range(len(points)):
        # Get neighbors
        neighbor_points = points[indices[i]]
        
        # Calculate covariance matrix
        centered_points = neighbor_points - np.mean(neighbor_points, axis=0)
        cov_matrix = centered_points.T @ centered_points
        
        # Find eigenvector with smallest eigenvalue (normal direction)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        normal = eigenvectors[:, 0]  # Smallest eigenvalue
        
        # Ensure consistent orientation (pointing outward)
        if normal[2] < 0:
            normal = -normal
        
        normals[i] = normal
    
    return normals

# Main demonstration
def demonstrate_point_cloud_processing():
    """Demonstrate various point cloud processing techniques."""
    # Create synthetic point cloud
    points = create_synthetic_point_cloud(2000)
    
    # Visualize original point cloud
    visualize_point_cloud(points, "Original Point Cloud")
    
    # Downsampling
    downsampled = downsample_point_cloud(points, target_size=500)
    visualize_point_cloud(downsampled, "Downsampled Point Cloud")
    
    # Farthest Point Sampling
    fps_points = farthest_point_sampling(points, num_samples=300)
    visualize_point_cloud(fps_points, "FPS Sampled Point Cloud")
    
    # Clustering
    labels = cluster_point_cloud(points, eps=0.15, min_samples=10)
    
    # Visualize clusters
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for label, color in zip(unique_labels, colors):
        if label == -1:  # Noise points
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                      c='black', s=1, alpha=0.3)
        else:
            mask = labels == label
            ax.scatter(points[mask, 0], points[mask, 1], points[mask, 2], 
                      c=[color], s=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Clustered Point Cloud')
    plt.show()
    
    # Voxelization
    voxel_centers, voxel_indices = voxelize_point_cloud(points, voxel_size=0.1)
    visualize_point_cloud(voxel_centers, "Voxelized Point Cloud")
    
    # Normal estimation
    normals = estimate_normals(points, k_neighbors=15)
    
    # Visualize normals
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Sample points for visualization
    sample_indices = np.random.choice(len(points), 100, replace=False)
    sample_points = points[sample_indices]
    sample_normals = normals[sample_indices]
    
    ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2], 
              c='blue', s=10)
    
    # Draw normals
    for point, normal in zip(sample_points, sample_normals):
        end_point = point + normal * 0.1
        ax.plot([point[0], end_point[0]], 
                [point[1], end_point[1]], 
                [point[2], end_point[2]], 'r-', linewidth=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud with Normals')
    plt.show()
    
    print(f"Original points: {len(points)}")
    print(f"Downsampled points: {len(downsampled)}")
    print(f"FPS points: {len(fps_points)}")
    print(f"Voxels: {len(voxel_centers)}")
    print(f"Clusters: {len(np.unique(labels)) - 1}")  # Exclude noise

# Stereo vision simulation
def simulate_stereo_vision():
    """Simulate stereo vision for 3D reconstruction."""
    # Create synthetic 3D points
    x = np.linspace(-2, 2, 50)
    y = np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * np.sin(X) * np.cos(Y)  # Simple surface
    
    # Flatten and create point cloud
    points_3d = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
    
    # Camera parameters
    focal_length = 1000
    baseline = 0.1
    image_width = 800
    image_height = 600
    
    # Project to left and right cameras
    left_points = []
    right_points = []
    
    for point in points_3d:
        # Left camera (at origin)
        if point[2] > 0:  # Only points in front of camera
            x_left = focal_length * point[0] / point[2] + image_width / 2
            y_left = focal_length * point[1] / point[2] + image_height / 2
            
            # Right camera (translated by baseline)
            x_right = focal_length * (point[0] - baseline) / point[2] + image_width / 2
            y_right = focal_length * point[1] / point[2] + image_height / 2
            
            if (0 <= x_left < image_width and 0 <= y_left < image_height and
                0 <= x_right < image_width and 0 <= y_right < image_height):
                left_points.append([x_left, y_left])
                right_points.append([x_right, y_right])
    
    left_points = np.array(left_points)
    right_points = np.array(right_points)
    
    # Calculate disparity
    disparity = left_points[:, 0] - right_points[:, 0]
    
    # Reconstruct 3D points
    reconstructed_z = focal_length * baseline / disparity
    reconstructed_x = (left_points[:, 0] - image_width / 2) * reconstructed_z / focal_length
    reconstructed_y = (left_points[:, 1] - image_height / 2) * reconstructed_z / focal_length
    
    reconstructed_points = np.column_stack([reconstructed_x, reconstructed_y, reconstructed_z])
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Original 3D surface
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=points_3d[:, 2], cmap='viridis', s=1)
    ax1.set_title('Original 3D Surface')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Left camera view
    axes[0, 1].scatter(left_points[:, 0], left_points[:, 1], c=disparity, cmap='viridis', s=1)
    axes[0, 1].set_title('Left Camera View')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].invert_yaxis()
    
    # Right camera view
    axes[1, 0].scatter(right_points[:, 0], right_points[:, 1], c=disparity, cmap='viridis', s=1)
    axes[1, 0].set_title('Right Camera View')
    axes[1, 0].set_xlabel('X')
    axes[1, 0].set_ylabel('Y')
    axes[1, 0].invert_yaxis()
    
    # Reconstructed 3D surface
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(reconstructed_points[:, 0], reconstructed_points[:, 1], reconstructed_points[:, 2], 
               c=reconstructed_points[:, 2], cmap='viridis', s=1)
    ax4.set_title('Reconstructed 3D Surface')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')
    
    plt.tight_layout()
    plt.show()
    
    # Calculate reconstruction error
    error = np.mean(np.linalg.norm(points_3d[:len(reconstructed_points)] - reconstructed_points, axis=1))
    print(f"Average reconstruction error: {error:.4f}")

# Main execution
if __name__ == "__main__":
    # Demonstrate point cloud processing
    demonstrate_point_cloud_processing()
    
    # Demonstrate stereo vision
    simulate_stereo_vision()
```

### Advanced 3D Vision Techniques

```python
# Multi-view stereo reconstruction
def multi_view_stereo_reconstruction(images, camera_matrices, depths):
    """Simulate multi-view stereo reconstruction."""
    # This is a simplified simulation
    # In practice, this would involve more complex algorithms
    
    # Create cost volume
    min_depth = np.min(depths)
    max_depth = np.max(depths)
    num_depths = 50
    depth_values = np.linspace(min_depth, max_depth, num_depths)
    
    # For each depth hypothesis, compute photometric consistency
    cost_volume = np.zeros((len(images[0]), len(images[0][0]), num_depths))
    
    for d_idx, depth in enumerate(depth_values):
        for i in range(len(images[0])):
            for j in range(len(images[0][0])):
                # Project point to other views
                costs = []
                for view_idx in range(1, len(images)):
                    # Simplified projection
                    # In practice, this would use proper camera projection
                    projected_i = int(i + np.random.normal(0, 2))
                    projected_j = int(j + np.random.normal(0, 2))
                    
                    if (0 <= projected_i < len(images[view_idx]) and 
                        0 <= projected_j < len(images[view_idx][0])):
                        cost = abs(images[0][i][j] - images[view_idx][projected_i][projected_j])
                        costs.append(cost)
                
                if costs:
                    cost_volume[i, j, d_idx] = np.mean(costs)
    
    # Find best depth for each pixel
    best_depths = depth_values[np.argmin(cost_volume, axis=2)]
    
    return best_depths

# Point cloud registration with RANSAC
def ransac_point_cloud_registration(source, target, num_iterations=1000, threshold=0.1):
    """Implement RANSAC-based point cloud registration."""
    best_transformation = None
    best_inliers = 0
    
    for _ in range(num_iterations):
        # Randomly sample 3 points
        sample_indices = np.random.choice(len(source), 3, replace=False)
        sample_points = source[sample_indices]
        
        # Find corresponding points in target
        distances = cdist(sample_points, target)
        correspondences = np.argmin(distances, axis=1)
        target_points = target[correspondences]
        
        # Estimate transformation
        try:
            # Calculate transformation using SVD
            source_centroid = np.mean(sample_points, axis=0)
            target_centroid = np.mean(target_points, axis=0)
            
            source_centered = sample_points - source_centroid
            target_centered = target_points - target_centroid
            
            H = source_centered.T @ target_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T
            
            t = target_centroid - R @ source_centroid
            
            # Transform all source points
            transformed_source = (R @ source.T).T + t
            
            # Count inliers
            distances = np.linalg.norm(transformed_source - target, axis=1)
            inliers = np.sum(distances < threshold)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_transformation = np.eye(4)
                best_transformation[:3, :3] = R
                best_transformation[:3, 3] = t
                
        except np.linalg.LinAlgError:
            continue
    
    return best_transformation, best_inliers

# Surface reconstruction using Poisson
def poisson_surface_reconstruction(points, normals):
    """Simulate Poisson surface reconstruction."""
    # This is a simplified simulation
    # In practice, this would involve solving a Poisson equation
    
    # Create a simple mesh from points and normals
    from scipy.spatial import Delaunay
    
    # Project points to 2D for triangulation
    # Use PCA to find principal components
    centered_points = points - np.mean(points, axis=0)
    cov_matrix = centered_points.T @ centered_points
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Project to 2D using first two principal components
    projection_matrix = eigenvectors[:, :2]
    points_2d = centered_points @ projection_matrix
    
    # Triangulate
    try:
        tri = Delaunay(points_2d)
        triangles = tri.simplices
    except:
        # Fallback: create simple triangles
        triangles = []
        for i in range(0, len(points) - 2, 3):
            if i + 2 < len(points):
                triangles.append([i, i+1, i+2])
        triangles = np.array(triangles)
    
    return triangles

# 3D object detection simulation
def simulate_3d_object_detection(point_cloud):
    """Simulate 3D object detection in point cloud."""
    # Simple clustering-based detection
    
    # Cluster points
    labels = cluster_point_cloud(point_cloud, eps=0.2, min_samples=10)
    
    # Find bounding boxes for each cluster
    bounding_boxes = []
    
    for label in np.unique(labels):
        if label == -1:  # Skip noise
            continue
        
        cluster_points = point_cloud[labels == label]
        
        if len(cluster_points) < 10:  # Skip small clusters
            continue
        
        # Calculate bounding box
        min_coords = np.min(cluster_points, axis=0)
        max_coords = np.max(cluster_points, axis=0)
        
        # Calculate center and dimensions
        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords
        
        bounding_boxes.append({
            'center': center,
            'dimensions': dimensions,
            'points': cluster_points,
            'label': label
        })
    
    return bounding_boxes

# Visualize 3D bounding boxes
def visualize_3d_bounding_boxes(point_cloud, bounding_boxes):
    """Visualize 3D bounding boxes."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot point cloud
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], 
              c='gray', s=1, alpha=0.5)
    
    # Plot bounding boxes
    colors = plt.cm.tab10(np.linspace(0, 1, len(bounding_boxes)))
    
    for bbox, color in zip(bounding_boxes, colors):
        center = bbox['center']
        dimensions = bbox['dimensions']
        
        # Create bounding box vertices
        x_min, y_min, z_min = center - dimensions / 2
        x_max, y_max, z_max = center + dimensions / 2
        
        # Define vertices
        vertices = np.array([
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max]
        ])
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
            (4, 5), (5, 6), (6, 7), (7, 4),  # top
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical
        ]
        
        # Draw edges
        for edge in edges:
            start = vertices[edge[0]]
            end = vertices[edge[1]]
            ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                   color=color, linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Object Detection')
    plt.show()
```

This comprehensive guide covers various 3D vision techniques, from basic point cloud processing to advanced reconstruction methods. The mathematical foundations provide understanding of the algorithms, while the Python implementations demonstrate practical applications in 3D computer vision. 