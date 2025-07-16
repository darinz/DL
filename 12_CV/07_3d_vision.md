# 3D Vision

3D vision involves understanding and processing three-dimensional data from the world. This guide covers point cloud processing, voxel-based methods, and multi-view geometry.

## Table of Contents

1. [Point Cloud Processing](#point-cloud-processing)
2. [Voxel-Based Methods](#voxel-based-methods)
3. [Multi-View Geometry](#multi-view-geometry)
4. [3D Reconstruction](#3d-reconstruction)

## Point Cloud Processing

### PointNet

PointNet directly processes point clouds using symmetric functions:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def pointnet_simulation():
    # Create synthetic point cloud
    np.random.seed(42)
    
    # Generate points for a cube
    n_points = 1000
    points = np.random.rand(n_points, 3) * 2 - 1  # Points in [-1, 1]^3
    
    # Filter points to create cube-like shape
    cube_mask = np.all(np.abs(points) < 0.8, axis=1)
    points = points[cube_mask]
    
    # Add some noise
    points += np.random.normal(0, 0.05, points.shape)
    
    # Simulate PointNet processing
    def pointnet_encoder(points, num_features=64):
        """Simulate PointNet encoder"""
        # Global feature extraction using max pooling
        global_features = np.max(points, axis=0)
        
        # MLP-like transformation (simplified)
        features = np.zeros((len(points), num_features))
        
        for i, point in enumerate(points):
            # Simple feature transformation
            features[i, :3] = point  # Original coordinates
            features[i, 3:6] = point**2  # Squared coordinates
            features[i, 6:9] = np.sin(point)  # Trigonometric features
            features[i, 9:12] = np.cos(point)  # More trigonometric features
            
            # Global context features
            features[i, 12:15] = global_features
            features[i, 15:18] = point - global_features  # Relative to global
        
        # Max pooling to get global feature
        global_feature = np.max(features, axis=0)
        
        return features, global_feature
    
    def pointnet_classifier(global_feature, num_classes=10):
        """Simulate PointNet classifier"""
        # Simple classification using global features
        scores = np.dot(global_feature[:20], np.random.randn(20, num_classes))
        probabilities = np.softmax(scores)
        return probabilities
    
    # Run PointNet pipeline
    features, global_feature = pointnet_encoder(points)
    class_probs = pointnet_classifier(global_feature)
    
    # Visualize results
    fig = plt.figure(figsize=(15, 5))
    
    # Original point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1)
    ax1.set_title('Input Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Feature visualization (first 3 dimensions)
    ax2 = fig.add_subplot(132, projection='3d')
    colors = features[:, :3]  # Use first 3 features as colors
    colors = (colors - colors.min()) / (colors.max() - colors.min())  # Normalize
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax2.set_title('Point Features')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Classification results
    ax3 = fig.add_subplot(133)
    classes = range(len(class_probs))
    ax3.bar(classes, class_probs)
    ax3.set_title('Classification Probabilities')
    ax3.set_xlabel('Class')
    ax3.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Point cloud size: {len(points)} points")
    print(f"Feature dimension: {features.shape[1]}")
    print(f"Predicted class: {np.argmax(class_probs)} (confidence: {np.max(class_probs):.3f})")

pointnet_simulation()
```

### PointNet++

PointNet++ uses hierarchical sampling and grouping:

```python
def pointnet_plus_plus_simulation():
    # Create more complex point cloud
    np.random.seed(42)
    
    # Generate points for multiple objects
    n_points = 2000
    points = np.random.rand(n_points, 3) * 4 - 2  # Points in [-2, 2]^3
    
    # Create multiple clusters
    cluster_centers = np.array([
        [-1, -1, -1],
        [1, 1, 1],
        [0, 0, 0]
    ])
    
    # Assign points to clusters
    labels = np.zeros(n_points)
    for i, point in enumerate(points):
        distances = np.linalg.norm(point - cluster_centers, axis=1)
        labels[i] = np.argmin(distances)
    
    # Add cluster-specific noise
    for i, center in enumerate(cluster_centers):
        mask = labels == i
        points[mask] += np.random.normal(0, 0.1, points[mask].shape)
    
    def farthest_point_sampling(points, num_samples):
        """Farthest Point Sampling"""
        n_points = len(points)
        sampled_indices = np.zeros(num_samples, dtype=int)
        distances = np.full(n_points, np.inf)
        
        # Start with random point
        sampled_indices[0] = np.random.randint(n_points)
        
        for i in range(1, num_samples):
            # Update distances
            last_point = points[sampled_indices[i-1]]
            new_distances = np.linalg.norm(points - last_point, axis=1)
            distances = np.minimum(distances, new_distances)
            
            # Find farthest point
            sampled_indices[i] = np.argmax(distances)
        
        return sampled_indices
    
    def ball_query(points, query_points, radius, max_points):
        """Ball Query for grouping"""
        groups = []
        
        for query_point in query_points:
            distances = np.linalg.norm(points - query_point, axis=1)
            indices = np.where(distances <= radius)[0]
            
            if len(indices) > max_points:
                indices = np.random.choice(indices, max_points, replace=False)
            
            groups.append(indices)
        
        return groups
    
    def hierarchical_processing(points, num_levels=3):
        """Simulate PointNet++ hierarchical processing"""
        current_points = points.copy()
        current_features = np.zeros((len(points), 3))  # Initial features are coordinates
        
        for level in range(num_levels):
            # Farthest point sampling
            num_samples = max(10, len(current_points) // (2 ** level))
            sampled_indices = farthest_point_sampling(current_points, num_samples)
            sampled_points = current_points[sampled_indices]
            
            # Ball query grouping
            radius = 0.5 * (0.8 ** level)  # Decreasing radius
            max_points = 32
            groups = ball_query(current_points, sampled_points, radius, max_points)
            
            # Process each group
            new_features = np.zeros((len(sampled_points), 64))
            
            for i, group_indices in enumerate(groups):
                if len(group_indices) > 0:
                    group_points = current_points[group_indices]
                    group_features = current_features[group_indices]
                    
                    # PointNet-like processing within group
                    group_global = np.max(group_features, axis=0)
                    group_local = group_features - group_global
                    
                    # Combine local and global features
                    combined = np.concatenate([group_local, group_global])
                    new_features[i, :len(combined)] = combined
            
            current_points = sampled_points
            current_features = new_features
        
        return current_features
    
    # Run PointNet++ pipeline
    hierarchical_features = hierarchical_processing(points)
    
    # Visualize results
    fig = plt.figure(figsize=(15, 5))
    
    # Original point cloud with clusters
    ax1 = fig.add_subplot(131, projection='3d')
    colors = plt.cm.tab10(labels / labels.max())
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=1)
    ax1.set_title('Input Point Cloud (Clustered)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Sampled points
    ax2 = fig.add_subplot(132, projection='3d')
    sampled_indices = farthest_point_sampling(points, 50)
    sampled_points = points[sampled_indices]
    ax2.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], 
               c='red', s=20)
    ax2.set_title('Farthest Point Sampling')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Feature visualization
    ax3 = fig.add_subplot(133)
    feature_colors = hierarchical_features[:, :3]
    feature_colors = (feature_colors - feature_colors.min()) / (feature_colors.max() - feature_colors.min())
    ax3.scatter(hierarchical_features[:, 0], hierarchical_features[:, 1], c=feature_colors[:, 2])
    ax3.set_title('Hierarchical Features')
    ax3.set_xlabel('Feature 1')
    ax3.set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original points: {len(points)}")
    print(f"Final features: {len(hierarchical_features)}")
    print(f"Feature dimension: {hierarchical_features.shape[1]}")

pointnet_plus_plus_simulation()
```

## Voxel-Based Methods

### VoxelNet

VoxelNet processes point clouds through voxelization:

```python
def voxelnet_simulation():
    # Create point cloud
    np.random.seed(42)
    n_points = 1000
    points = np.random.rand(n_points, 3) * 2 - 1
    
    # Create car-like shape
    car_mask = (np.abs(points[:, 0]) < 0.8) & (np.abs(points[:, 1]) < 0.4) & (points[:, 2] > -0.5)
    points = points[car_mask]
    
    def voxelize_points(points, voxel_size=0.1, max_points_per_voxel=32):
        """Convert points to voxels"""
        # Calculate voxel indices
        voxel_indices = np.floor(points / voxel_size).astype(int)
        
        # Create voxel dictionary
        voxels = {}
        
        for i, point in enumerate(points):
            voxel_key = tuple(voxel_indices[i])
            
            if voxel_key not in voxels:
                voxels[voxel_key] = []
            
            if len(voxels[voxel_key]) < max_points_per_voxel:
                voxels[voxel_key].append(point)
        
        return voxels
    
    def voxel_feature_learning(voxels, voxel_size=0.1):
        """Simulate Voxel Feature Learning (VFL)"""
        voxel_features = {}
        
        for voxel_key, voxel_points in voxels.items():
            if len(voxel_points) > 0:
                voxel_points = np.array(voxel_points)
                
                # Calculate voxel features
                centroid = np.mean(voxel_points, axis=0)
                relative_coords = voxel_points - centroid
                
                # Simple feature extraction
                features = np.concatenate([
                    centroid,  # Voxel centroid
                    np.std(voxel_points, axis=0),  # Standard deviation
                    np.max(relative_coords, axis=0),  # Max relative coordinates
                    np.min(relative_coords, axis=0),  # Min relative coordinates
                    [len(voxel_points)]  # Point count
                ])
                
                voxel_features[voxel_key] = features
        
        return voxel_features
    
    def sparse_convolution(voxel_features, conv_size=3):
        """Simulate sparse convolution"""
        # Create 3D grid
        voxel_keys = list(voxel_features.keys())
        if len(voxel_keys) == 0:
            return {}
        
        coords = np.array(voxel_keys)
        min_coords = np.min(coords, axis=0)
        max_coords = np.max(coords, axis=0)
        
        grid_size = max_coords - min_coords + 1
        feature_grid = np.zeros((*grid_size, len(next(iter(voxel_features.values())))))
        
        # Fill grid with features
        for voxel_key, features in voxel_features.items():
            grid_idx = np.array(voxel_key) - min_coords
            feature_grid[tuple(grid_idx)] = features
        
        # Simple 3D convolution (simplified)
        convolved_features = {}
        
        for voxel_key in voxel_features:
            grid_idx = np.array(voxel_key) - min_coords
            
            # Extract local neighborhood
            start_idx = np.maximum(0, grid_idx - conv_size // 2)
            end_idx = np.minimum(grid_size, grid_idx + conv_size // 2 + 1)
            
            neighborhood = feature_grid[start_idx[0]:end_idx[0], 
                                      start_idx[1]:end_idx[1], 
                                      start_idx[2]:end_idx[2]]
            
            # Simple convolution: average pooling
            if neighborhood.size > 0:
                convolved_feature = np.mean(neighborhood.reshape(-1, neighborhood.shape[-1]), axis=0)
                convolved_features[voxel_key] = convolved_feature
        
        return convolved_features
    
    # Run VoxelNet pipeline
    voxels = voxelize_points(points)
    voxel_features = voxel_feature_learning(voxels)
    convolved_features = sparse_convolution(voxel_features)
    
    # Visualize results
    fig = plt.figure(figsize=(15, 5))
    
    # Original point cloud
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=1)
    ax1.set_title('Input Point Cloud')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Voxelization
    ax2 = fig.add_subplot(132, projection='3d')
    voxel_centers = []
    for voxel_key in voxels:
        if len(voxels[voxel_key]) > 0:
            center = np.mean(voxels[voxel_key], axis=0)
            voxel_centers.append(center)
    
    if voxel_centers:
        voxel_centers = np.array(voxel_centers)
        ax2.scatter(voxel_centers[:, 0], voxel_centers[:, 1], voxel_centers[:, 2], 
                   c='red', s=20)
    ax2.set_title('Voxel Centers')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    
    # Feature distribution
    ax3 = fig.add_subplot(133)
    if convolved_features:
        features_array = np.array(list(convolved_features.values()))
        ax3.hist(features_array[:, 0], bins=20, alpha=0.7, label='Centroid X')
        ax3.hist(features_array[:, 1], bins=20, alpha=0.7, label='Centroid Y')
        ax3.hist(features_array[:, 2], bins=20, alpha=0.7, label='Centroid Z')
        ax3.legend()
    ax3.set_title('Feature Distribution')
    ax3.set_xlabel('Feature Value')
    ax3.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Input points: {len(points)}")
    print(f"Voxels: {len(voxels)}")
    print(f"Features after convolution: {len(convolved_features)}")

voxelnet_simulation()
```

## Multi-View Geometry

### Structure from Motion (SfM)

```python
def structure_from_motion_simulation():
    # Create 3D points
    np.random.seed(42)
    n_points = 50
    points_3d = np.random.rand(n_points, 3) * 2 - 1
    
    # Create camera poses
    n_cameras = 5
    camera_poses = []
    
    for i in range(n_cameras):
        # Random camera position
        position = np.random.rand(3) * 4 - 2
        position[2] = 2  # Camera above the scene
        
        # Look at origin
        look_at = np.array([0, 0, 0])
        up = np.array([0, 0, 1])
        
        # Create rotation matrix
        z_axis = look_at - position
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        
        camera_poses.append({
            'position': position,
            'rotation': rotation
        })
    
    def project_points(points_3d, camera_pose, focal_length=1000):
        """Project 3D points to 2D"""
        # Transform points to camera coordinates
        points_cam = []
        for point in points_3d:
            # Translate
            point_translated = point - camera_pose['position']
            # Rotate
            point_cam = camera_pose['rotation'].T @ point_translated
            points_cam.append(point_cam)
        
        points_cam = np.array(points_cam)
        
        # Project to 2D
        points_2d = []
        for point_cam in points_cam:
            if point_cam[2] > 0:  # Point in front of camera
                x = focal_length * point_cam[0] / point_cam[2]
                y = focal_length * point_cam[1] / point_cam[2]
                points_2d.append([x, y])
            else:
                points_2d.append([np.nan, np.nan])
        
        return np.array(points_2d)
    
    def estimate_fundamental_matrix(points1, points2):
        """Estimate fundamental matrix (simplified)"""
        # Remove NaN points
        valid_mask = ~(np.isnan(points1).any(axis=1) | np.isnan(points2).any(axis=1))
        points1_valid = points1[valid_mask]
        points2_valid = points2[valid_mask]
        
        if len(points1_valid) < 8:
            return None
        
        # Normalize points
        mean1 = np.mean(points1_valid, axis=0)
        mean2 = np.mean(points2_valid, axis=0)
        
        std1 = np.std(points1_valid, axis=0)
        std2 = np.std(points2_valid, axis=0)
        
        points1_norm = (points1_valid - mean1) / std1
        points2_norm = (points2_valid - mean2) / std2
        
        # Build constraint matrix
        A = np.zeros((len(points1_norm), 9))
        for i in range(len(points1_norm)):
            x1, y1 = points1_norm[i]
            x2, y2 = points2_norm[i]
            A[i] = [x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1]
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        F = Vt[-1].reshape(3, 3)
        
        # Enforce rank 2 constraint
        U, S, Vt = np.linalg.svd(F)
        S[2] = 0
        F = U @ np.diag(S) @ Vt
        
        return F
    
    # Project points to all cameras
    all_projections = []
    for camera_pose in camera_poses:
        projections = project_points(points_3d, camera_pose)
        all_projections.append(projections)
    
    # Estimate fundamental matrix between first two cameras
    F = estimate_fundamental_matrix(all_projections[0], all_projections[1])
    
    # Visualize results
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scene
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=20)
    
    # Draw cameras
    for i, camera_pose in enumerate(camera_poses):
        pos = camera_pose['position']
        ax1.scatter(pos[0], pos[1], pos[2], c='red', s=100, marker='^')
        ax1.text(pos[0], pos[1], pos[2], f'Cam{i+1}', fontsize=8)
    
    ax1.set_title('3D Scene and Cameras')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Projections in first camera
    ax2 = fig.add_subplot(132)
    projections = all_projections[0]
    valid_mask = ~np.isnan(projections).any(axis=1)
    valid_projections = projections[valid_mask]
    ax2.scatter(valid_projections[:, 0], valid_projections[:, 1], c='blue', s=20)
    ax2.set_title('Projections in Camera 1')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    
    # Projections in second camera
    ax3 = fig.add_subplot(133)
    projections = all_projections[1]
    valid_mask = ~np.isnan(projections).any(axis=1)
    valid_projections = projections[valid_mask]
    ax3.scatter(valid_projections[:, 0], valid_projections[:, 1], c='red', s=20)
    ax3.set_title('Projections in Camera 2')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    print(f"3D points: {len(points_3d)}")
    print(f"Cameras: {len(camera_poses)}")
    print(f"Fundamental matrix estimated: {F is not None}")

structure_from_motion_simulation()
```

## 3D Reconstruction

### Stereo Vision

```python
def stereo_vision_simulation():
    # Create 3D scene
    np.random.seed(42)
    
    # Create depth map
    height, width = 100, 100
    depth_map = np.zeros((height, width))
    
    # Create simple 3D scene
    for i in range(height):
        for j in range(width):
            # Create a plane with some variation
            depth_map[i, j] = 2.0 + 0.5 * np.sin(i * 0.1) * np.cos(j * 0.1)
    
    # Add some objects
    depth_map[30:50, 30:50] = 1.5  # Closer object
    depth_map[60:80, 60:80] = 3.0  # Farther object
    
    # Camera parameters
    focal_length = 100
    baseline = 0.1  # Distance between cameras
    
    def create_stereo_images(depth_map, focal_length, baseline):
        """Create left and right stereo images"""
        height, width = depth_map.shape
        
        # Create left image (reference)
        left_image = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                # Simple intensity based on depth
                left_image[i, j] = 255 * np.exp(-depth_map[i, j] / 5.0)
        
        # Create right image with disparity
        right_image = np.zeros((height, width))
        for i in range(height):
            for j in range(width):
                # Calculate disparity
                disparity = focal_length * baseline / depth_map[i, j]
                right_j = int(j - disparity)
                
                if 0 <= right_j < width:
                    right_image[i, right_j] = left_image[i, j]
        
        return left_image, right_image
    
    def compute_disparity(left_image, right_image, window_size=5):
        """Compute disparity using block matching"""
        height, width = left_image.shape
        disparity_map = np.zeros((height, width))
        
        half_window = window_size // 2
        
        for i in range(half_window, height - half_window):
            for j in range(half_window, width - half_window):
                # Extract left window
                left_window = left_image[i-half_window:i+half_window+1, 
                                       j-half_window:j+half_window+1]
                
                best_disparity = 0
                best_cost = float('inf')
                
                # Search for best match
                for d in range(0, min(50, j - half_window)):
                    right_j = j - d
                    if right_j >= half_window:
                        right_window = right_image[i-half_window:i+half_window+1, 
                                                 right_j-half_window:right_j+half_window+1]
                        
                        # Compute similarity (SSD)
                        cost = np.sum((left_window - right_window)**2)
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_disparity = d
                
                disparity_map[i, j] = best_disparity
        
        return disparity_map
    
    def reconstruct_3d(left_image, disparity_map, focal_length, baseline):
        """Reconstruct 3D points from disparity"""
        height, width = left_image.shape
        
        # Calculate 3D coordinates
        points_3d = []
        colors = []
        
        for i in range(0, height, 5):  # Sample every 5th pixel
            for j in range(0, width, 5):
                if disparity_map[i, j] > 0:
                    # Calculate depth
                    depth = focal_length * baseline / disparity_map[i, j]
                    
                    # Calculate 3D coordinates
                    x = (j - width/2) * depth / focal_length
                    y = (i - height/2) * depth / focal_length
                    z = depth
                    
                    points_3d.append([x, y, z])
                    colors.append(left_image[i, j])
        
        return np.array(points_3d), np.array(colors)
    
    # Create stereo images
    left_image, right_image = create_stereo_images(depth_map, focal_length, baseline)
    
    # Compute disparity
    disparity_map = compute_disparity(left_image, right_image)
    
    # Reconstruct 3D
    points_3d, colors = reconstruct_3d(left_image, disparity_map, focal_length, baseline)
    
    # Visualize results
    fig = plt.figure(figsize=(15, 10))
    
    # Original depth map
    ax1 = fig.add_subplot(231)
    ax1.imshow(depth_map, cmap='viridis')
    ax1.set_title('Ground Truth Depth')
    ax1.axis('off')
    
    # Left image
    ax2 = fig.add_subplot(232)
    ax2.imshow(left_image, cmap='gray')
    ax2.set_title('Left Image')
    ax2.axis('off')
    
    # Right image
    ax3 = fig.add_subplot(233)
    ax3.imshow(right_image, cmap='gray')
    ax3.set_title('Right Image')
    ax3.axis('off')
    
    # Disparity map
    ax4 = fig.add_subplot(234)
    ax4.imshow(disparity_map, cmap='viridis')
    ax4.set_title('Disparity Map')
    ax4.axis('off')
    
    # 3D reconstruction
    ax5 = fig.add_subplot(235, projection='3d')
    if len(points_3d) > 0:
        scatter = ax5.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                            c=colors, s=1)
    ax5.set_title('3D Reconstruction')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')
    
    # Depth comparison
    ax6 = fig.add_subplot(236)
    reconstructed_depth = np.zeros_like(depth_map)
    for i in range(0, height, 5):
        for j in range(0, width, 5):
            if disparity_map[i, j] > 0:
                reconstructed_depth[i, j] = focal_length * baseline / disparity_map[i, j]
    
    ax6.imshow(reconstructed_depth, cmap='viridis')
    ax6.set_title('Reconstructed Depth')
    ax6.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Original depth range: {depth_map.min():.2f} - {depth_map.max():.2f}")
    print(f"Reconstructed points: {len(points_3d)}")
    if len(points_3d) > 0:
        print(f"Reconstructed depth range: {points_3d[:, 2].min():.2f} - {points_3d[:, 2].max():.2f}")

stereo_vision_simulation()
```

## Summary

This guide covered 3D vision techniques:

1. **Point Cloud Processing**: PointNet, PointNet++ for direct point processing
2. **Voxel-Based Methods**: VoxelNet for structured 3D representation
3. **Multi-View Geometry**: SfM for camera pose estimation
4. **3D Reconstruction**: Stereo vision for depth estimation

### Key Takeaways

- **PointNet** processes unordered point sets using symmetric functions
- **PointNet++** uses hierarchical sampling for better local structure
- **VoxelNet** provides structured 3D representation through voxelization
- **SfM** estimates camera poses and 3D structure from multiple views
- **Stereo vision** reconstructs 3D from disparity between two cameras

### Next Steps

With 3D vision mastered, explore:
- SLAM and visual odometry
- 3D object detection
- Point cloud registration
- Neural radiance fields (NeRF) 