"""
Assembly Swarm Environment Configuration Module

This module processes target shape images and configures parameters for the 
multi-agent assembly swarm environment.
"""

from typing import Union
import numpy as np
import numpy.linalg as lg
import pickle
import argparse
import time
import os
import matplotlib.pyplot as plt
import cv2
import glob

# Track configuration loading time
start_time = time.time()
parser = argparse.ArgumentParser("Gym-AssemblySwarm Arguments")

# Global results container for processed image data
results = {
    "l_cell": [],              # Grid cell sizes after scaling
    "grid_coords": [],         # Normalized grid coordinates
    "binary_image": [],        # Processed binary images
    "shape_bound_points": []   # Boundary points of target shapes
}


def process_image(image_path):
    """
    Process a single image to extract grid coordinates for assembly targets.
    
    This function converts an image to binary, crops it to the shape boundaries,
    extracts grid centers from black regions, and scales the coordinates to
    the desired target size.
    
    Args:
        image_path (str): Path to the input image file
    """
    # Load and binarize the input image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Crop image to shape boundaries and flip vertically
    black_pixels = np.argwhere(binary_image == 0)
    min_y, min_x = black_pixels.min(axis=0)
    max_y, max_x = black_pixels.max(axis=0)
    binary_image = binary_image[min_y:max_y + 1, min_x:max_x + 1]

    height, width = binary_image.shape
    binary_image = np.dot(np.fliplr(np.eye(height)), binary_image)

    # Define grid size for discretization (36 pixels for star/shi shapes)
    grid_size = 36

    # Extract grid centers from black regions
    black_grid_coords = []
    for i in range(grid_size, height - grid_size, grid_size):
        for j in range(grid_size, width - grid_size, grid_size):
            # Extract current grid section
            grid_section = binary_image[i:i + grid_size, j:j + grid_size]
            
            # Calculate grid center coordinates
            center_x = j + grid_size / 2
            center_y = i + grid_size / 2

            # Check if grid is entirely within black region
            black_pixel_count = np.sum(grid_section == 0)
            total_pixel_count = grid_size * grid_size
            black_pixel_ratio = black_pixel_count / total_pixel_count

            # Save grid center if fully within target shape
            if black_pixel_ratio >= 1:
                black_grid_coords.append([center_x, center_y])

    # Convert to numpy array for processing
    black_grid_coords = np.array(black_grid_coords, dtype=np.float64)
    print("The number of grid: ", len(black_grid_coords))

    # Center the coordinates at origin
    x_mean_grid = np.mean(black_grid_coords[:,0])
    y_mean_grid = np.mean(black_grid_coords[:,1])
    black_grid_coords[:,0] -= x_mean_grid
    black_grid_coords[:,1] -= y_mean_grid

    # Calculate shape boundaries
    x_min = np.min(black_grid_coords[:,0])
    x_max = np.max(black_grid_coords[:,0])
    y_min = np.min(black_grid_coords[:,1])
    y_max = np.max(black_grid_coords[:,1])

    # Scale coordinates to target physical size
    target_hight = 2.2  # Target height in physical units
    real_hight = y_max - y_min
    h_scale = target_hight / real_hight
    grid_coords = h_scale * black_grid_coords
    print(grid_size * h_scale)

    # Visualize the processed results
    fig, ax = plt.subplots(figsize=(8, 8))

    # Get original image extent and adjust for centering and scaling
    img = plt.imshow(binary_image, cmap='gray', origin='lower', aspect='equal')
    origin_extent = img.get_extent()
    img.remove()

    # Plot scaled and centered image with grid points
    new_extent = [origin_extent[0] - x_mean_grid, origin_extent[1] - x_mean_grid, 
                  origin_extent[2] - y_mean_grid, origin_extent[3] - y_mean_grid]
    plt.imshow(binary_image, cmap='gray', 
               extent=[new_extent[0]*h_scale, new_extent[1]*h_scale, 
                      new_extent[2]*h_scale, new_extent[3]*h_scale], 
               origin='lower', aspect='equal')
    plt.scatter(grid_coords[:, 0], grid_coords[:, 1], color='green', 
                marker='o', alpha=0.8, label='Black Area Grids')

    plt.legend()
    plt.title('Grid Centers in Black Areas and Edge Grids')
    ax.relim()           # Recalculate data limits
    ax.autoscale_view()  # Auto-adjust axis range
    plt.close()

    # Calculate scaled boundary points
    shape_bound_points = np.array([new_extent[0]*h_scale, new_extent[1]*h_scale, 
                                  new_extent[2]*h_scale, new_extent[3]*h_scale])

    # Store results for this image
    results["l_cell"].append(grid_size * h_scale)
    results["grid_coords"].append(grid_coords)
    results["binary_image"].append(binary_image)
    results["shape_bound_points"].append(shape_bound_points)


# Process all images in the specified folder
image_folder = 'Your/Image/Folder/Path'  # Replace with your image folder path
image_paths = sorted(glob.glob(os.path.join(image_folder, '*.png')), 
                    key=lambda x: int(os.path.basename(x).split('.')[0]))

# Process each image file
for image_path in image_paths:
    process_image(image_path)

# Save processed results to pickle file
results_file = os.path.join(image_folder, 'results.pkl')
with open(results_file, 'wb') as f:
    pickle.dump(results, f)

## ==================== Environment Configuration ====================
# Multi-agent system parameters
parser.add_argument("--n_a", type=int, default=30, help='Number of agents in the swarm') 
parser.add_argument("--is_boundary", type=bool, default=True, help='Enable wall boundaries (vs periodic boundaries)') 
parser.add_argument("--is_con_self_state", type=bool, default=True, help="Include agent's own state in observation") 
parser.add_argument("--is_feature_norm", type=bool, default=False, help="Normalize input features") 
parser.add_argument("--dynamics_mode", type=str, default='Cartesian', help="Agent dynamics model (Cartesian coordinates)") 

# Visualization parameters
parser.add_argument("--render-traj", type=bool, default=True, help="Render agent trajectories during simulation") 
parser.add_argument("--traj_len", type=int, default=15, help="Length of trajectory history to display")

# Agent behavior configuration
parser.add_argument("--agent_strategy", type=str, default='input', help="Agent control strategy: input/random/llm/rule")
parser.add_argument("--training_method", default="llm_rl", type=str, choices=['llm_rl', 'pid', 'manual_rl', 'irl'],help="Training methodology")
parser.add_argument("--is_collected", type=bool, default=False, help="Collect expert data for IRL or imitation learning")
parser.add_argument("--results_file", type=type(results_file), default=results_file, help="Path to processed image results file")
parser.add_argument("--video", type=bool, default=False, help="Enable video recording of simulations")
## ==================== End of Environment Configuration ====================

## ==================== Training Hyperparameters ====================
# Basic training setup
parser.add_argument("--env_name", default="assembly", type=str,help="Environment name identifier")
parser.add_argument("--seed", default=226, type=int, help="Random seed for reproducibility")
parser.add_argument("--n_rollout_threads", default=1, type=int,help="Number of parallel environment threads")
parser.add_argument("--n_training_threads", default=5, type=int,help="Number of CPU threads for training")

# Training schedule and memory
parser.add_argument("--buffer_length", default=int(2e4), type=int,help="Replay buffer capacity")
parser.add_argument("--n_episodes", default=3000, type=int,help="Total number of training episodes")
parser.add_argument("--episode_length", default=200, type=int,help="Maximum steps per episode")
parser.add_argument("--batch_size", default=512, type=int, help="Batch size for neural network updates")

# Network architecture and learning rates
parser.add_argument("--hidden_dim", default=180, type=int,help="Hidden layer dimension for neural networks")
parser.add_argument("--lr_actor", default=1e-4, type=float,help="Learning rate for actor networks")
parser.add_argument("--lr_critic", default=1e-3, type=float,help="Learning rate for critic networks")

# Exploration and regularization
parser.add_argument("--epsilon", default=0.1, type=float,help="Epsilon for epsilon-greedy exploration")
parser.add_argument("--noise_scale", default=0.9, type=float,help="Scale factor for action noise")
parser.add_argument("--tau", default=0.01, type=float,help="Soft update rate for target networks")

# Algorithm and hardware configuration
parser.add_argument("--agent_alg", default="MADDPG", type=str, choices=['MADDPG', 'DDPG'],help="Multi-agent reinforcement learning algorithm")
parser.add_argument("--device", default="cpu", type=str, choices=['cpu', 'gpu'],help="Compute device for training")
parser.add_argument("--save_interval", default=10, type=int, help="Episode interval for saving checkpoints")

# Inverse Reinforcement Learning parameters
parser.add_argument("--lr_discriminator", default=1e-3, type=float,help="Learning rate for IRL discriminator")
parser.add_argument("--disc_use_linear_lr_decay", default=False, type=bool,help="Enable linear learning rate decay for discriminator")
parser.add_argument("--hidden_num", default=4, type=int,help="Number of hidden layers in discriminator")
## ==================== End of Training Hyperparameters ====================

# Parse all arguments and measure loading time
gpsargs = parser.parse_args()
end_time = time.time()
print('Load config parameters takes ', end_time - start_time)