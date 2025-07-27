import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import matplotlib.pyplot as plt
import numpy as np
import torch
# pip install perlin-noise
from perlin_noise import PerlinNoise
import numpy as np

def generate_perlin_dirs(B, W, H, scale=0.1, device='cpu'):
    dirs_list = []
    for _ in range(B):
        noise_x = PerlinNoise(octaves=4)
        noise_y = PerlinNoise(octaves=4)
        noise_z = PerlinNoise(octaves=4)
        field = torch.zeros(W,H,3)
        for i in range(W):
            for j in range(H):
                field[i,j,0] = noise_x([i*scale,j*scale])
                field[i,j,1] = noise_y([i*scale,j*scale])
                field[i,j,2] = noise_z([i*scale,j*scale])
        field = torch.nn.functional.normalize(field, dim=-1)
        dirs_list.append(field)
    dirs = torch.stack(dirs_list,0).to(device)
    return dirs

import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_dirs(B=1, W=64, H=64, sigma_range=(2.0, 8.0), visualize=True, batch_to_show=0, step=4, device='cuda'):
    """
    Generate synthetic direction fields with spatial continuity and random sigma on CUDA.

    Args:
        B (int): Number of batches.
        W (int): Width.
        H (int): Height.
        sigma_range (tuple): (min_sigma, max_sigma) range for random smoothing.
        visualize (bool): If True, display quiver and RGB visualization.
        batch_to_show (int): Index of batch to visualize.
        step (int): Downsampling step for quiver plot.
        device (str): device to use ('cuda' or 'cpu')

    Returns:
        dirs (torch.Tensor): Synthetic directions of shape [B, W, H, 3], normalized, on given device.
    """
    # Create random directions on GPU
    dirs = torch.randn(B, W, H, 3, device=device)

    # For smoothing, move to CPU (scipy doesn't support GPU tensors)
    dirs_cpu = dirs.cpu().numpy()

    for b in range(B):
        sigma_b = np.random.uniform(sigma_range[0], sigma_range[1])
        for c in range(3):
            dirs_cpu[b, :, :, c] = gaussian_filter(dirs_cpu[b, :, :, c], sigma=sigma_b)

    # Convert back to torch tensor on device
    dirs = torch.tensor(dirs_cpu, device=device, dtype=torch.float32)
    #print(dirs)
    # Normalize to unit vectors
    norm = torch.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-8)
    dirs = dirs / norm

    if visualize:
        # Move data for visualization to CPU and numpy
        dirs_b = dirs[batch_to_show].cpu().numpy()
        X, Y = np.meshgrid(np.arange(0, H, step), np.arange(0, W, step))
        U = dirs_b[::step, ::step, 0]
        V = dirs_b[::step, ::step, 1]

        plt.figure(figsize=(6, 6))
        plt.quiver(X, Y, U, V, angles='xy', scale_units='xy')
        plt.gca().invert_yaxis()
        plt.title(f"Batch {batch_to_show} - Direction Field (Quiver)")
        plt.tight_layout()
        plt.show()

        dirs_color = (dirs_b + 1.0) / 2.0
        rgb_img = np.clip(dirs_color, 0, 1)

        plt.figure(figsize=(6, 6))
        plt.imshow(rgb_img)
        plt.title(f"Batch {batch_to_show} - Orientation Encoded as RGB")
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    return dirs


import torch
import torch.nn.functional as F

# -------------------------------------------
# 1. Helper: build orthonormal rotation matrix from direction field
# -------------------------------------------

def build_rotation_from_dirs(dirs_smooth: torch.Tensor) -> torch.Tensor:
    """
    Build rotation matrices from smooth direction field.

    Args:
        dirs_smooth: [B, W, H, 3] normalized direction vectors
                     (principal axis for each voxel)

    Returns:
        rotations: [B, W, H, 3, 3] rotation matrices
                   first column = dirs_smooth (principal direction)
    """
    device = dirs_smooth.device
    B, W, H, _ = dirs_smooth.shape

    # normalize the input directions
    v1 = dirs_smooth / (torch.norm(dirs_smooth, dim=-1, keepdim=True) + 1e-8)
    # pick a helper vector (e.g., [0,0,1]) and fix collinearity
    helper = torch.tensor([0.0, 0.0, 1.0], device=device).view(1,1,1,3)
    helper = helper.expand(B, W, H, 3)
    parallel_mask = (torch.abs((v1 * helper).sum(-1, keepdim=True)) > 0.99).float()
    # blend with [1,0,0] if too parallel
    alt = torch.tensor([1.0, 0.0, 0.0], device=device).view(1,1,1,3)
    helper = parallel_mask * alt + (1-parallel_mask) * helper

    # make v2 orthogonal to v1
    v2 = helper - (helper * v1).sum(-1, keepdim=True) * v1
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True) + 1e-8)

    # third vector is cross product
    v3 = torch.cross(v1, v2, dim=-1)

    # stack into rotation matrix [v1 v2 v3]
    R = torch.stack([v1, v2, v3], dim=-1)  # [B,W,H,3,3]

    return R

# -------------------------------------------
# 2. Your provided sampling function
# -------------------------------------------
def sample_eigenvalues(eigen_dists, batch_size, w, h):
    total_samples = batch_size * w * h
    device = eigen_dists[0][0].device
    sampled = []
    for values, probs in eigen_dists:
        indices = torch.multinomial(probs, total_samples, replacement=True)
        samples = values[indices]
        sampled.append(samples)
    samples_stack = torch.stack(sampled, dim=1)
    return samples_stack.view(batch_size, w, h, 3)

# -------------------------------------------
# 3. Combine everything into a single generator
# -------------------------------------------
def generate_white_matter_like_tensors(eigen_dists, dirs):
    """
    eigen_dists: output from prepare_eigenvalue_distributions
    dirs: [B, W, H, 3] direction field
    returns: D [B,W,H,3,3] diffusion tensors
    """
    B, W, H, _ = dirs.shape
    device = dirs.device

    # Sample raw eigenvalues
    raw_eigs = sample_eigenvalues(eigen_dists, B, W, H)  # [B,W,H,3]
    vals, _ = torch.sort(raw_eigs, dim=-1, descending=True)

    # Enforce white-matter shape
    λ1 = vals[..., 0]
    λ2 = 0.3 * vals[..., 1]
    λ3 = 0.3 * vals[..., 2]

    # Build diagonal Λ
    Λ = torch.zeros(B, W, H, 3, 3, device=device)
    Λ[..., 0, 0] = λ1
    Λ[..., 1, 1] = λ2
    Λ[..., 2, 2] = λ3

    # Build rotation matrices
    R = build_rotation_from_dirs(dirs)  # [B,W,H,3,3]

    # Compute D = R Λ Rᵀ
    R_T = R.transpose(-1, -2)
    D = torch.matmul(R, torch.matmul(Λ, R_T))  # [B,W,H,3,3]
    return D

# -------------------------------------------
# 4. Example usage
# -------------------------------------------
if __name__ == "__main__":
    B, W, H = 1, 64, 64

    # Example: a constant direction field (aligned along x-axis)
    dirs = torch.tensor([1.,0.,0.]).view(1,1,1,3).expand(B,W,H,3)

    # Fake eigenvalue distribution for testing
    values = torch.linspace(0.001, 0.002, 100)  # just for demo
    probs = torch.ones_like(values) / len(values)
    eigen_dists = [(values, probs), (values, probs), (values, probs)]

    D = generate_white_matter_like_tensors(eigen_dists, dirs)
    print("D shape:", D.shape)  # [B,W,H,3,3]

for i in range(5):
 #sigma_range=(20, 40)
 dirs = generate_dirs(B=4, W=64, H=64, batch_to_show=np.random.randint(4),sigma_range=(45, 50), visualize=True)
 print(torch.norm(dirs[2,32,32,:]))
 rots = build_rotation_from_dirs(dirs)  # [B,W,H,3,3]
 #print(rots[2,32,32,:,:]@rots[2,32,32,:,:].T)
