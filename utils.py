import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
import torch.linalg as linalg
import os
import random
from torch.utils.data import Dataset
#import torchvision.transforms as T
#from dipy.data import get_sphere 
import seaborn as sns
import nibabel as nib

import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress specific warning by setting ignore for all warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate, approximate err =")

import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_FA(tensor3x3: torch.Tensor) -> torch.Tensor:
    """
    Compute FA from a diffusion tensor of shape [X, Y, Z, 3, 3].

    Args:
        tensor3x3: Diffusion tensor, shape [X, Y, Z, 3, 3]

    Returns:
        FA: Fractional anisotropy, shape [X, Y, Z]
    """
    X, Y, Z, _, _ = tensor3x3.shape
    tensor_flat = tensor3x3.reshape(-1, 3, 3)  # [N_voxels, 3, 3]

    # Compute eigenvalues for each tensor
    eigvals, _ = torch.linalg.eigh(tensor_flat)  # [N_voxels, 3], sorted ascending

    # Sort descending for clarity (optional)
    #eigvals = eigvals[:, ::-1]
    eigvals, _ = torch.sort(eigvals, descending=True)
    eigvals=eigvals.abs()
    # Calculate FA
    lambda_mean = eigvals.mean(dim=1, keepdim=True)  # [N,1]

    numerator = torch.sqrt( ((eigvals - lambda_mean)**2).sum(dim=1) * 1.5 )  # sqrt(3/2 * sum(...))
    denominator = torch.sqrt((eigvals**2).sum(dim=1) + 1e-15)  # add small to avoid div zero

    FA = numerator / denominator
    FA[FA>1]=1
    FA[FA<0]=0


    # Reshape back to [X,Y,Z]
    FA = FA.reshape(X, Y, Z)

    return FA, eigvals



def amatrix(bvecs: torch.Tensor) -> torch.Tensor:
    
    gx, gy, gz = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    A = torch.stack([
        gx * gx,
        2 * gx * gy,
        2 * gx * gz,
        gy * gy,
        2 * gy * gz,
        gz * gz
    ], dim=1)
    return A  # Shape: [G, 6]

def fit_dti_tensor(dwi: torch.Tensor, mean_b0: torch.Tensor,  bvecs: torch.Tensor) -> torch.Tensor:
    """
    Fit DTI tensor using linear least squares.

    Args:
        dwi: DWI image of shape [X, Y, Z, G]
        mean_b0: Mean b=0 image, shape [X, Y, Z]
        bvals: 1D tensor, shape [G]
        bvecs: 2D tensor, shape [G, 3] or [3, G]

    Returns:
        tensor_6d: Tensor coefficients of shape [X, Y, Z, 6]
    """
    eps = 1e-6
    X, Y, Z, G = dwi.shape

    # Reshape for vectorized computation
    S = dwi.reshape(-1, G)          # [N_voxels, G]
    S0 = mean_b0.reshape(-1, 1)     # [N_voxels, 1]

    # Compute log signal normalized by mean b0
    
    log_signal = torch.log((S + eps) / (S0 + eps))  # [N_voxels, G]

    # Ensure bvals shape is [G]
    #bvals = bvals.flatten()         # [G]

    # Normalize log signal by negative bvals to get ADCs
    # Avoid division by zero by adding eps
    log_signal = log_signal / (-1000. + eps)  # [N_voxels, G]

    # Make sure bvecs shape is [G, 3]
    if bvecs.shape[0] == 3 and bvecs.shape[1] == G:
        bvecs = bvecs.T  # transpose to [G, 3]

    # Compute diffusion tensor design matrix A of shape [G, 6]
    A = amatrix(bvecs)  # [G, 6]

    # Compute pseudo-inverse of A: shape [6, G]
    A_pinv = torch.pinverse(A)

    # Multiply A_pinv [6, G] by log_signal.T [G, N_voxels]
    tensor_vec = torch.matmul(A_pinv, log_signal.T)  # [6, N_voxels]

    # Transpose to [N_voxels, 6]
    tensor_vec = tensor_vec.T

    # Reshape back to 4D volume tensor coefficients [X, Y, Z, 6]
    tensor = tensor_vec.reshape(X, Y, Z, 6)

    # Replace NaN and Inf with zero
    tensor6 = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    X, Y, Z, _ = tensor6.shape

    # Initialize output tensor
    tensor3x3 = torch.zeros((X, Y, Z, 3, 3), device=tensor6.device, dtype=tensor6.dtype)

    # Assign values to symmetric tensor components
    tensor3x3[..., 0, 0] = tensor6[..., 0]  # Dxx
    tensor3x3[..., 0, 1] = tensor6[..., 1]  # Dxy
    tensor3x3[..., 0, 2] = tensor6[..., 2]  # Dxz

    tensor3x3[..., 1, 0] = tensor6[..., 1]  # Dxy
    tensor3x3[..., 1, 1] = tensor6[..., 3]  # Dyy
    tensor3x3[..., 1, 2] = tensor6[..., 4]  # Dyz

    tensor3x3[..., 2, 0] = tensor6[..., 2]  # Dxz
    tensor3x3[..., 2, 1] = tensor6[..., 4]  # Dyz
    tensor3x3[..., 2, 2] = tensor6[..., 5]  # Dzz

    
    return tensor3x3



def lambdas_boxplots(eigvals_gt, eigvals_est):
    """
    Generates a boxplot comparing ground truth and estimated eigenvalues.
    
    Parameters:
        eigvals_gt (numpy.ndarray): Ground truth eigenvalues of shape (H, W, D, 3).
        eigvals_est (numpy.ndarray): Estimated eigenvalues of shape (H, W, D, 3).
    """
    # Validate inputs
    if eigvals_gt.shape != eigvals_est.shape:
        raise ValueError("The ground truth and estimated eigenvalues must have the same shape.")
    if eigvals_gt.shape[-1] != 3:
        raise ValueError("The last dimension of the eigenvalue arrays must be 3 (for λ1, λ2, λ3).")
    
    # Flatten the eigenvalue arrays to (N_voxels, 3)
    eigvals_gt_flat = eigvals_gt.reshape(-1, 3)
    eigvals_est_flat = eigvals_est.reshape(-1, 3)

    # Prepare data for the boxplot
    data = []
    labels = []
    colors = []
    for i in range(3):  # Loop over λ1, λ2, λ3
        data.append(eigvals_gt_flat[:, i])  # Ground truth
        labels.append(f'λ{i+1} (GT)')
        colors.append('red')  # Red for GT
        data.append(eigvals_est_flat[:, i])  # Estimated
        labels.append(f'λ{i+1} (Est)')
        colors.append('green')  # Green for Est

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    bplot = plt.boxplot(data, labels=labels,  showfliers=False, patch_artist=True)

    # Apply colors to the boxes
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Rotate x-ticks for better readability
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Eigenvalues (mm²/s)')
    plt.title('Comparison of Ground Truth (Red) and Estimated (Green) Eigenvalues')
    #plt.grid(axis='y',  alpha=0.7)
    plt.tight_layout()
    plt.show()


def plot_eigenvalue_boxplots(eigvals_gt, eigvals_est, eigvals_fit):
    """
    Plot boxplots for eigenvalues from three tensors (gt, est, fit) with lambda 1, 2, and 3 on the x-axis.
    
    Parameters:
        eigvals_gt (torch.Tensor): Ground truth eigenvalues of shape [240, 240, 16, 3].
        eigvals_est (torch.Tensor): Estimated eigenvalues of shape [240, 240, 16, 3].
        eigvals_fit (torch.Tensor): Fitted eigenvalues of shape [240, 240, 16, 3].
    """
    # Move tensors to CPU and reshape to 2D for easier processing (flatten spatial dimensions)
    eigvals_gt = eigvals_gt.cpu().numpy().reshape(-1, 3)
    eigvals_est = eigvals_est.cpu().numpy().reshape(-1, 3)
    eigvals_fit = eigvals_fit.cpu().numpy().reshape(-1, 3)
    
    # Prepare data for each lambda (1, 2, 3) across gt, est, and fit
    data = {
        "Lambda 1": [eigvals_gt[:, 0], eigvals_est[:, 0], eigvals_fit[:, 0]],
        "Lambda 2": [eigvals_gt[:, 1], eigvals_est[:, 1], eigvals_fit[:, 1]],
        "Lambda 3": [eigvals_gt[:, 2], eigvals_est[:, 2], eigvals_fit[:, 2]],
    }
    
    # Plot box plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)  # One subplot per lambda
    for i, (label, values) in enumerate(data.items()):
        ax[i].boxplot(values, labels=["GT", "Est", "Fit"], patch_artist=True, boxprops=dict(facecolor='lightblue'))
        ax[i].set_title(label, fontsize=14)
        ax[i].set_xlabel("Type", fontsize=12)
        if i == 0:  # Set shared y-axis label
            ax[i].set_ylabel("Eigenvalue Magnitude", fontsize=12)
    
    plt.tight_layout()
    plt.show()



def calculate_md(eigvals):
    return eigvals.mean(axis=-1)

def calculate_ad(eigvals):
     return eigvals[:,:,:, 0]

def calculate_ad2(eigvals):
     return eigvals[:,:,:, 1]

def calculate_rd(eigvals):
    return eigvals[:,:,:, 1:3].mean(axis=3)




mean_diffusivity = lambda eigvals: eigvals.mean(axis=-1, keepdims=True)
fa_numerator = lambda eigvals, md: np.sqrt(((eigvals - md) ** 2).sum(axis=-1))
fa_denominator = lambda eigvals: np.sqrt((eigvals ** 2).sum(axis=-1))

calculate_fa = lambda eigvals: np.sqrt(3 / 2) * (
    fa_numerator(eigvals, mean_diffusivity(eigvals)) /
    (fa_denominator(eigvals) + 1e-12) )

 

def scale_to_01(arr):
    epsilon=1e-10
    min_val = torch.min(arr)
    max_val = torch.max(arr)
        
    scaled_arr = (arr - min_val) / (max_val - min_val)
    return scaled_arr + epsilon

                        
def scale_to_match_eigvals(est, gt):
    mean_gt = gt.mean()
    std_gt = gt.std()
    est_norm = (est - est.mean()) / (est.std() + 1e-8)
    return est_norm * std_gt + mean_gt


def scale_to_match_eigvals_per_channel(eigvals_est, eigvals_gt):
    """
    Differentiable version: scales each eigenvalue channel in eigvals_est to match
    the corresponding mean and std of eigvals_gt without in-place modification.
    
    Args:
        eigvals_est: [D, H, W, B, 3] or [B, 3, D, H, W] depending on your pipeline.
        eigvals_gt:  same shape as eigvals_est.

    Returns:
        Scaled eigvals_est: tensor with same shape.
    """
    channels = eigvals_est.shape[-1]
    scaled_channels = []

    for i in range(channels):
        gt = eigvals_gt[..., i]
        est = eigvals_est[..., i]

        gt_mean = gt.mean()
        gt_std = gt.std()
        est_mean = est.mean()
        est_std = est.std()

        # Avoid division by zero
        est_std = est_std if est_std != 0 else torch.tensor(1.0, device=est.device)

        est_norm = (est - est.mean()) / (est.std() + 1e-8)
        scaled=est_norm * gt_std + gt_mean
        scaled_channels.append(scaled.unsqueeze(-1))  # Shape: [..., 1]


    return torch.cat(scaled_channels, dim=-1)  # Shape: [..., 3]


def map_to_range(y, min_val, max_val):
        
        y=scale_to_01(y)
        epsilon=1e-10
        #min = torch.min(y)
        #max = torch.max(y)
        #scaled_y= (y - min) / (max - min) + epsilon

        return min_val + y * (max_val - min_val)




def dtimodel(g=None, b=1000, D=None):
     
    S=torch.zeros(g.shape(0)) 
    for i in range(g.shape(0)):
            S[i]=torch.exp(b*(g[i,:].T*D*g[i,:]))
    return S
    


def random_rotation_matrices2(w=240, h=240, batchsize=16, device=None):
    """
    Generates a grid of random 3D rotation matrices of shape (240, 240, 16, 3, 3).
    
    Parameters:
    grid_size (tuple): The size of the grid, default is (240, 240, 16).
    device: Device for the tensors (e.g., 'cpu' or 'cuda').
    
    Returns:
    torch.Tensor: A tensor of shape (240, 240, 16, 3, 3) where each entry is a random 3D rotation matrix.
    """
    grid_size=(w,h, batchsize)
    # Step 1: Generate random unit vectors (axes of rotation) for each element in the grid
    axes = torch.randn(*grid_size, 3, device=device)
    axes = axes / axes.norm(dim=-1, keepdim=True)  # Normalize to make them unit vectors
    
    # Step 2: Generate random rotation angles between 0 and 2π
    angles = torch.rand(*grid_size,  device=device) * 2 * torch.pi

    # Step 3: Calculate trigonometric values for the Rodrigues' rotation formula
    cos_theta = torch.cos(angles)
    sin_theta = torch.sin(angles)
    one_minus_cos = 1 - cos_theta

    # Step 4: Extract x, y, z components of each axis for broadcasting
    x = axes[..., 0]
    y = axes[..., 1]
    z = axes[..., 2]

    # Step 5: Calculate the rotation matrices using the Rodrigues' rotation formula in batch
    rotation_matrices = torch.empty((*grid_size, 3, 3), device=device)
    rotation_matrices[..., 0, 0] = cos_theta + x * x * one_minus_cos
    rotation_matrices[..., 0, 1] = x * y * one_minus_cos - z * sin_theta
    rotation_matrices[..., 0, 2] = x * z * one_minus_cos + y * sin_theta

    rotation_matrices[..., 1, 0] = y * x * one_minus_cos + z * sin_theta
    rotation_matrices[..., 1, 1] = cos_theta + y * y * one_minus_cos
    rotation_matrices[..., 1, 2] = y * z * one_minus_cos - x * sin_theta

    rotation_matrices[..., 2, 0] = z * x * one_minus_cos - y * sin_theta
    rotation_matrices[..., 2, 1] = z * y * one_minus_cos + x * sin_theta
    rotation_matrices[..., 2, 2] = cos_theta + z * z * one_minus_cos

    return rotation_matrices


def gen_orthonormal_rots(batch_size, device="cuda"):
    """
    Generate a batch of random 3x3 rotation matrices using PyTorch.
    
    Parameters:
        batch_size (tuple): Shape of the batch (e.g., (128, 128, 16)).
        device (str): Device to run the computation on ("cuda" or "cpu").
        
    Returns:
        torch.Tensor: Batch of rotation matrices of shape (*batch_size, 3, 3).
    """
    # Flatten the batch size for easier processing
    total_batches = torch.prod(torch.tensor(batch_size))
    
    # Generate random 3x3 matrices
    random_matrices = torch.randn((total_batches, 3, 3), device=device)
    
    # Perform QR decomposition to ensure orthonormality
    Q, R = torch.linalg.qr(random_matrices)
    
    # Ensure Q is a proper rotation matrix (det(Q) = 1)
    det = torch.det(Q)
    correction = torch.diag_embed(torch.sign(det)).to(device)  # Correction for improper rotations
    Q = Q @ correction
    
    # Reshape back to the original batch size
    rotation_matrices = Q.view(*batch_size, 3, 3)
    return rotation_matrices
import torch

def generate_random_rots_uniform(w=128, h=128, batch_size=16, device="cuda"):
    """
    Generate random 3x3 rotation matrices uniformly sampled from SO(3) for a 128x128x16 grid.

    Parameters:
        w (int): First grid dimension (default: 128).
        h (int): Second grid dimension (default: 128).
        batch_size (int): Third grid dimension (default: 16).
        device (str): Device to perform computation on ("cuda" or "cpu").

    Returns:
        torch.Tensor: Tensor of shape (w, h, batch_size, 3, 3) containing rotation matrices.
    """
    total_batches = w * h * batch_size  # Flattened total number of matrices to generate

    # Step 1: Generate random vectors
    x1 = torch.randn((total_batches, 3), device=device)
    x2 = torch.randn((total_batches, 3), device=device)

    # Step 2: Normalize the first vector
    v1 = x1 / torch.norm(x1, dim=-1, keepdim=True)

    # Step 3: Make the second vector orthogonal to the first
    proj = torch.sum(x2 * v1, dim=-1, keepdim=True) * v1  # Projection of x2 onto v1
    v2 = x2 - proj
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    # Step 4: Compute the third vector using the cross product
    v3 = torch.cross(v1, v2)

    # Step 5: Stack the vectors to form the rotation matrix
    Q = torch.stack((v1, v2, v3), dim=-1)  # Shape: (total_batches, 3, 3)

    # Step 6: Reshape to the desired grid shape
    rotation_matrices = Q.view(w, h, batch_size, 3, 3)
    return rotation_matrices

# Example usage
#rotation_matrices = generate_random_rotation_matrix_uniform(128, 128, 16)
#print("Generated rotation matrices shape:", rotation_matrices.shape)

def display_gradients(unit_directions=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(unit_directions[:, 0].cpu(), unit_directions[:, 1].cpu(), unit_directions[:, 2].cpu(), s=1)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.show()





def add_rician_noise(images, snr):
    
    sigma = 1 / snr
    #sigma=0.005
    # Generate the noise components as independent Gaussian random variables 
    real_noise = torch.normal(0, sigma, images.size(), device=images.device)
    imag_noise = torch.normal(0, sigma, images.size(), device=images.device)

    # Add the noise components to the images 
    noisy_images = torch.sqrt((images + real_noise) ** 2 + imag_noise ** 2)


    return noisy_images


def extract_right_part(vector):
    right_parts = []
    for element in vector:
        right_part = element.split('_')[1]
        right_parts.append(right_part)
    return [int(right) for right in right_parts]


def vector(ind, total=13):
    result_vector = []
    for num in range(total):
        random_index = random.randint(2, ind)
        result_vector.append(f"{num}_{random_index}")
    #print("\nResulting vector:", result_vector)
    return extract_right_part(result_vector)



def load_subject_data(subject_folder_path):
    """
    Loads diffusion data, mask, bvals, and bvecs for a given subject folder into PyTorch tensors.

    Args:
        subject_folder_path (str): Full path to the subject folder (e.g., .../subject_1)

    Returns:
        data_tensor (torch.Tensor): Diffusion data tensor (shape [X, Y, Z, D])
        mask_tensor (torch.Tensor): Brain mask tensor (shape [X, Y, Z])
        bvals_tensor (torch.Tensor): b-values (shape [D])
        bvecs_tensor (torch.Tensor): b-vectors (shape [3, D])
    """
    # Helper to load .nii or .nii.gz
    def load_nii_tensor(file_base_name):
        for ext in ['.nii.gz', '.nii']:
            full_path = os.path.join(subject_folder_path, file_base_name + ext)
            if os.path.exists(full_path):
                nii_img = nib.load(full_path)
                data = nii_img.get_fdata(dtype=np.float32)
                return torch.from_numpy(data)
        raise FileNotFoundError(f"{file_base_name}.nii or .nii.gz not found in {subject_folder_path}")

    # Helper to load text files
    def load_txt_tensor(file_name):
        file_path = os.path.join(subject_folder_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_name} not found in {subject_folder_path}")
        data = np.loadtxt(file_path)
        return torch.tensor(data, dtype=torch.float32)

    # Load data
    data_tensor = load_nii_tensor("data")
    mask_tensor = load_nii_tensor("nodif_brain_mask")
    bvals_tensor = load_txt_tensor("bvals")
    bvecs_tensor = load_txt_tensor("bvecs")

    return data_tensor, mask_tensor, bvals_tensor, bvecs_tensor



#import torchvision.transforms as T
#import torchvision.transforms.functional as TF
import os
import math
import random
from torch.utils.data import Dataset
import nibabel as nib
from pathlib import Path
import torch
import glob


import torch
from torch.utils.data import Dataset
import random

class DTIbatch(Dataset):
    def __init__(self, data_all: torch.Tensor, mask: torch.Tensor, bvecs: torch.Tensor,
                 z_slices: int, total_samples: int, z_offset: int = 0):
        """
        Args:
            data_all    : torch.Tensor of shape [W, H, Z, G]
            mask        : torch.Tensor of shape [W, H, Z]
            bvecs       : torch.Tensor of shape [G, 3]
            z_slices    : int, number of consecutive Z‐slices to sample
            total_samples: int, how many random samples this dataset should yield
            z_offset    : int, minimal distance from volume boundary (optional; default 0)
        """
        super().__init__()
        self.data_all = data_all
        self.mask = mask
        self.bvecs = bvecs          # shape [G, 3]
        self.z_slices = z_slices
        self.total_samples = total_samples
        self.z_offset = z_offset

        # Infer dimensions from data_all
        # data_all is [W, H, Z, G]
        _, _, _, self.Z, self.G = data_all.shape

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Each call picks:
          - A random starting z index between [z_offset, Z - z_slices - z_offset]
          - For each gradient direction g in [0..G−1], a voxel block of shape [W,H,z_slices]
            at slice range [z_start : z_start+z_slices]
          - The corresponding b‐vector for that g
        Returns:
            data_block : torch.Tensor of shape [W, H, z_slices, G]
            mask_block : torch.Tensor of shape [W, H, z_slices]
            bvecs      : torch.Tensor of shape [3, G]
        """
        
        
        
        # 1. Pick a random z_start
        z_min = self.z_offset
        z_max = self.Z - self.z_slices - self.z_offset
        if z_max < z_min:
            raise ValueError(
                f"z_offset={self.z_offset} + z_slices={self.z_slices} "
                f"exceeds volume depth={self.Z}"
            )
        
        
        
        z_start = random.randint(z_min, z_max)
        z_indices = list(range(z_start, z_start + self.z_slices))
        total_subjects, W, H, Z, G = self.data_all.shape
        
        data_block = torch.empty((W, H, self.z_slices, G), dtype=self.data_all.dtype)
        mask_block = torch.empty((W, H, self.z_slices), dtype=self.data_all.dtype)
        bvecs_out = torch.empty((3, G), dtype=self.bvecs.dtype)
        
                
        """ for g in range(91):
            subj = random.randint(0, total_subjects-1)  
            data_block[..., g] = self.data_all[subj, :, :, z_indices, g]
            mask_block= self.mask[subj,:,:,z_indices]
            #print(self.bvecs.shape)
            bvecs_out[:, g] = self.bvecs[subj, :, g]
         """

        subj = random.randint(0, total_subjects-1)
        data_block = self.data_all[subj, :, :, z_indices, :]        # [W, H, z_slices, G]
        mask_block = self.mask[subj, :, :, z_indices]               # [W, H, z_slices]
        bvecs_out = self.bvecs[subj,...]                              # originally [G,3], transpose→ [3, G]
        
        return data_block, mask_block, bvecs_out



class LogEuclideanMSELoss(nn.Module):
    def __init__(self, eps=1e-10):
        super(LogEuclideanMSELoss, self).__init__()
        self.eps = eps

    def matrix_log(self, tensor):
        # Eigen decomposition
        eigvals, eigvecs = torch.linalg.eigh(tensor)
        # Clamp eigenvalues to avoid log(0) or negative values
        log_eigvals = torch.log(torch.clamp(eigvals, min=self.eps))
        # Reconstruct log-matrix
        return eigvecs @ torch.diag_embed(log_eigvals) @ eigvecs.transpose(-1, -2)

    def forward(self, D_est, D_gt):
        log_D_est = self.matrix_log(D_est)
        log_D_gt = self.matrix_log(D_gt)
        diff = log_D_est - log_D_gt
        loss = torch.mean((diff ** 2).sum(dim=(-2, -1)))  # Frobenius norm squared
        return loss




class DTINet3D(nn.Module):
    def __init__(self, in_channels=91):  # Set in_channels=G
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 6, kernel_size=1)  # Directly output 6 channels
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # Input: [1, G, W, H, D] → Output: [1, 6, W, H, D]
        params = self.encoder(x)
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, D, 6]

        # Extract Cholesky parameters (lower-triangular)
        l11 = torch.abs(params[..., 0])  # Ensure positivity
        l21 = params[..., 1]
        l31 = params[..., 2]
        l22 = torch.abs(params[..., 3])  # Ensure positivity
        l32 = params[..., 4]
        l33 = torch.abs(params[..., 5])  # Ensure positivity

        # Build Cholesky factor L
        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33

        D_spd = L @ L.transpose(-1, -2)  # SPD by construction
        
        return D_spd

import torch
import torch.nn as nn

class DTINet3D_PI2(nn.Module):
    def __init__(self, in_channels=91, bvals=None):
        super().__init__()
        # Depthwise processing (no mixing of signals)
        self.depthwise = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.ReLU(),
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.ReLU(),
            #nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            #nn.ReLU()
        )
        # Minimal mixing at the end to get tensor parameters
        # Note: out_params=6 is not valid; use out_channels=6
        #self.final = nn.Conv3d(in_channels, 6, kernel_size=1)
        
        self.final = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            #nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm3d(256),
            #nn.Conv3d(256, 256, kernel_size=1, padding=0),
            #nn.ReLU(),
            #nn.Conv3d(256, 256, kernel_size=1, padding=0),
            #nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 6, kernel_size=1)
        )
        
        self.bvals = bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_dti_signal(self, D_est, bvecs, S0=1.0):
        """Predict DTI signal from estimated tensor and bvecs.
        Args:
            D_est: [W, H, B, 3, 3]
            bvecs: [N, 3]
            S0: [W, H, B] or scalar
        Returns:
            pred_signal: [W, H, B, N]
        """
        W, H, B, _, _ = D_est.shape
        N = bvecs.shape[0]
        D_flat = D_est.reshape(-1, 3, 3)  # [V, 3, 3]
        gdg = torch.einsum('nj,vjk,nk->vn', bvecs, D_flat, bvecs)  # [V, N]
        exponent = -self.bvals * gdg  # [V, N]
        signal = torch.exp(exponent)  # [V, N]
        if not torch.is_tensor(S0):
            S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
        S0 = S0.reshape(-1, 1)  # [V, 1]
        signal = S0 * signal    # [V, N]
        return signal.reshape(W, H, B, N)  # [W, H, B, N]

    def forward(self, x, bvecs):
        # x: [N, in_channels, W, H, B]
        mean_b0 = x[0, 0, ...]    # [W, H, B]
        x = self.depthwise(x)      # [N, in_channels, W, H, B]
        params = self.final(x)     # [N, 6, W, H, B]

        # Move params to [W, H, B, 6] if N=1, else generalize for batch
        if params.shape[0] == 1:
            params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]
        else:
            pass

        # Tensor construction (as before)
        l11 = torch.abs(params[..., 0])
        l21 = params[..., 1]
        l31 = params[..., 2]
        l22 = torch.abs(params[..., 3])
        l32 = params[..., 4]
        l33 = torch.abs(params[..., 5])

        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33

        D = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]
        pred_signal = self.predict_dti_signal(D, bvecs, S0=mean_b0)
        return D, pred_signal





class DTINet3D_PI(nn.Module):
    def __init__(self, in_channels=91, bvals=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
          
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.Conv3d(32, 6, kernel_size=1)
        )
        #self.register_buffer('bvals', torch.as_tensor(bvals, dtype=torch.float32))  # [N]
        self.bvals=bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def predict_dti_signal(self, D_est, bvecs, S0=1.0):
        """
        D_est: [W, H, B, 3, 3]
        bvecs: [N, 3] (passed at runtime)
        S0: [W, H, B] or scalar
        Returns: [W, H, B, N]
        """
        W, H, B, _, _ = D_est.shape
        N = bvecs.shape[0]
        
        D_flat = D_est.reshape(-1, 3, 3)  # [V, 3, 3]
        gdg = torch.einsum('nj,vjk,nk->vn', bvecs, D_flat, bvecs)  # [V, N]
        exponent = -self.bvals * gdg  # [V, N]
        signal = torch.exp(exponent)  # [V, N]
        # Handle S0
        if not torch.is_tensor(S0):
            S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
        S0 = S0.reshape(-1, 1)
        signal = S0 * signal
        return signal.reshape(W, H, B, N)

    def forward(self, x, bvecs):
        """
        x: [1, 91, W, H, B]
        bvecs: [N, 3] (passed at runtime)
        Returns:
            pred_tensor: [W, H, B, 3, 3]
            pred_signal: [W, H, B, N]
        """
        # Compute mean_b0 from the first channel (assumed b=0)
        x=torch.abs(x)
        mean_b0 = x[0,0,...]  # [W, H, B]
        #print(x.shape)
        # Predict tensor using network (SPD)
        params = self.encoder(x)  # [1, 6, W, H, B]
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]
        l123max=0.041; l123min=0.011
        l456min=-0.15; l456max=0.15

        """ # Cholesky construction
        l11 =l123min + torch.sigmoid(params[..., 0])*(l123max-l123min)
        l21 =l123min + torch.sigmoid(params[..., 1])*(l123max-l123min)
        l31 =l123min + torch.sigmoid(params[..., 2])*(l123max-l123min)
        l22 =l456min + torch.sigmoid(params[..., 3])*(l456max-l456min)
        l32 =l456min + torch.sigmoid(params[..., 4])*(l456max-l456min)
        l33 =l456min + torch.sigmoid(params[..., 5])*(l456max-l456min)
         """
        l11 =torch.abs(params[..., 0])
        l21 =params[..., 1]
        l31 =params[..., 2]
        l22 =torch.abs(params[..., 3])
        l32 =params[..., 4]
        l33 =torch.abs(params[..., 5])
 
        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33
        D = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]

        # Predict signal from network tensor using current bvecs and mean_b0
        pred_signal = self.predict_dti_signal(D, bvecs, S0=mean_b0)  # [W, H, B, N]
        x=pred_signal.permute(3,0,1,2)[None,...]

        return D,  pred_signal


""" 
class DTINet3D_PI_old(nn.Module):
    def __init__(self, in_channels=91, bvals=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=1, padding=0),
            nn.BatchNorm3d(256),
            
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 6, kernel_size=1)
        )
        #self.register_buffer('bvals', torch.as_tensor(bvals, dtype=torch.float32))  # [N]
        self.bvals=bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    
    def predict_dti_signal(self, D_est, bvecs, S0=1.0):
        
        #D_est: [W, H, B, 3, 3]
        #bvecs: [N, 3] (passed at runtime)
        #S0: [W, H, B] or scalar
        #Returns: [W, H, B, N]
        
        W, H, B, _, _ = D_est.shape
        N = bvecs.shape[0]
        
        D_flat = D_est.reshape(-1, 3, 3)  # [V, 3, 3]
        gdg = torch.einsum('nj,vjk,nk->vn', bvecs, D_flat, bvecs)  # [V, N]
        exponent = -self.bvals * gdg  # [V, N]
        signal = torch.exp(exponent)  # [V, N]
        # Handle S0
        if not torch.is_tensor(S0):
            S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
        S0 = S0.reshape(-1, 1)
        signal = S0 * signal
        return signal.reshape(W, H, B, N)

    def forward(self, x, bvecs):
        
        #x: [1, 91, W, H, B]
        #bvecs: [N, 3] (passed at runtime)
        #Returns:
         #   pred_tensor: [W, H, B, 3, 3]
         #   pred_signal: [W, H, B, N]
        
        # Compute mean_b0 from the first channel (assumed b=0)
        x=torch.abs(x)
        mean_b0 = x[0,0,...]  # [W, H, B]
        #print(x.shape)
        # Predict tensor using network (SPD)
        params = self.encoder(x)  # [1, 6, W, H, B]
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]
        l123max=0.041; l123min=0.011
        l456min=-0.15; l456max=0.15

         # Cholesky construction
        #l11 =l123min + torch.sigmoid(params[..., 0])*(l123max-l123min)
        #l21 =l123min + torch.sigmoid(params[..., 1])*(l123max-l123min)
        #l31 =l123min + torch.sigmoid(params[..., 2])*(l123max-l123min)
        #l22 =l456min + torch.sigmoid(params[..., 3])*(l456max-l456min)
        #l32 =l456min + torch.sigmoid(params[..., 4])*(l456max-l456min)
        #l33 =l456min + torch.sigmoid(params[..., 5])*(l456max-l456min)
        
        l11 =torch.abs(params[..., 0])
        l21 =params[..., 1]
        l31 =params[..., 2]
        l22 =torch.abs(params[..., 3])
        l32 =params[..., 4]
        l33 =torch.abs(params[..., 5])
 
        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33
        D = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]

        # Predict signal from network tensor using current bvecs and mean_b0
        pred_signal = self.predict_dti_signal(D, bvecs, S0=mean_b0)  # [W, H, B, N]
        x=pred_signal.permute(3,0,1,2)[None,...]

        return D,  pred_signal
 """


import torch
import torch.nn as nn
import torch.nn.functional as F

class DTINet3D_PI_old(nn.Module):
    def __init__(self, in_channels=91, bvals=None, dropout_p=0.3):
        super().__init__()
        self.dropout_p = dropout_p
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p),

            nn.Conv3d(64, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p),

            nn.Conv3d(128, 256, kernel_size=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(256),

            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p),

            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),

            nn.Conv3d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),

            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            nn.ReLU(),

            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 6, kernel_size=1)
        )

        self.bvals = bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_dti_signal(self, D_est, bvecs, S0=1.0):
        W, H, B, _, _ = D_est.shape
        N = bvecs.shape[0]
        
        D_flat = D_est.reshape(-1, 3, 3)  # [V, 3, 3]
        gdg = torch.einsum('nj,vjk,nk->vn', bvecs, D_flat, bvecs)  # [V, N]
        exponent = -self.bvals * gdg  # [V, N]
        signal = torch.exp(exponent)  # [V, N]
        
        if not torch.is_tensor(S0):
            S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
        S0 = S0.reshape(-1, 1)
        signal = S0 * signal
        return signal.reshape(W, H, B, N)

    def forward(self, x, bvecs):
        x = torch.abs(x)
        mean_b0 = x[0, 0, ...]  # [W, H, B]

        params = self.encoder(x)  # [1, 6, W, H, B]
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]

        l11 = torch.abs(params[..., 0])
        l21 = params[..., 1]
        l31 = params[..., 2]
        l22 = torch.abs(params[..., 3])
        l32 = params[..., 4]
        l33 = torch.abs(params[..., 5])

        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33

        D = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]
        pred_signal = self.predict_dti_signal(D, bvecs, S0=mean_b0)
        x = pred_signal.permute(3, 0, 1, 2)[None, ...]
        return D, pred_signal


def enable_mc_dropout(model):
    """ Enables dropout during inference for MC Dropout """
    for m in model.modules():
        if isinstance(m, nn.Dropout3d):
            m.train()


# Example usage:
# x: [1, 91, W, H, B], mean_b0: [W, H, B], bvecs: [91, 3], bvals: [91]
# Example shapes
""" W, H, B = 32, 32, 32
x = torch.randn(1, 91, W, H, B)      # Example input signal
bvecs = torch.randn(91, 3)           # Example bvecs for this batch
bvals = torch.randn(91)           # Example bvecs for this batch

model = DTINet3D_PI(in_channels=91, bvals=bvals.cpu())

pred_tensor, pred_signal = model(x, bvecs)
 """

class DTINet3D_Eig(nn.Module):
    def __init__(self, in_channels=None):
        super(DTINet3D_Eig, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 256, kernel_size=5, padding=2),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 128, kernel_size=5, padding=2),
            nn.ReLU(),

            nn.Conv3d(128, 128, kernel_size=7, padding=3),
            nn.Conv3d(128, 64, kernel_size=7, padding=3),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
        )

        # Directly predict 3 eigenvalues
        self.final = nn.Conv3d(32, 3, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = torch.abs(x)                      # Ensure positive inputs
        x = self.encoder(x)                   # [B, 32, D, H, W]
        eigvals = self.final(x)               # [B, 3, D, H, W]
        eigvals = torch.abs(eigvals)          # Optionally ensure non-negative eigenvalues
        eigvals = eigvals.permute(2, 3, 4, 0, 1)  # [D, H, W, B, 3]
        return eigvals


def FA(eigvals: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute Fractional Anisotropy (FA) from eigenvalues.
    
    Args:
        eigvals (torch.Tensor): Tensor of shape [..., 3] representing the 3 eigenvalues.
        eps (float): Small value to avoid division by zero.

    Returns:
        torch.Tensor: FA values with shape [...].
    """
    λ1, λ2, λ3 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]

    # Mean diffusivity
    md = (λ1 + λ2 + λ3) / 3.0

    # FA numerator: sum of squared differences from mean
    numerator = (λ1 - md)**2 + (λ2 - md)**2 + (λ3 - md)**2

    # FA denominator: sum of squares
    denominator = λ1**2 + λ2**2 + λ3**2

    fa = torch.sqrt(1.5 * numerator / (denominator + eps))
    fa = torch.clamp(fa, 0.0, 1.0)  # Ensure FA in [0, 1]
    return fa


class DTINet(nn.Module):
    def __init__(self, in_channels=None):
        super(DTINet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 32, kernel_size=3, padding=1),
        )

        self.final = nn.Conv2d(32, 6, kernel_size=1)

        # Apply He initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.permute(2, 3, 0, 1)  # [B, G, W, H]
        x = torch.abs(x)

        x = self.encoder(x)  # [B, 32, W, H]
        params = self.final(x)  # [B, 6, W, H]
        params = params.permute(2, 3, 0, 1)  # [W, H, B, 6]

        Dxx = params[..., 0]
        Dyy = params[..., 1]
        Dzz = params[..., 2]
        Dxy = params[..., 3]
        Dxz = params[..., 4]
        Dyz = params[..., 5]

        D = torch.stack([
            torch.stack([Dxx, Dxy, Dxz], dim=-1),
            torch.stack([Dxy, Dyy, Dyz], dim=-1),
            torch.stack([Dxz, Dyz, Dzz], dim=-1)
        ], dim=-2)

        eps = 1e-10
        D = torch.nan_to_num(D, nan=eps, posinf=eps, neginf=eps)

        eigvals, eigvecs = torch.linalg.eigh(D)
        eigvecs, _ = torch.linalg.qr(eigvecs)
        eigvals = torch.abs(eigvals)

        D_spd = torch.matmul(eigvecs, torch.matmul(torch.diag_embed(eigvals), eigvecs.transpose(-1, -2)))

        return D_spd




def build_design_matrix(bvals, bvecs):
    """
    Build the design matrix for DTI signal prediction.
    Returns [N, 6]: [b_x^2, b_y^2, b_z^2, 2*b_x*b_y, 2*b_x*b_z, 2*b_y*b_z] * bval for each measurement.
    """
    bvecs = torch.as_tensor(bvecs, dtype=torch.float32)
    bvals = torch.as_tensor(bvals, dtype=torch.float32)
    bx, by, bz = bvecs[:, 0], bvecs[:, 1], bvecs[:, 2]
    design = torch.stack([
        bx**2, by**2, bz**2,
        2*bx*by, 2*bx*bz, 2*by*bz
    ], dim=-1)  # [N, 6]
    return design * bvals[:, None]

def predict_dti_signal2(D_est, bvals, bvecs, S0=1.0):
    """
    Predict DTI signal using D_est tensors via S = S0 * exp(-bmat @ d.T)
    Args:
        D_est: tensor of shape [..., 3, 3] — full diffusion tensor
        bvals: array-like of shape [N] — diffusion weighting
        bvecs: array-like of shape [N, 3] — gradient directions
        S0: scalar or tensor broadcastable to D_est[..., 0, 0] — baseline signal
    Returns:
        Predicted signal: tensor of shape [..., N]
    """
    # Extract unique tensor elements (order: Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
    Dxx = D_est[..., 0, 0]
    Dyy = D_est[..., 1, 1]
    Dzz = D_est[..., 2, 2]
    Dxy = D_est[..., 0, 1]
    Dxz = D_est[..., 0, 2]
    Dyz = D_est[..., 1, 2]
    d = torch.stack([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], dim=-1).float()  # [..., 6]

    # Build design matrix [N, 6]
    bmat = build_design_matrix(bvals, bvecs).float()  # [N, 6]

    # Compute exponent: [..., N] = -1000 * (d @ bmat.T)
    # (d shape: [..., 6], bmat.T shape: [6, N])
    exponent = -1000 * torch.matmul(d, bmat.T)  # [..., N]

    # Compute signal
    signal = torch.exp(exponent)  # [..., N]

    # Handle S0 broadcasting
    if not torch.is_tensor(S0):
        S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
    while S0.dim() < signal.dim():
        S0 = S0.unsqueeze(-1)
    signal = S0 * signal  # [..., N]

    return signal


def predict_dti_signal(D_est, bvals, bvecs, S0=1.0):
    """
    D_est: [num_voxels, 3, 3]
    bvals: [N]
    bvecs: [N, 3]
    S0: scalar or [num_voxels] or broadcastable
    Returns: [num_voxels, N]
    """
    W=D_est.shape[0]
    H=D_est.shape[1]
    B=D_est.shape[2]
    V=W*H*B    
    
    D_est = torch.as_tensor(D_est, dtype=torch.float32)        # [V, 3, 3]
    bvals = torch.as_tensor(bvals, dtype=torch.float32)        # [N]
    bvecs = torch.as_tensor(bvecs.T, dtype=torch.float32)        # [N, 3]
    G=bvecs.shape[0]
    
    gdg = torch.einsum('nj,vjk,nk->vn', bvecs, D_est.reshape(V,3,3), bvecs)   # [V, N]
    exponent = -bvals * gdg                       # [V, N]
    signal = torch.exp(exponent)                               # [V, N]

    # Handle S0
    if not torch.is_tensor(S0):
        S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
    if S0.ndim == 0:
        S0 = S0.view(1, 1)
    elif S0.ndim == 1:
        S0 = S0.view(-1, 1)
 
    signal = S0* signal.reshape(W, H, B, G)
    #signal = signal.reshape(W,H,B, G)
    
    return signal 



def dti_signal_estimate(D_est, bvec, bval=1000):
    """
    Computes DTI signal S = S0 * exp(-b * g^T D g)
    
    Args:
        D_est: Tensor of shape [X, Y, Z, 3, 3]
        bvecs: Tensor of shape [G, 3]
        bval:  Scalar b-value (e.g., 1000)
        S0:    Scalar or tensor broadcastable to [X, Y, Z]

    Returns:
        Signal: Tensor of shape [X, Y, Z, G]
    """
    X, Y, Z, _, _ = D_est.shape
    G = bvec.shape[1]

      
    # Ensure bvecs is on the same device
    bvec = bvec.to(D_est.device)

    # Expand tensors for broadcasting
    D = D_est.unsqueeze(3)  # [X, Y, Z, 1, 3, 3]
    g = bvec.reshape(1, 1, 1, G, 3, 1)  # [1, 1, 1, G, 3, 1]

    # Compute g^T D g
    Dg = torch.matmul(D, g)                      # [X, Y, Z, G, 3, 1]
    gDg = torch.matmul(g.transpose(-2, -1), Dg)  # [X, Y, Z, G, 1, 1]
    gDg = gDg.squeeze(-1).squeeze(-1)            # [X, Y, Z, G]

    # Signal equation
    exponent = -bval * gDg
    
    S = torch.exp(exponent)
    
    """ 
    X, Y, Z, _, _ = D_est.shape
    G = bvec.shape[1]
    D_est_flat=D_est.reshape(-1,3,3)
    S=torch.zeros(G, X*Y*Z).to(D_est.device)
    
    for i in range(G):
       for j in range(X*Y*Z):
         S[i,j]= -bval*torch.exp((bvec[:,i:i+1].T@D_est_flat[j,...]@bvec[:,i:i+1]))

    S=S.reshape([X, Y, Z, G])   
    """    
    return S  # shape: [X, Y, Z, G]



def eigenvalue_distributions(est=None, gt=None, gt_fname=None, est_fname=None, mask=None):
    
    labels = ["λ₁", "λ₂", "λ₃"]
    colors = ["green", "red"]
    
    if mask is not None:
        est = est[mask]
        gt = gt[mask]

    plt.figure(figsize=(15, 4))

    for i in range(3):
        plt.subplot(1, 3, i+1)
        sns.histplot(gt[:, i], kde=True, color=colors[1], label=gt_fname, stat="density", bins=50, alpha=0.6)
        sns.histplot(est[:, i], kde=True, color=colors[0], label=est_fname, stat="density", bins=50, alpha=0.6)
        plt.title(f"Distribution of {labels[i]}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()

    plt.tight_layout()
    plt.show()



def lambdas_boxplots(eigvals_gt, eigvals_est):
    """
    Generates a boxplot comparing ground truth and estimated eigenvalues.
    
    Parameters:
        eigvals_gt (numpy.ndarray): Ground truth eigenvalues of shape (H, W, D, 3).
        eigvals_est (numpy.ndarray): Estimated eigenvalues of shape (H, W, D, 3).
    """
    # Validate inputs
    if eigvals_gt.shape != eigvals_est.shape:
        raise ValueError("The ground truth and estimated eigenvalues must have the same shape.")
    if eigvals_gt.shape[-1] != 3:
        raise ValueError("The last dimension of the eigenvalue arrays must be 3 (for λ1, λ2, λ3).")
    
    # Flatten the eigenvalue arrays to (N_voxels, 3)
    eigvals_gt_flat = eigvals_gt.reshape(-1, 3)
    eigvals_est_flat = eigvals_est.reshape(-1, 3)

    # Prepare data for the boxplot
    data = []
    labels = []
    colors = []
    for i in range(3):  # Loop over λ1, λ2, λ3
        data.append(eigvals_gt_flat[:, i])  # Ground truth
        labels.append(f'λ{i+1} (GT)')
        colors.append('red')  # Red for GT
        data.append(eigvals_est_flat[:, i])  # Estimated
        labels.append(f'λ{i+1} (Est)')
        colors.append('green')  # Green for Est

    # Plot the boxplot
    plt.figure(figsize=(10, 6))
    bplot = plt.boxplot(data, labels=labels,  showfliers=False, patch_artist=True)

    # Apply colors to the boxes
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Rotate x-ticks for better readability
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Eigenvalues (mm²/s)')
    plt.title('Comparison of Ground Truth (Red) and Estimated (Green) Eigenvalues')
    #plt.grid(axis='y',  alpha=0.7)
    plt.tight_layout()
    plt.show()


def calculate_md(eigvals):
    return eigvals.mean(axis=-1)

def calculate_ad(eigvals):
     return eigvals[:,:,:, 0]

def calculate_ad2(eigvals):
     return eigvals[:,:,:, 1]

def calculate_rd(eigvals):
    return eigvals[:,:,:, 1:3].mean(axis=3)

from fury import window, actor

def visualize_two_bvecs(bvecs1, bvecs2):
    """
    Visualize two sets of b-vectors (Nx3) on a unit sphere using lines and points.
    
    Parameters:
    - bvecs1: numpy array of shape (N, 3)
    - bvecs2: numpy array of shape (M, 3)
    """
    # Create lines from origin to each vector
    lines1 = np.array([[np.zeros(3), v] for v in bvecs1])
    lines2 = np.array([[np.zeros(3), v] for v in bvecs2])

    # Create line actors
    line_actor1 = actor.line(lines1, colors=(0, 1, 0))  # Green lines for bvecs1
    line_actor2 = actor.line(lines2, colors=(1, 0, 0))  # Red lines for bvecs2

    # Create point actors
    point_actor1 = actor.point(bvecs1, colors=(0, 1, 0), point_radius=0.01)
    point_actor2 = actor.point(bvecs2, colors=(1, 0, 0), point_radius=0.01)

    # Create the scene and add all actors
    scene = window.Scene()
    scene.add(line_actor1)
    scene.add(line_actor2)
    scene.add(point_actor1)
    scene.add(point_actor2)

    # Show the visualization
    window.show(scene)



def load_processed_subject(base_path, subject_id):
    """
    Load DWI data, bvals, bvecs, and brain mask from .pt files in a flat directory.

    Parameters:
    - base_path (str or Path): Path to the directory containing all `.pt` files.
    - subject_id (str): e.g., "sub1", "sub2", ...

    Returns:
    - data (torch.Tensor): 4D diffusion data [W, H, Z, G]
    - bvals (torch.Tensor): 1D b-values [G]
    - bvecs (torch.Tensor): 2D gradient directions [3, G]
    - mask (torch.Tensor): 3D brain mask [W, H, Z]
    """
    base_path = Path(base_path)

    data = torch.load(base_path / f"{subject_id}_data.pt")
    bvals = torch.load(base_path / f"{subject_id}_bvals.pt")
    bvecs = torch.load(base_path / f"{subject_id}_bvecs.pt")
    mask = torch.load(base_path / f"{subject_id}_mask.pt")

    return data, bvals, bvecs, mask




def map_to_range_tensors(D_est, D_gt):
    """
    Element-wise min-max scale each component of symmetric 3x3 D_est
    to match the range of corresponding component in D_gt.

    Args:
        D_est: [..., 3, 3] tensor (e.g. predicted)
        D_gt:  [..., 3, 3] tensor (e.g. ground truth)

    Returns:
        Scaled D_est: [..., 3, 3]
    """

    def tensor3x3_to_vec6(D):
        Dxx = D[..., 0, 0]
        Dyy = D[..., 1, 1]
        Dzz = D[..., 2, 2]
        Dxy = D[..., 0, 1]
        Dxz = D[..., 0, 2]
        Dyz = D[..., 1, 2]
        return torch.stack([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz], dim=-1)

    def vec6_to_tensor3x3(v):
        Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = [v[..., i] for i in range(6)]
        zeros = torch.zeros_like(Dxx)
        D = torch.stack([
            torch.stack([Dxx, Dxy, Dxz], dim=-1),
            torch.stack([Dxy, Dyy, Dyz], dim=-1),
            torch.stack([Dxz, Dyz, Dzz], dim=-1)
        ], dim=-2)
        return D

    # Convert to 6D vector form
    vec_est = tensor3x3_to_vec6(D_est)
    vec_gt = tensor3x3_to_vec6(D_gt)
    
    # Scale each component of vec_est to match the min/max of vec_gt
    vec_scaled = []
    for i in range(6):
        est_i = vec_est[..., i]
        gt_i = vec_gt[..., i]
        gt_min = gt_i.amin()
        gt_max = gt_i.amax()
        est_min = est_i.amin()
        est_max = est_i.amax()
        # Avoid divide-by-zero
        scale = (gt_max - gt_min) / (est_max - est_min + 1e-8)
        vec_i_scaled = (est_i - est_min) * scale + gt_min
        vec_scaled.append(vec_i_scaled)

    vec_scaled = torch.stack(vec_scaled, dim=-1)
    D_scaled = vec6_to_tensor3x3(vec_scaled)
    return D_scaled

def sample_unit_sphere(n_points, device=None):
    """
    Uniformly sample n_points directions over a unit sphere using PyTorch.
    
    Parameters:
        n_points (int): Number of unit directions to generate.
        device (str): The device to use ("cpu" or "cuda").
    
    Returns:
        torch.Tensor: Tensor of shape (n_points, 3) with unit direction vectors.
    """
    # Generate random azimuthal angles (phi) uniformly in [0, 2*pi]
    #phi = torch.rand(n_points, device=device) * 2 * torch.pi
    
    theta = 2 * torch.pi * torch.rand(n_points, device=device)
    phi = torch.acos(1 - 2 * torch.rand(n_points, device=device))

    # Generate random cos(theta) uniformly in [-1, 1]
    #cos_theta = torch.rand(n_points, device=device) * 2 - 1
    #theta = torch.acos(cos_theta)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    
    return torch.stack((x, y, z), dim=1)




def batch_rand_perlin_2d(b, shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3, device=None):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
   
    grid = torch.stack(torch.meshgrid(torch.arange(0, res[0], delta[0], device=device), torch.arange(0, res[1], delta[1], device=device)), dim = -1) % 1
    angles = 2*math.pi*torch.rand(b, res[0]+1, res[1]+1, device=device)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)
   
    tile_grads = lambda slice1, slice2: gradients[:,slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 1).repeat_interleave(d[1], 2)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:, :shape[0], :shape[1]]).sum(dim = -1)
   
    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])

    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def batch_rand_perlin_2d_octaves2(b, shape, res, octaves=1, persistence=0.5, device=None):
    noise = torch.zeros((b,) + shape, device=device)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * batch_rand_perlin_2d(b, shape, (frequency*res[0], frequency*res[1]), device=device)
        frequency *= 2
        amplitude *= persistence
    return noise


def batch_rand_perlin_2d_octaves(lac, b, shape, res, octaves=1, persistence=0.5, device=None):
    noise = torch.zeros((b,) + shape, device=device)
    frequency = 1
    amplitude = 1
    #lac=1
    #persistence=.2
    
    for _ in range(octaves):
        
        noise += amplitude * batch_rand_perlin_2d(b, shape, (int(frequency*res[0]), int(frequency*res[1])), device=device)
        frequency *= lac
        amplitude *= persistence
    return noise

def generate_mask(batchsize=None, w=None,h=None, device=None):
    
    
        mask = torch.zeros((batchsize, w,h), device=device)
        
        resolutions = [(2**i,2**i) for i in range(0,4)] 
        
        r=resolutions[torch.randint(2, 4, (1,),device=device).item()] 
        o=torch.randint(2, 4, (1,),device=device).item()
        o=4
        r=(4,4)
        persistence= 0.5 #* (torch.rand(1,device=device) + 0.2)
        noise=batch_rand_perlin_2d_octaves(lac=2, b=batchsize, shape=(w, h), res=r, octaves=o, persistence=persistence, device=device) 
        #noise=(noise).squeeze()
        # Assign each region a unique value or color for each image
        point=noise.max()/3
        m1=noise >0 
        m2=noise<point 
        mask[ m1 & m2] = 1
        m3=noise <=0
        m4= noise>-point
        mask[ m3 & m4] = 2
        mask[((noise <-point )  | (noise > point))]=3

        return mask

def generate_mask2(batchsize=None, w=None,h=None, device=None):
    
    
    mask = torch.zeros((batchsize, w,h), device=device)
        
    resolutions = [(2**i,2**i) for i in range(1,3)] 
    point=0.954
    r=resolutions[torch.randint(0, 2, (1,),device=device).item()] 
    o=torch.randint(1, 3, (1,),device=device).item()
    persistence= 0.7 * (torch.rand(1,device=device) + 0.2)
    noise=batch_rand_perlin_2d_octaves(lac=2,b=batchsize, shape=(w,h), res=r, octaves=o, persistence=persistence, device=device) 
    # Assign each region a unique value or color for each image
    point=0.954/3
    m1=noise >0 
    m2=noise<point 
    mask[ m1 & m2] = 1
    m3=noise <=0
    m4= noise>-point
    mask[ m3 & m4] = 2
    mask[((noise <-point )  | (noise > point))]=3

    return mask

def generate_random_rots_uniform(w=128, h=128, batch_size=16, device="cuda"):
    """
    Generate random 3x3 rotation matrices uniformly sampled from SO(3) for a 128x128x16 grid.

    Parameters:
        w (int): First grid dimension (default: 128).
        h (int): Second grid dimension (default: 128).
        batch_size (int): Third grid dimension (default: 16).
        device (str): Device to perform computation on ("cuda" or "cpu").

    Returns:
        torch.Tensor: Tensor of shape (w, h, batch_size, 3, 3) containing rotation matrices.
    """
    total_batches = w * h * batch_size  # Flattened total number of matrices to generate

    # Step 1: Generate random vectors
    x1 = torch.randn((total_batches, 3), device=device)
    x2 = torch.randn((total_batches, 3), device=device)

    # Step 2: Normalize the first vector
    v1 = x1 / torch.norm(x1, dim=-1, keepdim=True)

    # Step 3: Make the second vector orthogonal to the first
    proj = torch.sum(x2 * v1, dim=-1, keepdim=True) * v1  # Projection of x2 onto v1
    v2 = x2 - proj
    v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

    # Step 4: Compute the third vector using the cross product
    v3 = torch.cross(v1, v2)

    # Step 5: Stack the vectors to form the rotation matrix
    Q = torch.stack((v1, v2, v3), dim=-1)  # Shape: (total_batches, 3, 3)

    # Step 6: Reshape to the desired grid shape
    rotation_matrices = Q.view(w, h, batch_size, 3, 3)
    return rotation_matrices

def display_gradients(unit_directions=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(unit_directions[:, 0].cpu(), unit_directions[:, 1].cpu(), unit_directions[:, 2].cpu(), s=1)
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    plt.show()




def add_rician_noise(images, snr):
    
    sigma = 1 / snr
    #sigma=0.005
    # Generate the noise components as independent Gaussian random variables 
    real_noise = torch.normal(0, sigma, images.size(), device=images.device)
    imag_noise = torch.normal(0, sigma, images.size(), device=images.device)

    # Add the noise components to the images 
    noisy_images = torch.sqrt((images + real_noise) ** 2 + imag_noise ** 2)


    return noisy_images


def gen_perlin( w=None, h=None,  batchsize=None, lims=None, lacunarity=None,  device=None):

    #resolutions = [(2**i,2**i) for i in range(1,3)] 
    #r=resolutions[torch.randint(0, 2, (1,),device=device).item()]
       
    while(True):
        #resolutions = [(2**i,2**i) for i in range(0,4)] 
        #r=resolutions[torch.randint(2, 4, (1,),device=device).item()] 
        #o=torch.randint(2, 4, (1,),device=device).item()
        #o=4; 
        persistence=0.5; 
        resolutions = [(2**i,2**i) for i in range(1,3)]
        r=resolutions[torch.randint(0, 2, (1,),device=device).item()] 
        o=torch.randint(1, 3, (1,),device=device).item()
        persistence= 0.7 * (torch.rand(1,device=device) + 0.2)
        #r=(4, 4)
        #lacunarity=4    
        image=batch_rand_perlin_2d_octaves(lac=lacunarity, b=(batchsize), shape=(w, h), res=r, octaves=o, persistence=persistence, device=device) 
        image= 2*(image - torch.min(image))/(torch.max(image) - torch.min(image))-1 
        image=map_to_range(image,  lims.min(),  lims.max())
        # Apply a sinusoidal transformation to introduce negative values
        #image = torch.cos(2*math.pi*image/lims.max() ) * (lims.max()  - lims.min())
        #print(image.max(), image.min(), lims.max(), lims.min())
        #exit()    
            
        yield image    
   


   
def genbatch3(batchsize=None,  gradients=None, b_value=None, device=None, 
             w=None, h=None, snr=None, N=None, mask=None):
     
    epsilon=1e-10      
      
    if mask==None:        
         mask=generate_mask(batchsize=batchsize, w=w, h=h, device=device)
         mask=mask.permute(1,2,0) 
    else:
         mask=mask    

    D_gmrange=torch.tensor([0.5* 10**(-3), 2.0* 10**(-3)], device=device)
    D_wmrange=torch.tensor([0.3* 10**(-3), 1.3* 10**(-3)], device=device)
    D_csfrange=torch.tensor([1.0* 10**(-3), 3.0* 10**(-3)], device=device)
    
    #
    #ivimGM(0.0005, 0.002), (-0.0006, 0.0044)
    #ivimWM(0.0003, 0.0013),(-0.0010, 0.0031)
    #ivimCSF(0.001, 0.003),  (-0.0003,0.0046)
    
    full_range_D=torch.tensor([0., 3.0* 10**(-3)], device=device)
    D_wmrange=D_gmrange=D_csfrange=full_range_D
    
    full_range_S0=torch.tensor([0.05, 1.0], device=device)
    S0_wmrange=S0_gmrange=S0_csfrange=full_range_S0
    
    while(True):
        
        
        Dcomps_wm=torch.zeros(batchsize, w, h, 6, device=device)
        Dcomps_gm=torch.zeros(batchsize, w, h, 6, device=device)
        Dcomps_csf=torch.zeros(batchsize,w, h, 6, device=device)
        
        for i in range(6):
            Dcomps_wm[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=D_wmrange , device=device))
            Dcomps_gm[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=D_gmrange , device=device))
            Dcomps_csf[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=D_csfrange , device=device))

        
        #D_gm=ensure_spd_tensor2(D_components=Dcomps_gm, tissue_type='gm') # 128x128x32x3x3
        #D_wm=ensure_spd_tensor2(D_components=Dcomps_wm, tissue_type='wm') # 128x128x32x3x3
        #D_csf=ensure_spd_tensor2(D_components=Dcomps_csf, tissue_type='csf') # 128x128x32x3x3
        
        #D = torch.zeros_like(D_wm).to(device=device)

        # Set the vectors based on the regions in the mask
        #D[mask==1,:,:] = D_wm[mask==1,:,:]  # White matter region
        #D[mask==2,:,:] = D_gm[mask==2,:,:]  # Gray matter region
        #D[mask==3,:,:] = D_csf[mask==3,:,:] # Cerebrospinal fluid region

        
        """         
        plt.imshow(mask[:,:,10].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D_wm[:,:,10,1,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D_gm[:,:,10,1,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D_csf[:,:,10,1,1].cpu())
        plt.colorbar()
        plt.show()
        exit()
        """  
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
       
     
        #S0_wmrange=torch.tensor([0.05, 0.25], device=device)
        #S0_gmrange=torch.tensor([0.2, 0.5], device=device)
        #S0_csfrange=torch.tensor([0.5, 1.0], device=device)
        
        
        S0wm_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_wmrange , device=device))
        S0gm_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_gmrange , device=device))
        S0csf_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_csfrange , device=device))
             
        #print(mask.shape, S0csf_mask.shape)       
        
        S0 = torch.zeros(( w, h, batchsize), device=device)
        
        S0[mask==1]=S0wm_mask.permute(1,2,0)[mask==1]
        S0[mask==2]=S0gm_mask.permute(1,2,0)[mask==2]
        S0[mask==3]=S0csf_mask.permute(1,2,0)[mask==3]
        
        #S0 = torch.rand(( w, h, batchsize), device=device) # all random

        #random_indices = np.random.choice(sphere.vertices.shape[0], size=N-1, replace=False)
        
        #selected_gradients = torch.tensor(sphere.vertices[random_indices], device=device) # Fixed 756!
        
        #selected_gradients = sample_unit_semisphere(n_points=N-1, device=device) # Not fixed, over sphere
        selected_gradients = sample_unit_sphere(n_points=N-1, device=device) # Not fixed, over sphere

        #selected_gradients=torch.tensor(grads64).to(device)# Fixed
        selected_gradients=torch.nan_to_num(selected_gradients, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        grads=torch.zeros((N, 3)).to(device)
        grads[1:,:]=selected_gradients
        
        #S=gen_dti_signals(b_values=b_value, gradients=grads, D=D, S0=S0,device=device)
        #print(S.shape, S.max(), S.min(), S0.max())
        #exit()
        #S=map_to_range(S, 0., 1.)
        
        if snr is not None:
            S=add_rician_noise(S, snr=snr)
         
        #S=map_to_range(S, 0., 1.)
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
        
        yield D, S, grads, mask   

def pad_and_resize_signal(signal, target_size=128):
    """
    signal: tensor of shape (height, width, channels, batch)
           e.g., (125, 154, 1, 91)
    target_size: int, output spatial size (square)
    returns: tensor of shape (target_size, target_size, channels, batch)
    """
    # Permute to (batch, channels, height, width) if needed, but here we keep original order
    # signal is (height, width, channels, batch) = (125, 154, 1, 91)
    # For padding, we need to move spatial dims to the end, but PyTorch F.pad expects (..., H, W)
    # So, permute to (channels, batch, height, width)
    signal = signal.permute(2, 3, 0, 1)  # (1, 91, 125, 154)
    
    # Pad to square: (height, width) to (max_side, max_side)
    h, w = signal.shape[-2], signal.shape[-1]
    max_side = max(h, w)
    pad_h = (max_side - h) // 2
    pad_w = (max_side - w) // 2
    padded = F.pad(signal, (pad_w, max_side - w - pad_w, pad_h, max_side - h - pad_h), mode='constant')
    
    # Resize to target_size
    resized = F.interpolate(padded, size=(target_size, target_size), mode='bilinear', align_corners=False)
    
    # Permute back to original order (optional)
    # Here, output is (channels, batch, target_size, target_size)
    # To get (target_size, target_size, channels, batch):
    resized = resized.permute(2, 3, 0, 1)
    return resized


def L_to_D(L=None, matter=None):        
        
    
    """ l123max=0.041; l123min=0.011
    l456min=-0.15; l456max=0.15

    # Cholesky construction
    l11 =l123min + torch.abs(L[...,0])*(l123max-l123min)
    l21 =l123min + L[...,1]*(l123max-l123min)
    l31 =l123min + L[...,2]*(l123max-l123min)
    l22 =l456min + torch.abs(L[...,3])*(l456max-l456min)
    l32 =l456min + L[...,4]*(l456max-l456min)
    l33 =l456min + torch.abs(L[...,5])*(l456max-l456min)
     """
    l11 =torch.abs(L[..., 0])
    l21 =L[..., 1]
    l31 =L[..., 2]
    l22 =torch.abs(L[..., 3])
    l32 =L[..., 4]
    l33 =torch.abs(L[..., 5])
    
    L3x3 = torch.zeros(L.shape[0], L.shape[1], L.shape[2], 3, 3, device=L.device)
    
    L3x3[..., 0, 0] = l11
    L3x3[..., 1, 0] = l21
    L3x3[..., 2, 0] = l31
    L3x3[..., 1, 1] = l22
    L3x3[..., 2, 1] = l32
    L3x3[..., 2, 2] = l33
    
    D = L3x3 @ L3x3.transpose(-1, -2)  # [W, H, B, 3, 3]
    D_wmrange=torch.tensor([0.3* 10**(-3), 1.3* 10**(-3)], device=L.device)
    D_gmrange=torch.tensor([0.5* 10**(-3), 2.0* 10**(-3)], device=L.device)
    D_wmrange=torch.tensor([0.3* 10**(-3), 1.3* 10**(-3)], device=L.device)
    D_csfrange=torch.tensor([1.0* 10**(-3), 3.0* 10**(-3)], device=L.device)
    
    if matter=='wm':
        D = map_to_range(D,  D_wmrange.min(),  D_wmrange.max()) #To check
    if matter=='gm':
        D = map_to_range(D,  D_gmrange.min(),  D_gmrange.max()) #To check
    if matter=='csf':
        D = map_to_range(D,  D_csfrange.min(),  D_csfrange.max()) #To check
    
    

    """ 
    if matter=='wm':
        eigvals, eigvecs = torch.linalg.eigh(D)
        eigvals, _=torch.sort(eigvals, dim=-1, descending=True)
        #eigvals[...,0]=eigvals[...,1]=eigvals[...,2]
        #eigvals=0.0040*(eigvals/eigvals.max())
        eigvals=map_to_range(eigvals,0.0003, 0.004)
        D= eigvecs@ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
    elif(matter=='gm'):
        eigvals, eigvecs = torch.linalg.eigh(D)
        #eigvals=0.0040*(eigvals/eigvals.max())
        eigvals=map_to_range(eigvals,0.0003, 0.0037)
        #eigvals[...,0]=eigvals[...,1]=eigvals[...,2]
        D= eigvecs@ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
    else:
        eigvals, eigvecs = torch.linalg.eigh(D)
        #eigvals=0.0040*(eigvals/eigvals.max())
        eigvals=map_to_range(eigvals,0.0003, 0.0037)
        eigvals[...,0]=eigvals[...,1]=eigvals[...,2]
        D= eigvecs@ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
     """
 
    return D    


def L_to_D_2(L=None, matter=None):        
        
    
    if matter=='wm':
        eigvals, eigvecs = torch.linalg.eigh(D)
        eigvals, _=torch.sort(eigvals, dim=-1, descending=True)
        D= eigvecs@ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
    
    elif(matter=='gm'):
        eigvals, eigvecs = torch.linalg.eigh(D)
        #eigvals[...,0]=eigvals[...,1]=eigvals[...,2]
        D= eigvecs@ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
    else:
        eigvals, eigvecs = torch.linalg.eigh(D)
        eigvals[...,0]=eigvals[...,1]=eigvals[...,2]
        D= eigvecs@ torch.diag_embed(eigvals) @ eigvecs.transpose(-2, -1)
 
    return D    


def gen_synth(batchsize=None,  bvecs=None, bvals=None, device=None, 
                w=None, h=None, snr=None, N=None, mask=None):
     
    epsilon=1e-10      
      
    if mask==None:        
         mask=generate_mask2(batchsize=batchsize, w=w, h=h, device=device)
         mask=mask.permute(1, 2, 0) 
    else:
         mask=mask    

    #wmranges=torch.tensor([0.0173, 0.054], device=device)
    #gmranges=torch.tensor([0.0173, 0.054], device=device)
    #csfranges=torch.tensor([0.0173, 0.054], device=device)
    
    #lambda_wmrange=torch.tensor([0.0* 10**(-3), 4.0* 10**(-3)], device=device)
    #lambda_gmrange=torch.tensor([0.0* 10**(-3), 4.0* 10**(-3)], device=device)
    #lambda_csfrange=torch.tensor([0.0* 10**(-3), 4.0* 10**(-3)], device=device)
    
    #wm_ranges  = [(1.3*10**(-3), 1.7*10**(-3)), (0.3*10**(-3), 0.6*10**(-3)), (0.3*10**(-3), 0.6*10**(-3))]
    #gm_ranges  = [(0.8*10**(-3), 1.1*10**(-3)), (0.6*10**(-3), 0.9*10**(-3)), (0.6*10**(-3), 0.9*10**(-3))]
    #csf_ranges = [(2.5*10**(-3), 3.5*10**(-3)), (2.5*10**(-3), 3.5*10**(-3)), (2.5*10**(-3), 3.5*10**(-3))]
       
    
    unitrange=torch.tensor([0., 1.], device=device)
    S0_wmrange=torch.tensor([0.05, 0.25], device=device)
    S0_gmrange=torch.tensor([0.2, 0.5], device=device)
    S0_csfrange=torch.tensor([0.5, 1.0], device=device)
    
    #S0_wmrange=torch.tensor([0., 1.], device=device)
    #S0_gmrange=torch.tensor([0., 1.], device=device)
    #S0_csfrange=torch.tensor([0., 1.], device=device)
    
    
    while(True):
        
        
        L_wm_components=torch.zeros(batchsize, w, h, 6, device=device)
        L_gm_components=torch.zeros(batchsize, w, h, 6, device=device)
        L_csf_components=torch.zeros(batchsize, w, h, 6, device=device)
        
               
        for i in range(6):
            L_wm_components[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=unitrange , device=device))
            L_gm_components[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=unitrange , device=device))
            L_csf_components[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=unitrange , device=device))
           
        D_wm=L_to_D(L_wm_components, matter='wm')
        D_gm=L_to_D(L_gm_components, matter='gm')
        D_csf=L_to_D(L_csf_components, matter='csf')
        
         
        R=generate_random_rots_uniform(w=w, h=h, batch_size=batchsize, device=device)
        #R=random_rotation_matrices2(w=w, h=h, batchsize=batchsize, device=device)
        
        
        D_wm=D_wm.permute(1,2,0,3,4)
        D_gm=D_gm.permute(1,2,0,3,4)
        D_csf=D_csf.permute(1,2,0,3,4)
        
        
        """ 
        plt.imshow(mask[:,:,10].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D_gm[:, :, 10, 0,0].cpu())
        plt.colorbar()
        plt.show()
        
        exit()
         """
        
        D = torch.zeros(( w, h, batchsize,3,3), device=device)
        
        D[mask==1]=D_wm[mask==1]
        D[mask==2]=D_gm[mask==2]
        D[mask==3]=D_csf[mask==3]
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
    
        D=R.transpose(-1,-2)@D@R
        
        S0wm_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_wmrange , device=device))
        S0gm_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_gmrange , device=device))
        S0csf_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_csfrange , device=device))
             
        S0 = torch.zeros(( w, h, batchsize), device=device)
        
        S0[mask==1]=S0wm_mask.permute(1,2,0)[mask==1]
        S0[mask==2]=S0gm_mask.permute(1,2,0)[mask==2]
        S0[mask==3]=S0csf_mask.permute(1,2,0)[mask==3]
        

        bvecs=torch.nan_to_num(bvecs, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        S=predict_dti_signal(D_est=D, bvals=bvals, bvecs=bvecs, S0=S0[...,None])

        if snr is not None:
            #snr= random.uniform(20, 60)
            S=add_rician_noise(S, snr=snr)
        
        #S=S/S.max() 
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
        
        yield D, S, bvecs, mask  

def sample_eig_triplets(eig_dist, num_samples=1):
    """
    Randomly picks 'num_samples' triplets of eigenvalues from the eigenvalue distribution.
    eig_dist: tensor of shape [N, 3]
    num_samples: int, number of triplets to sample
    Returns: tensor of shape [num_samples, 3]
    """
    N = eig_dist.shape[0]
    idx = torch.randint(low=0, high=N, size=(num_samples,))
    return eig_dist[idx]

from scipy.ndimage import gaussian_filter

def generate_dirs(batchsize=1, w=64, h=64, sigma_range=(2.0, 8.0), visualize=True, batch_to_show=0, step=4, device='cuda'):
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
    dirs = torch.randn(batchsize, w, h, 3, device=device)

    # For smoothing, move to CPU (scipy doesn't support GPU tensors)
    dirs_cpu = dirs.cpu().numpy()

    for b in range(batchsize):
        sigma_b = np.random.uniform(sigma_range[0], sigma_range[1])
        for c in range(3):
            dirs_cpu[b, :, :, c] = gaussian_filter(dirs_cpu[b, :, :, c], sigma=sigma_b)

    # Convert back to torch tensor on device
    dirs = torch.tensor(dirs_cpu, device=device, dtype=torch.float32)

    # Normalize to unit vectors
    norm = torch.norm(dirs, dim=-1, keepdim=True).clamp(min=1e-8)
    dirs = dirs / norm

    if visualize:
        # Move data for visualization to CPU and numpy
        dirs_b = dirs[batch_to_show].cpu().numpy()
        X, Y = np.meshgrid(np.arange(0, h, step), np.arange(0, w, step))
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


import torch.nn.functional as F
import torch

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


""" def gen_synth2(batchsize=None,  bvecs=None, bvals=None, device=None, 
                w=None, h=None, snr=None, N=None, mask=None):
     
    epsilon=1e-10      
      
    if mask==None:        
         mask=generate_mask2(batchsize=batchsize, w=w, h=h, device=device)
         mask=mask.permute(1, 2, 0) 
    else:
         mask=mask    

    eigvals_all=torch.load("invivo_eig_dist")
    wm_eigvals=torch.load("wm_eigs.pt")
    gm_eigvals=torch.load("gm_eigs.pt")
    csf_eigvals=torch.load("csf_eigs.pt")
     
    unitrange=torch.tensor([0., 1.], device=device)
    
    #S0_wmrange=torch.tensor([0.05, 0.25], device=device)
    #S0_gmrange=torch.tensor([0.2, 0.5], device=device)
    #S0_csfrange=torch.tensor([0.5, 1.0], device=device)
      
    S0_wmrange=torch.tensor([0.0, 1.], device=device)
    S0_gmrange=torch.tensor([0.0, 1.], device=device)
    S0_csfrange=torch.tensor([0.0, 1.], device=device)
    

    while(True):
        
        
        lambdas=torch.zeros(batchsize, w, h, 3, device=device)
        
        
        wm_eigvals=eigvals_all
        values, wm_dists = prepare_eigenvalue_distributions(
                           wm_eigvals, batch_size=batchsize, w=w, h=h, num_bins=1000  )
        lambdas_wm = sample_eigenvalues( wm_dists, batch_size=batchsize, w=w, h=h, keep_fraction=0.8 )
        lambdas_wm=lambdas_wm.permute(1, 2, 0, 3).to(device)
        lambdas_wm, _ = torch.sort(lambdas_wm, dim=-1, descending=True)
        #lambdas_wm=map_to_range(lambdas_wm, 0.3e-3, 1.3e-3)
        
        dirs = generate_dirs(batchsize, w, h, sigma_range=(5, 40), visualize=False) # 5,7
        R = build_rotation_from_dirs(dirs)  # [B,W,H,3,3]
        R=R.permute(1,2,0,3,4)
        
        L_wm = torch.diag_embed(lambdas_wm).to(device)
        
        L_wm=R.transpose(-1,-2)@L_wm@R
        
        # For Gray Matter (GM)
        gm_eigvals=eigvals_all
        values, gm_dists = prepare_eigenvalue_distributions(
                           gm_eigvals, batch_size=batchsize, w=w, h=h, num_bins=1000  )
        lambdas_gm = sample_eigenvalues( gm_dists, batch_size=batchsize, w=w, h=h, keep_fraction=0.9 )
        lambdas_gm=lambdas_gm.permute(1, 2, 0, 3) # w, h, b, 3
        lambdas_gm[..., 0]=lambdas_gm[..., 1]=lambdas_gm[..., 2]
        #lambdas_gm=map_to_range(lambdas_gm, 0.5e-3, 2.0e-3)
        
        L_gm = torch.diag_embed(lambdas_gm).to(device)
        R1=generate_random_rots_uniform(w=w, h=h, batch_size=batchsize, device=device)
        
        dirs_gm = generate_dirs(batchsize, w, h, sigma_range=(5, 40), visualize=False) # 25,25
        R1 = build_rotation_from_dirs(dirs_gm)  # [B,W,H,3,3]
        R1=R1.permute(1,2,0,3,4)
        L_gm=R1.transpose(-1,-2)@L_gm@R1
        
        # For Cerebrospinal Fluid (CSF)
        csf_eigvals=eigvals_all
        values, csf_dists = prepare_eigenvalue_distributions(
                            csf_eigvals, batch_size=batchsize, w=w, h=h, num_bins=400  )
        lambdas_csf = sample_eigenvalues( csf_dists, batch_size=batchsize, w=w, h=h, keep_fraction=1.0 )
        lambdas_csf=lambdas_csf.permute(1, 2, 0, 3) # w, h, b, 3
        #lambdas_csf=map_to_range(lambdas_csf, 1.0e-3, 3.0e-3)
        lambdas_csf[..., 0]=lambdas_csf[..., 1]=lambdas_csf[..., 2]
        L_csf = torch.diag_embed(lambdas_csf).to(device)
        
        dirs_csf = generate_dirs(batchsize, w, h, sigma_range=(5, 40), visualize=False) # 5,7
        R2 = build_rotation_from_dirs(dirs_csf)  # [B,W,H,3,3]
        R2=R2.permute(1,2,0,3,4)
        L_csf=R2.transpose(-1,-2)@L_csf@R2
        
        
        #R2=generate_random_rots_uniform(w=w, h=h, batch_size=batchsize, device=device)
        #L_csf=R2.transpose(-1,-2)@L_csf@R2
        
        
        
        #print(R1.shape, R.shape)
        D = torch.zeros(( w, h, batchsize, 3, 3), device=device)
        
        
        D[mask==1]=L_wm[mask==1]
        D[mask==2]=L_gm[mask==2]
        D[mask==3]=L_csf[mask==3]
        
        
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
        D=D.float()
        #D=R.transpose(-1,-2)@D@R
        #D=R2.transpose(-1,-2)@D@R2
        s0=torch.rand(w, h, batchsize, device=device)
        S0wm_mask =torch.rand(w, h, batchsize, device=device) * (S0_wmrange.max() - S0_wmrange.min()) + S0_wmrange.min()
        S0gm_mask =torch.rand(w, h, batchsize, device=device)* (S0_gmrange.max() - S0_gmrange.min()) + S0_gmrange.min()
        S0csf_mask =torch.rand(w, h, batchsize, device=device) * (S0_csfrange.max() - S0_csfrange.min()) + S0_csfrange.min()
 
        S0 = torch.zeros(( w, h, batchsize), device=device)

        S0[mask==1]=S0wm_mask[mask==1]
        S0[mask==2]=S0gm_mask[mask==2]
        S0[mask==3]=S0csf_mask[mask==3]
        

        bvecs=torch.nan_to_num(bvecs, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        S=predict_dti_signal(D_est=D, bvals=bvals, bvecs=bvecs, S0=S0[...,None])
        #print(S.shape, S0.shape)
        #S=predict_dti_signal(D_est=D, bvals=bvals, bvecs=bvecs, S0=1)

        if snr is not None:
            S=add_rician_noise(S, snr=snr)
        
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
           
        yield D, S, bvecs, mask  

"""

def gen_synth2(batchsize=None, bvecs=None, bvals=None, device=None,
               w=None, h=None, snr=None, N=None, mask=None):
    epsilon = 1e-10

    if mask is None:
        mask = generate_mask2(batchsize=batchsize, w=w, h=h, device=device)
        mask = mask.permute(1, 2, 0)

    eigvals_all = torch.load("invivo_eig_dist")
    lambdas = eigvals_all
    lambdas, _ = torch.sort(lambdas, dim=-1, descending=True)
    
    wm_eigvals=torch.load("wm_eigs.pt")
    wm_eigvals, _ = torch.sort(wm_eigvals, dim=-1, descending=True)
    
    gm_eigvals=torch.load("gm_eigs.pt")
    gm_eigvals, _ = torch.sort(gm_eigvals, dim=-1, descending=True)
    
    csf_eigvals=torch.load("csf_eigs.pt")
    csf_eigvals, _ = torch.sort(csf_eigvals, dim=-1, descending=True)
    
            
    S0_wmrange = torch.tensor([0.0, 1.], device=device)
    S0_gmrange = torch.tensor([0.0, 1.], device=device)
    S0_csfrange = torch.tensor([0.0, 1.], device=device)

    num_signals = 1  # average over these many signals

    while True:
        # reset signal accumulator each new yield
        S_accum = torch.zeros((w, h, batchsize, bvals.shape[0]), device=device)
        D_accum = torch.zeros((w, h, batchsize, 3, 3), device=device)
        

        for i in range(num_signals):
            # ===== WM =====
            _, wm_dists = prepare_eigenvalue_distributions(
                          wm_eigvals, batch_size=batchsize, w=w, h=h, num_bins=50000)
            lambdas_wm = sample_eigenvalues(wm_dists, batch_size=batchsize, w=w, h=h, keep_fraction=0.8)
            #lambdas_wm = sample_eigenvalues2(wm_dists, batch_size=batchsize, w=w, h=h)
            lambdas_wm = lambdas_wm.permute(1, 2, 0, 3).to(device)
            lambdas_wm, _ = torch.sort(lambdas_wm, dim=-1, descending=True)
            lambdas_wm[..., 0] = lambdas_wm[..., 0] + ( 0.1)*lambdas_wm[..., 1]
            lambdas_wm[..., 1]=lambdas_wm[..., 1]*(torch.rand(1).to(device) * (0.7 - 0.5) + 0.5)
            lambdas_wm[..., 2]=lambdas_wm[..., 2]*(torch.rand(1).to(device) * (0.7 - 0.5) + 0.5)
            L_wm = torch.diag_embed(lambdas_wm).to(device)
            
            R_wm = build_rotation_from_dirs(
                                generate_dirs(batchsize, w, h, sigma_range=(5, 10), visualize=False)).permute(1, 2, 0, 3, 4)
            L_wm = R_wm.transpose(-1, -2) @ L_wm @ R_wm

            # ===== GM =====
            _, gm_dists = prepare_eigenvalue_distributions(
                            gm_eigvals, batch_size=batchsize, w=w, h=h, num_bins=50000)
            lambdas_gm = sample_eigenvalues(gm_dists, batch_size=batchsize, w=w, h=h, keep_fraction=0.9)
            lambdas_gm = lambdas_gm.permute(1, 2, 0, 3).to(device)
            lambdas_gm[..., 0] = lambdas_gm[..., 1] = lambdas_gm[..., 2]
            
            lambdas_gm[..., 0] = lambdas_gm[..., 0]*(torch.rand(1).to(device) * (0.99 - 0.8) + 0.8)
            lambdas_gm[..., 1] = lambdas_gm[..., 1]*(torch.rand(1).to(device) * (0.99 - 0.8) + 0.8)
            lambdas_gm[..., 2] = lambdas_gm[..., 2]*(torch.rand(1).to(device) * (0.99 - 0.8) + 0.8)
            
            lambdas_gm, _ = torch.sort(lambdas_gm, dim=-1, descending=True)
            
            L_gm = torch.diag_embed(lambdas_gm).to(device)
            R_gm = build_rotation_from_dirs(
                            generate_dirs(batchsize, w, h, sigma_range=(50, 52), visualize=False)).permute(1, 2, 0, 3, 4)
            L_gm = R_gm.transpose(-1, -2) @ L_gm @ R_gm

            # ===== CSF =====
            _, csf_dists = prepare_eigenvalue_distributions(
                                 csf_eigvals, batch_size=batchsize, w=w, h=h, num_bins=50000)
            lambdas_csf = sample_eigenvalues(csf_dists, batch_size=batchsize, w=w, h=h, keep_fraction=1.0)
            lambdas_csf = lambdas_csf.permute(1, 2, 0, 3).to(device)
            lambdas_csf[..., 0] = lambdas_csf[..., 1] = lambdas_csf[..., 2]
            
            #lambdas_csf[..., 0] = lambdas_csf[..., 0]*(torch.rand(1).to(device) * (0.99 - 0.90) + 0.90)
            #lambdas_csf[..., 1] = lambdas_csf[..., 1]*(torch.rand(1).to(device) * (0.99 - 0.90) + 0.90)
            #lambdas_csf[..., 2] = lambdas_csf[..., 2]*(torch.rand(1).to(device) * (0.99 - 0.90) + 0.90)
            
            lambdas_csf, _ = torch.sort(lambdas_csf, dim=-1, descending=True)
            
            L_csf = torch.diag_embed(lambdas_csf).to(device)
            R_csf = build_rotation_from_dirs(
                                    generate_dirs(batchsize, w, h, sigma_range=(50, 52), visualize=False)).permute(1, 2, 0, 3, 4)
            L_csf = R_csf.transpose(-1, -2) @ L_csf @ R_csf

            # ===== D_i for this iteration =====
            D_i = torch.zeros((w, h, batchsize, 3, 3), device=device)
            D_i[mask == 1] = L_wm[mask == 1]
            D_i[mask == 2] = L_gm[mask == 2]
            D_i[mask == 3] = L_csf[mask == 3]
            D_i = torch.nan_to_num(D_i, nan=epsilon, posinf=epsilon, neginf=epsilon).float()
            #R=generate_random_rots_uniform(w=w, h=h, batch_size=batchsize, device=device)
            #D_i = R.transpose(-1, -2) @ D_i @ R


            # ===== S0 for this iteration =====
            S0wm_mask = torch.rand(w, h, batchsize, device=device) * (S0_wmrange.max() - S0_wmrange.min()) + S0_wmrange.min()
            S0gm_mask = torch.rand(w, h, batchsize, device=device) * (S0_gmrange.max() - S0_gmrange.min()) + S0_gmrange.min()
            S0csf_mask = torch.rand(w, h, batchsize, device=device) * (S0_csfrange.max() - S0_csfrange.min()) + S0_csfrange.min()
            S0_i = torch.zeros((w, h, batchsize), device=device)
            S0_i[mask == 1] = S0wm_mask[mask == 1]
            S0_i[mask == 2] = S0gm_mask[mask == 2]
            S0_i[mask == 3] = S0csf_mask[mask == 3]

            # ===== compute S_i and accumulate =====
            S_i = predict_dti_signal(D_est=D_i, bvals=bvals, bvecs=bvecs, S0=S0_i[..., None])
            S_accum += S_i
            D_accum += D_i

            #D_last = D_i  # save last one (or you can store all if needed)

        # average after loop
        S = S_accum / num_signals
        D=fit_dti_tensor(S, mean_b0=S[...,0], bvecs=bvecs )
        #D = D_accum / num_signals

        if snr is not None:
            S = add_rician_noise(S, snr=snr)

        S = torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon

        yield D, S, bvecs, mask





def gen_synth3(batchsize=None,  bvecs=None, bvals=None, device=None, 
                w=None, h=None, snr=None, N=None, mask=None):
     
    epsilon=1e-10      
      
    if mask==None:        
         mask=generate_mask2(batchsize=batchsize, w=w, h=h, device=device)
         mask=mask.permute(1, 2, 0) 
    else:
         mask=mask    

    eigvals_all=torch.load("invivo_eig_dist")
    wm_eigvals=torch.load("wm_eigs.pt")
    gm_eigvals=torch.load("gm_eigs.pt")
    csf_eigvals=torch.load("csf_eigs.pt")
     
    unitrange=torch.tensor([0., 1.], device=device)
    
    S0_wmrange=torch.tensor([0.05, 0.25], device=device)
    S0_gmrange=torch.tensor([0.2, 0.5], device=device)
    S0_csfrange=torch.tensor([0.5, 1.0], device=device)
     
    #S0_wmrange=torch.tensor([0.0, 1.], device=device)
    #S0_gmrange=torch.tensor([0.0, 1.], device=device)
    #S0_csfrange=torch.tensor([0.0, 1.], device=device)
    
    l123max=0.041; l123min=0.011
    l456min=-0.15; l456max=0.15
    params=torch.zeros(batchsize, w, h, 6, device=device)
    

    while(True):
        
        image=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=unitrange , device=device))
                 
        for i in range(6):
            params[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=unitrange , device=device))
        
        # Cholesky construction
        l11 =l123min + (params[..., 0])*(l123max-l123min)
        l21 =l123min + (params[..., 1])*(l123max-l123min)
        l31 =l123min + (params[..., 2])*(l123max-l123min)
        l22 =l456min + (params[..., 3])*(l456max-l456min)
        l32 =l456min + (params[..., 4])*(l456max-l456min)
        l33 =l456min + (params[..., 5])*(l456max-l456min)
           

        # First 3 are diagonal (positive)
        l11 = torch.abs(l11) 
        l22 = torch.abs(l22) 
        l33 = torch.abs(l33)

        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        

        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33
        
        LL = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]
        
        LL=LL.permute(1, 2, 0, 3, 4)    
        
        R1=generate_random_rots_uniform(w=w, h=h, batch_size=batchsize, device=device)
        R2=generate_random_rots_uniform(w=w, h=h, batch_size=batchsize, device=device)
        
        #R=random_rotation_matrices2(w=w, h=h, batchsize=batchsize, device=device)
        
        D = torch.zeros(( w, h, batchsize, 3, 3), device=device)
        
        
        D[mask==1]=LL[mask==1]
        D[mask==2]=LL[mask==2]
        D[mask==3]=LL[mask==3]
        
        
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
        D=D.float()
        #D=R1.transpose(-1,-2)@D@R1
        #D=R2.transpose(-1,-2)@D@R2
       
        S0_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=unitrange , device=device))
             
        S0 = torch.zeros(( w, h, batchsize), device=device)
        
        S0[mask==1]=S0_mask.permute(1,2,0)[mask==1]
        S0[mask==2]=S0_mask.permute(1,2,0)[mask==2]
        S0[mask==3]=S0_mask.permute(1,2,0)[mask==3]
        

        bvecs=torch.nan_to_num(bvecs, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        S=predict_dti_signal(D_est=D, bvals=bvals, bvecs=bvecs, S0=S0[...,None])
        #print(S.shape, S0.shape)
        #S=predict_dti_signal(D_est=D, bvals=bvals, bvecs=bvecs, S0=1)

        if snr is not None:
            S=add_rician_noise(S, snr=snr)
        
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
        
        yield D, S, bvecs, mask  



class DtiSynth(nn.Module):

    def __init__(self, in_channels=91, bvals=None):
        super().__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv3d(in_channels, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            #nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=1, padding=0),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 6, kernel_size=1)
        )
        #self.register_buffer('bvals', torch.as_tensor(bvals, dtype=torch.float32))  # [N]
        self.bvals=bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, bvecs):
        """
        x: [1, 91, W, H, B]
        bvecs: [N, 3] (passed at runtime)
        Returns:
            pred_tensor: [W, H, B, 3, 3]
            pred_signal: [W, H, B, N]
        """
        # Compute mean_b0 from the first channel (assumed b=0)
        x=torch.abs(x)
        S0 = x[0,0,...]  # [W, H, B]
        #print(x.shape)
        # Predict tensor using network (SPD)
        params = self.encoder(x)  # [1, 6, W, H, B]
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]
         
        """ l123max=0.041; l123min=0.011
        l456min=-0.15; l456max=0.15

        # Cholesky construction
        l11 =l123min + torch.sigmoid(params[..., 0])*(l123max-l123min)
        l21 =l123min + torch.sigmoid(params[..., 1])*(l123max-l123min)
        l31 =l123min + torch.sigmoid(params[..., 2])*(l123max-l123min)
        l22 =l456min + torch.sigmoid(params[..., 3])*(l456max-l456min)
        l32 =l456min + torch.sigmoid(params[..., 4])*(l456max-l456min)
        l33 =l456min + torch.sigmoid(params[..., 5])*(l456max-l456min)
         """   

        # First 3 are diagonal (positive)
        l11 = torch.abs(params[..., 0]) 
        l21 = params[..., 1] 
        l31 = params[..., 2] 
        l22 = torch.abs(params[..., 3]) 
        l32 = params[..., 4]
        l33 = torch.abs(params[..., 5])

        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        

        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33
        D = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]
        
        return D

""" class DtiSynth(nn.Module):

    def __init__(self, in_channels=91, bvals=None):
        super().__init__()
        self.encoder = nn.Sequential(
            
            nn.Conv3d(in_channels, 64, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            #nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.ReLU(),
            #nn.BatchNorm3d(256),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            #nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 6, kernel_size=1)
        )
        #self.register_buffer('bvals', torch.as_tensor(bvals, dtype=torch.float32))  # [N]
        self.bvals=bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, bvecs):
        x: [1, 91, W, H, B]
        bvecs: [N, 3] (passed at runtime)
        Returns:
            pred_tensor: [W, H, B, 3, 3]
            pred_signal: [W, H, B, N]
        # Compute mean_b0 from the first channel (assumed b=0)
        
        x=torch.abs(x)
        S0 = x[0,0,...]  # [W, H, B]
        #print(x.shape)
        # Predict tensor using network (SPD)
        params = self.encoder(x)  # [1, 6, W, H, B]
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]
      
        # First 3 are diagonal (positive)
        l11 = torch.abs(params[..., 0]) 
        l21 = params[..., 1] 
        l31 = params[..., 2] 
        l22 = torch.abs(params[..., 3]) 
        l32 = params[..., 4]
        l33 = torch.abs(params[..., 5])

        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        

        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33
        D = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]
        
        return D

 """
class MixedConv3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MixedConv3DBlock, self).__init__()
        split = out_channels // 3
        self.conv1 = nn.Conv3d(in_channels, split, kernel_size=1, padding=0)
        self.conv3 = nn.Conv3d(in_channels, split, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(in_channels, out_channels - 2 * split, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv3(x)
        x3 = self.conv5(x)
        out = torch.cat([x1, x2, x3], dim=1)
        return self.relu(self.bn(out))


class DtiSynth2(nn.Module):

    def __init__(self, in_channels=91, bvals=None):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=1),
            nn.ReLU(),
            MixedConv3DBlock(64, 128),
            MixedConv3DBlock(128, 256),
            MixedConv3DBlock(256, 256),
            nn.Conv3d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 32, kernel_size=1),
            nn.Conv3d(32, 16, kernel_size=1),
            nn.Conv3d(16, 6, kernel_size=1)
        )
        self.bvals = bvals
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def predict_dti_signal(self, D_est, bvecs, S0=1.0):
        """
        D_est: [W, H, B, 3, 3]
        bvecs: [N, 3] (passed at runtime)
        S0: [W, H, B] or scalar
        Returns: [W, H, B, N]
        """
        W, H, B, _, _ = D_est.shape
        N = bvecs.shape[0]
        
        D_flat = D_est.reshape(-1, 3, 3)  # [V, 3, 3]
        gdg = torch.einsum('nj,vjk,nk->vn', bvecs, D_flat, bvecs)  # [V, N]
        exponent = -self.bvals * gdg  # [V, N]
        signal = torch.exp(exponent)  # [V, N]
        # Handle S0
        if not torch.is_tensor(S0):
            S0 = torch.tensor(S0, dtype=signal.dtype, device=signal.device)
        S0 = S0.reshape(-1, 1)
        signal = S0 * signal
        return signal.reshape(W, H, B, N)

    def forward(self, x, bvecs):
        """
        x: [1, 91, W, H, B]
        bvecs: [N, 3] (passed at runtime)
        Returns:
            pred_tensor: [W, H, B, 3, 3]
            pred_signal: [W, H, B, N]
        """
        # Compute mean_b0 from the first channel (assumed b=0)
        x=torch.abs(x)
        mean_b0 = x[0,0,...]  # [W, H, B]
        #print(x.shape)
        # Predict tensor using network (SPD)
        params = self.encoder(x)  # [1, 6, W, H, B]
        params = params.squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]
        """ l123max=0.041; l123min=0.011
        l456min=-0.15; l456max=0.15

        # Cholesky construction
        l11 =l123min + torch.sigmoid(params[..., 0])*(l123max-l123min)
        l21 =l123min + torch.sigmoid(params[..., 1])*(l123max-l123min)
        l31 =l123min + torch.sigmoid(params[..., 2])*(l123max-l123min)
        l22 =l456min + torch.sigmoid(params[..., 3])*(l456max-l456min)
        l32 =l456min + torch.sigmoid(params[..., 4])*(l456max-l456min)
        l33 =l456min + torch.sigmoid(params[..., 5])*(l456max-l456min)
         """  
        l11 =torch.abs(params[..., 0])
        l21 =params[..., 1]
        l31 =params[..., 2]
        l22 =torch.abs(params[..., 3])
        l32 =params[..., 4]
        l33 =torch.abs(params[..., 5])
      
        L = torch.zeros(*params.shape[:-1], 3, 3, device=params.device)
        L[..., 0, 0] = l11
        L[..., 1, 0] = l21
        L[..., 2, 0] = l31
        L[..., 1, 1] = l22
        L[..., 2, 1] = l32
        L[..., 2, 2] = l33
        pred_tensor = L @ L.transpose(-1, -2)  # [W, H, B, 3, 3]

        # Predict signal from network tensor using current bvecs and mean_b0
        pred_signal = self.predict_dti_signal(pred_tensor, bvecs, S0=mean_b0)  # [W, H, B, N]
        #x=pred_signal.permute(3,0,1,2)[None,...]

        return pred_tensor


import torch

def prepare_eigenvalue_distributions(eigvals_all, batch_size, w, h, num_bins=200, eps=1e-8):
    """
    Compute normalized discrete distributions for λ1, λ2, λ3 based on eigvals_all,
    and prepare them for sampling.

    Args:
        eigvals_all (Tensor): [N, 3] eigenvalue tensor, sorted per row.
        batch_size (int): Number of batches.
        w (int): Width.
        h (int): Height.
        num_bins (int): Number of histogram bins.
        eps (float): Small value to avoid empty bins during normalization.

    Returns:
        values (Tensor): Support values (bin centers) [num_bins]
        eigen_dists (tuple): [(values, probs1), (values, probs2), (values, probs3)]
    """
    device = eigvals_all.device

    eigvals_min = eigvals_all.min().item()
    eigvals_max = eigvals_all.max().item()
    values = torch.linspace(eigvals_min, eigvals_max, steps=num_bins, device=device)

    # Extract each eigenvalue set
    λ1 = eigvals_all[:, 0]
    λ2 = eigvals_all[:, 1]
    λ3 = eigvals_all[:, 2]

    # Compute histograms
    hist1 = torch.histc(λ1, bins=num_bins, min=eigvals_min, max=eigvals_max)
    hist2 = torch.histc(λ2, bins=num_bins, min=eigvals_min, max=eigvals_max)
    hist3 = torch.histc(λ3, bins=num_bins, min=eigvals_min, max=eigvals_max)

    # Add eps to avoid zero probabilities
    probs1 = (hist1 + eps) / (hist1.sum() + eps * num_bins)
    probs2 = (hist2 + eps) / (hist2.sum() + eps * num_bins)
    probs3 = (hist3 + eps) / (hist3.sum() + eps * num_bins)

    eigen_dists = [(values, probs1), (values, probs2), (values, probs3)]
    return values, eigen_dists


import torch

def sample_eigenvalues(eigen_dists, batch_size, w, h, keep_fraction=0.5):
    """
    Sample eigenvalue triplets but only from the middle part of the distributions.

    Args:
        eigen_dists (tuple): Tuple of 3 (values, probs) pairs for each eigenvalue λ1, λ2, λ3.
        batch_size (int): Number of batches.
        w (int): Width of output.
        h (int): Height of output.
        keep_fraction (float): Fraction of the central bins to keep (0<keep_fraction<=1).

    Returns:
        Tensor: Sampled eigenvalues of shape [batch_size, w, h, 3]
    """
    total_samples = batch_size * w * h
    device = eigen_dists[0][0].device  # assume all on same device

    sampled = []

    for values, probs in eigen_dists:
        num_bins = values.shape[0]
        # Determine central region indices
        keep_bins = int(num_bins * keep_fraction)
        start = (num_bins - keep_bins) // 2
        end = start + keep_bins

        # Slice to middle region
        values_mid = values[start:end]
        probs_mid = probs[start:end]
        # Normalize probs again so they sum to 1
        probs_mid = probs_mid / probs_mid.sum()

        # Sample indices within this smaller region
        indices = torch.multinomial(probs_mid, total_samples, replacement=True)
        samples = values_mid[indices]
        sampled.append(samples)

    # Stack to [total_samples, 3], then reshape
    samples_stack = torch.stack(sampled, dim=1)
    return samples_stack.view(batch_size, w, h, 3)


def sample_eigenvalues2(wm_dist, batch_size, w, h, r_range=(0.3, 0.4)):
    """
    Smart sampling of eigenvalues with FA > 0.4:
    - λ1 sampled from wm_dist[0]
    - λ2, λ3 derived as ratios of λ1 within r_range to enforce anisotropy

    Args:
        wm_dist (tuple): (values, probs) pairs for λ1, λ2, λ3 distributions.
                         We will only use wm_dist[0] for λ1 sampling.
        batch_size (int): Number of batches.
        w (int): Width.
        h (int): Height.
        r_range (tuple): Min and max ratio for λ2, λ3 relative to λ1.

    Returns:
        torch.Tensor: [batch_size, w, h, 3] eigenvalues satisfying FA > 0.4.
    """
    total_samples = batch_size * w * h
    device = wm_dist[0][0].device

    # --- Sample λ1 from its distribution
    values1, probs1 = wm_dist[0]
    indices1 = torch.multinomial(probs1, total_samples, replacement=True)
    l1 = values1[indices1]

    # --- Sample ratios r for λ2, λ3
    r_min, r_max = r_range
    # uniform random in [r_min, r_max]
    r = torch.rand(total_samples, device=device) * (r_max - r_min) + r_min
    l2 = l1 * r
    l3 = l1 * r

    # Stack
    eigs = torch.stack([l1, l2, l3], dim=-1)
    return eigs.view(batch_size, w, h, 3)

