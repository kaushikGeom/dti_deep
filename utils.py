import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
import torch.linalg as linalg
import numpy as np
import torch
import random
import math
import scipy.linalg
import dipy
import os
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms as T
from dipy.data import get_sphere 
import seaborn as sns
import nibabel as nib
import random
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F
import warnings
# Suppress specific warning by setting ignore for all warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate, approximate err =")

import torch
import matplotlib.pyplot as plt
import numpy as np

import torch

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

def fit_dti_tensor(dwi: torch.Tensor, mean_b0: torch.Tensor, bvals: torch.Tensor, bvecs: torch.Tensor) -> torch.Tensor:
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
    bvals = bvals.flatten()         # [G]

    # Normalize log signal by negative bvals to get ADCs
    # Avoid division by zero by adding eps
    log_signal = log_signal / (-bvals + eps)  # [N_voxels, G]

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


def map_to_range(y, min_val, max_val):
        y=scale_to_01(y)
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






def gen_dti_signals(D=None, gradients=None, b_values=None, S0=None, device=None):
    
        epsilon=1e-10; 
        #S0 = S0  # Shape: [128, 128, 32, 1]
        #S0=1
        if S0 is not None:
            S0=S0[...,None]
        else:
            S0=1    
        #gradients = gradients / torch.norm(gradients, dim=1, keepdim=True)
        #exponent = -b_values * torch.einsum('ij,...jk,ik->...i', gradients, D, gradients)  # Shape: [128, 128, 32, 65]
        print(S0.shape,  gradients.shape )
        for i in range(S0.shape[0]):
         exponent1=gradients[i,...] @D.reshape(-1,3,3)@gradients[i,...].T
         print(exponent1.shape)
         exit()
         exponent=-b_values*exponent1.reshape(D.shape[0], D.shape[1], D.shape[2], gradients.shape[0],gradients.shape[0]).diagonal(dim1=-2, dim2=-1) 
         S=S0*torch.exp(exponent)
        print(S0.shape, S.shape)
        #S=S/S0[...,None]
              
        return S
 



import torch

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


import os
import nibabel as nib
import torch
import numpy as np

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

def gen_batch(batchsize=None, b_value=None, device=None, w=None, 
              h=None, N=None, mask=None, snr=None, dir=None):
    
    data, mask, bvals, bvecs = load_subject_data(dir)

    



import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os
import math
import random
from torch.utils.data import Dataset
import nibabel as nib


from pathlib import Path

import torch
from torch.utils.data import Dataset
from pathlib import Path
import random

import torch
from pathlib import Path
import random
import glob

class DWIDataset:
    def __init__(self, data_dir, num_subjects=32, z_slices=20, g_dirs=15):
        self.data_dir = Path(data_dir)
        self.subjects = sorted(list(self.data_dir.glob("*_data.pt")))[:num_subjects]
        self.z_slices = z_slices
        self.g_dirs = g_dirs

    def __call__(self):
        # Randomly select g_dirs different subjects
        subject_paths = random.choices(self.subjects, k=self.g_dirs)

        # Use the first subject to determine shape
        base_data = torch.load(subject_paths[0])
        X, Y, Z, G = base_data.shape

        # Random z starting index
        start_z = random.randint(0, Z - self.z_slices)
        z_indices = list(range(start_z, start_z + self.z_slices))

        data_volumes = []
        mask_volumes = []
        bvecs = []

        for subj_data_path in subject_paths:
            subj_id = subj_data_path.stem.replace("_data", "")

            data = torch.load(subj_data_path)
            mask = torch.load(self.data_dir / f"{subj_id}_mask.pt")
            bvec = torch.load(self.data_dir / f"{subj_id}_bvecs.pt")

            g_idx = random.randint(0, G - 1)

            vol = data[:, :, z_indices, g_idx]         # shape: [X, Y, z_slices]
            msk = mask[:, :, z_indices]
            bvec=bvec.T                # shape: [X, Y, z_slices]
            bvec_g = bvec[g_idx]                       # shape: [3]

            data_volumes.append(vol.unsqueeze(-1))     # -> [X, Y, z_slices, 1]
            mask_volumes.append(msk.unsqueeze(-1))     # -> [X, Y, z_slices, 1]
            bvecs.append(bvec_g.unsqueeze(0))          # -> [1, 3]

        # Stack along the g_dir dimension
        data_block = torch.cat(data_volumes, dim=-1)    # [X, Y, z_slices, g_dirs]
        mask_block = torch.cat(mask_volumes, dim=-1)    # [X, Y, z_slices, g_dirs]
        bvecs_tensor = torch.cat(bvecs, dim=0)          # [g_dirs, 3]

        return data_block, mask_block, bvecs_tensor, z_indices
           

class DWIDataset1:
    def __init__(self, base_path, num_subjects, z_indices, device='cpu', rotation_angle=5):
        """
        Custom DWI data loader for flat .pt file structure.

        Args:
            base_path (str): Path to folder with sub1_data.pt, sub1_mask.pt, ...
            num_subjects (int): Number of subjects (e.g., 32).
            z_indices (list of int): Fixed list of z-slices to load.
            device (str): 'cpu' or 'cuda'.
        """
        self.base_path = Path(base_path)
        self.subject_ids = list(range(1, num_subjects + 1))
        self.z_indices = z_indices
        self.device = device
        self.rotation_angle = rotation_angle

    def get_sample(self):
        subj_id = random.choice(self.subject_ids)
        prefix = f"sub{subj_id}"

        # Load flat-named files
        data = torch.load(self.base_path / f"{prefix}_data.pt").to(self.device)
        mask = torch.load(self.base_path / f"{prefix}_mask.pt").to(self.device)
        bvals = torch.load(self.base_path / f"{prefix}_bvals.pt").to(self.device)
        bvecs = torch.load(self.base_path / f"{prefix}_bvecs.pt").to(self.device)

        # Select fixed z-slices
        #data_block = torch.stack([data[:, :, z, :] for z in self.z_indices], dim=2)
        #mask_block = torch.stack([mask[:, :, z] for z in self.z_indices], dim=2)
        
        data_block = data[:, :, self.z_indices, :] 
        mask_block = mask[:, :, self.z_indices] 

        angle = random.uniform(-self.rotation_angle, self.rotation_angle)
        bvecs_rotated = self.rotate_bvecs(bvecs, angle)

        return data_block, mask_block, bvecs_rotated, bvals

    def rotate_bvecs(self, bvecs, angle):
        return bvecs  # no rotation implemented

import torch
import torch.nn as nn

class DTINet1(nn.Module):
    def __init__(self, in_channels=91):
        super(DTINet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            #nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(64, 256, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            nn.ReLU(),


            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            #nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            #nn.BatchNorm3d(128),
            
            nn.ReLU(),

            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            #nn.BatchNorm3d(128),
            
            nn.ReLU(),

            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
        )

        self.final = nn.Conv3d(32, 6, kernel_size=1)  

    def forward(self, x):

       
        x = x.permute(3, 2, 0, 1)  # [W, H, B, G] -> [G, B, W, H]
        x = x.unsqueeze(0).permute(0, 1, 3, 4, 2)  # [1, G, W, H, B]
        #x=x/x.max()
        x = self.encoder(torch.abs(x))
        params = self.final(x).squeeze(0).permute(1, 2, 3, 0)  # [W, H, B, 6]

        # Symmetric matrix elements
        Dxx = params[..., 0]
        Dyy = params[..., 1]
        Dzz = params[..., 2]
        Dxy = params[..., 3]
        Dxz = params[..., 4]
        Dyz = params[..., 5]

        # Build symmetric matrix
        D = torch.stack([
            torch.stack([Dxx, Dxy, Dxz], dim=-1),
            torch.stack([Dxy, Dyy, Dyz], dim=-1),
            torch.stack([Dxz, Dyz, Dzz], dim=-1)
        ], dim=-2)  # [W, H, B, 3, 3]

        # Make SPD: D' = Q * diag(ReLU(λ) + ε) * Q^T
        eps = 1e-10
        D = torch.nan_to_num(D, nan=eps, posinf=eps, neginf=eps)

        eigvals, eigvecs = torch.linalg.eigh(D)  # symmetric eigendecomp
        eigvecs, _ = torch.linalg.qr(eigvecs) 
        eigvals = torch.abs(eigvals)   # ensure positivity
        D_spd = torch.matmul(eigvecs, torch.matmul(torch.diag_embed(eigvals), eigvecs.transpose(-1, -2)))

        return D_spd

import torch
import torch.nn as nn

class DTINet(nn.Module):
    def __init__(self, in_channels=91):
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
            # no ReLU here, as original commented out
        )

        self.final = nn.Conv2d(32, 6, kernel_size=1)  

    def forward(self, x):
        # Original input: [W, H, B, G]
        # Rearrange to [B, G, W, H] as 2D images with channels=G
        x = x.permute(2, 3, 0, 1)  # [B, G, W, H]

        # Optionally take absolute value as in original
        x = torch.abs(x)

        # Pass through encoder (2D convs)
        x = self.encoder(x)  # [B, 32, W, H]

        # Final conv to get 6 tensor coefficients per pixel
        params = self.final(x)  # [B, 6, W, H]

        # Rearrange to [W, H, B, 6] to match original format
        params = params.permute(2, 3, 0, 1)  # [W, H, B, 6]

        # Extract tensor components
        Dxx = params[..., 0]
        Dyy = params[..., 1]
        Dzz = params[..., 2]
        Dxy = params[..., 3]
        Dxz = params[..., 4]
        Dyz = params[..., 5]

        # Build symmetric tensor matrix: [W, H, B, 3, 3]
        D = torch.stack([
            torch.stack([Dxx, Dxy, Dxz], dim=-1),
            torch.stack([Dxy, Dyy, Dyz], dim=-1),
            torch.stack([Dxz, Dyz, Dzz], dim=-1)
        ], dim=-2)

        # Ensure no NaN/Inf
        eps = 1e-10
        D = torch.nan_to_num(D, nan=eps, posinf=eps, neginf=eps)

        # Symmetric eigendecomposition
        eigvals, eigvecs = torch.linalg.eigh(D)
        eigvecs, _ = torch.linalg.qr(eigvecs)
        eigvals = torch.abs(eigvals)  # force positivity

        # Rebuild SPD matrix
        D_spd = torch.matmul(eigvecs, torch.matmul(torch.diag_embed(eigvals), eigvecs.transpose(-1, -2)))

        return D_spd


def dti_signal_estimate(D_est, bvecs, bvals):
            
            W, H, B, _, _ = D_est.shape
            G = bvecs.shape[-1]

            g=bvecs[None,:,:]
            g = g.permute(0, 2, 1)  # [B, G, 3]
            
            g = g.unsqueeze(-1)        # [B, G, 3, 1]
            g_T = g.transpose(-2, -1)  # [B, G, 1, 3]

            # Expand D_tensor to match: [W, H, B, 1, 3, 3]
            D_exp = D_est.unsqueeze(3)

            # Compute g^T * D * g → scalar → [W, H, B, G]
            tmp = torch.matmul(D_exp, g)         # [W, H, B, G, 3, 1]
            #bvals=torch.tensor(bvals).to(D_tensor.device)
            #bvals = bvals.view(1, 1, 1, G)
            #S_est = S0[...,None]*torch.exp(-bvals * torch.matmul(g_T, tmp).squeeze(-1).squeeze(-1) )  # [W, H, B, G]
            S_est = torch.exp(-bvals * torch.matmul(g_T, tmp).squeeze(-1).squeeze(-1) )  # [W, H, B, G]
            #S_est=S_est/S0.max()
            #print(S_est.shape, S0.shape)
            #exit()
            return S_est



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


import torch
import os
import random
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

import numpy as np
import nibabel as nib
from pathlib import Path

import numpy as np
import nibabel as nib
from pathlib import Path
import torch
from pathlib import Path


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



def get_test_data(path, num_subjects=37, device='cpu'):
    # Subjects 37 and 38 are considered as test subjects
    test_subjects = [num_subjects]
    
    data_batch, mask_batch, bvec_batch = [], [], []

    # Loop over each test subject (37 and 38)
    for subj_idx in test_subjects:
        base = os.path.join(path, f"data{subj_idx}.pt")
        data = torch.load(base, map_location=device)  # Shape: [W, H, Z, G]
        mask = torch.load(base.replace("data", "mask"), map_location=device)  # Shape: [W, H, Z]
        bvec = torch.load(base.replace("data", "bvecs"), map_location=device)  # Shape: [3, G]

        W, H, Z, G = data.shape
        
        # Randomly pick a slice index along the Z axis
        z = random.randint(87, 90)
        #z=82
        # Extract the random slice for the current subject
        data_slice = data[:, :, z, :]  # Shape: [W, H, G]
        mask_slice = mask[:, :, z]  # Shape: [W, H]
        bvec_slice = bvec  # Shape: [3, G]

        # Add slices to the batch list
        data_batch.append(data_slice.unsqueeze(2))  # [W, H, 1, G]
        mask_batch.append(mask_slice.unsqueeze(2))  # [W, H, 1]
        bvec_batch.append(bvec_slice.unsqueeze(0))  # [1, 3, G]

    # Stack the slices to create a batch
    x = torch.cat(data_batch, dim=2).to(device)  # Shape: [W, H, 2, G] for 2 subjects
    m = torch.cat(mask_batch, dim=2).to(device)  # Shape: [W, H, 2]
    bv = torch.cat(bvec_batch, dim=0).to(device)  # Shape: [2, 3, G]

    return x, m, bv


class DWIDataset2(Dataset):

    def __init__(self, data_dir, num_slices=16, device='cpu', rotation_angle=5):
        self.device = device
        self.num_slices = num_slices
        self.rotation_angle = rotation_angle
        self.data = torch.load(os.path.join(data_dir, 'data_all.pt'), map_location=device)      # [N, W, H, Z, G]
        self.mask = torch.load(os.path.join(data_dir, 'mask_all.pt'), map_location=device)      # [N, W, H, Z]
        self.bvecs = torch.load(os.path.join(data_dir, 'bvecs_all.pt'), map_location=device)     # [N, 3, G]
        self.N = self.data.shape[0]

        self.transform = T.Compose([
            T.RandomRotation(degrees=rotation_angle),
        ])

    def __len__(self):
        return 20  # virtual length

    def rotate_bvecs_z(self, bvecs, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        Rz = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,     1]
        ], dtype=bvecs.dtype, device=bvecs.device)
        return Rz @ bvecs  # [3, G]

    def __getitem__(self, _):
        data_batch, mask_batch, bvec_batch = [], [], []

        for _ in range(self.num_slices):
            subj_idx = random.randint(0, self.N - 1)
            d = self.data[subj_idx]     # [W, H, Z, G]
            m = self.mask[subj_idx]     # [W, H, Z]
            bvec = self.bvecs[subj_idx] # [3, G]

            W, H, Z, G = d.shape
            z = random.randint(0, Z - 1)
            data_slice = d[:, :, z, :]     # [W, H, G]
            mask_slice = m[:, :, z]        # [W, H]

            # Permute for torchvision: [G, H, W]
            data_slice = data_slice.permute(2, 1, 0)
            data_slice = self.transform(data_slice)
            data_slice = data_slice.permute(2, 1, 0)

            mask_slice = self.transform(mask_slice.unsqueeze(0)).squeeze(0)

            bvec_rotated = self.rotate_bvecs_z(bvec, -self.rotation_angle)

            data_batch.append(data_slice.unsqueeze(2))   # [W, H, 1, G]
            mask_batch.append(mask_slice.unsqueeze(2))   # [W, H, 1]
            bvec_batch.append(bvec_rotated.unsqueeze(0)) # [1, 3, G]

        x = torch.cat(data_batch, dim=2).to(self.device)  # [W, H, B, G]
        m = torch.cat(mask_batch, dim=2).to(self.device)  # [W, H, B]
        bv = torch.cat(bvec_batch, dim=0).to(self.device) # [B, 3, G]

        return x, m, bv
