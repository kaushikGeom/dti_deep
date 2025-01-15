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
import numpy as np

from dipy.data import get_sphere 

sphere=get_sphere('symmetric724')

import warnings
# Suppress specific warning by setting ignore for all warnings
warnings.filterwarnings("ignore", message="logm result may be inaccurate, approximate err =")

import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F


def compute_ssim(D_est, D_gt, data_range=None, eps=1e-8):
    """
    Compute SSIM between two tensors without explicit loops.
    Args:
        D_est (torch.Tensor): Estimated tensor of shape (128, 128, 32, 3, 3).
        D_gt (torch.Tensor): Ground truth tensor of the same shape as D_est.
        data_range (float, optional): Range of input data (default: max - min of D_gt).
        eps (float): Small value to avoid division by zero (default: 1e-8).
    Returns:
        float: Mean SSIM across all slices.
    """
    # Ensure the shapes are identical
    assert D_est.shape == D_gt.shape, "D_est and D_gt must have the same shape!"

    if data_range is None:
        data_range = D_gt.max() - D_gt.min()

    # Unpack dimensions for clarity
    H, W, D, C1, C2 = D_est.shape

    # Reshape tensors to combine depth and extra dimensions into batch size
    D_est = D_est.permute(2, 3, 4, 0, 1).reshape(-1, 1, H, W)
    D_gt = D_gt.permute(2, 3, 4, 0, 1).reshape(-1, 1, H, W)

    # Compute mean and variance for each patch
    mu1 = F.avg_pool2d(D_est, kernel_size=11, stride=1, padding=5)
    mu2 = F.avg_pool2d(D_gt, kernel_size=11, stride=1, padding=5)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(D_est * D_est, kernel_size=11, stride=1, padding=5) - mu1_sq
    sigma2_sq = F.avg_pool2d(D_gt * D_gt, kernel_size=11, stride=1, padding=5) - mu2_sq
    sigma12 = F.avg_pool2d(D_est * D_gt, kernel_size=11, stride=1, padding=5) - mu1_mu2
    
    # Constants for SSIM (using default values from the original paper)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Compute SSIM for each slice
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + eps)
    
    # Compute mean SSIM over all slices
    return ssim_map.mean().item()

#grads64=torch.load('64grads.pt') 

def compute_pixel_smpe(gt, pred):
    """
    Compute per-pixel Symmetric Mean Percentage Error (SMPE) between ground truth and predictions.
    
    Parameters:
        gt (np.ndarray): Ground truth values of shape [H, W, D, 3].
        pred (np.ndarray): Predicted values of shape [H, W, D, 3].
    
    Returns:
        np.ndarray: SMPE for each eigenvalue per pixel [H, W, D, 3].
    """
    smpe = np.abs(gt - pred) / ((np.abs(gt) + np.abs(pred)) / 2) * 100
    return smpe  

def metrics_per_pixel_smpe_boxplots(md_sim, md_est, md_fit,
                                 fa_sim, fa_est, fa_fit,
                                 ad_sim, ad_est, ad_fit,
                                 rd_sim, rd_est, rd_fit):
    """
    Plot per-pixel SMPE boxplots for MD, FA, AD, RD metrics comparing sim vs est and sim vs fit.
    
    Parameters:
        md_sim, md_est, md_fit, fa_sim, fa_est, fa_fit, ad_sim, ad_est, ad_fit, rd_sim, rd_est, rd_fit: 
        These are the tensors of shape [240, 240, 16].
    """
    """ # Convert tensors to numpy for easier processing
    md_sim, md_est, md_fit = md_sim.numpy(), md_est.numpy(), md_fit.numpy()
    fa_sim, fa_est, fa_fit = fa_sim.numpy(), fa_est.numpy(), fa_fit.numpy()
    ad_sim, ad_est, ad_fit = ad_sim.numpy(), ad_est.numpy(), ad_fit.numpy()
    rd_sim, rd_est, rd_fit = rd_sim.numpy(), rd_est.numpy(), rd_fit.numpy()
    """
    # Compute per-pixel SMPE for each pair
    md_smpe_est = compute_pixel_smpe(md_sim, md_est)
    md_smpe_fit = compute_pixel_smpe(md_sim, md_fit)

    fa_smpe_est = compute_pixel_smpe(fa_sim, fa_est)
    fa_smpe_fit = compute_pixel_smpe(fa_sim, fa_fit)

    ad_smpe_est = compute_pixel_smpe(ad_sim, ad_est)
    ad_smpe_fit = compute_pixel_smpe(ad_sim, ad_fit)

    rd_smpe_est = compute_pixel_smpe(rd_sim, rd_est)
    rd_smpe_fit = compute_pixel_smpe(rd_sim, rd_fit)

    # Reshape to 2D: [H*W*D, 3] for easier plotting
    md_smpe_est = md_smpe_est.reshape(-1)
    md_smpe_fit = md_smpe_fit.reshape(-1)

    fa_smpe_est = fa_smpe_est.reshape(-1)
    fa_smpe_fit = fa_smpe_fit.reshape(-1)

    ad_smpe_est = ad_smpe_est.reshape(-1)
    ad_smpe_fit = ad_smpe_fit.reshape(-1)

    rd_smpe_est = rd_smpe_est.reshape(-1)
    rd_smpe_fit = rd_smpe_fit.reshape(-1)

    # Prepare the data for boxplots (per metric comparisons)
    data = [
        (md_smpe_est, md_smpe_fit),  # MD comparison (sim vs est, sim vs fit)
        (fa_smpe_est, fa_smpe_fit),  # FA comparison (sim vs est, sim vs fit)
        (ad_smpe_est, ad_smpe_fit),  # AD comparison (sim vs est, sim vs fit)
        (rd_smpe_est, rd_smpe_fit)   # RD comparison (sim vs est, sim vs fit)
    ]

    # Create positions for each boxplot group
    positions = [1, 2,  4, 5,  7, 8,  10, 11,  13, 14]  # Group positions

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["lightblue", "lightgreen"]

    # Boxplot properties
    for i, (sim_vs_est, sim_vs_fit) in enumerate(data):
        
        
        ax.boxplot(
            [sim_vs_est],
            positions=[positions[2 * i]],  # Different positions for each comparison
            widths=0.6,
            patch_artist=True,
            showfliers=False,  # Disable outliers
            boxprops=dict(facecolor=colors[0], color="blue"),
            medianprops=dict(color="red")
        ) # Add x-tick labels at the center of each group
    
        ax.boxplot(
            [sim_vs_fit],
            positions=[positions[2 * i + 1]],  # Different positions for each comparison
            widths=0.6,
            patch_artist=True,
            showfliers=False,  # Disable outliers
            boxprops=dict(facecolor=colors[1], color="blue"),
            medianprops=dict(color="red")
        ) # Add x-tick labels at the center of each group
    
    ax.set_xticks([1.5, 4.5, 7.5, 10.5])  # Midpoints of grouped boxes
    ax.set_xticklabels(["MD", "FA", "AD", "RD"])

    # Set y-label and title
    ax.set_ylabel("Symmetric Mean Percentage Error (%)", fontsize=12)
    ax.set_title("Per-Pixel SMPE for MD, FA, AD, RD Metrics", fontsize=14)

    # Add legend manually
    handles = [
        plt.Line2D([0], [0], color='lightblue', lw=10, label='Sim vs Est'),
        plt.Line2D([0], [0], color='lightgreen', lw=10, label='Sim vs Fit'),
    ]
    ax.legend(handles=handles, loc="upper right")

    # Tight layout for better spacing
    plt.tight_layout()
    plt.show()



def plot_per_pixel_smpe_boxplots(eigvals_gt, eigvals_est, eigvals_fit):
    """
    Plot per-pixel Symmetric Mean Percentage Error (SMPE) for lambda 1, 2, and 3.
    
    Parameters:
        eigvals_gt (torch.Tensor): Ground truth eigenvalues of shape [240, 240, 16, 3].
        eigvals_est (torch.Tensor): Estimated eigenvalues of shape [240, 240, 16, 3].
        eigvals_fit (torch.Tensor): Fitted eigenvalues of shape [240, 240, 16, 3].
    """
    # Move tensors to CPU and convert to numpy arrays
    eigvals_gt = eigvals_gt
    eigvals_est = eigvals_est
    eigvals_fit = eigvals_fit

    # Compute per-pixel SMPE
    smpe_est = compute_pixel_smpe(eigvals_gt, eigvals_est)  # Shape: [240, 240, 16, 3]
    smpe_fit = compute_pixel_smpe(eigvals_gt, eigvals_fit)  # Shape: [240, 240, 16, 3]

    # Reshape to 2D for easier boxplot data preparation: [H*W*D, 3]
    smpe_est = smpe_est.reshape(-1, 3)
    smpe_fit = smpe_fit.reshape(-1, 3)

    # Prepare data for each eigenvalue
    data = [
        smpe_est[:, 0], smpe_fit[:, 0],  # Lambda 1
        smpe_est[:, 1], smpe_fit[:, 1],  # Lambda 2
        smpe_est[:, 2], smpe_fit[:, 2],  # Lambda 3
    ]

    # Create positions for boxplots
    positions = [1, 2, 4, 5, 7, 8]  # Two boxes per eigenvalue

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define colors for each eigenvalue pair (GT vs EST and GT vs FIT)
    colors = ["lightblue", "lightgreen"]

    ax.boxplot(
        [data[0]],  
        positions=[positions[0]],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=colors[0], color='blue'),
        medianprops=dict(color='red')
    )
    ax.boxplot(
        [data[2]],  
        positions=[positions[2]],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=colors[0], color='blue'),
        medianprops=dict(color='red')
    )
    
    ax.boxplot(
        [data[4]],  
        positions=[positions[4]],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=colors[0], color='blue'),
        medianprops=dict(color='red')
    )
    
    
    ax.boxplot(
        [data[1]],  
        positions=[positions[1]],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=colors[1], color='blue'),
        medianprops=dict(color='red')
    )
    ax.boxplot(
        [data[3]],  
        positions=[positions[3]],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=colors[1], color='blue'),
        medianprops=dict(color='red')
    )
    
    ax.boxplot(
        [data[5]],  
        positions=[positions[5]],
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=colors[1], color='blue'),
        medianprops=dict(color='red')
    )
    
    # Add x-tick labels at the center of groups
    ax.set_xticks([1.5, 4.5, 7.5])  # Midpoints of grouped boxes
    ax.set_xticklabels(["Lambda 1", "Lambda 2", "Lambda 3"])

    # Add labels and title
    ax.set_ylabel("Symmetric Mean Percentage Error (%)", fontsize=12)
    ax.set_title("Per-Pixel SMPE for Eigenvalues", fontsize=14)

    # Add legend manually
    handles = [
        plt.Line2D([0], [0], color='lightblue', lw=10, label='GT vs EST'),
        plt.Line2D([0], [0], color='lightgreen', lw=10, label='GT vs FIT'),
    ]
    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

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


def sample_unit_semisphere(n_points, device="cuda"):
    """
    Uniformly sample n_points directions over the upper half of a unit sphere (z >= 0) using PyTorch.
    
    Parameters:
        n_points (int): Number of unit directions to generate.
        device (str): The device to use ("cpu" or "cuda").
    
    Returns:
        torch.Tensor: Tensor of shape (n_points, 3) with unit direction vectors in the upper hemisphere.
    """
    # Generate random azimuthal angles (phi) uniformly in [0, 2*pi]
    phi = torch.rand(n_points, device=device) * 2 * torch.pi
    
    # Generate random cos(theta) uniformly in [0, 1] for upper hemisphere
    cos_theta = torch.rand(n_points, device=device)
    theta = torch.acos(cos_theta)
    
    # Convert spherical coordinates to Cartesian coordinates
    x = torch.sin(theta) * torch.cos(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(theta)  # Always non-negative
    
    return torch.stack((x, y, z), dim=1)

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

def calculate_md(eigvals):
    return eigvals.mean(axis=-1)

def calculate_ad(eigvals):
     return eigvals[:,:,:, 0]

def calculate_ad2(eigvals):
     return eigvals[:,:,:, 1]

def calculate_rd(eigvals):
    return eigvals[:,:,:, 1:3].mean(axis=3)



# Define FA calculation directly with lambdas
""" def calculate_fa(eigvals): 
 md=calculate_md(eigvals)
 return np.sqrt(3 / 2) *   (np.sqrt(((eigvals - md) ** 2).sum(axis=-1)) /
                           (np.sqrt((eigvals ** 2).sum(axis=-1)) + 1e-12))
 """       

mean_diffusivity = lambda eigvals: eigvals.mean(axis=-1, keepdims=True)
fa_numerator = lambda eigvals, md: np.sqrt(((eigvals - md) ** 2).sum(axis=-1))
fa_denominator = lambda eigvals: np.sqrt((eigvals ** 2).sum(axis=-1))

calculate_fa = lambda eigvals: np.sqrt(3 / 2) * (
    fa_numerator(eigvals, mean_diffusivity(eigvals)) /
    (fa_denominator(eigvals) + 1e-12) )

 
def calculate_fa_tensor2(eigvals): 
            fa= torch.sqrt(torch.tensor(3.0) / 2) * (
                torch.sqrt(((eigvals - eigvals.mean(dim=-1, keepdim=True)) ** 2).sum(dim=-1)) /
                eigvals.norm(dim=-1))
            return fa

def calculate_fa_tensor(evals): 

    mean_diffusivity = evals.mean(dim=-1, keepdim=True)
    numerator = torch.sqrt(((evals - mean_diffusivity) ** 2).sum(dim=-1))
    denominator = torch.sqrt((evals ** 2).sum(dim=-1))
    fa = torch.sqrt(torch.tensor(3. / 2)) * (numerator / denominator)
    return fa

def euclidean_mse(D_sim, D):
    """
    Compute the Mean Squared Error (MSE) on the Euclidean distance between two tensors D_sim and D.
    
    :param D_sim: Predicted tensor of shape [batch_size, height, width, 3, 3]
    :param D: Ground truth tensor of shape [batch_size, height, width, 3, 3]
    :return: MSE of the Euclidean distances between the tensors
    """
    # Flatten the last two dimensions (3x3 matrices) for each voxel
    D_sim_flat = D_sim.reshape(D_sim.size(0), D_sim.size(1), D_sim.size(2), -1)  # Flatten to [batch_size, height, width, 9]
    D_flat = D.reshape(D.size(0), D.size(1), D.size(2), -1)  # Flatten to [batch_size, height, width, 9]

    # Compute the squared differences (Euclidean distance squared between corresponding 3x3 matrices)
    squared_diff = (D_sim_flat - D_flat) ** 2  # Shape: [batch_size, height, width, 9]

    # Sum the squared differences over the last dimension (for each 3x3 matrix)
    sum_squared_diff = squared_diff.sum(dim=-1)  # Shape: [batch_size, height, width]

    # Compute MSE: average over all elements
    mse = sum_squared_diff.mean()  # Scalar value: Mean squared error over all voxels

    return mse


#import warnings
#warnings.filterwarnings("ignore", message="logm result may be inaccurate")


def logm_approx(A):
        """ Approximate the matrix logarithm using eigenvalue decomposition. """
        eigvals, eigvecs = linalg.eigh(A)  # Ensure it's symmetric and positive definite.
        eigvals = eigvals.real  # Use real part of eigenvalues for positive-definite matrices
        log_eigvals = torch.log(eigvals)  # Logarithm of eigenvalues
        
        # Reconstruct the matrix logarithm
        logA = torch.matmul(eigvecs, torch.matmul(torch.diag_embed(log_eigvals), eigvecs.transpose(-2, -1)))
        return logA
    
# Function to compute the log-Euclidean distance
def log_euclidean(D, D_sim):
    """
    Compute the Log-Euclidean distance between two SPD matrices D and D_sim.
    
    Arguments:
    D -- A tensor of shape (batch_size, height, width, 3, 3) representing D matrices.
    D_sim -- A tensor of shape (batch_size, height, width, 3, 3) representing D_sim matrices.
    
    Returns:
    log_euclidean_dist -- The Log-Euclidean distance between D and D_sim.
    """
    
    # Step 1: Compute the matrix logarithms of D and D_sim using eigenvalue decomposition.
    
    
    # Compute the logarithms of D and D_sim (this will be done element-wise across the batch)
    log_D = logm_approx(D)
    log_D_sim = logm_approx(D_sim)
    # Step 2: Compute the Frobenius norm of the difference between the log matrices.
    
    # Frobenius norm is just the square root of the sum of squared entries
    dist = torch.norm(log_D - log_D_sim, p='fro')  # Frobenius norm (L2 norm)
    
    return dist

""" def le_mse(D1, D2):

    device=D1.device
    #log_e_dist = le_metric.dist(D1.detach().cpu().numpy(), D2.detach().cpu().numpy())
    D1, D2= D1.detach().cpu().numpy(), D2.detach().cpu().numpy()
    
    return torch.tensor(np.mean(le_metric.dist(D1, D2)**2), device=device)
 """

def scale_to_01(arr):
    epsilon=1e-10
    min_val = torch.min(arr)
    max_val = torch.max(arr)
        
    scaled_arr = (arr - min_val) / (max_val - min_val)
    return scaled_arr + epsilon


def map_to_range(y, min_val, max_val):
        y=scale_to_01(y)
        return min_val + y * (max_val - min_val)





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

def eigvals_generator_matrix(target_matrix, batchsize=None, w=None, h=None):
    """
    Generates three matrices (lambda1, lambda2, lambda3) where each element in the matrices
    sums to the corresponding element in the target matrix, with one dominant value in each set.

    Parameters:
    target_matrix (torch.Tensor): A 2D tensor of shape (240, 240) representing target values.

    Returns:
    tuple: Three tensors (lambda1, lambda2, lambda3) of shape (240, 240) where:
           - lambda1[i, j] + lambda2[i, j] + lambda3[i, j] = target_matrix[i, j]
           - lambda1[i, j] is the dominant value in each triplet
    """
    #if target_matrix.ndim != 2 or target_matrix.shape != (w, h, batchsize):
    #    raise ValueError("Input must be a 240x240 matrix.")

    # Step 1: Generate lambda1 as a random fraction between 50% and 90% of each target cell
    lambda1 = torch.empty_like(target_matrix).uniform_(0.5, 0.9) * target_matrix

    # Step 2: Calculate the remaining values needed to reach each target value
    remaining = target_matrix - lambda1

    # Step 3: Split the remaining values between lambda2 and lambda3 randomly
    lambda2 = torch.empty_like(target_matrix).uniform_(0, 1) * remaining
    lambda3 = remaining - lambda2  # Ensures that lambda1 + lambda2 + lambda3 = target_matrix
    lambdas = torch.stack((lambda1, lambda2, lambda3), dim=-1)
    return lambdas
import torch

import torch

def eigvals_generator_matrix2(target_matrix):
    """
    Generates eigenvalue matrices randomly following one of the three cases for each batch:
    - Case 1: lambda1 > lambda2 > lambda3
    - Case 2: lambda1 > lambda2 ≈ lambda3
    - Case 3: lambda1 ≈ lambda2 > lambda3

    Parameters:
    -----------
    target_matrix : torch.Tensor
        A 3D tensor of shape (w, h, batchsize) representing target values.

    Returns:
    --------
    torch.Tensor
        A tensor of shape (w, h, batchsize, 3) containing the eigenvalues for each point.
    """
    w, h, batchsize = target_matrix.shape

    # Generate random numbers for the cases
    case = torch.randint(1, 4, (w, h, batchsize), dtype=torch.long)  # Randomly pick case for each (w, h, batch)

    # Step 1: Generate the random values for lambda1, lambda2, and lambda3
    # Case 1: lambda1 > lambda2 > lambda3
    lambda1 = torch.empty((w, h, batchsize)).uniform_(0.5, 0.9) * target_matrix
    remaining_1 = target_matrix - lambda1
    lambda2_1 = torch.empty((w, h, batchsize)).uniform_(0.3, 0.5) * remaining_1
    lambda3_1 = remaining_1 - lambda2_1

    # Case 2: lambda1 > lambda2 ≈ lambda3
    lambda1_2 = torch.empty((w, h, batchsize)).uniform_(0.5, 0.9) * target_matrix
    remaining_2 = target_matrix - lambda1_2
    lambda2_2 = 0.5 * remaining_2
    lambda3_2 = 0.5 * remaining_2

    # Case 3: lambda1 ≈ lambda2 > lambda3
    lambda2_3 = torch.empty((w, h, batchsize)).uniform_(0.4, 0.5) * target_matrix
    lambda1_3 = lambda2_3 + torch.empty((w, h, batchsize)).uniform_(0.0, 0.1) * target_matrix
    lambda3_3 = target_matrix - lambda1_3 - lambda2_3

    # Step 2: Choose the correct values based on the case
    lambda1_final = torch.zeros((w, h, batchsize)).to(device=target_matrix.device)
    lambda2_final = torch.zeros((w, h, batchsize)).to(device=target_matrix.device)
    lambda3_final = torch.zeros((w, h, batchsize)).to(device=target_matrix.device)

    # Use `case` tensor to select values
    lambda1_final = torch.where(case == 1, lambda1, lambda1_final)
    lambda2_final = torch.where(case == 1, lambda2_1, lambda2_final)
    lambda3_final = torch.where(case == 1, lambda3_1, lambda3_final)

    lambda1_final = torch.where(case == 2, lambda1_2, lambda1_final)
    lambda2_final = torch.where(case == 2, lambda2_2, lambda2_final)
    lambda3_final = torch.where(case == 2, lambda3_2, lambda3_final)

    lambda1_final = torch.where(case == 3, lambda1_3, lambda1_final)
    lambda2_final = torch.where(case == 3, lambda2_3, lambda2_final)
    lambda3_final = torch.where(case == 3, lambda3_3, lambda3_final)

    # Stack the lambdas into a single tensor
    lambdas = torch.stack((lambda1_final, lambda2_final, lambda3_final), dim=-1)

    # Sort lambdas in descending order along the last dimension (ensuring lambda1 > lambda2 > lambda3)
    lambdas, _ = torch.sort(lambdas, descending=True, dim=-1)

    return torch.abs(lambdas)


def dtimodel(g=None, b=1000, D=None):
     
    S=torch.zeros(g.shape(0)) 
    for i in range(g.shape(0)):
            S[i]=torch.exp(b*(g[i,:].T*D*g[i,:]))
    return S
    

def select_values_normal_distr(range_tensor, size=None, std_dev=None):

    lower_bound, upper_bound = range_tensor[0].item(), range_tensor[1].item()

    mean = (lower_bound + upper_bound) / 2

    # If no standard deviation is provided, set it based on the range
    if std_dev is None:
        std_dev = (upper_bound - lower_bound) / 6  # 99.7% of values lie within 3 std deviations

    
    random_values = torch.normal(mean=torch.full((size), mean), std=std_dev * torch.ones(size))

    random_values = torch.clamp(random_values, min=lower_bound, max=upper_bound)

    return random_values



def random_rotation_matrices2(grid_size=(240, 240, 16), device=None):
    """
    Generates a grid of random 3D rotation matrices of shape (240, 240, 16, 3, 3).
    
    Parameters:
    grid_size (tuple): The size of the grid, default is (240, 240, 16).
    device: Device for the tensors (e.g., 'cpu' or 'cuda').
    
    Returns:
    torch.Tensor: A tensor of shape (240, 240, 16, 3, 3) where each entry is a random 3D rotation matrix.
    """
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

import torch

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




def apply_region_vectors(mask, lambda_wm, lambda_gm, lambda_csf, S0=None,device=None):
    """
    Replaces pixels in the mask with vectors from lambda_wm, lambda_gm, and lambda_csf based on regions.
    
    Parameters:
    mask (torch.Tensor): A tensor of shape (240, 240) with region labels:
                         0 for background, 1 for white matter (wm), 2 for gray matter (gm), 3 for cerebrospinal fluid (csf).
    lambda_wm (torch.Tensor): A tensor of shape (240, 240, 3) with vectors for white matter.
    lambda_gm (torch.Tensor): A tensor of shape (240, 240, 3) with vectors for gray matter.
    lambda_csf (torch.Tensor): A tensor of shape (240, 240, 3) with vectors for cerebrospinal fluid.

    Returns:
    torch.Tensor: A tensor of shape (240, 240, 3) where each pixel has the corresponding vector based on the mask.
    """
    # Initialize an output tensor with the same shape as lambda_wm, lambda_gm, lambda_csf (240, 240, 3)
    output = torch.zeros_like(lambda_wm).to(device=device)

    # Set the vectors based on the regions in the mask
    output[mask == 0] = lambda_wm[mask == 0]  # White matter region
    output[mask == 1] = lambda_gm[mask == 1]  # Gray matter region
    output[mask == 2] = lambda_csf[mask == 2] # Cerebrospinal fluid region
    
    return output

import matplotlib.pyplot as plt

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
    mask[ m1 & m2] = 0
    m3=noise <=0
    m4= noise>-point
    mask[ m3 & m4] = 1
    mask[((noise <-point )  | (noise > point))]=2

    return mask

def compute_T(R, D):
    """
    Computes T = R^T * D * R for each voxel in a 240x240x16 grid.
    
    Parameters:
    R (torch.Tensor): Rotation matrices with shape [240, 240, 16, 3, 3].
    D (torch.Tensor): Diagonal elements of matrices with shape [240, 240, 16, 3].
    
    Returns:
    torch.Tensor: Resulting tensor T of shape [240, 240, 16, 3, 3].
    """
    # Step 1: Expand D to a full 3x3 diagonal matrix with shape [240, 240, 16, 3, 3]
    D_full = torch.zeros(D.shape + (3,), device=D.device)  # shape [240, 240, 16, 3, 3]
    D_full[..., 0, 0] = D[..., 0]  # Set the first diagonal element
    D_full[..., 1, 1] = D[..., 1]  # Set the second diagonal element
    D_full[..., 2, 2] = D[..., 2]  # Set the third diagonal element
    R_T = R.transpose(-1, -2)  # shape [240, 240, 16, 3, 3]

    T = torch.matmul(torch.matmul(R_T, D_full) , R)  # shape [240, 240, 16, 3, 3]

    return T


def is_positive_definite_sylvester(D):
    """
    Checks if all 3x3 matrices in the tensor D are positive definite using the Sylvester criterion.

    Args:
        D (torch.Tensor): A tensor of shape (..., 3, 3), where each 3x3 matrix will be checked.

    Returns:
        torch.Tensor: A boolean tensor of the same shape as D[..., 0, 0] indicating whether each matrix is positive definite.
    """
    if D.shape[-2:] != (3, 3):
        raise ValueError("The last two dimensions of the tensor must be 3x3.")

    # Extract the minors
    minor_1 = D[..., 0, 0]  # Top-left element
    minor_2 = D[..., 0, 0] * D[..., 1, 1] - D[..., 0, 1] * D[..., 1, 0]  # Determinant of the top-left 2x2 submatrix
    minor_3 = torch.det(D)  # Determinant of the full 3x3 matrix

    # Sylvester criterion: All minors must be positive
    is_positive_definite = (minor_1 > 0) & (minor_2 > 0) & (minor_3 > 0)

    return is_positive_definite

import torch

def correct_to_pure_rotation(evecs_est):
    """
    Ensure that evecs_est is a pure rotation matrix (det = +1).
    
    Args:
        evecs_est: Tensor of shape (..., 3, 3) representing a batch of 3x3 matrices.
    
    Returns:
        A corrected version of evecs_est with det = +1.
    """
    # Calculate determinant along the last two dimensions
    det = torch.det(evecs_est)  # Shape: (...,)

    # Find matrices with det = -1
    mask = (det < 0)  # Boolean mask for improper rotations, Shape: (...)

    # Expand mask to match the last column of evecs_est
    mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (..., 1, 1)

    # Correct the matrices by flipping the sign of the last column
    corrected_evecs = evecs_est.clone()
    corrected_evecs[mask.expand_as(evecs_est)] *= -1  # Flip the last column if det = -1

    return corrected_evecs


def gen_dti_signals(D=None, gradients=None, b_values=None, S0=None, device=None):
    
        epsilon=1e-10; 
        #S0 = S0  # Shape: [128, 128, 32, 1]
        #S0=1
        if S0 is not None:
            S0=S0[...,None]
        else:
            S0=1    

        #exponent = -b_values * torch.einsum('ij,...jk,ik->...i', gradients, D, gradients)  # Shape: [128, 128, 32, 65]
        exponent1=gradients@D.reshape(-1,3,3)@gradients.T
        exponent=-b_values*exponent1.reshape(D.shape[0], D.shape[1], D.shape[2], gradients.shape[0],gradients.shape[0]).diagonal(dim1=-2, dim2=-1) 
        S=S0*torch.exp(exponent)
        #Snorm=S/S0
        #print(exponent1[], S.shape)
        #exit()
        
        return S
 



grads64=np.array([      [ 0.999979  , -0.00504001, -0.00402795],
       [ 0.        ,  0.999992  , -0.00398794],
       [-0.0257055 ,  0.653861  , -0.756178  ],
       [ 0.589518  , -0.769236  , -0.246462  ],
       [-0.235785  , -0.529095  , -0.815147  ],
       [-0.893578  , -0.263559  , -0.363394  ],
       [ 0.79784   ,  0.133726  , -0.587851  ],
       [ 0.232937  ,  0.931884  , -0.278087  ],
       [ 0.93672   ,  0.144139  , -0.31903   ],
       [ 0.50413   , -0.846694  ,  0.170183  ],
       [ 0.345199  , -0.850311  ,  0.397252  ],
       [ 0.456765  , -0.635672  ,  0.622323  ],
       [-0.487481  , -0.393908  , -0.779229  ],
       [-0.617033  ,  0.676849  , -0.40143   ],
       [-0.578512  , -0.109347  ,  0.808311  ],
       [-0.825364  , -0.525034  , -0.207636  ],
       [ 0.895076  , -0.0448242 ,  0.443655  ],
       [ 0.289992  , -0.545473  ,  0.786361  ],
       [ 0.115014  , -0.96405   ,  0.239541  ],
       [-0.799934  ,  0.407767  ,  0.440264  ],
       [ 0.512494  ,  0.842139  , -0.167785  ],
       [-0.790005  ,  0.157993  ,  0.592394  ],
       [ 0.949281  , -0.237695  , -0.20583   ],
       [ 0.232318  ,  0.787051  , -0.571472  ],
       [-0.0196707 , -0.192031  ,  0.981192  ],
       [ 0.215969  , -0.957123  , -0.193061  ],
       [ 0.772645  , -0.607534  , -0.18418   ],
       [-0.160153  ,  0.360413  , -0.918941  ],
       [-0.146167  ,  0.735274  ,  0.661821  ],
       [ 0.88737   ,  0.421111  , -0.187724  ],
       [-0.562989  ,  0.236482  ,  0.791909  ],
       [-0.381313  ,  0.147037  , -0.912678  ],
       [-0.305954  , -0.203793  ,  0.929979  ],
       [-0.332682  , -0.134113  , -0.933454  ],
       [-0.962239  , -0.269464  , -0.0385391 ],
       [-0.959532  ,  0.20977   , -0.187871  ],
       [ 0.450964  , -0.890337  , -0.0627015 ],
       [-0.771192  ,  0.631175  , -0.0829533 ],
       [ 0.709816  ,  0.413159  , -0.570492  ],
       [-0.694543  ,  0.0279395 , -0.718908  ],
       [ 0.681549  ,  0.533101  ,  0.501293  ],
       [-0.141689  , -0.729241  , -0.669427  ],
       [-0.740351  ,  0.393223  , -0.545212  ],
       [-0.102756  ,  0.825367  , -0.555167  ],
       [ 0.583913  , -0.600782  , -0.545992  ],
       [-0.087755  , -0.339651  , -0.936449  ],
       [-0.550506  , -0.795484  , -0.253276  ],
       [ 0.837443  , -0.462202  ,  0.291648  ],
       [ 0.362929  , -0.56593   , -0.740274  ],
       [-0.183611  ,  0.397081  ,  0.89923   ],
       [-0.718323  , -0.695701  , -0.00354897],
       [ 0.432782  ,  0.686361  ,  0.584473  ],
       [ 0.501837  ,  0.694337  , -0.515805  ],
       [-0.170518  , -0.513769  ,  0.840812  ],
       [ 0.463195  ,  0.428052  , -0.776029  ],
       [ 0.383713  , -0.812572  , -0.438738  ],
       [-0.714166  , -0.251467  , -0.653247  ],
       [ 0.259205  ,  0.887258  ,  0.381557  ],
       [ 0.        ,  0.0813186 ,  0.996688  ],
       [ 0.0363633 , -0.904616  , -0.424675  ],
       [ 0.570854  , -0.308597  ,  0.760851  ],
       [-0.282205  ,  0.149795  ,  0.947588  ],
       [ 0.720351  ,  0.611914  , -0.326583  ],
       [ 0.265891  ,  0.960683  ,  0.0799352 ]])

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

import torch

def adjust_eigenvalues_white_matter(lambda_wm):
    """
    Enforces specific eigenvalue relationships on the tensor.

    Parameters:
        lambda_wm (torch.Tensor): Tensor of shape [32, 128, 128, 3].

    Returns:
        torch.Tensor: Modified tensor with eigenvalues following the specified cases.
    """
    # Copy the input tensor to avoid modifying the original
    modified_lambda = lambda_wm.clone()

    # Randomly choose a case for each voxel
    case_choices = torch.randint(0, 3, size=lambda_wm.shape[:-1], device=lambda_wm.device)

    # Case 1: λ1 > λ2 > λ3
    case1 = case_choices == 0
    if case1.any():
        sorted_vals = torch.sort(lambda_wm[case1], descending=True, dim=-1).values
        modified_lambda[case1] = sorted_vals

    # Case 2: λ1 = λ2 > λ3
    case2 = case_choices == 1
    if case2.any():
        sorted_vals = torch.sort(lambda_wm[case2], descending=True, dim=-1).values
        sorted_vals[..., 0] = sorted_vals[..., 1]  # λ1 = λ2
        modified_lambda[case2] = sorted_vals

    # Case 3: λ1 > λ2 = λ3
    case3 = case_choices == 2
    if case3.any():
        sorted_vals = torch.sort(lambda_wm[case3], descending=True, dim=-1).values
        sorted_vals[..., 1] = sorted_vals[..., 2]  # λ2 = λ3
        modified_lambda[case3] = sorted_vals

    return modified_lambda


def adjust_eigenvalues_gray_csf_matter(lambda_tensor, epsilon=1e-3):
    """
    Adjust eigenvalues for gray matter to enforce the following cases:
    1. All eigenvalues are exactly equal (isotropic diffusion).
    2. Eigenvalues are nearly equal with small random variations.
    3. One eigenvalue is slightly larger, while the other two are nearly equal.

    Parameters:
        lambda_tensor (torch.Tensor): Tensor of shape [batch_size, w, h, 3].
        epsilon (float): Small variation for near equality.

    Returns:
        torch.Tensor: Modified tensor with eigenvalues adjusted for gray matter.
    """
    batch_size, w, h, _ = lambda_tensor.shape

    # Randomly assign each voxel to one of the 3 cases
    choices = torch.randint(0, 3, size=(batch_size, w, h), device=lambda_tensor.device)  # 0, 1, 2 correspond to cases

    # Clone the tensor to store modified eigenvalues
    modified_lambda = lambda_tensor.clone()

    # Case 1: All eigenvalues are exactly equal
    case_1_mask = choices == 0
    if case_1_mask.any():
        mean_vals = modified_lambda[case_1_mask].mean(dim=-1, keepdim=True)  # Calculate mean along eigenvalue dimension
        modified_lambda[case_1_mask] = mean_vals.expand(-1, 3)  # Expand to match eigenvalue dimension

    # Case 2: Eigenvalues are nearly equal with small random variations
    case_2_mask = choices == 1
    if case_2_mask.any():
        mean_vals = modified_lambda[case_2_mask].mean(dim=-1, keepdim=True)  # Mean eigenvalue
        noise = torch.empty_like(modified_lambda[case_2_mask]).uniform_(-epsilon, epsilon)  # Small variations
        modified_lambda[case_2_mask] = mean_vals + noise

    # Case 3: One eigenvalue slightly larger, others nearly equal
    case_3_mask = choices == 2
    if case_3_mask.any():
        sorted_vals, _ = modified_lambda[case_3_mask].sort(dim=-1, descending=True)
        mean_vals = sorted_vals[..., 1:].mean(dim=-1, keepdim=True)  # Average λ₂ and λ₃
        noise = torch.empty_like(mean_vals).uniform_(-epsilon, epsilon)  # Small noise
        sorted_vals[..., 0] = sorted_vals[..., 0] + epsilon  # Slightly increase λ₁
        sorted_vals[..., 1:] = mean_vals + noise  # Nearly equal λ₂ and λ₃
        modified_lambda[case_3_mask] = sorted_vals

    return modified_lambda



import torch

def scale_with_random_negatives(image, prob_range=(0.5, 0.6)):
    """
    Scale an image from [0, 1] to [a, b] and randomly introduce negative values 
    with a randomly chosen probability.
    
    Parameters:
    - image (torch.Tensor): Input tensor with values in range [0, 1].
    - a (float): Lower bound of the desired positive range (positive number).
    - b (float): Upper bound of the desired positive range (positive number).
    - prob_range (tuple): Range of probabilities for flipping values to negative (min, max).
    
    Returns:
    - torch.Tensor: Scaled tensor with both positive and negative values.
    """
    
    # Randomly choose a probability for negatives within the specified range
    negative_prob = torch.rand(1).item() * (prob_range[1] - prob_range[0]) + prob_range[0]
    
    # Generate random probabilities for each value in the image
    random_probs = torch.rand_like(image)  # Random values between 0 and 1, same shape as image
    
    # Create a mask for values to be made negative
    negative_mask = random_probs < negative_prob  # Boolean mask for negatives
    
    # Flip selected values to negative
    image[negative_mask] *= -1  # Apply mask to flip values
    
    return image


def gen_perlin( w=None, h=None,  batchsize=None, lims=None, lacunarity=None,  device=None):

    #resolutions = [(2**i,2**i) for i in range(1,3)] 
    #r=resolutions[torch.randint(0, 2, (1,),device=device).item()]

    #o=torch.randint(1, 4, (1,),device=device).item()
    #persistence= 0.7 * (torch.rand(1,device=device) + 0.2)
    while(True):
        resolutions = [(2**i,2**i) for i in range(0,4)] 
            
        r=resolutions[torch.randint(2, 4, (1,),device=device).item()] 
        o=torch.randint(2, 4, (1,),device=device).item()
            
        o=4; 
        persistence=0.5; 
        r=(4, 4)
        #lacunarity=4    
        image=batch_rand_perlin_2d_octaves(lac=lacunarity, b=(batchsize), shape=(w, h), res=r, octaves=o, persistence=persistence, device=device) 
        image= 2*(image - torch.min(image))/(torch.max(image) - torch.min(image))-1 
        image=map_to_range(image,  lims.min(),  lims.max())
        # Apply a sinusoidal transformation to introduce negative values
        #image = torch.cos(2*math.pi*image/lims.max() ) * (lims.max()  - lims.min())
        #print(image.max(), image.min(), lims.max(), lims.min())
        #exit()    
            
        yield image    
   
def ensure_spd_tensor(tissue_type=None,   batchsize=None, width=None, height=None, device=None):
        
        epsilon = 1e-30
        #D =torch.zeros(height, width, batchsize, 3, 3, device) +epsilon
                               
        if tissue_type=='gm' :
           
            #ivimGM(0.0005, 0.002), (-0.0006, 0.0044)
    
            gmrange=torch.tensor([0.5* 10**(-3), 2.0* 10**(-3)], device=device)
            lambdas_gm=torch.zeros(batchsize, width, height, 3, device=device)
        
            for i in range(3):
              lambdas_gm[...,i]=next(gen_perlin(width, height, batchsize, lacunarity=2,  lims=gmrange , device=device))
            
            lambdas_gm=lambdas_gm.permute(1,2,0,3)
            #lambdas_gm[...,0]=lambdas_gm[...,1]=lambdas_gm[...,2]
            #print(lambdas_gm[10,70,10,0], lambdas_gm[10,70,10,1], lambdas_gm[10,70,10,2])
            #fa=calculate_fa_tensor(evals=lambdas_gm)
            #print(fa.shape, fa[10,70,10], fa.min(), fa.max(), fa[fa<0.2].sum())
            #exit()
            
            evecs=generate_random_rots_uniform(w=width, h=height, batch_size=batchsize, device=device)
            D=evecs@torch.diag_embed(lambdas_gm)@evecs.transpose(-1,-2)
                    
        
        if tissue_type=='csf':
            
            #ivimCSF(0.001, 0.003),  (-0.0003,0.0046)
    
            csfrange=torch.tensor([1.0* 10**(-3), 3.0* 10**(-3)], device=device)
            lambdas_csf=torch.zeros(batchsize, width, height, 3, device=device)
        
            for i in range(3):
                lambdas_csf[...,i]=next(gen_perlin(width, height, batchsize, lacunarity=2,  lims=csfrange , device=device))
            
            lambdas_csf=lambdas_csf.permute(1,2,0,3)
            lambdas_csf[...,0]=lambdas_csf[...,1]=lambdas_csf[...,2]
            #evecs=generate_random_rots_uniform(w=width, h=height, batch_size=batchsize, device=device)
            #D=evecs@torch.diag_embed(lambdas_csf)@evecs.transpose(-1,-2)
            D=torch.diag_embed(lambdas_csf)
            
        elif(tissue_type=='wm'):
            
            #ivimWM(0.0003, 0.0013),(-0.0010, 0.0031)
            
            wmrange=torch.tensor([0.3* 10**(-3), 1.3* 10**(-3)], device=device)
            #wmrange=torch.tensor([0.3* 10**(-3), 3.0* 10**(-3)], device=device)
            
            #lambdas_wm=torch.zeros(batchsize, width, height, 3, device=device)
            D_components=torch.zeros(batchsize, width, height, 6, device=device)
            D =torch.zeros(height, width, batchsize, 3, 3, device=device) +epsilon
        
            for i in range(6):
               D_components[...,i]=next(gen_perlin(width, height, batchsize, lacunarity=2,  lims=wmrange , device=device))
        
            D_components=D_components.permute(1,2,0,3)
        
            """ lambdas_wm=lambdas_wm.permute(1,2,0,3)
            
            lambdas_wm=torch.sort(lambdas_wm, descending=True).values

            #lambdas_wm[...,1:2]=lambdas_wm[...,2:3]
            evecs=generate_random_rots_uniform(w=width, h=height, batch_size=batchsize, device=device)
            
            lambdas_wm=torch.sort(lambdas_wm, descending=True).values
            """
            
            D[..., 0, 0] =  D_components[..., 0]   # D11
            D[..., 1, 1] =  D_components[..., 1]   # D22
            D[..., 2, 2] =  D_components[..., 2]   # D33
            
            D[..., 0, 1] =  D_components[..., 3]
            D[..., 0, 2] =  D_components[..., 4]
            D[..., 1, 2] =  D_components[..., 5]
            
            D[..., 1, 0] =  D_components[..., 3]
            D[..., 2, 0] =  D_components[..., 4]
            D[..., 2, 1] =  D_components[..., 5]
            
            D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
            
            eigvecs=generate_random_rots_uniform(w=width, h=height, batch_size=batchsize, device=device)
            D=eigvecs@D@eigvecs.transpose(-1,-2) 
            
            D[..., 0, 0] = torch.abs(D[..., 0, 0] )   # D11
            D[..., 1, 1] = torch.abs(D[..., 1, 1] )   # D22
            D[..., 2, 2] = torch.abs(D[..., 2, 2] )   # D33
            D[..., 0, 1] =  0
            D[..., 0, 2] =  0
            D[..., 1, 2] =  0
            D[..., 1, 0] = D[..., 1, 0] 
            D[..., 2, 0] = D[..., 2, 0] 
            D[..., 2, 1] = D[..., 2, 1] 
            
            D2=D@D.transpose(-1,-2)
            
            evals, evecs = torch.linalg.eigh(D2)
            evals=torch.nan_to_num(evals, nan=epsilon, posinf=epsilon, neginf=epsilon) +epsilon
            evals=torch.sqrt(evals)
            evals=torch.sort(evals, descending=True).values
            #evals[...,1:2]=evals[...,2:3]
            #evecs=generate_random_rots_uniform(w=width, h=height, batch_size=batchsize, device=device)
            D=evecs@torch.diag_embed(evals)@evecs.transpose(-1,-2)
            
            
            
        return D + epsilon



def ensure_spd_tensor2(D_components=None, tissue_type=None):
        
        epsilon = 1e-30
        batch_size, height, width, _ = D_components.shape
        # Adjust the ranges to account for square root!!
        D =torch.zeros(height, width, batch_size, 3, 3, device=D_components.device) +epsilon
        D_components=D_components.permute(1,2,0,3)
        
        #D_components=scale_with_random_negatives(D_components, prob_range=(0.1, 0.9))
        if torch.isnan(D_components).any():
            print("Warning: D_components contains NaN values")
            D_components = torch.nan_to_num(D_components, nan=epsilon, posinf=epsilon, neginf=epsilon)  # Replace NaNs with 0

        D[..., 0, 0] =  D_components[..., 0]   # D11
        D[..., 1, 1] =  D_components[..., 1]   # D22
        D[..., 2, 2] =  D_components[..., 2]   # D33
        
        D[..., 0, 1] =  D_components[..., 3]
        D[..., 0, 2] =  D_components[..., 4]
        D[..., 1, 2] =  D_components[..., 5]
        
        D[..., 1, 0] =  D_components[..., 3]
        D[..., 2, 0] =  D_components[..., 4]
        D[..., 2, 1] =  D_components[..., 5]
        
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon) +epsilon
        
        eigvecs=generate_random_rots_uniform(w=width, h=height, batch_size=batch_size, device=D_components.device)
        D=eigvecs@D@eigvecs.transpose(-1,-2) 
        
        D[..., 0, 0] = torch.abs(D[..., 0, 0] )   # D11
        D[..., 1, 1] = torch.abs(D[..., 1, 1] )   # D22
        D[..., 2, 2] = torch.abs(D[..., 2, 2] )   # D33
        D[..., 0, 1] =  0
        D[..., 0, 2] =  0
        D[..., 1, 2] =  0
        D[..., 1, 0] = D[..., 1, 0] 
        D[..., 2, 0] = D[..., 2, 0] 
        D[..., 2, 1] = D[..., 2, 1] 
        
        D2=D@D.transpose(-1,-2)
        
        
        evals, evecs = torch.linalg.eigh(D2)
        evals=torch.nan_to_num(evals, nan=epsilon, posinf=epsilon, neginf=epsilon) +epsilon
        evals=torch.sqrt(evals)
        #eigvecs=generate_random_rots_uniform(w=width, h=height, batch_size=batch_size, device=D_components.device)
        #D=evecs@torch.diag_embed(evals)@evecs.transpose(-1,-2) 
        #print("WM", D.min(), D.max())
        
        n=torch.rand(1)
                               
        if tissue_type=='gm' :
            
            #evecs=generate_random_rots_uniform(w=width, h=height, batch_size=batch_size, device=D_components.device)
            #evals=map_to_range(evals, D_components.min(), D_components.max())
            
            evals[...,0]=(evals[...,0]+evals[...,1]+evals[...,2]) /3
            evals[...,2]=evals[...,1]=evals[...,0]
            ind=torch.randint(0,3, (1,))
            #evals[...,0]=evals[...,ind.item()]
            #evals[...,1]=evals[...,ind.item()]
            #evals[...,2]=evals[...,ind.item()]
            
            D=evecs@torch.diag_embed(evals)@evecs.transpose(-1,-2) 
            #D=torch.diag_embed(evals) 
            #print("GM", D.min(), D.max())
        if tissue_type=='csf':
            
            #evecs=generate_random_rots_uniform(w=width, h=height, batch_size=batch_size, device=D_components.device)
            #D=eigvecs@torch.diag_embed(evals)@eigvecs.transpose(-1,-2)
            #evals=map_to_range(evals, D_components.min(), D_components.max())
            evals[...,0]=(evals[...,0]+evals[...,1]+evals[...,2]) /3
            evals[...,2]=evals[...,1]=evals[...,0]
            ind=torch.randint(0,3, (1,))
            #evals[...,0]=evals[...,ind.item()]
            #evals[...,1]=evals[...,ind.item()]
            #evals[...,2]=evals[...,ind.item()]
            
            #D=torch.diag_embed(evals) 
            D=evecs@torch.diag_embed(evals)@evecs.transpose(-1,-2) 
            #print("CSF", D.min(), D.max())
        
        elif(tissue_type=='wm'):
            
            D =torch.zeros(height, width, batch_size, 3, 3, device=D_components.device) +epsilon
            D_wmrange=torch.tensor([0.3* 10**(-3), 1.3* 10**(-3)], device=D_components.device)
            #D_gmrange=torch.tensor([0.5* 10**(-3), 2.0* 10**(-3)], device=D_components.device)
            D_components=torch.zeros(batch_size, width, height, 6, device=D_components.device)
        
            for i in range(6):
              D_components[...,i]=next(gen_perlin(height, width, batch_size, lacunarity=2,  lims=D_wmrange , device=D_components.device))
        
            if torch.isnan(D_components).any():
                print("Warning: D_components contains NaN values")
                D_components = torch.nan_to_num(D_components, nan=epsilon, posinf=epsilon, neginf=epsilon)  # Replace NaNs with 0
            
            D_components=D_components.permute(1,2,0,3)
            
            D[..., 0, 0] =  D_components[..., 0]   # D11
            D[..., 1, 1] =  D_components[..., 1]   # D22
            D[..., 2, 2] =  D_components[..., 2]   # D33
            
            D[..., 0, 1] =  D_components[..., 3]
            D[..., 0, 2] =  D_components[..., 4]
            D[..., 1, 2] =  D_components[..., 5]
            
            D[..., 1, 0] =  D_components[..., 3]
            D[..., 2, 0] =  D_components[..., 4]
            D[..., 2, 1] =  D_components[..., 5]
            
            D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon) +epsilon
            
            eigvecs=generate_random_rots_uniform(w=width, h=height, batch_size=batch_size, device=D_components.device)
            D=eigvecs@D@eigvecs.transpose(-1,-2) 
            
            evals, evecs = torch.linalg.eig(D)
            evals=torch.abs(evals)
            
            evals=torch.nan_to_num(evals, nan=epsilon, posinf=epsilon, neginf=epsilon) +epsilon
            #evals[...,2]=evals[...,1]=evals[...,0]
            #rint(evals[10,70,10,0],evals[10,70,10,1],evals[10,70,10,2])
            fa=calculate_fa_tensor(evals=evals)
            #print(fa.shape, fa[10,70,10], fa.min(), fa.max(), fa[fa<0.2].sum())
            #exit()
            D=evecs@torch.diag_embed(evals)@evecs.transpose(-1,-2) 
            #D=torch.diag_embed(evals) 

            #eigvecs=generate_random_rots_uniform(w=width, h=height, batch_size=batch_size, device=D_components.device)
            #D=eigvecs@torch.diag_embed(torch.sqrt(evals))@eigvecs.transpose(-1,-2)
            
            
        return D + epsilon


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


def genbatch(batchsize=None,  gradients=None, b_value=None, device=None, 
             w=None, h=None, snr=None, N=None, masks=None):
     
    epsilon=1e-10      
      
    if masks==None:        
         mask=generate_mask(batchsize=batchsize, w=w, h=h, device=device)
         mask=mask.permute(1,2,0) 
    else:
     
      mask = torch.zeros((batchsize, w, h), device=device)
      masks=masks.permute(2,0,1) 
      #print(masks.shape)
      #exit()     
      for bs in range(batchsize):
      
        bs_rep_vector=torch.tensor(vector(754, total=batchsize))
        #bs_rep_vector=torch.tensor(vector(754, total=batchsize))
        #masks=masks.permute(2,0,1)
        #filter_mask=torch.isin(array, bs_rep_vector)
        mask[bs, :, :]= masks[bs_rep_vector[bs], :, : ]
      mask=mask.permute(1,2,0)
      #print(mask.shape)
      #exit()
        #mask=mask    

    
    #ivimGM(0.0005, 0.002), (-0.0006, 0.0044)
    #ivimWM(0.0003, 0.0013),(-0.0010, 0.0031)
    #ivimCSF(0.001, 0.003),  (-0.0003,0.0046)
    
    
    while(True):
        
        
        D_gm=ensure_spd_tensor(tissue_type='gm',   batchsize=batchsize, width=w, height=h, device=device) # 128x128x32x3x3
        D_wm=ensure_spd_tensor(tissue_type='wm',   batchsize=batchsize, width=w, height=h, device=device) # 128x128x32x3x3
        D_csf=ensure_spd_tensor(tissue_type='csf', batchsize=batchsize, width=w, height=h, device=device) # 128x128x32x3x3
        
        D = torch.zeros_like(D_wm).to(device=device)

        # Set the vectors based on the regions in the mask
        D[mask==3,:,:] = D_wm[mask==3,:,:]  # White matter region 1
        D[mask==2,:,:] = D_gm[mask==2,:,:]  # Gray matter region 2
        D[mask==1,:,:] = D_csf[mask==1,:,:] # Cerebrospinal fluid region 3

        
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
        plt.imshow(D[:,:,10,1,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D[:,:,10,2,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D[:,:,10,0,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D[:,:,10,1,2].cpu())
        plt.colorbar()
        plt.show()
               
        exit()
  """
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
       
     
        S0_wmrange=torch.tensor([0.05, 0.25], device=device)
        S0_gmrange=torch.tensor([0.2, 0.5], device=device)
        S0_csfrange=torch.tensor([0.5, 1.0], device=device)
        
        S0wm_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_wmrange , device=device))
        S0gm_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_gmrange , device=device))
        S0csf_mask=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=S0_csfrange , device=device))
             
        #print(mask.shape, S0csf_mask.shape)       
        
        S0 = torch.zeros(( w, h, batchsize), device=device)
        
        S0[mask==3]=S0wm_mask.permute(1,2,0)[mask==3]
        S0[mask==2]=S0gm_mask.permute(1,2,0)[mask==2]
        S0[mask==1]=S0csf_mask.permute(1,2,0)[mask==1]
        
        #S0 = torch.rand(( w, h, batchsize), device=device) # all random

        #random_indices = np.random.choice(sphere.vertices.shape[0], size=N-1, replace=False)
        
        #selected_gradients = torch.tensor(sphere.vertices[random_indices], device=device) # Fixed 756!
        
        #selected_gradients = sample_unit_semisphere(n_points=N-1, device=device) # Not fixed, over sphere
        selected_gradients = sample_unit_sphere(n_points=N-1, device=device) # Not fixed, over sphere

        #selected_gradients=torch.tensor(grads64).to(device)# Fixed
        selected_gradients=torch.nan_to_num(selected_gradients, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        grads=torch.zeros((N, 3)).to(device)
        grads[1:,:]=selected_gradients
        
        S=gen_dti_signals(b_values=b_value, gradients=grads, D=D, S0=S0,device=device)
        #print(S.shape, S.max(), S.min(), S0.max())
        #exit()
        S=map_to_range(S, 0., 1.)
        
        if snr is not None:
            S=add_rician_noise(S, snr=snr)
         
        #S=map_to_range(S, 0., 1.)
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
        
        yield D, S, grads, mask   


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
    
    #D_gmrange= torch.sqrt(D_gmrange)
    #D_wmrange= torch.sqrt(D_wmrange)
    #D_csfrange= torch.sqrt(D_csfrange)
    # from in vivo DTI data based on FA..
    #D_gmrange=torch.tensor([0.0006, 0.0044], device=device)
    #D_wmrange=torch.tensor([0.0010, 0.0031], device=device)
    #D_csfrange=torch.tensor([0.0003,0.0046], device=device)
    
    #ivimGM(0.0005, 0.002), (-0.0006, 0.0044)
    #ivimWM(0.0003, 0.0013),(-0.0010, 0.0031)
    #ivimCSF(0.001, 0.003),  (-0.0003,0.0046)
    
    
    while(True):
        
        
        Dcomps_wm=torch.zeros(batchsize, w, h, 6, device=device)
        Dcomps_gm=torch.zeros(batchsize, w, h, 6, device=device)
        Dcomps_csf=torch.zeros(batchsize,w, h, 6, device=device)
        
        
        for i in range(6):
            Dcomps_wm[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=D_wmrange , device=device))
            Dcomps_gm[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2,  lims=D_gmrange , device=device))
            Dcomps_csf[...,i]=next(gen_perlin(w, h, batchsize, lacunarity=2, lims=D_csfrange , device=device))

        
        D_gm=ensure_spd_tensor2(D_components=Dcomps_gm, tissue_type='gm') # 128x128x32x3x3
        D_wm=ensure_spd_tensor2(D_components=Dcomps_wm, tissue_type='wm') # 128x128x32x3x3
        D_csf=ensure_spd_tensor2(D_components=Dcomps_csf, tissue_type='csf') # 128x128x32x3x3
        
        D = torch.zeros_like(D_wm).to(device=device)

        # Set the vectors based on the regions in the mask
        D[mask==1,:,:] = D_wm[mask==1,:,:]  # White matter region
        D[mask==2,:,:] = D_gm[mask==2,:,:]  # Gray matter region
        D[mask==3,:,:] = D_csf[mask==3,:,:] # Cerebrospinal fluid region

        
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
        plt.imshow(D[:,:,10,1,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D[:,:,10,2,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D[:,:,10,0,1].cpu())
        plt.colorbar()
        plt.show()
        plt.imshow(D[:,:,10,1,2].cpu())
        plt.colorbar()
        plt.show()
               
        exit()
  """
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
       
     
        S0_wmrange=torch.tensor([0.05, 0.25], device=device)
        S0_gmrange=torch.tensor([0.2, 0.5], device=device)
        S0_csfrange=torch.tensor([0.5, 1.0], device=device)
        
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
        
        S=gen_dti_signals(b_values=b_value, gradients=grads, D=D, S0=S0,device=device)
        #print(S.shape, S.max(), S.min(), S0.max())
        #exit()
        S=map_to_range(S, 0., 1.)
        
        if snr is not None:
            S=add_rician_noise(S, snr=snr)
         
        #S=map_to_range(S, 0., 1.)
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon) + epsilon
        
        yield D, S, grads, mask   



#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
     
#d, t= next(genbatch(batchsize=16, device=device, w=240, h=240, b_value=1000))
#eigenvalues = torch.linalg.eigvals(d[100,110,15,:,:]) 
