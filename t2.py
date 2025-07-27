
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from tqdm import tqdm    
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

from utils import   eigenvalue_distributions , DtiSynth, pad_and_resize_signal,\
  lambdas_boxplots, load_processed_subject, gen_synth

from utils import calculate_ad, calculate_ad2, calculate_fa, calculate_md, calculate_rd, fit_dti_tensor

from tensorboardX import SummaryWriter

import nibabel as nib
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


base_path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\data_dti"
coefficient_labels = ['Dxx', 'Dxy', 'Dxz', 'Dyy', 'Dyz', 'Dzz']

# Initialize lists to store coefficients from all subjects
all_invivo_coeffs = []
all_sim_coeffs = []

for subject_num in range(1, 33):  # subjects 1-32
    subject_id = f"sub{subject_num}"
    test_data, test_bvals, test_bvecs, test_mask = load_processed_subject(base_path, subject_id)
    test_data = torch.nan_to_num(test_data, nan=0, posinf=0, neginf=0)
    a, b, c, d = 10, 145-10, 10, 174-10
    test_data = test_data / test_data.max()
    test_data = test_data[a:b, c:d, :, :]
    test_mask = test_mask[a:b, c:d, :]
    S = torch.tensor(test_data)
    
    # Process bvecs
    test_bvecs = torch.cat([torch.tensor([0.,0.,0.]).unsqueeze(-1), test_bvecs], dim=-1)
    
    # Fit in vivo tensor
    D_invivo = fit_dti_tensor(S, mean_b0=S[...,0:1], bvecs=test_bvecs)
    mask = test_mask.bool()
    # Extract coefficients for in vivo data (masked)
    invivo_coeffs = torch.stack([
        D_invivo[...,0,0][mask],  # Dxx
        D_invivo[...,0,1][mask],  # Dxy
        D_invivo[...,0,2][mask],  # Dxz
        D_invivo[...,1,1][mask],  # Dyy
        D_invivo[...,1,2][mask],  # Dyz
        D_invivo[...,2,2][mask]   # Dzz
    ], dim=1)
    all_invivo_coeffs.append(invivo_coeffs)
    
    # Generate simulated data
    D_sim, S_gt, grads, _ = next(gen_synth(batchsize=32, bvals=1000, bvecs=test_bvecs,
                                        device=None, w=128, h=128, N=91, mask=None, snr=None))
    # Extract coefficients for simulated data (all voxels)
    sim_coeffs = torch.stack([
        D_sim[...,0,0].flatten(),  # Dxx
        D_sim[...,0,1].flatten(),  # Dxy
        D_sim[...,0,2].flatten(),  # Dxz
        D_sim[...,1,1].flatten(),  # Dyy
        D_sim[...,1,2].flatten(),  # Dyz
        D_sim[...,2,2].flatten()   # Dzz
    ], dim=1)
    all_sim_coeffs.append(sim_coeffs)

# Combine all subject data
invivo_tensor = torch.cat(all_invivo_coeffs)
sim_tensor = torch.cat(all_sim_coeffs)

# Calculate min and max for each coefficient
results = {
    'In Vivo': {
        'min': invivo_tensor.min(dim=0).values.numpy(),
        'max': invivo_tensor.max(dim=0).values.numpy()
    },
    'Simulated': {
        'min': sim_tensor.min(dim=0).values.numpy(),
        'max': sim_tensor.max(dim=0).values.numpy()
    }
}

# Print results in table format
print("| Coefficient | In Vivo Min | In Vivo Max | Simulated Min | Simulated Max |")
print("|-------------|-------------|-------------|---------------|---------------|")
for idx, label in enumerate(coefficient_labels):
    row = f"| {label} | {results['In Vivo']['min'][idx]:.4f} | {results['In Vivo']['max'][idx]:.4f} | "
    row += f"{results['Simulated']['min'][idx]:.4f} | {results['Simulated']['max'][idx]:.4f} |"
    print(row)
import torch
import matplotlib.pyplot as plt

# Load tensors
wm_eigvals = torch.load("wm_eigs.pt")  # shape [N_wm, 3]
gm_eigvals = torch.load("gm_eigs.pt")  # shape [N_gm, 3]
csf_eigvals = torch.load("csf_eigs.pt")  # shape [N_csf, 3]

# Convert to numpy
wm = wm_eigvals.cpu().numpy()
gm = gm_eigvals.cpu().numpy()
csf = csf_eigvals.cpu().numpy()

labels = ['λ1', 'λ2', 'λ3']
colors = ['blue', 'green', 'red']

# Plot for each eigenvalue component
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for i in range(3):
    axs[i].hist(wm[:, i], bins=100, alpha=0.6, label='WM', color='blue', density=True)
    axs[i].hist(gm[:, i], bins=100, alpha=0.6, label='GM', color='green', density=True)
    axs[i].hist(csf[:, i], bins=100, alpha=0.6, label='CSF', color='red', density=True)
    axs[i].set_title(f'Distribution of {labels[i]}')
    axs[i].set_xlabel('Eigenvalue')
    axs[i].set_ylabel('Density')
    axs[i].legend()

plt.tight_layout()
plt.show()




