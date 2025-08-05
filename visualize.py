import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from utils import generate_random_rots_uniform, \
     prepare_eigenvalue_distributions, sample_eigenvalues, \
     build_rotation_from_dirs, generate_dirs, generate_dirs2

import numpy as np
from dipy.viz import window, actor
from dipy.data import get_sphere
from dipy.reconst.dti import color_fa

import matplotlib.pyplot as plt
import numpy as np
import torch
# pip install perlin-noise
from perlin_noise import PerlinNoise
import numpy as np



wm_eigvals=torch.load("wm_eigs.pt") #np.percentile(csf_eigvals, 99.99) ==> 0.003639444402103531
print(wm_eigvals.min(), np.percentile(wm_eigvals, 99.99))
#exit()
w=h=128
_, wm_dists = prepare_eigenvalue_distributions(
                          wm_eigvals, batch_size=1, w=w, h=h, num_bins=100000)
lambdas_wm = sample_eigenvalues(wm_dists, batch_size=1,  w=w, h=h, keep_fraction=1.0)

lambdas_wm = lambdas_wm.permute(1, 2, 0, 3)
lambdas_wm, _ = torch.sort(lambdas_wm, dim=-1, descending=True)
#lambdas_wm[...,0]=lambdas_wm[...,1]=lambdas_wm[...,2]
#lambdas_wm[...,1]=lambdas_wm[...,2]

alpha=0.7
#lambdas_wm[..., 0] = lambdas_wm[..., 0]*(torch.rand(1) * (1 - alpha) + alpha)
#lambdas_wm[..., 1] = lambdas_wm[..., 1]*(torch.rand(1) * (1 - alpha) + alpha)
#lambdas_wm[..., 2] = lambdas_wm[..., 2]*(torch.rand(1) * (1 - alpha) + alpha)
            
            
L_wm = torch.diag_embed(lambdas_wm)
            
#R_wm = build_rotation_from_dirs(
#                                generate_dirs( batchsize=1, w=w, h=h, sigma_range=(3, 30), visualize=True)).permute(1, 2, 0, 3, 4)
R_wm = build_rotation_from_dirs(
                   generate_dirs2(batchsize=1, w=w, h=h,  visualize=True)).permute(1, 2, 0, 3, 4)


L_wm = R_wm.transpose(-1, -2).cpu() @ L_wm.cpu() @ R_wm.cpu()
#R=generate_random_rots_uniform(w=w, h=h, batch_size=3, device='cpu')
#L_wm = R.cpu() @ L_wm.cpu() @ R.transpose(-1, -2).cpu()
        
print(L_wm.shape) 
data=L_wm[:,:,0,:,:]


# === 2. Compute eigenvalues/vectors ===
evals = np.zeros((w, h, 3))
evecs = np.zeros((w, h, 3, 3))

for i in range(w):
    for j in range(h):
        e_val, e_vec = np.linalg.eigh(data[i, j])
        idx = np.argsort(e_val)[::-1]
        evals[i, j] = e_val[idx]
        evecs[i, j] = e_vec[:, idx]

# === 3. Compute FA and RGB ===
mean_evals = np.mean(evals, axis=2, keepdims=True)
numerator = np.sqrt(1.5) * np.linalg.norm(evals - mean_evals, axis=2)
denominator = np.linalg.norm(evals, axis=2) + 1e-10
fa_sample = numerator / denominator

# Take the principal eigenvector (first column of eigenvector matrix)
# Result shape: (128, 128, 3)
principal_directions = evecs[:, :, :, 0]

# Normalize vectors to unit length (avoid overly bright or dim colors)
norm = np.linalg.norm(principal_directions, axis=2, keepdims=True) + 1e-10
rgb_sample = np.abs(principal_directions / norm)  # take abs to avoid negative RGB

# Expand for 3D (Z=1 slice)
rgb_sample = rgb_sample[:, :, np.newaxis, :]  # shape: (128, 128, 1, 3)

rgb_sample = color_fa(fa_sample, evecs)

# === 4. Expand dimensions to mimic 3D volume ===
evals = evals[:, :, np.newaxis, :]
evecs = evecs[:, :, np.newaxis, :, :]
rgb_sample = rgb_sample[:, :, np.newaxis, :]
rgb_sample = np.ones((128, 128, 1, 3)) * np.array([0.2, 0.4, 0.8])

step = 2  # Change to 2 or 3 for smaller gaps

evals_sub = evals[::step, ::step, :, :]
evecs_sub = evecs[::step, ::step, :, :, :]
rgb_sub   = rgb_sample[::step, ::step, :, :]  # if you're coloring   

# === 5. Visualize ===
sphere = get_sphere(name='repulsion724')
scene = window.Scene()


tensor_actor = actor.tensor_slicer(
    evals_sub,
    evecs_sub,
    scalar_colors=rgb_sub,
    sphere=sphere,
    scale=1.0
)

scene.add(tensor_actor)
window.show(scene)
