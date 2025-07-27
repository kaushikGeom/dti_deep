import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
from tqdm import tqdm    
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
from utils import load_processed_subject

import torch
def basisN(argument, segment_id, total_segments):
    argument = 1 - argument
    argument = argument.view(-1)  # Ensure 1D tensor
    out = torch.zeros_like(argument)
    for i in range(len(argument)):
        arg = argument[i]
        if (arg >= (segment_id - 2) / total_segments) and (arg < (segment_id - 1) / total_segments):
            t = (arg - (segment_id - 2) / total_segments) / (1 / total_segments)
            out[i] = 0.5 * t * t
        elif (arg >= (segment_id - 1) / total_segments) and (arg < segment_id / total_segments):
            t = (arg - (segment_id - 1) / total_segments) / (1 / total_segments)
            out[i] = -t * t + t + 0.5
        elif (arg >= segment_id / total_segments) and (arg < (segment_id + 1) / total_segments):
            t = (arg - segment_id / total_segments) / (1 / total_segments)
            out[i] = 0.5 * (1 - t) * (1 - t)
        else:
            out[i] = 0
    return out

def SimulateDWMRI(fiber_orientation, dwmri_gradient_orientation, device='cuda'):
    fiber = torch.tensor(fiber_orientation[:3], dtype=torch.float32, device=device)
    gradient = torch.tensor(dwmri_gradient_orientation[:3], dtype=torch.float32, device=device)
    fiber = fiber / torch.norm(fiber)
    gradient = gradient / torch.norm(gradient)
    cosine = torch.dot(fiber, gradient)
    coefficients = torch.tensor([0.0672, 0.1521, 0.3091, 0.4859, 0.6146], dtype=torch.float32, device=device)
    num_of_cp = len(coefficients)
    out = torch.tensor(0.0, device=device)
    for k in range(1, num_of_cp + 1):
        out = out + coefficients[k - 1] * basisN(torch.abs(cosine), k, num_of_cp)
    out = out + coefficients[-1] * basisN(torch.abs(cosine), num_of_cp + 1, num_of_cp)
    return out

def generate_signals_with_fibers(GradientOrientations, fiber_orientation1, fiber_orientation2, size=32, device='cuda'):

    S = torch.ones(size, size, 1, len(GradientOrientations), device=device)
    
    for i in range(len(GradientOrientations)):
        for idx in range(size*size):
            x_idx = idx // size
            y_idx = idx % size
            s1 = SimulateDWMRI(fiber1, GradientOrientations[i,:], device)
            s2 = SimulateDWMRI(fiber2, GradientOrientations[i,:], device)
            S[x_idx, y_idx, 0, i] = (s1 + s2) / 2

    return S

base_path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\data_dti"
subject_id = f"sub{33}"
test_data, test_bvals, test_bvecs, test_mask = load_processed_subject(base_path, subject_id)

bv0=torch.tensor([1.,0.,0.])
bvecs91 = torch.cat([bv0.unsqueeze(-1), test_bvecs], dim=-1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_gradients = 91
GradientOrientations = bvecs91.T#torch.randn(num_gradients, 3, device=device)

# Example: single fiber orientation for all voxels
fiber1 = torch.tensor([1., 0., 0.], device=device)
fiber2 = torch.tensor([0., 1., 0.], device=device)
S = generate_signals_with_fibers(GradientOrientations, fiber1, fiber2, size=32, device=device)

# Example: spatially varying fiber orientations (e.g., [32,32,3] tensors)
# fiber1 = torch.randn(32,32,3, device=device)
# fiber2 = torch.randn(32,32,3, device=device)
# S = generate_signals_with_fibers(GradientOrientations, fiber1, fiber2, size=32, device=device)

print(S.shape)  # Output: torch.Size([32, 32, 1, 64])
