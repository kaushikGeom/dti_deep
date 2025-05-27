import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)

import torch
import random
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
"""
This scripts extracts 60 b =1000 from the 38 subjects.
All subjects have 90 each for 1000,2000,3000 but two had only 60 for b=1000
so chosen all 38 subjects for one b=0 and 60 (gradient directions) b=1000 values
"""

def group_bvals(bvals, shells=[0, 1000, 2000, 3000], tol=20):
   
    shell_indices = {}

    for shell in shells:
        idx = np.where((bvals >= shell - tol) & (bvals <= shell + tol))[0]
        shell_indices[shell] = idx

    return shell_indices


def load_subject_tensor(subject_path):
    data_nii = nib.load(os.path.join(subject_path, 'data.nii.gz'))
    data_tensor = torch.tensor(data_nii.get_fdata(), dtype=torch.float32)

    mask_nii = nib.load(os.path.join(subject_path, 'nodif_brain_mask.nii.gz'))
    mask_tensor = torch.tensor(mask_nii.get_fdata(), dtype=torch.float32)

    bvals = np.loadtxt(os.path.join(subject_path, 'bvals'))
    bvecs = np.loadtxt(os.path.join(subject_path, 'bvecs'))

    bvals_tensor = torch.tensor(bvals, dtype=torch.float32)
    bvecs_tensor = torch.tensor(bvecs, dtype=torch.float32)

    return data_tensor, mask_tensor, bvals_tensor, bvecs_tensor


base_dir = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data"
path1=r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data\individual_38_subjects"
n_subjects =38
shells = [0, 1000]  # desired b-values
tol = 20

data_list = []
mask_list = []
bvecs_list = []

for i in range(1, n_subjects + 1):
    subj = f"subject_{i}"
    path = os.path.join(base_dir, subj)
    print(f"Loading {subj}...")

    # Load subject tensors
    data, mask, bvals, bvecs = load_subject_tensor(path)

    if i == 1:
        # Only once: choose bval indices from first subject
        bval_groups = group_bvals(bvals.numpy())
        for shell, idx in bval_groups.items():
         print(f"b={shell}: {len(idx)} directions")


        selected_idx = np.concatenate([[bval_groups[0][0]], bval_groups[1000][0:60]])
        selected_idx = np.sort(selected_idx)  # Optional: sort if needed

        bvals_sel = bvals[selected_idx]
        bvecs_sel = bvecs[:, selected_idx]
    #if i==8:
    #continue
    # Select bval directions
    data_sel = data[..., selected_idx]  # [W, H, Z, G']
    #data_list.append(data_sel.unsqueeze(0))  # [1, W, H, Z, G']
    #mask_list.append(mask.unsqueeze(0))
    #bvecs_list.append(bvecs_sel.unsqueeze(0))      # [1, W, H, Z]
    torch.save(torch.tensor(data_sel), os.path.join(path1, f"data{i}.pt"))
    torch.save(torch.tensor(mask), os.path.join(path1, f"mask{i}.pt"))
    torch.save(torch.tensor(bvecs_sel), os.path.join(path1, f"bvecs{i}.pt"))
    
exit()
# Stack all
data_all = torch.cat(data_list, dim=0)  # [N, W, H, Z, G']
mask_all = torch.cat(mask_list, dim=0)  # [N, W, H, Z]
bvecs_all=torch.cat(bvecs_list, dim=0)
# Save
torch.save(data_all, os.path.join(base_dir, "data_all.pt"))
torch.save(mask_all, os.path.join(base_dir, "mask_all.pt"))
torch.save(bvecs_all, os.path.join(base_dir, "bvecs_all.pt"))
print(bvecs_all.shape)
print("✅ All subjects saved in combined tensors.")

path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tensors
data = torch.load(os.path.join(path, "data_all.pt"), device)       # [38, W, H, Z, G]
mask = torch.load(os.path.join(path, "mask_all.pt"), device)       # [38, W, H, Z]
bvecs = torch.load(os.path.join(path, "bvecs_all.pt"), device)   


path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data\individual_38_subjects"

for i in range(38):
    torch.save(data[i], os.path.join(path, f"data{i+1}.pt"), device)
    torch.save(mask[i], os.path.join(path, f"mask_{i+1}.pt"), device)
    torch.save(bvecs[i], os.path.join(path, f"bvecs_{i+1}.pt"), device)
    