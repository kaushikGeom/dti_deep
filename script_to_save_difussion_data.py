import nibabel as nib
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

path=r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\HealthyVolunteers38\103818_3T_Diffusion_preproc\103818\T1w\Diffusion"
import os
import numpy as np
import nibabel as nib
from pathlib import Path

# Base folder where all subject folders are stored
input_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data')
output_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data')
output_base.mkdir(exist_ok=True)

import os
import numpy as np
import nibabel as nib
from pathlib import Path
import shutil

# Define your base paths
input_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data')
output_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data1')
output_base.mkdir(exist_ok=True)
import numpy as np
import nibabel as nib
import shutil
from pathlib import Path

# Paths
import numpy as np
import nibabel as nib
import shutil
from pathlib import Path

# Define input/output paths
import numpy as np
import nibabel as nib
import shutil
from pathlib import Path

# Define input/output paths
input_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data')
output_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data1')
output_base.mkdir(exist_ok=True)

# Subjects to skip
import shutil
from pathlib import Path

# Define base paths
input_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data')
output_base = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data1')
output_base.mkdir(exist_ok=True)

# Subjects to skip
from pathlib import Path

# Base folder containing the existing subfolders
base_folder = Path(r'C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data')

# Get all folders starting with 'sub' and sort them by numeric suffix
subfolders = sorted(
    [f for f in base_folder.iterdir() if f.is_dir() and f.name.startswith('sub')],
    key=lambda x: int(x.name.replace('sub', ''))
)

# Renaming folders to sub1, sub2, ... consecutively
for idx, folder in enumerate(subfolders, start=1):
    new_name = base_folder / f"sub{idx}"
    if folder != new_name:
        folder.rename(new_name)
        print(f"🔁 Renamed {folder.name} → {new_name.name}")
    else:
        print(f"✅ {folder.name} already in correct order")


import os
import shutil

# Set your source and destination paths

import os
import shutil

root_dir = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\HealthyVolunteers38\data_unzipped"
output_dir = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Data_Subjects"

files_to_copy = ['data.nii', 'bvals', 'bvecs', 'nodif_brain_mask.nii']
os.makedirs(output_dir, exist_ok=True)

# Loop through top-level folders like 861456_3T_Diffusion_preproc
for wrapper_folder in os.listdir(root_dir):
    wrapper_path = os.path.join(root_dir, wrapper_folder)
    if not os.path.isdir(wrapper_path):
        continue

    # Inside that, expect a subject folder like 861456
    for subject_folder in os.listdir(wrapper_path):
        subject_path = os.path.join(wrapper_path, subject_folder)
        diffusion_path = os.path.join(subject_path, 'T1w', 'Diffusion')

        if not os.path.isdir(diffusion_path):
            print(f"Diffusion path missing: {diffusion_path}")
            continue

        dest_subject_folder = os.path.join(output_dir, subject_folder)
        os.makedirs(dest_subject_folder, exist_ok=True)

        for file in files_to_copy:
            src_file = os.path.join(diffusion_path, file)

            # Check for .nii.gz if needed
            if not os.path.exists(src_file) and file.endswith('.nii'):
                src_file += '.gz'

            if os.path.exists(src_file):
                shutil.copy(src_file, dest_subject_folder)
                print(f"Copied {os.path.basename(src_file)} for subject {subject_folder}")
            else:
                print(f"Missing {file} for subject {subject_folder}")

import os

data_subjects_dir = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data"

# List all current folder names
subject_folders = [f for f in os.listdir(data_subjects_dir) if os.path.isdir(os.path.join(data_subjects_dir, f))]

# Sort to ensure consistent numbering (optional)
subject_folders.sort()

# Rename each to subject_1, subject_2, etc.
for idx, old_name in enumerate(subject_folders, start=1):
    old_path = os.path.join(data_subjects_dir, old_name)
    new_name = f"subject_{idx}"
    new_path = os.path.join(data_subjects_dir, new_name)

    os.rename(old_path, new_path)
    print(f"Renamed '{old_name}' to '{new_name}'")


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

subject_dir = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data\subject_1"
data, mask, bvals, bvecs = load_subject_data(subject_dir)

print("Data shape:", data.shape)
print("Mask shape:", mask.shape)
print("bvals shape:", bvals.shape)  # [D]
print("bvecs shape:", bvecs.shape)  # [3, D]


# Assuming you've already loaded:
# data, mask, bvals, bvecs = load_subject_data(subject_dir)


import os
import torch
import nibabel as nib
import numpy as np
from glob import glob

# Input folder with all subject folders
input_base = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data"

# Output folder for separate .pt files
output_base = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_data_pt"
os.makedirs(output_base, exist_ok=True)

# Find subject directories like sub1, sub2, ...
subject_dirs = sorted(glob(os.path.join(input_base, "sub*")))

for sub_path in subject_dirs:
    sub_name = os.path.basename(sub_path)
    print(f"Processing {sub_name}...")

    # Load files
    data = torch.tensor(nib.load(os.path.join(sub_path, "data.nii.gz")).get_fdata(), dtype=torch.float32)
    mask = torch.tensor(nib.load(os.path.join(sub_path, "mask.nii.gz")).get_fdata(), dtype=torch.uint8)
    bvals = torch.tensor(np.loadtxt(os.path.join(sub_path, "bvals")), dtype=torch.float32)
    bvecs = torch.tensor(np.loadtxt(os.path.join(sub_path, "bvecs")), dtype=torch.float32)

    # Save each as a separate .pt file
    torch.save(data, os.path.join(output_base, f"{sub_name}_data.pt"))
    torch.save(mask, os.path.join(output_base, f"{sub_name}_mask.pt"))
    torch.save(bvals, os.path.join(output_base, f"{sub_name}_bvals.pt"))
    torch.save(bvecs, os.path.join(output_base, f"{sub_name}_bvecs.pt"))

    print(f"Saved: {sub_name}_*.pt")



import torch
import os
from pathlib import Path
import shutil

import torch
from pathlib import Path

# Linux-style paths
input_dir = Path("/Processed_data_pt/")
output_dir = Path("/data_dti/")
output_dir.mkdir(parents=True, exist_ok=True)

# Get all subject IDs by scanning for *_data.pt files
subject_ids = sorted(set(f.name.split("_")[0] for f in input_dir.glob("*_data.pt")))

for subj in subject_ids:
    print(f"Processing {subj}...")

    # Load subject-specific files
    data = torch.load(input_dir / f"{subj}_data.pt")       # [X, Y, Z, G]
    bvals = torch.load(input_dir / f"{subj}_bvals.pt")     # [G]
    bvecs = torch.load(input_dir / f"{subj}_bvecs.pt")     # [G, 3]
    mask = torch.load(input_dir / f"{subj}_masks.pt")      # [X, Y, Z]

    if bvals.ndim > 1:
        bvals = bvals.squeeze()

    # Calculate S0 (mean of b=0 images)
    S0 = data[..., bvals == 0].mean(dim=-1)  # [X, Y, Z]

    # Select b=1000 shell
    S = data[..., bvals == 1000]  # [X, Y, Z, N]

    # Stack S0 as the first channel
    S0 = S0.unsqueeze(-1)  # [X, Y, Z, 1]
    S_stacked = torch.cat([S0, S], dim=-1)  # [X, Y, Z, N+1]

    # Save processed data
    torch.save(S_stacked, output_dir / f"{subj}_S.pt")
    torch.save(bvals[bvals == 1000], output_dir / f"{subj}_bvals.pt")
    torch.save(bvecs[bvals == 1000], output_dir / f"{subj}_bvecs.pt")
    torch.save(mask, output_dir / f"{subj}_mask.pt")

print("✅ All subjeccd ts processed and saved to /data_dti.")
