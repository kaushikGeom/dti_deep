import torch
from pathlib import Path

# Linux-style paths
input_dir = Path("/home/sumit/recons2/dti2/Processed_data_pt")
output_dir = Path("/home/sumit/recons2/dti2/data_dti")
output_dir.mkdir(parents=True, exist_ok=True)
# Get all subject IDs by scanning for *_data.pt files
subject_ids = sorted(set(f.name.split("_")[0] for f in input_dir.glob("*_data.pt")))

for subj in subject_ids:
    print(f"Processing {subj}...")

    # Load subject-specific files
    data = torch.load(input_dir / f"{subj}_data.pt")       # [X, Y, Z, G]
    bvals = torch.load(input_dir / f"{subj}_bvals.pt")     # [G]
    bvecs = torch.load(input_dir / f"{subj}_bvecs.pt")     # [G, 3]
    mask = torch.load(input_dir / f"{subj}_mask.pt")      # [X, Y, Z]

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
    torch.save(S_stacked, output_dir / f"{subj}_data.pt")
    torch.save(bvals[bvals == 1000], output_dir / f"{subj}_bvals.pt")
    torch.save(bvecs[:, bvals == 1000], output_dir / f"{subj}_bvecs.pt")
    torch.save(mask, output_dir / f"{subj}_mask.pt")

print("✅ All subjects processed and saved to /data_dti.")
