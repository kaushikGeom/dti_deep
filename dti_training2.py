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
from utils import  map_to_range, gen_dti_signals, DWIDataset2, rotate_data_and_mask, DTINet, dti_signal_estimate
#import geomstats.geometry.spd_matrices as spd
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batchsize=4
epochs=500

#path = r"individual_38_subjects"
path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Subjects_Data"

data_all = torch.load(os.path.join(path, "data_all.pt"))     # [38, W, H, Z, G]
mask_all = torch.load(os.path.join(path, "mask_all.pt"))     # [38, W, H, Z]
bvec_all = torch.load(os.path.join(path, "mask_all.pt"))    # [38, 3, G]

ds = DWIDataset2(path,  num_slices=batchsize, device=device)
#dl = DataLoader(ds, batch_size=1, shuffle=True)  

#xb, mask, bvec = ds[0]
#grads=torch.rand(30, 3).to(device)
bvals=1000; N=61
model = DTINet().to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr=0.001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

criterion = torch.nn.MSELoss()
#criterion = LogEuclideanLoss()  

psnr=PeakSignalNoiseRatio().to(device)
                                

modelname="testinvivo" 
modelboard=f"runs_ivim/" + modelname
writer = SummaryWriter(modelboard)
model_dir="models_ivim/"
modelpath = os.path.join(model_dir, modelname + '.pt')


if not os.path.exists(model_dir):
# Create a new directory if  does not exist
 os.makedirs(model_dir)
 print("The new directory is created!")

epsilon=1e-10

for epoch in tqdm(range(epochs)): 
    print(f"\nEpoch: {epoch+1}\n" + "-"*30)
    ds = DWIDataset2(path, num_subjects=36, num_slices=batchsize, device=device)
# Training Phase
    train_loss = 0
    model.train()

    for i in tqdm(range(len(ds))):
        # Move data to GPU/CPU
        xb, mb, bv= ds[i]
        xb, mb, bv = xb.to(device).squeeze(), mb.to(device), bv.to(device).squeeze()
        #print("train..")
        # Normalize input signals
        S_gt = xb / (xb.max() + epsilon) 

        # Forward pass
        D_est = model(S_gt)  # Predict diffusion tensor
        S_est = dti_signal_estimate(D_est, bv, bvals)  

        # Handle NaNs/Infs
        S_est = torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
        S_gt = torch.nan_to_num(S_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)

        # Compute loss
        optimizer.zero_grad()
        loss = criterion(S_est, S_gt) 
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # Adjust learning rate
    scheduler.step(train_loss)
    
    # Compute average loss
    train_loss /= len(ds)
    writer.add_scalar('Training_Loss', train_loss, epoch+1)
    print(f"Train Loss: {train_loss:.5f}")

    # Validation Phase
    val_loss = 0
    psnr_value = 0
    str_sym = 0
 
    #ds = DWIDataset(path, num_subjects=36, num_slices=batchsize, device=device)
 
 
    model.eval()

    with torch.inference_mode(): 
        for i in tqdm(range(len(ds))):
     
            xb, mb, bv= ds[i]
    
            xb, mb, bv = xb.to(device).squeeze(), mb.to(device), bv.to(device).squeeze()
        
            # Normalize input
            S_gt = xb / (xb.max() + epsilon)

            # Forward pass
            D_est = model(S_gt)
            S_est = dti_signal_estimate(D_est, bv, bvals)
            
            # Handle NaNs/Infs
            S_est = torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            S_gt = torch.nan_to_num(S_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)

            # Compute validation loss
            vloss = criterion(S_est, S_gt)
            val_loss += vloss.item()

            # Compute PSNR
            psnr_value += psnr(S_est, S_gt)

        # Average metrics
        val_loss /= len(ds)
        psnr_value /= len(ds)

        # Save model periodically
        #if epoch % 10 == 9:
        torch.save(model.state_dict(), modelpath)

        writer.add_scalar('Validation_Loss', val_loss, epoch+1)
        writer.add_scalar('PSNR', psnr_value, epoch+1)

        # Print results
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: LR: {current_lr:.6f}")
        print(f"Val Loss: {val_loss:.9f} | PSNR: {psnr_value:.5f}\n")

