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
from utils import  map_to_range,fit_dti_tensor , DWIDataset, DTINet, dti_signal_estimate, load_processed_subject
from utils import compute_FA
#import geomstats.geometry.spd_matrices as spd
from torchmetrics import StructuralSimilarityIndexMeasure as ssim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import random
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_data_pt"
base_path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\data_dti"

subject_id = "sub1"

data, bvals, bvecs, mask = load_processed_subject(base_path, subject_id)

#data, bvals, bvecs, mask = load_processed_subject(Path('Processed_Data/sub1'))
base_path = Path('Processed_data_pt')  # or full absolute path

#xb, mask, bvec = ds[0]
#grads=torch.rand(30, 3).to(device)
batches=1

model = DTINet().to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

#criterion = LogEuclideanLoss()  
#criterion = torch.nn.MSELoss(reduction='none')  # no internal averaging
criterion = torch.nn.MSELoss() 

psnr=PeakSignalNoiseRatio().to(device)
                                
modelname="run" 
modelboard=f"runs_ivim/" + modelname
writer = SummaryWriter(modelboard)
model_dir="models_ivim/"
modelpath = os.path.join(model_dir, modelname + '.pt')


if not os.path.exists(model_dir):
# Create a new directory if  does not exist
 os.makedirs(model_dir)
 print("The new directory is created!")


epsilon = 1e-10
Z=120
b = 20  # batch size
epochs = 1000

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}\n" + "-"*100)
    
    model.train()
    train_loss = 0
    z_start = random.randint(20, Z - b)
    z_indices = list(range(z_start, z_start + b))
    
    for i in tqdm(range(batches), desc="Training"):
        
        train_ds =DWIDataset(base_path=base_path, num_subjects=32, z_indices=z_indices, device='cuda')


        xb, mb, bv, bvals = train_ds.get_sample()

        xb = xb.to(device)
        mb = mb.to(device)
        bv = bv.to(device)

        S0 = xb[..., 0]
        S1000=   xb[..., 1:]
        bvals1000=bvals
        bv1000=bv
        S = xb
        bvals91 = torch.cat([torch.tensor([0.]).to(device), bvals1000], dim=-1)
        
        S_gt = S/S0.max()
        
        D_gt = fit_dti_tensor(dwi=S1000, mean_b0=S0, bvals=bvals1000, bvecs=bv1000)
        D_gt=D_gt*mb[...,None,None]
        FA_gt, eigvals_gt=compute_FA(D_gt)
        """ 
        S = torch.cat([S0.unsqueeze(-1), S1000], dim=-1)
        bv = torch.cat([bv0.unsqueeze(-1), bv1000], dim=-1)
        #S_gt = S/S0.max()
        S = S / (S0[..., None].max() )  # voxel-wise division
        
        S_gt = torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon)
        """
        S_gt = torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        D_est = model(S_gt)
        D_est=D_est*mb[...,None,None]
        
        D_est = torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
        FA_est, eigvals_est=compute_FA(D_est)
       
        #S_est = dti_signal_estimate(D_est, bv, bvals=bvals91)
        #S_est = dti_signal_estimate(D_est, bv,S0, bvals=1000)
        
        #S_est = torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
        #print(S_est.max(), S_gt.max())
        
        #S_est=S_est*mb[...,None]
        #S_est = torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
        #S_est = (S_est - S_est.mean()) / S_est.std()
        #S_est = S_est/S_est.max()
            
        optimizer.zero_grad()
        #mb=mb.bool()
        #mb_exp = mb.unsqueeze(-1).expand_as(S_gt) 
        #loss = criterion(S_est, S_gt)  # [145, 174, 8, 91]
        #loss = criterion(eigvals_est, eigvals_gt)  # [145, 174, 8, 91]
        loss = criterion(D_est, D_gt)  # [145, 174, 8, 91]
        
        #loss =  criterion(FA_est, FA_gt)
        #loss = loss[mb_exp].mean()
            
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batches
    writer.add_scalar('Training_Loss', train_loss, epoch + 1)
    print(f"Train Loss: {train_loss:.5f}")

    # --------------------- Validation ---------------------
    model.eval()
    val_loss, psnr_total = 0, 0

    with torch.inference_mode():
        for i in tqdm(range(batches), desc="Validation"):

            val_ds =DWIDataset(base_path=base_path, num_subjects=32, z_indices=z_indices, device='cuda')

            xb, mb, bv, bvals = val_ds.get_sample()

            xb = xb.to(device)
            mb = mb.to(device)
            bv = bv.to(device)
            
                
            S0 = xb[..., bvals == 0].mean(dim=-1)
            S1000=   xb[..., bvals == 1000]
            bv1000=bv[:, bvals == 1000]
            bv0=bv[:, bvals == 0].mean(dim=-1)
            bvals1000=bvals[bvals==1000]
            S = torch.cat([S0.unsqueeze(-1), S1000], dim=-1)
            bv = torch.cat([bv0.unsqueeze(-1), bv1000], dim=-1)
            S_gt = S/S0.max()
        
            D_gt = fit_dti_tensor(dwi=S1000, mean_b0=S0, bvals=bvals1000, bvecs=bv1000)
            D_gt=D_gt*mb[...,None,None]
            FA_gt, eigvals_gt=compute_FA(D_gt)
            #print(tensor[70,78,10,:,:])
            """ 
            S = torch.cat([S0.unsqueeze(-1), S1000], dim=-1)
            bv = torch.cat([bv0.unsqueeze(-1), bv1000], dim=-1)
            #S_gt = S/S0.max()
            S = S / (S0[..., None].max() )  # voxel-wise division
            
            S_gt = torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon)
            """
            D_est = model(S_gt)
            D_est = torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            D_est=D_est*mb[...,None,None]
            FA_est, eigvals_est=compute_FA(D_est)
            #S_est = dti_signal_estimate(D_est, bv, bvals=bvals91)
            
            """ plt.imshow(S1000[:,:,4, 40].cpu().numpy())
            plt.show()
            plt.imshow(FA_gt[:,:,4].cpu().numpy())
            plt.show()
            plt.imshow(FA_est[:,:,4].detach().cpu().numpy())
            plt.show()
             """
            #S_est = dti_signal_estimate(D_est, bv,S0, bvals=1000)
            #S_est = torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            #print(S_est.max(), S_gt.max())
            
            #S_est=S_est*mb[...,None]
            #S_est = torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            #S_est = (S_est - S_est.mean()) / S_est.std()
            #S_est = S_est/S_est.max()
                 
            optimizer.zero_grad()
            #mb=mb.bool()
            #mb_exp = mb.unsqueeze(-1).expand_as(S_gt) 
            #vloss = criterion(D_est, D_gt)  # [145, 174, 8, 91]
            #vloss = criterion(eigvals_est, eigvals_gt)  # [145, 174, 8, 91]
            
            vloss = criterion(D_est, D_gt)  # [145, 174, 8, 91]
            #vloss = criterion(S_est, S_gt)  # [145, 174, 8, 91]
          
            #vloss =  criterion(FA_est, FA_gt)# [145, 174, 8, 3,3]
            #vloss = vloss[mb_exp].mean()
            val_loss += vloss.item()
            psnr_total = psnr_total + psnr(FA_est.clone() , FA_gt.clone() )
            
            #psnr_total = psnr_total + psnr(D_est.clone() , D_gt.clone() )


    val_loss =val_loss/ batches
    psnr_total = psnr_total / batches


    writer.add_scalar('Validation_Loss', val_loss, epoch + 1)
    writer.add_scalar('PSNR', psnr_total, epoch + 1)

    # Save model every epoch
    #torch.save(model.state_dict(), modelpath)
    torch.save(model, modelpath)
    
    print(f"Val Loss: {val_loss:.7f} | PSNR: {psnr_total:.2f}")
