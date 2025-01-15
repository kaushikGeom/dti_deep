import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from tqdm import tqdm    
from Nets import DtiEigs, DtiNet
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from utils import genbatch, map_to_range, calculate_fa_tensor
import torch.nn.functional as F

#from torchmetrics import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

# Initialize the SSIM metric

from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


batchsize=16#16#20; 
no_of_batches=100#100#70 # 500
N=batchsize*no_of_batches
h=w=96#128#240; 
epochs=500

snr=100; 


#grads=torch.rand(30, 3).to(device)
b_value=1000; N=65
model = DtiNet(b_value=b_value, N=N).to(device)

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

criterion = torch.nn.MSELoss()
#criterion = LogEuclideanLoss()  

psnr=PeakSignalNoiseRatio().to(device)
                                
#ssim=ssim().to(device=device)


modelname=f"rots_Sopt_ranges=128x128_dirs{N}DTI_bs{batchsize}_bpe{no_of_batches}_snr{snr}_lr_{lr}"

#model.load_state_dict(torch.load(os.path.join("dti//models_ivim",  modelname + '.pt')))
masks=torch.load('masks_240x240x756.pt')
masks = masks.permute(2, 0, 1).unsqueeze(1)  # Shape: [756, 1, 240, 240]

# Resize to [96, 96] using bilinear interpolation
resized_masks = F.interpolate(masks, size=(96, 96), mode='bilinear', align_corners=False)

# Restore to [96, 96, 756] by permuting back
resized_masks = resized_masks.squeeze(1).permute(1, 2, 0).long()  # Shape: [96, 96, 756]
print(resized_masks.shape) 
""" for i in range(756):
 plt.imshow(resized_masks[:,:,i+200])
 plt.show()
exit()            
 """
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
    
        print(f" Epoch: {epoch+1}\n---------")
        train_loss= 0
        model.train(True)
        #t1=time.time()
        for batch in tqdm(range(no_of_batches)):

                D_gt, S_gt, grads, _= next(genbatch(batchsize=batchsize, b_value=b_value,  
                                            device=device, w=w, h=h, N=N, masks=resized_masks, snr=snr)) 
                #D_est, _ =model(S_gt.permute(2,3,0,1)) 
                D_est, _ =model(S_gt.permute(2,3,0,1)/S_gt[...,0:1].permute(2,3,0,1)) 
                #print(S_gt.shape, D_gt.shape)
                S_gt0=S_gt[...,0:1]
                #print(S_gt[...,0:1].shape, S_gt[...,0:1].permute(2,3,0,1).shape)
                
                D_est=D_est.permute(1,2,0,3,4)
                D_est=map_to_range(D_est, D_gt.min(), D_gt.max())
                #print(S_gt.shape, D_gt.shape, D_est.shape)
                #exit()
                #S_est=torch.exp(-b_value * torch.einsum('ij,...jk,ik->...i', grads, D_est, grads)) 
                exponent1=grads@D_est.reshape(-1,3,3)@grads.T
                S_est=S_gt0*torch.exp(-b_value*exponent1.reshape(D_est.shape[0], D_est.shape[1], D_est.shape[2], grads.shape[0],grads.shape[0]).diagonal(dim1=-2, dim2=-1) )
                
                D_est=torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                D_gt=torch.nan_to_num(D_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                 
                #S_est=map_to_range(S_est, S_gt.min(), S_gt.max())
                
                """ eigvals_est, _ = torch.linalg.eigh(D_est)
                eigvals_gt, _ = torch.linalg.eigh(D_gt)
                eigvals_est=torch.abs(eigvals_est) + epsilon
                eigvals_gt=torch.abs(eigvals_gt)+ epsilon
                 
                eigvals_gt=torch.sort(eigvals_gt, dim=-1, descending=True).values                     
                eigvals_est=torch.sort(eigvals_est, dim=-1, descending=True).values                     
                

                eigvals_est[...,0]=map_to_range(eigvals_est[...,0] + epsilon, eigvals_gt[...,0].min(), eigvals_gt[...,0].max())
                eigvals_est[...,1]=map_to_range(eigvals_est[...,1] + epsilon, eigvals_gt[...,1].min(), eigvals_gt[...,1].max())
                eigvals_est[...,2]=map_to_range(eigvals_est[...,2] + epsilon, eigvals_gt[...,2].min(), eigvals_gt[...,2].max())
                
                fa_est, fa_gt =calculate_fa_tensor(eigvals_est), calculate_fa_tensor(eigvals_gt)

                fa_est=torch.nan_to_num(fa_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                fa_gt=torch.nan_to_num(fa_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                 """
                
                """ loss1 = (criterion(eigvals_gt[...,0], eigvals_est[...,0]) + epsilon + \
                               criterion(eigvals_gt[...,1], eigvals_est[...,1]) + epsilon + \
                               criterion(eigvals_gt[...,2], eigvals_est[...,2]) + epsilon)
                 """
                
                S_gt=torch.nan_to_num(S_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                S_est=torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                
                #loss = criterion(D_est, D_gt) 
                loss = criterion(S_est, S_gt) 
            
                
                train_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        scheduler.step(train_loss)
        # Calculate loss and accuracy per epoch 
        train_loss /= no_of_batches
        writer.add_scalar('Training_Loss', train_loss, epoch+1)
        print(f"Train loss: {train_loss:.5} ")
        
        val_loss=0; psnr_value=0; str_sym=0
        model.eval() # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode(): 
            
            for batch in tqdm(range(no_of_batches)):
        
                D_gt, S_gt, grads, _= next(genbatch(batchsize=batchsize, b_value=b_value,  
                                            device=device, w=w, h=h, N=N, masks=resized_masks, snr=snr)) 
                #D_est, _ =model(S_gt.permute(2,3,0,1)) 
                D_est, _ =model(S_gt.permute(2,3,0,1)/S_gt[...,0:1].permute(2,3,0,1)) 
                
                S_gt0=S_gt[...,0:1]
                D_est=D_est.permute(1,2,0,3,4)
                D_est=map_to_range(D_est, D_gt.min(), D_gt.max())
                
                #S_est=torch.exp(-b_value * torch.einsum('ij,...jk,ik->...i', grads, D_est, grads)) 
                exponent1=grads@D_est.reshape(-1,3,3)@grads.T
                S_est=S_gt0*torch.exp(-b_value*exponent1.reshape(D_est.shape[0], D_est.shape[1], D_est.shape[2], grads.shape[0],grads.shape[0]).diagonal(dim1=-2, dim2=-1) )
                #print(S_gt.shape, S_est.shape, S_gt0.shape)
                D_est=torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                D_gt=torch.nan_to_num(D_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                 
                #S_est=map_to_range(S_est, S_gt.min(), S_gt.max())
                
                """ eigvals_est, _ = torch.linalg.eigh(D_est)
                eigvals_gt, _ = torch.linalg.eigh(D_gt)
                eigvals_est=torch.abs(eigvals_est) + epsilon
                eigvals_gt=torch.abs(eigvals_gt)+ epsilon
                 
                eigvals_gt=torch.sort(eigvals_gt, dim=-1, descending=True).values                     
                eigvals_est=torch.sort(eigvals_est, dim=-1, descending=True).values                     
                

                eigvals_est[...,0]=map_to_range(eigvals_est[...,0] + epsilon, eigvals_gt[...,0].min(), eigvals_gt[...,0].max())
                eigvals_est[...,1]=map_to_range(eigvals_est[...,1] + epsilon, eigvals_gt[...,1].min(), eigvals_gt[...,1].max())
                eigvals_est[...,2]=map_to_range(eigvals_est[...,2] + epsilon, eigvals_gt[...,2].min(), eigvals_gt[...,2].max())
                 
                fa_est, fa_gt =calculate_fa_tensor(eigvals_est), calculate_fa_tensor(eigvals_gt)

                fa_est=torch.nan_to_num(fa_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                fa_gt=torch.nan_to_num(fa_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                 """
                """ loss1 = (criterion(eigvals_gt[...,0], eigvals_est[...,0]) + epsilon + \
                               criterion(eigvals_gt[...,1], eigvals_est[...,1]) + epsilon + \
                               criterion(eigvals_gt[...,2], eigvals_est[...,2]) + epsilon)
                 """
                
                S_gt=torch.nan_to_num(S_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                S_est=torch.nan_to_num(S_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                
                vloss = criterion(S_est, S_gt) 
                #vloss = criterion(D_est, D_gt) 
                 
            
                #psnr_value= psnr_value + psnr(eigvals_est, eigvals_gt)
                
                #psnr_value= psnr_value +  psnr(D_est, D_gt) 
                psnr_value= psnr_value +  psnr(S_est, S_gt) 
                str_sym= str_sym + ssim(S_est, S_gt)
                val_loss += vloss.item()
                        
            # Adjust metrics and print out
            scheduler.step(val_loss)
            val_loss /= no_of_batches
            str_sym  /= no_of_batches
            psnr_value /= no_of_batches
            #if(epoch) % 10 == 9:
            torch.save(model.state_dict(), modelpath)
            writer.add_scalar('Validation_Loss', val_loss, epoch+1)
            writer.add_scalar('PSNR', psnr_value, epoch+1)
            #writer.add_scalar('Str. Symm.', str_sym, epoch+1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Learning rate: {current_lr}")
            print(f"Val. loss: {val_loss:.9f} | PSNR: {psnr_value:.9} | Str.Symm.: {str_sym:.9f}\n")


