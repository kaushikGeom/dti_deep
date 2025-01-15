import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from tqdm import tqdm    
from Nets import DtiEigs, DtiEigsTrans
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from utils import genbatch2, map_to_range, genbatch, genbatch3
#import geomstats.geometry.spd_matrices as spd
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


from torchmetrics import StructuralSimilarityIndexMeasure as ssim
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


batchsize=16#16#20; 
no_of_batches=100#100#70 # 500
N=batchsize*no_of_batches
h=w=96#128; 
epochs=500
snr=100; 


#grads=torch.rand(30, 3).to(device)
b_value=1000; N=65
model = DtiEigs(b_value=b_value, N=N).to(device)
#model = DtiEigsTrans(b_value=b_value, N=N).to(device)

#model = DtiEigs_2(b_value=b_value, N=N).to(device) #To test

from torch.optim.lr_scheduler import ReduceLROnPlateau
lr=0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

criterion = torch.nn.MSELoss()
#criterion = LogEuclideanLoss()  

psnr=PeakSignalNoiseRatio().to(device)
                                
#ssim=ssim().to(device=device)


modelname=f"Eigs_128x128_dirs{N}DTI_bs{batchsize}_bpe{no_of_batches}_snr{snr}_lr_{lr}" #Delta With random rots in wm only

            
modelboard=f"runs_ivim/" + modelname
writer = SummaryWriter(modelboard)
model_dir="models_ivim/"
modelpath = os.path.join(model_dir, modelname + '.pt')

r1, r2=0.3*1e-3, 3.0*1e-3

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

                D_gt, S_gt, grads, _= next(genbatch3(batchsize=batchsize, b_value=b_value,  
                                            device=device, w=w, h=h, N=N, mask=None, snr=snr)) 
                
                eigvals_est=model(S_gt.permute(2,3,0,1)/S_gt[...,0:1].permute(2,3,0,1)) 
                #print( S_gt.permute(2,3,0,1).shape, S_gt[...,0:1].permute(2,3,0,1).shape)
                #exit()
                eigvals_est=torch.nan_to_num(eigvals_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
                eigvals_gt, _ = torch.linalg.eigh(D_gt) 
                eigvals_gt=torch.nan_to_num(eigvals_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                #print("hh", eigvals_est.min(), eigvals_est.max(), eigvals_est.shape, eigvals_gt.shape)
                #print("gg", eigvals_gt.min(), eigvals_gt.max())
                #exit()
                
                eigvals_gt=torch.sort(eigvals_gt, dim=-1, descending=True).values
                eigvals_est=torch.sort(eigvals_est, dim=-1, descending=True).values 
                #eigvals_est=map_to_range(eigvals_est, eigvals_gt.min(), eigvals_gt.max())
                
                eigvals_est[...,0]=map_to_range(eigvals_est[...,0] + epsilon, eigvals_gt[...,0].min(), eigvals_gt[...,0].max())
                eigvals_est[...,1]=map_to_range(eigvals_est[...,1] + epsilon, eigvals_gt[...,1].min(), eigvals_gt[...,1].max())
                eigvals_est[...,2]=map_to_range(eigvals_est[...,2] + epsilon, eigvals_gt[...,2].min(), eigvals_gt[...,2].max())
                
                #eigvals_gt[...,0]=map_to_range(eigvals_gt[...,0] + epsilon, r1, r2 )
                #eigvals_gt[...,1]=map_to_range(eigvals_gt[...,1] + epsilon, r1, r2)
                #eigvals_gt[...,2]=map_to_range(eigvals_gt[...,2] + epsilon, r1, r2)
                
                
                loss = (criterion(eigvals_gt[...,0], eigvals_est[...,0]) + \
                               criterion(eigvals_gt[...,1], eigvals_est[...,1]) + \
                               criterion(eigvals_gt[...,2], eigvals_est[...,2]) )
                #S=torch.tensor(S_est, requires_grad=True)
                #S_gt=torch.tensor(S_gt, requires_grad=True)
                #loss = criterion(eigvals_est, eigvals_gt) 
                #loss = criterion(S_est, S_gt) 
                
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
        
                D_gt, S_gt, grads, _= next(genbatch3(batchsize=batchsize, b_value=b_value,  
                                            device=device, w=w, h=h, N=N, mask=None, snr=snr)) 
                
                eigvals_est=model(S_gt.permute(2,3,0,1)/S_gt[...,0:1].permute(2,3,0,1)) 
                #eigvals_est=model(S_gt.permute(2,3,0,1)) 
                
                eigvals_est=torch.nan_to_num(eigvals_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
                eigvals_gt, _ = torch.linalg.eigh(D_gt) 
                eigvals_gt=torch.nan_to_num(eigvals_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                
                eigvals_gt=torch.sort(eigvals_gt, dim=-1, descending=True).values
                eigvals_est=torch.sort(eigvals_est, dim=-1, descending=True).values 
                #eigvals_est=map_to_range(eigvals_est, eigvals_gt.min(), eigvals_gt.max())
                
                eigvals_est[...,0]=map_to_range(eigvals_est[...,0] + epsilon, eigvals_gt[...,0].min(), eigvals_gt[...,0].max())
                eigvals_est[...,1]=map_to_range(eigvals_est[...,1] + epsilon, eigvals_gt[...,1].min(), eigvals_gt[...,1].max())
                eigvals_est[...,2]=map_to_range(eigvals_est[...,2] + epsilon, eigvals_gt[...,2].min(), eigvals_gt[...,2].max())
                
                """ eigvals_gt[...,0]=map_to_range(eigvals_gt[...,0] + epsilon, r1, r2 )
                eigvals_gt[...,1]=map_to_range(eigvals_gt[...,1] + epsilon, r1, r2)
                eigvals_gt[...,2]=map_to_range(eigvals_gt[...,2] + epsilon, r1, r2)
                 """
                vloss = (criterion(eigvals_gt[...,0], eigvals_est[...,0]) + \
                                criterion(eigvals_gt[...,1], eigvals_est[...,1]) + \
                                criterion(eigvals_gt[...,2], eigvals_est[...,2]) )
                #S=torch.tensor(S_est, requires_grad=True)
                #S_gt=torch.tensor(S_gt, requires_grad=True)
                
                #vloss = criterion(S, S_gt)
                #vloss=criterion(eigvals_est, eigvals_gt)
                
                psnr_value= psnr_value + psnr(eigvals_est, eigvals_gt)
                #str_sym= str_sym + ssim(eigvals_est, eigvals_gt)
                #str_sym= str_sym + ssim(eigvals_est.reshape(-1,3), eigvals_gt.reshape(-1,3))
                
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
            print(f"Val. loss: {val_loss:.9f} | PSNR: {psnr_value:.9} | Str.Symm.: {str_sym:.9}\n")


