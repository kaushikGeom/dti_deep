import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from tqdm import tqdm    
import torch
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from utils import  scale_to_match_eigvals_per_channel,  compute_FA, gen_synth2, LogEuclideanMSELoss
from utils import  fit_dti_tensor, predict_dti_signal, DtiSynth2D_1, DtiSynth2D_3, DtiSynth2D_3_1

#import geomstats.geometry.spd_matrices as spd
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


from torchmetrics import StructuralSimilarityIndexMeasure as ssim
from tensorboardX import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

no_of_batches=4
#125, 154, 145, 91 (The in vivo data has this shape)
h, w, batchsize =128, 128, 16 # to keep the dimension same
epochs=5000
snr=80; #20 to 60 for dti

N=91, 
b_value=1000
bvals=b_value*torch.ones(90)
bvals = torch.cat([torch.tensor([0.]), bvals]).to(device)
bvecs=torch.load('bvecs.pt').to(device)

wm_eigvals=torch.load("wm_eigs.pt") #np.percentile(csf_eigvals, 99.99) ==> 0.003639444402103531
gm_eigvals=torch.load("gm_eigs.pt")
csf_eigvals=torch.load("csf_eigs.pt")
    

model = DtiSynth2D_1(91, 1000).to(device) #To test
#model = DtiSynth2D_3(91, 1000).to(device) #To test
#model = DtiSynth2D_3_1(91, 1000).to(device) #To test

from torch.optim.lr_scheduler import ReduceLROnPlateau
#lr=0.0005
lr=0.001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True)

criterion_mse = torch.nn.MSELoss() 
criterion_le = LogEuclideanMSELoss()

psnr=PeakSignalNoiseRatio().to(device)
                                
#ssim=ssim().to(device=device)

#np.percentile(eig
# vals_all,99.99) , 0.0035784252783395984

#modelname=f"2test_{snr}perlin_cnn2d_3_eig_perlin_loss=s+fa" 
modelname=f"2test_{snr}perlin_cnn2d_1_eig_perlin_loss=s+fa" 
modelname=f"2test_{snr}perlin_cnn2d_3_eig_perlin_loss=s+fa" 


modelboard=f"runs_ivim/" + modelname
writer = SummaryWriter(modelboard)
model_dir="models_ivim/"
modelpath = os.path.join(model_dir, modelname + '.pt')

eval_max=np.percentile(csf_eigvals, 99.99)
eval_min=np.percentile(wm_eigvals, 0.01)

#eigvals_all=torch.load("invivo_eig_dist")
#eval_max=np.percentile(eigvals_all, 99.99)
#eval_min=np.percentile(eigvals_all, 0.01)
    

if not os.path.exists(model_dir):
# Create a new directory if  does not exist
 os.makedirs(model_dir)
 print("The new directory is created!")

epsilon=1e-10

eigvals_all=torch.load("invivo_eig_dist")
    
for epoch in tqdm(range(epochs)): 
    
        print(f" Epoch: {epoch+1}\n---------")
        train_loss= 0
        model.train(True)
        #t1=time.time()
        for batch in tqdm(range(no_of_batches)):

                D_gt, S_gt, grads, _ = next(gen_synth2(batchsize=batchsize, bvals=bvals, bvecs=bvecs,  
                                            device=device, w=w, h=h, N=N, mask=None, snr=snr)) 
                
                optimizer.zero_grad()
                D_est =model(S_gt.permute(2, 3, 0, 1), bvecs.T).permute(1,2,0,3,4)
                #D_est= map_to_range(D_est, D_gt.min(), D_gt.max()) 
                 
                eigvals_est, eigvecs_est = torch.linalg.eigh(D_est)
                eigvals_gt, eigvecs_gt = torch.linalg.eigh(D_gt)
                

                #eigvals_est= map_to_range(eigvals_est, eigvals_all.min(), 0.0036) 
                #eigvals_gt= map_to_range(eigvals_gt, eigvals_all.min(),0.0036) 
                
                eigvals_est= scale_to_match_eigvals_per_channel(eigvals_est, eigvals_gt) 
                
                eigvals_gt, _=torch.sort(eigvals_gt, dim=-1, descending=True)
                eigvals_est, _=torch.sort(eigvals_est, dim=-1, descending=True)
                
                
                D_est= eigvecs_est @ torch.diag_embed(eigvals_est) @ eigvecs_est.transpose(-2, -1)
                D_gt= eigvecs_gt @ torch.diag_embed(eigvals_gt) @ eigvecs_gt.transpose(-2, -1)
             
                
                D_est = torch.nan_to_num(D_est,nan=0.0, posinf=0.0, neginf=0.0).clone()
                D_gt = torch.nan_to_num(D_gt,nan=0.0, posinf=0.0, neginf=0.0).clone()
                
                FA_est, _ =compute_FA(D_est)
                FA_gt, _=compute_FA(D_gt)

                S_est=predict_dti_signal(D_est, bvals=bvals, bvecs=bvecs, S0=S_gt[..., 0:1])

                S_est = torch.nan_to_num(S_est,nan=0.0, posinf=0.0, neginf=0.0).clone()
                S_gt = torch.nan_to_num(S_gt,nan=0.0, posinf=0.0, neginf=0.0).clone()
            
                #loss = 0.5*criterion_mse(D_est, D_gt) + 0.5*criterion_mse(S_est, S_gt)
                #loss = criterion_mse(D_est, D_gt)
                #loss = criterion_le(D_est, D_gt) 
                FA_est_expanded = FA_est.unsqueeze(-1)  # Now shape: [B, W, H, 1]
                est_concat = torch.cat([S_est, eigvals_est, FA_est_expanded], dim=-1)  # Shape: [B, W, H, 92]
                
                FA_gt_expanded = FA_gt.unsqueeze(-1)  # Now shape: [B, W, H, 1]
                gt_concat = torch.cat([S_gt, eigvals_gt, FA_gt_expanded], dim=-1)  # Shape: [B, W, H, 92]

                loss=criterion_mse(est_concat, gt_concat) 
                
                #loss=criterion_mse(S_est, S_gt)
                
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
        
        scheduler.step(train_loss)
        # Calculate loss and accuracy per epoch 
        train_loss /= no_of_batches
        writer.add_scalar('Training_Loss', train_loss, epoch+1)
        print(f"Train loss: {train_loss:.5} ")
        
        val_loss=0; psnr_value=0; str_sym=0
        psnr_value1=0
        model.eval() # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode(): 
            
            for batch in tqdm(range(no_of_batches)):
        
                D_gt_val, S_gt_val, grads,_ = next(gen_synth2(batchsize=batchsize, bvals=bvals, bvecs=bvecs,  
                                            device=device, w=w, h=h, N=N, mask=None, snr=snr)) 
                
                
                D_est_val =model(S_gt_val.permute(2, 3, 0, 1), bvecs.T).permute(1,2,0,3,4)

                eigvals_est_val, eigvecs_est_val = torch.linalg.eigh(D_est_val)
                eigvals_gt_val, eigvecs_gt_val = torch.linalg.eigh(D_gt_val)
                
                
                #eigvals_est_val= map_to_range(eigvals_est_val, eigvals_gt_val.min(), eigvals_gt_val.max()) 
                #eigvals_est_val= map_to_range(eigvals_est_val,  eigvals_all.min(), 0.0036) 
                #eigvals_gt_val= map_to_range(eigvals_gt_val,  eigvals_all.min(), 0.0036) 
                
                eigvals_est_val=scale_to_match_eigvals_per_channel(eigvals_est_val, eigvals_gt_val) 
                
                eigvals_gt_val, _=torch.sort(eigvals_gt_val, dim=-1, descending=True)
                eigvals_est_val, _=torch.sort(eigvals_est_val, dim=-1, descending=True)
                
                
                D_est_val= eigvecs_est_val @ torch.diag_embed(eigvals_est_val) @ eigvecs_est_val.transpose(-2, -1)
                D_gt_val= eigvecs_gt_val @ torch.diag_embed(eigvals_gt_val) @ eigvecs_gt_val.transpose(-2, -1)
                 
                D_est_val = torch.nan_to_num(D_est_val,nan=0.0, posinf=0.0, neginf=0.0).clone()
                D_gt_val = torch.nan_to_num(D_gt_val,nan=0.0, posinf=0.0, neginf=0.0).clone()
                
                FA_est_val, _=compute_FA(D_est_val)
                FA_gt_val, _=compute_FA(D_gt_val)

                S_est_val=predict_dti_signal(D_est_val, bvals=bvals, bvecs=bvecs, S0=S_gt_val[..., 0:1])

                S_est_val = torch.nan_to_num(S_est_val,nan=0.0, posinf=0.0, neginf=0.0).clone()
                S_gt_val = torch.nan_to_num(S_gt_val,nan=0.0, posinf=0.0, neginf=0.0).clone()
            
                #vloss = 0.5*criterion_mse(D_est_val, D_gt_val) + 0.5*criterion_mse(S_est_val, S_gt_val)
                #vloss = criterion_mse(D_est_val, D_gt_val) 
                #vloss=criterion_mse(S_est_val, S_gt_val)
                
                FA_est_expanded = FA_est_val.unsqueeze(-1)  # Now shape: [B, W, H, 1]
                est_concat = torch.cat([S_est_val ,eigvals_est_val, FA_est_expanded], dim=-1)  # Shape: [B, W, H, 92]
                
                FA_gt_expanded = FA_gt_val.unsqueeze(-1)  # Now shape: [B, W, H, 1]
                gt_concat = torch.cat([S_gt_val, eigvals_gt_val, FA_gt_expanded], dim=-1)  # Shape: [B, W, H, 92]

                vloss=criterion_mse(est_concat, gt_concat) 
                
                #vloss=criterion_mse(S_est_val, S_gt_val) + criterion_mse(FA_est_val, FA_gt_val)
                
                psnr_value= psnr_value + psnr(FA_est_val, FA_gt_val)
                psnr_value1= psnr_value1 + psnr(eigvals_est_val, eigvals_gt_val)
                
                #str_sym= str_sym + ssim(eigvals_est, eigvals_gt)
                #str_sym= str_sym + ssim(eigvals_est.reshape(-1,3), eigvals_gt.reshape(-1,3))
                
                val_loss += vloss.item()
                        
            # Adjust metrics and print out
            scheduler.step(val_loss)
            val_loss /= no_of_batches
            str_sym  /= no_of_batches
            psnr_value /= no_of_batches
            psnr_value1 /= no_of_batches
            
            #if(epoch) % 10 == 9:
            torch.save(model.state_dict(), modelpath)
            writer.add_scalar('Validation_Loss', val_loss, epoch+1)
            writer.add_scalar('PSNR', psnr_value, epoch+1)
            #writer.add_scalar('Str. Symm.', str_sym, epoch+1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}: Learning rate: {current_lr}")
            print(f"Val. loss: {val_loss:.9f} | PSNR FA: {psnr_value:.9} | PSNR Evals.: {psnr_value1:.9}\n")


