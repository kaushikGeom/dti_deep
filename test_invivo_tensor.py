import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from tqdm import tqdm    
from Nets import DtiNet, DtiEigs, DtiEigs_31
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from utils import map_to_range, calculate_ad, calculate_fa, calculate_ad2, \
calculate_md, calculate_rd, display_gradients, lambdas_boxplots,\
metrics_per_pixel_smpe_boxplots

import torch
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
ssim_metric = StructuralSimilarityIndexMeasure()

import nibabel as nib
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table


# Move images to the same device (e.g., CUDA if using GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


epochs = 800# 100
batchsize=64#16#32
snr=100; 


#grads=torch.rand(30, 3).to(device)
b_value=1000; N=65 #7, 15, 31[redo31], 65(batchsize=32)
model = DtiNet(b_value=b_value, N=N)
psnr=PeakSignalNoiseRatio()
                                
#ssim=ssim().to(device=device)


#modelname="2Sopt_ranges=128x128_dirs65DTI_bs16_bpe100_snr100_lr_0.0001"
modelname="rots_Sopt_ranges=128x128_dirs65DTI_bs16_bpe100_snr100_lr_0.0001"

model.load_state_dict(torch.load(os.path.join("models_ivim",  modelname + '.pt'),map_location=torch.device('cpu')))
            
modelpath = os.path.join("models_ivim/", modelname + '.pt')

#mask=torch.load('mask.pt')
mask=None
#print(mask.shape)
#exit()
h=w=128; epsilon=1e-20

path=r"C:\Users\Sumit\.dipy\sherbrooke_3shell"

# Load b-values and b-vectors from files
bvals, bvecs = read_bvals_bvecs(path+ "//HARDI193.bval", path+ "//HARDI193.bvec")

# Create a mask to keep only b-values 0 and 1000
mask_bvals =  (bvals == 0) | (bvals == 1000)
mask_bvecs =  (bvals == 1000)

# Filter the bvals and bvecs using the mask
bvals = bvals[mask_bvals]
bvecs = bvecs[mask_bvals]
#torch.save(bvecs, '64grads.pt')
#bvecs=np.vstack(([0,0,0], bvecs))
bvecs=torch.tensor(bvecs)
# Create a GradientTable object
gtab = gradient_table(bvals, bvecs)

img = nib.load(path+ "//3shells-1000-2000-3500-N193.nii")
# Get the data as a NumPy array
data = img.get_fdata()
mask=data[:,:,:,0]>data[:,:,:,0].mean()

data=data[15:108,15:108,:,:]

# Apply the mask to select corresponding signals
#data=data/data.max()
#mask=data[:,:,:,0]>data[:,:,:,0].mean()
mask=mask[15:108,15:108,:]

data = data[..., mask_bvals]
#mask=mask[:,:,mask_bvals]
data=data*mask[:,:,:, None]
#data[:,:,:,0]=data[:,:,:,0]/data[:,:,:,0].max()
#data=data/data.max()

#print("gg",data.min(),data.max())
#exit()

#print(mask.shape, data.shape, data.max(), data.min())

#data=map_to_range(torch.tensor(data), 0.,1 )

S_invivo=torch.tensor(data).float()
S0=torch.tensor(data[:,:,:,0:1]).float()
S_invivo_norm=(S_invivo )/(S0 + epsilon)
S_invivo_norm=S_invivo_norm[:,:,20:32,:].float()
print(S_invivo_norm.max(), S0.min())
#exit()

#S_invivo_norm=S_invivo[:,:,30:32,:].float() # No norm

S_invivo_norm=map_to_range(S_invivo_norm, 0.,1 )
#S_invivo_norm[S_invivo_norm>1]=1

tenmodel = dti.TensorModel(gtab)
tenfit = tenmodel.fit(S_invivo_norm.numpy())
#eigvals_sqfit=tenfit.evals
#eigvals_sqfit=torch.abs(torch.tensor(eigvals_sqfit))
print(S_invivo.shape)
D_gt=torch.tensor(tenfit.quadratic_form)
mask=mask[:,:,20:32]
#print(mask.shape)
#eigvals_gt,_=torch.linalg.eigh(D_gt)
#eigvals_gt=torch.nan_to_num(eigvals_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
#D_gt = torch.diag_embed(eigvals_gt) 

print("smax",bvecs[0,:])  
D_gt=D_gt*mask[:,:,:,None,None]

S_invivo_norm=S_invivo_norm.permute(2,3,0,1)
print(S_invivo.shape)

model.eval()
with torch.inference_mode(): 
            
                D_est, S =model(S_invivo_norm, gradients=bvecs)
                D_est=torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon) +epsilon
                
                eigvals_gt, _ = torch.linalg.eigh(D_gt) 
                eigvals_gt=torch.nan_to_num(eigvals_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
                eigvals_gt=torch.sort(eigvals_gt, dim=-1, descending=True).values                     
                #print(D_est.shape, D_gt.shape)
                D_est=D_est.permute(1,2,0,3,4)
                print(D_est[10,10,10,:,:])
                print(D_gt[10,10,10,:,:])
                D_est=map_to_range(D_est, D_gt.min(), D_gt.max()) 
                
                eigvals_est, _ = torch.linalg.eigh(D_est)
                #eigvals_est=eigvals_est.permute(1,2,0,3)
                print(eigvals_est.shape, eigvals_gt.shape)
                
                eigvals_est=torch.nan_to_num(eigvals_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
                eigvals_est=torch.sort(eigvals_est, dim=-1, descending=True).values                    
                
                gtrange=eigvals_gt[mask,:]
                print(gtrange[:,0].min(),gtrange[:,0].max() )
                eigvals_est[...,0]=map_to_range(eigvals_est[...,0], gtrange[:,0].min(), gtrange[:,0].max())
                eigvals_est[...,1]=map_to_range(eigvals_est[...,1], gtrange[:,1].min(), gtrange[:,1].max())
                eigvals_est[...,2]=map_to_range(eigvals_est[...,2], gtrange[:,2].min(), gtrange[:,2].max())
                
                eigvals_est=eigvals_est*mask[:,:,:, None]
                eigvals_gt=eigvals_gt*mask[:,:,:, None]
                 
                 

est = eigvals_est.detach().numpy()
fit = eigvals_gt
print(est.min(),est.max())
print(fit.min(),fit.max())

#est=np.sort(est, axis=-1) 
#fit=np.sort(fit, axis=-1)

lambdas_boxplots(fit, est)


md_est, md_fit =calculate_md(est), calculate_md(fit)
#fa_est, fa_fit =calculate_ad2(est), calculate_ad2(fit)
fa_est, fa_fit =calculate_fa(est), calculate_fa(fit)

ad_est, ad_fit =calculate_ad(est), calculate_ad(fit)
rd_est, rd_fit =calculate_rd(est), calculate_rd(fit)

""" metrics_per_pixel_smpe_boxplots(md_fit, md_est, md_fit,
                                fa_fit, fa_est, fa_fit,
                                ad_fit, ad_est, ad_fit,
                                rd_fit, rd_est, rd_fit)
 """
#rd_sim--240, 240,16 should be 16,1 240,240
mask=torch.tensor(mask)
print(md_fit.shape, mask.shape)
#exit()
md_ssim = ssim_metric(torch.tensor(md_fit).permute(2, 0,1)[:, None, :, :],
                      torch.tensor(md_est).permute(2, 0,1)[:, None, :, :])
fa_ssim = ssim_metric(torch.tensor(fa_fit).permute(2, 0,1)[:, None, :, :] ,
                      torch.tensor(fa_est).permute(2, 0,1)[:, None, :, :])
ad_ssim = ssim_metric(torch.tensor(ad_fit).permute(2, 0,1)[:, None, :, :],
                      torch.tensor(ad_est).permute(2, 0,1)[:, None, :, :])
rd_ssim = ssim_metric(torch.tensor(rd_fit).permute(2, 0,1)[:, None, :, :] ,
                      torch.tensor(rd_est).permute(2, 0,1)[:, None, :, :])

k=6

md_est, md_fit =md_est[:,:, k], md_fit[:,:, k]
fa_est, fa_fit =fa_est[:,:, k], fa_fit[:,:, k]
ad_est, ad_fit =ad_est[:,:, k], ad_fit[:,:, k]
rd_est, rd_fit =rd_est[:,:, k], rd_fit[:,:, k]

# Create a figure with 2 rows and 4 columns
fig, axs = plt.subplots(2, 4, figsize=(8, 6))

# Titles for the columns
axs[0, 0].set_title(f"MD(wrt lsqFit), SSIM: {md_ssim:.3f}",fontsize=8)
axs[0, 1].set_title(f"FA, SSIM: {fa_ssim:.3f}",fontsize=8)
axs[0, 2].set_title(f"AD, SSIM: {ad_ssim:.3f}",fontsize=8)
axs[0, 3].set_title(f"RD, SSIM: {rd_ssim:.3f}",fontsize=8)


#axs[1, 0].set_title("MD")
#axs[1, 1].set_title("FA")
#axs[1, 2].set_title("AD")
#axs[1, 3].set_title("RD")

#fig.text(0.01, 0.80, "Ground Truths", va='center', ha='center', fontsize=10, rotation='vertical')
fig.text(0.01, 0.75, "Deep-Estimations", va='center', ha='center', fontsize=10, rotation='vertical')
fig.text(0.01, 0.25, "NNlSq Fit", va='center', ha='center', fontsize=10, rotation='vertical')


# Row 1: Estimations
im5 = axs[0, 0].imshow(md_est, cmap='viridis' )
im6 = axs[0, 1].imshow(fa_est, cmap='viridis')
im7 = axs[0, 2].imshow(ad_est, cmap='viridis')
im8 = axs[0, 3].imshow(rd_est, cmap='viridis')

im9  = axs[1, 0].imshow(md_fit, cmap='viridis')
im10 = axs[1, 1].imshow(fa_fit, cmap='viridis')
im11 = axs[1, 2].imshow(ad_fit, cmap='viridis')
im12 = axs[1, 3].imshow(rd_fit, cmap='viridis')

fig.colorbar(im5, ax=axs[0, 0], fraction=0.046, pad=0.01, orientation='horizontal')
fig.colorbar(im6, ax=axs[0, 1], fraction=0.046, pad=0.01, orientation='horizontal')
fig.colorbar(im7, ax=axs[0, 2], fraction=0.046, pad=0.01, orientation='horizontal')
fig.colorbar(im8, ax=axs[0, 3], fraction=0.046, pad=0.01, orientation='horizontal')


fig.colorbar(im9,  ax=axs[1, 0], fraction=0.046, pad=0.01, orientation='horizontal')
fig.colorbar(im10, ax=axs[1, 1], fraction=0.046, pad=0.01, orientation='horizontal')
fig.colorbar(im11, ax=axs[1, 2], fraction=0.046, pad=0.01, orientation='horizontal')
fig.colorbar(im12, ax=axs[1, 3], fraction=0.046, pad=0.01, orientation='horizontal')

# Turn off axis labels and ticks for all subplots
for ax in axs.flat:
    ax.axis('off')

# Adjust layout for better spacing
plt.tight_layout()
plt.savefig("test3.png")
plt.show()


