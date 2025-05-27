import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
current_dir = os.path.dirname(os.path.abspath(__name__))
sys.path.append(current_dir)
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
from utils import gen_lambdas
from dipy.segment.tissue import TissueClassifierHMRF

from Nets import  DtiCoeffs
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio 
from utils import genbatch, map_to_range, calculate_ad, calculate_fa, calculate_ad2, \
calculate_md, calculate_rd, display_gradients, lambdas_boxplots, eigenvalue_distributions


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
#snr=100; 


b_value=1000; N=65 #7, 15, 31[redo31], 65(batchsize=32)
psnr=PeakSignalNoiseRatio()


model = DtiCoeffs(b_value=b_value, N=N).to(device) #To test

#modelname="2D_Eigs_128x128_dirs65DTI_bs16_bpe20_snr100_lr_0.0001"
modelname="coeffstest100" #test5_snr40, test5_snr60, test5_snr40_60
model.load_state_dict(torch.load(os.path.join("models_ivim",  modelname + '.pt'),map_location=torch.device('cpu')))


#mask=torch.load('mask.pt')
h=w=128; epsilon=1e-10

path=r"C:\Users\Sumit\.dipy\sherbrooke_3shell"
# Load b-values and b-vectors from files
bvals, bvecs = read_bvals_bvecs(path+ "//HARDI193.bval", path+ "//HARDI193.bvec")

# Create a mask to keep only b-values 0 and 1000
mask_bvals =  (bvals == 0) | (bvals == 1000)
#mask_bvecs =  (bvals == 1000)

# Filter the bvals and bvecs using the mask
bvals = bvals[mask_bvals]
bvecs = bvecs[mask_bvals]
#torch.save(bvecs, '64grads.pt')
#bvecs=np.vstack(([0,0,0], bvecs))

# Create a GradientTable object
gtab = gradient_table(bvals, bvecs)

img = nib.load(path+ "//3shells-1000-2000-3500-N193.nii")
# Get the data as a NumPy array
data = img.get_fdata()

data=data[:,:,:,mask_bvals]

mask=data[:,:,:,0]>data[:,:,:,0].mean()

data=data[15:111,15:111,:, :]
mask=mask[15:111,15:111,:]

print(mask.shape, data.shape, data.max(), data.min())

#data=data*mask[...,None]
S_invivo=torch.tensor(data).float() 
#S_invivo=map_to_range(S_invivo, 0.02, 1 )
tenmodel = dti.TensorModel(gtab)
S_invivo=S_invivo/S_invivo[...,0:1]
#print(S_invivo.min(), S_invivo[...,0:1].max())

#S_invivo_numpy=S_invivo.numpy() + 1e-10
tenfit = tenmodel.fit(S_invivo)
eigvals_gt=torch.abs(torch.tensor(tenfit.evals))
D_gt=torch.tensor(tenfit.quadratic_form)
#eigvals_gt, _=torch.linalg.eigh(D_gt)

#eigvals_gt=torch.tensor(tenfit.evals)
eigvals_gt=torch.nan_to_num(eigvals_gt, nan=epsilon, posinf=epsilon, neginf=epsilon)
#D_gt = torch.diag_embed(eigvals_gt) 
mask=torch.tensor(mask)

print(S_invivo.min(), mask.shape)

model.eval()

with torch.inference_mode(): 
            
            D_est, _ =model(S_invivo.permute(2,3,0,1).to(device))

            print("IEst=",D_est.min(), D_est.max())
            #print("IGt=",_gt.min(), eigvals_gt.max())
            #exit()
            D_est=map_to_range(D_est + epsilon, D_gt.min(), D_gt.max())
            
            D_est=torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            eigvals_est, _ = torch.linalg.eigh(D_est) 
                 
            eigvals_est=eigvals_est.clone().detach().cpu()
            eigvals_est=eigvals_est*mask[...,None]
            eigvals_gt=eigvals_gt*mask[...,None]
           
            eigvals_est=torch.abs(eigvals_est)
            eigvals_gt=torch.abs(eigvals_gt)


            eigvals_est[...,0]=map_to_range(eigvals_est[...,0] + epsilon, eigvals_gt[...,0].min(), eigvals_gt[...,0].max())
            eigvals_est[...,1]=map_to_range(eigvals_est[...,1] + epsilon, eigvals_gt[...,1].min(), eigvals_gt[...,1].max())
            eigvals_est[...,2]=map_to_range(eigvals_est[...,2] + epsilon, eigvals_gt[...,2].min(), eigvals_gt[...,2].max())
            
            eigvals_gt, _=torch.sort(eigvals_gt, dim=-1, descending=True)
            eigvals_est, _=torch.sort(eigvals_est, dim=-1, descending=True)
     
                         


#print("Est=",eigvals_est.min(), eigvals_est.max())
#print("Gt=",eigvals_gt.min(), eigvals_gt.max())
#exit()
 

est = eigvals_est
fit = eigvals_gt.detach().numpy()
print(est.shape, fit.shape)
#exit()
print(est.min(),est.max())
print(fit.min(),fit.max())

#est=np.sort(est, axis=-1) 
#fit=np.sort(fit, axis=-1)

lambdas_boxplots(fit, est)


md_est, md_fit =(est[...,0]+ est[...,1]+est[...,2])/3, (fit[...,0]+fit[...,1]+fit[...,2])/3
#fa_est, fa_fit =calculate_ad2(est), calculate_ad2(fit)
fa_est, fa_fit =calculate_fa(est), calculate_fa(fit)

ad_est, ad_fit =est[...,0], fit[...,0]
lambda2_est, lambda2_fit =est[...,1], fit[...,1]
lambda3_est, lambda3_fit =est[...,2], fit[...,2]

rd_est, rd_fit =(est[...,1]+est[...,2])/2, (fit[...,1]+fit[...,2])/2


""" md_est, md_fit =calculate_md(est), calculate_md(fit)
#fa_est, fa_fit =calculate_ad2(est), calculate_ad2(fit)
fa_est, fa_fit =calculate_fa(est), calculate_fa(fit)

ad_est, ad_fit =calculate_ad(est), calculate_ad(fit)
rd_est, rd_fit =calculate_rd(est), calculate_rd(fit)
 """


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

lambda2_ssim = ssim_metric(torch.tensor(lambda2_fit).permute(2, 0,1)[:, None, :, :] ,
                      torch.tensor(lambda2_est).permute(2, 0,1)[:, None, :, :])

lambda3_ssim = ssim_metric(torch.tensor(lambda3_fit).permute(2, 0,1)[:, None, :, :] ,
                      torch.tensor(lambda3_est).permute(2, 0,1)[:, None, :, :])

rd_ssim = ssim_metric(torch.tensor(rd_fit).permute(2, 0,1)[:, None, :, :] ,
                      torch.tensor(rd_est).permute(2, 0,1)[:, None, :, :])

k=30
mask_metrics=mask[:,:,k].numpy()

md_est, md_fit =md_est[:,:, k]*mask_metrics, md_fit[:,:, k]*mask_metrics
fa_est, fa_fit =fa_est[:,:, k]*mask_metrics, fa_fit[:,:, k]*mask_metrics

ad_est, ad_fit =ad_est[:,:, k]*mask_metrics, ad_fit[:,:, k]*mask_metrics
lambda2_est, lambda2_fit =lambda2_est[:,:, k]*mask_metrics, lambda2_fit[:,:, k]*mask_metrics
lambda3_est, lambda3_fit =lambda3_est[:,:, k]*mask_metrics, lambda3_fit[:,:, k]*mask_metrics

#ad_est, ad_fit =ad_est[:,:, k]*mask_metrics, ad_fit[:,:, k]*mask_metrics

rd_est, rd_fit =rd_est[:,:, k]*mask_metrics, rd_fit[:,:, k]*mask_metrics

# Create a figure with 2 rows and 4 columns
fig, axs = plt.subplots(2, 4, figsize=(8, 6))

# Titles for the columns
#axs[0, 0].set_title(f"MD(wrt lsqFit), SSIM: {md_ssim:.3f}",fontsize=8)

axs[0, 0].set_title(f"FA, SSIM: {fa_ssim:.3f}",fontsize=8)
axs[0, 1].set_title(f"AD(EigValMax), SSIM: {ad_ssim:.3f}",fontsize=8)
axs[0, 2].set_title(f"(EigValMid), SSIM: {lambda2_ssim:.3f}",fontsize=8)
axs[0, 3].set_title(f"(EigValMin), SSIM: {lambda3_ssim:.3f}",fontsize=8)

#axs[0, 3].set_title(f"RD, SSIM: {rd_ssim:.3f}",fontsize=8)


#axs[1, 0].set_title("MD")
#axs[1, 1].set_title("FA")
#axs[1, 2].set_title("AD")
#axs[1, 3].set_title("RD")

#fig.text(0.01, 0.80, "Ground Truths", va='center', ha='center', fontsize=10, rotation='vertical')
fig.text(0.01, 0.75, "Deep-Estimations", va='center', ha='center', fontsize=10, rotation='vertical')
fig.text(0.01, 0.25, "NNlSq Fit", va='center', ha='center', fontsize=10, rotation='vertical')


# Row 1: Estimations
#im5 = axs[0, 0].imshow(md_est, cmap='viridis', vmax= 0.90*md_fit.max(), vmin=0)
im5 = axs[0, 0].imshow(fa_est, cmap='viridis', vmax= 0.90*fa_fit.max(), vmin=0)
im6 = axs[0, 1].imshow(ad_est, cmap='viridis', vmax= 0.90*ad_fit.max(), vmin=0)
im7 = axs[0, 2].imshow(lambda2_est, cmap='viridis', vmax= 0.90*lambda2_fit.max(), vmin=0)
im8 = axs[0, 3].imshow(lambda3_est, cmap='viridis', vmax= 0.90*lambda3_fit.max(), vmin=0)

#im7 = axs[0, 3].imshow(rd_est, cmap='viridis', vmax= 0.90*rd_fit.max(), vmin=0)

im9 = axs[1, 0].imshow(fa_fit, cmap='viridis', vmax= 0.90*fa_fit.max(), vmin=0)
im10 = axs[1, 1].imshow(ad_fit, cmap='viridis', vmax= 0.90*ad_fit.max(), vmin=0)
im11 = axs[1, 2].imshow(lambda2_fit, cmap='viridis', vmax= 0.90*lambda2_fit.max(), vmin=0)
im12 = axs[1, 3].imshow(lambda3_fit, cmap='viridis', vmax= 0.90*lambda3_fit.max(), vmin=0)

""" 
im5 = axs[0, 0].imshow(md_est, cmap='viridis')
im6 = axs[0, 1].imshow(fa_est, cmap='viridis')
im7 = axs[0, 2].imshow(ad_est, cmap='viridis')
im8 = axs[0, 3].imshow(rd_est, cmap='viridis')

im9  = axs[1, 0].imshow(md_fit, cmap='viridis')
im10 = axs[1, 1].imshow(fa_fit, cmap='viridis')
im11 = axs[1, 2].imshow(ad_fit, cmap='viridis')
im12 = axs[1, 3].imshow(rd_fit, cmap='viridis')

 """
fig.colorbar(im5, ax=axs[0, 0],  pad=0.01, orientation='horizontal')
fig.colorbar(im6, ax=axs[0, 1],  pad=0.01, orientation='horizontal')
fig.colorbar(im7, ax=axs[0, 2],  pad=0.01, orientation='horizontal')
fig.colorbar(im8, ax=axs[0, 3],  pad=0.01, orientation='horizontal')


fig.colorbar(im9,  ax=axs[1, 0],  pad=0.01, orientation='horizontal')
fig.colorbar(im10, ax=axs[1, 1],  pad=0.01, orientation='horizontal')
fig.colorbar(im11, ax=axs[1, 2],  pad=0.01, orientation='horizontal')
fig.colorbar(im12, ax=axs[1, 3],  pad=0.01, orientation='horizontal')

# Turn off axis labels and ticks for all subplots
for ax in axs.flat:
    ax.axis('off')

# Adjust layout for better spacing
plt.tight_layout()
#plt.savefig("eigen_synth.png")
plt.show()


#bvecs=torch.tensor(bvecs).to(device)
#display_gradients(bvecs)

########################################################################
eigenvalue_distributions(est=eigvals_est.reshape(-1, 3), gt=eigvals_gt.reshape(-1, 3),
                         est_fname="Deep Estimated",  gt_fname="Groundtruth (Invivo)",mask=mask.reshape(-1))

""" exit()

nclass = 3
beta = 0.1
#data=data/data[...,0:1]
hmrf = TissueClassifierHMRF()
initial_segmentation, final_segmentation, PVE = hmrf.classify(data[...,0], nclass, beta)

slice=30
fig = plt.figure()
a = fig.add_subplot(1, 2, 1)
img_ax = np.rot90(final_segmentation[..., slice])
imgplot = plt.imshow(img_ax)
a.axis("off")
a.set_title("Axial")
a = fig.add_subplot(1, 2, 2)
img_cor = np.rot90(final_segmentation[:, 46, :])
imgplot = plt.imshow(img_cor)
a.axis("off")
a.set_title("Coronal")
#plt.savefig("final_seg.png", bbox_inches="tight", pad_inches=0)
#PVE[PVE[..., 0]>0.5]=1
#PVE[PVE[..., 0]<0.001]=3
#PVE[PVE[..., 0]<0.5 and PVE[..., 0]>0.001]=2

fig = plt.figure()
a = fig.add_subplot(1, 3, 1)
img_ax = np.rot90(PVE[..., slice, 0])
imgplot = plt.imshow(img_ax, cmap="gray")
a.axis("off")
a.set_title("CSF")
a = fig.add_subplot(1, 3, 2)
img_cor = np.rot90(PVE[:, :, slice, 1])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis("off")
a.set_title("Gray Matter")
a = fig.add_subplot(1, 3, 3)
img_cor = np.rot90(PVE[:, :, slice, 2])
imgplot = plt.imshow(img_cor, cmap="gray")
a.axis("off")
a.set_title("White Matter")
#plt.savefig("probabilities.png", bbox_inches="tight", pad_inches=0)
plt.show()
 """



D_gt_sims, _, _, _, _,_,_ = next(gen_lambdas(batchsize=60, b_value=b_value,  
                                            device=device, w=96, h=96, N=N, mask=None, snr=None)) 
        
eigvals_gt_sims, _ = torch.linalg.eigh(D_gt_sims) 
eigvals_gt_sims=torch.abs(eigvals_gt_sims)
eigvals_gt_sims, _=torch.sort(eigvals_gt_sims, dim=-1, descending=True)
            
eigenvalue_distributions(est=eigvals_gt_sims.cpu().numpy().reshape(-1, 3),  gt=eigvals_gt.reshape(-1, 3),
                         est_fname="Simulated", gt_fname="In vivo", mask=mask.reshape(-1))                

