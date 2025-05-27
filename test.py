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
from skimage.metrics import structural_similarity as ssim

from utils import  map_to_range,eigenvalue_distributions , DTINet, \
                    lambdas_boxplots, load_processed_subject, compute_FA, fit_dti_tensor
from utils import calculate_ad, calculate_ad2, calculate_fa, calculate_md, calculate_rd
from tensorboardX import SummaryWriter

import nibabel as nib
import dipy.reconst.dti as dti
from dipy.io.gradients import read_bvals_bvecs
from dipy.core.gradients import gradient_table

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_Data_pt"

base_path = r"C:\Users\Sumit\Desktop\DWMRI_DATA_KFA_Article\Processed_data_pt"
subject_id = "sub34"

test_data, test_bvals, test_bvecs, test_mask = load_processed_subject(base_path, subject_id)

print(test_data.shape)
print(test_mask.shape)
print(test_bvals.shape)
print(test_bvecs.shape)

S0 = test_data[..., test_bvals == 0].mean(dim=-1)
S1000=   test_data[..., test_bvals == 1000]

bv1000=test_bvecs[:, test_bvals == 1000]
bv0=test_bvecs[:, test_bvals == 0].mean(dim=-1)
test_bvecs = torch.cat([bv0.unsqueeze(-1), bv1000], dim=-1)

test_bvals1000=test_bvals[test_bvals == 1000]
test_bvals = torch.cat([torch.tensor([0.]), test_bvals1000])

S = torch.cat([S0.unsqueeze(-1), S1000], dim=-1)

S=S/S0[...,None]

epsilon=1e-10

model = DTINet().to(device)

criterion = torch.nn.MSELoss()
#criterion = LogEuclideanLoss()  
psnr=PeakSignalNoiseRatio().to(device)
                                

modelname="run" 
modelboard=f"runs_ivim/" + modelname
writer = SummaryWriter(modelboard)
model_dir="models_ivim/"
modelpath = os.path.join(model_dir, modelname + '.pt')


#model.load_state_dict(torch.load(os.path.join("models_ivim",  modelname + '.pt'),map_location=torch.device('cpu')))
model = torch.load(os.path.join("models_ivim",  modelname + '.pt'), weights_only=False)

gtab = gradient_table(test_bvals, test_bvecs)

#S_gt=torch.tensor(test_data).clone() 
#S_invivo=map_to_range(S_invivo, 0.02, 1 )
tenmodel = dti.TensorModel(gtab)

slice=74
S_gt=S[:,:,slice:slice+1,:]
 
test_mask=test_mask[:,:,slice]

#test_mask=test_mask.squeeze().cpu().numpy().astype(bool)
tenfit = tenmodel.fit(S_gt.squeeze(), test_mask)


#D_gt=torch.tensor(tenfit.quadratic_form)
D_gt=tenfit.quadratic_form
plt.imshow(tenfit.fa)
plt.show()


""" D_gt =fit_dti_tensor(dwi=S1000[:,:, slice:slice+1,:], mean_b0=S0[:,:,slice:slice+1], bvals=test_bvals1000, bvecs=bv1000)
FA_gt, eigvals_gt=compute_FA(torch.tensor(D_gt))
plt.imshow(FA_gt.cpu().numpy().squeeze())
plt.show()
 """
eigvals_gt, _ = torch.linalg.eigh(torch.tensor(D_gt)) 
            
#plt.imshow(D_gt[:,:,2,2])
#plt.show()

model.eval()
with torch.inference_mode(): 

            D_est = model(torch.tensor(S_gt).float().to(device))  
            D_est = torch.nan_to_num(D_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            
            D_est=D_est.cpu()*torch.tensor(test_mask)[:,:,None, None, None]
            print(D_est.min(), D_est.max())
            print(D_gt.min(), D_gt.max())
            D_est=map_to_range(D_est, D_gt.min(), D_gt.max())
            print(D_est.min(), D_est.max())
            print(D_gt.min(), D_gt.max())
            

            eigvals_est, _ = torch.linalg.eigh(D_est) 
            eigvals_est = torch.nan_to_num(eigvals_est, nan=epsilon, posinf=epsilon, neginf=epsilon)
            eigvals_est=eigvals_est.abs()
            eigvals_est=eigvals_est.squeeze()*test_mask[...,None]
            
            print("IEst=",eigvals_est.min(), eigvals_est.max())
            print("IGt=",eigvals_gt.min(), eigvals_gt.max())
            eigvals_est=torch.tensor(eigvals_est)                
            eigvals_est[...,0]=map_to_range(eigvals_est[...,0] + epsilon, eigvals_gt[...,0].min(), eigvals_gt[...,0].max())
            eigvals_est[...,1]=map_to_range(eigvals_est[...,1] + epsilon, eigvals_gt[...,1].min(), eigvals_gt[...,1].max())
            eigvals_est[...,2]=map_to_range(eigvals_est[...,2] + epsilon, eigvals_gt[...,2].min(), eigvals_gt[...,2].max())
              
            eigvals_gt, _=torch.sort(eigvals_gt, dim=-1, descending=True)
            eigvals_est, _=torch.sort(eigvals_est, dim=-1, descending=True)

                         


#print("Est=",eigvals_est.min(), eigvals_est.max())
#print("Gt=",eigvals_gt.min(), eigvals_gt.max())
#exit()
 

est = eigvals_est.cpu().squeeze().numpy()
fit = eigvals_gt.detach().numpy().squeeze()
print(est.shape, fit.shape)
#exit()
print(est.min(),est.max())
print(fit.min(),fit.max())

#est=np.sort(est, axis=-1) 
#fit=np.sort(fit, axis=-1)
test_mask=test_mask>0
a,b,c,d=10,145-10,10,174-10
test_mask=test_mask[a:b,c:d]

est=est[a:b,c:d,:]
fit=fit[a:b,c:d,:]


est_l=est[test_mask.squeeze(),:]
fit_l=fit[test_mask.squeeze(),:]

lambdas_boxplots(fit_l, est_l)

#exit()
#md_est, md_fit =(est[...,0]+ est[...,1]+est[...,2])/3, (fit[...,0]+fit[...,1]+fit[...,2])/3
#fa_est, fa_fit =calculate_ad2(est), calculate_ad2(fit)
fa_est, fa_fit =calculate_fa(est), calculate_fa(fit)


ad_est, ad_fit =est[...,0], fit[...,0]
lambda2_est, lambda2_fit =est[...,1], fit[...,1]
lambda3_est, lambda3_fit =est[...,2], fit[...,2]

rd_est, rd_fit =(est[...,1]+est[...,2])/2, (fit[...,1]+fit[...,2])/2

#rd_sim--240, 240,16 should be 16,1 240,240


# Compute SSIM
#md_ssim, _ = ssim(md_fit, md_est, data_range=md_fit.max() - md_fit.min(), full=True)

fa_ssim, _ = ssim(fa_fit, fa_est, data_range=fa_fit.max() - fa_fit.min(), full=True)

ad_ssim, _ = ssim(ad_fit, ad_est, data_range=ad_fit.max() - ad_fit.min(), full=True)

rd_ssim, _ = ssim(rd_fit, rd_est, data_range=rd_fit.max() - rd_fit.min(), full=True)

lambda2_ssim, _ = ssim(lambda2_fit, lambda2_est, data_range=lambda2_fit.max() - lambda2_fit.min(), full=True)

lambda3_ssim, _ = ssim(lambda3_fit, lambda3_est, data_range=lambda3_fit.max() - lambda3_fit.min(), full=True)



# Compute absolute differences
fa_diff       = np.abs(fa_est - fa_fit)
ad_diff       = np.abs(ad_est - ad_fit)
lambda2_diff  = np.abs(lambda2_est - lambda2_fit)
lambda3_diff  = np.abs(lambda3_est - lambda3_fit)

fig, axs = plt.subplots(3, 4, figsize=(12, 9))

# Row 0: Estimations (Deep)
axs[0, 0].imshow(fa_est, cmap='viridis', vmax=np.percentile(fa_fit, 99), vmin=0)
axs[0, 1].imshow(ad_est, cmap='viridis', vmax=np.percentile(ad_fit, 99), vmin=0)
axs[0, 2].imshow(lambda2_est, cmap='viridis', vmax=np.percentile(lambda2_fit, 99), vmin=0)
axs[0, 3].imshow(lambda3_est, cmap='viridis', vmax=np.percentile(lambda3_fit, 99), vmin=0)

# Titles with SSIM
axs[0, 0].set_title(f"FA, SSIM: {fa_ssim:.3f}", fontsize=10)
axs[0, 1].set_title(f"AD (λ₁), SSIM: {ad_ssim:.3f}", fontsize=10)
axs[0, 2].set_title(f"λ₂ (Mid), SSIM: {lambda2_ssim:.3f}", fontsize=10)
axs[0, 3].set_title(f"λ₃ (Min), SSIM: {lambda3_ssim:.3f}", fontsize=10)

# Row 1: Fit (NN-LS)
axs[1, 0].imshow(fa_fit, cmap='viridis', vmax=np.percentile(fa_fit, 99), vmin=0)
axs[1, 1].imshow(ad_fit, cmap='viridis', vmax=np.percentile(ad_fit, 99), vmin=0)
axs[1, 2].imshow(lambda2_fit, cmap='viridis', vmax=np.percentile(lambda2_fit, 99), vmin=0)
axs[1, 3].imshow(lambda3_fit, cmap='viridis', vmax=np.percentile(lambda3_fit, 99), vmin=0)

# Row 2: Absolute Differences
axs[2, 0].imshow(fa_diff, cmap='viridis', vmin=0, vmax=0.1*np.percentile(fa_fit, 99))
axs[2, 1].imshow(ad_diff, cmap='viridis', vmin=0, vmax=0.1*np.percentile(ad_fit, 99))
axs[2, 2].imshow(lambda2_diff, cmap='viridis', vmin=0, vmax=0.1*np.percentile(lambda2_fit, 99))
axs[2, 3].imshow(lambda3_diff, cmap='viridis', vmin=0, vmax=0.1*np.percentile(lambda3_fit, 99))

# Short colorbars under each image
for i in range(3):
    for j in range(4):
        fig.colorbar(axs[i, j].images[0], ax=axs[i, j], orientation='horizontal', pad=0.01, shrink=0.9)

# Row labels
fig.text(0.01, 0.85, "Deep-Estimations", va='center', ha='left', fontsize=10, rotation='vertical')
fig.text(0.01, 0.52, "NN-LS Fit", va='center', ha='left', fontsize=10, rotation='vertical')
fig.text(0.01, 0.18, "Abs. Diff", va='center', ha='left', fontsize=10, rotation='vertical')

# Hide axes
for ax in axs.flat:
    ax.axis('off')

# Adjust layout
plt.subplots_adjust(top=0.93, bottom=0.05, left=0.08, right=0.99, wspace=0.1, hspace=0.3)

plt.show()


#bvecs=torch.tensor(bvecs).to(device)
#display_gradients(bvecs)

########################################################################
print(eigvals_est.shape, eigvals_gt.shape)
eigenvalue_distributions(est=est.reshape(-1, 3), gt=fit.reshape(-1, 3),
                         est_fname="Deep Estimated",  gt_fname="Groundtruth (Invivo)",mask=test_mask.reshape(-1))


"""
assert not torch.isnan(S_est).any()
            assert not torch.isinf(S_est).any()
            assert not torch.isnan(S_gt).any()
            assert not torch.isinf(S_gt).any()

"""