import torch
import torch.nn as nn
import torch.nn.functional as F
import geomstats.geometry.spd_matrices as spd
import math

manifold = spd.SPDMatrices(n=3)
manifold.n=3
le_metric = spd.SPDLogEuclideanMetric(space=manifold)

def scale_to_01(arr):
    min_val = torch.min(arr)
    max_val = torch.max(arr)
        
    scaled_arr = (arr - min_val) / (max_val - min_val)
    return scaled_arr


def map_to_range(y, min_val, max_val):
        y=scale_to_01(y)
        return min_val + y * (max_val - min_val)




def is_positive_definite(matrix):
    # Compute the eigenvalues of the matrix
    eigenvalues = torch.linalg.eigvals(matrix)
    
    # Check if all eigenvalues are positive (real part of eigenvalues should be positive)
    # Note: If using complex matrices, ensure the real part of eigenvalues is positive
    return torch.all(eigenvalues.real > 0)

class LogEuclideanLoss(nn.Module):
    def __init__(self, epsilon=1e-10):
        super(LogEuclideanLoss, self).__init__()
        self.epsilon = epsilon  # Small value to ensure numerical stability in log

    def forward(self, D1, D2):
        """
        Compute the Log-Euclidean MSE between two sets of 3x3 SPD matrices
        in tensors of shape [batch_size, height, width, depth, 3, 3].

        Args:
            D1: A tensor of shape [batch_size, height, width, depth, 3, 3], representing SPD matrices.
            D2: A tensor of shape [batch_size, height, width, depth, 3, 3], representing SPD matrices.

        Returns:
            A scalar tensor representing the Log-Euclidean MSE between D1 and D2.
        """
        
        D1=torch.nan_to_num(D1, nan=self.epsilon, posinf=self.epsilon, neginf=self.epsilon)
        D1=torch.abs(D1.real)
        D2=torch.nan_to_num(D2, nan=self.epsilon, posinf=self.epsilon, neginf=self.epsilon)
        D2=torch.abs(D2.real)
        
        # Eigen-decompose each 3x3 SPD matrix
        eigvals1, eigvecs1 = torch.linalg.eig(D1)  # Shape: [batch_size, height, width, depth, 3]
        eigvals2, eigvecs2 = torch.linalg.eig(D2)  # Shape: [batch_size, height, width, depth, 3]
        eigvals1, eigvecs1 =eigvals1.real, eigvecs1.real 
        eigvals2, eigvecs2 =eigvals2.real, eigvecs2.real 
        
        # Compute the log of the eigenvalues (add epsilon to avoid log(0))
        log_eigvals1 = torch.log(eigvals1 + self.epsilon)+ self.epsilon
        log_eigvals2 = torch.log(eigvals2 + self.epsilon)+ self.epsilon

        # Reconstruct log-transformed matrices
        log_D1 = eigvecs1 @ torch.diag_embed(log_eigvals1) @ eigvecs1.transpose(-2, -1)
        log_D2 = eigvecs2 @ torch.diag_embed(log_eigvals2) @ eigvecs2.transpose(-2, -1)

        # Compute the element-wise squared difference between log matrices
        log_euclidean_dist = (log_D1 - log_D2).pow(2)

        # Mean squared error over the last two dimensions (3x3 matrices) and all spatial dimensions
        return log_euclidean_dist.mean()


import torch
import torch.nn as nn
import torch.nn.functional as F


""" class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, H, W, C = x.shape  # Input shape [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # Change to [B, C, H, W] for Conv2d
        x = self.proj(x)  # Patch embedding
        _, C, H, W = x.shape  # Updated dimensions
        x = x.permute(0, 2, 3, 1).view(B, -1, C)  # Flatten patches
        x = self.norm(x)  # Apply layer norm
        return x, H, W
 """
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, depth=7):
        super(PatchEmbedding, self).__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

        # Adding additional layers
        self.depth = depth
        self.layers = nn.ModuleList([nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * 1.5)),
            nn.ReLU(),
            nn.Linear(int(embed_dim * 1.5), embed_dim)
        ) for _ in range(depth)])

    def forward(self, x):
        B, H, W, C = x.shape  # Input shape [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # Change to [B, C, H, W] for Conv2d
        x = self.proj(x)  # Patch embedding
        _, C, H, W = x.shape  # Updated dimensions
        x = x.permute(0, 2, 3, 1).view(B, -1, C)  # Flatten patches
        x = self.norm(x)  # Apply layer norm

        # Apply additional layers
        for layer in self.layers:
            x = layer(x)

        return x, H, W

""" class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio):
        super(SwinTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            #nn.GELU(),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # Feedforward MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        return x
 """
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, depth=7):
        super(SwinTransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        # Adding additional MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        self.norm2 = nn.LayerNorm(dim)

        # Additional MLP layers
        self.depth = depth
        self.mlp_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(dim, int(dim * 1.5)),
            nn.ReLU(),
            nn.Linear(int(dim * 1.5), dim)
        ) for _ in range(depth)])

    def forward(self, x):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        # Feedforward MLP with residual connection
        mlp_output = self.mlp(x)
        x = x + mlp_output
        x = self.norm2(x)

        # Apply additional MLP layers
        for layer in self.mlp_layers:
            mlp_output = layer(x)
            x = x + mlp_output
            x = self.norm2(x)

        return x

class SwinTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_heads, mlp_ratio, num_classes):
        super(SwinTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size=patch_size)

        self.swin_block = SwinTransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)

        self.head = nn.Sequential(
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B, H, W, C = x.shape  # Input: [B, 128, 128, 65]
        
        # Patch Embedding
        x, patch_H, patch_W = self.patch_embed(x)  # [B, num_patches, embed_dim]

        # Swin Transformer Block
        x = self.swin_block(x)  # [B, num_patches, embed_dim]

        # Classification Head
        x = self.head(x)  # [B, num_patches, num_classes]

        # Reshape back to spatial dimensions
        x = x.view(B, patch_H, patch_W, -1)  # [B, patch_H, patch_W, num_classes]

        # Upsample to match original dimensions
        x = F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode='bilinear', align_corners=False)
        x = x.permute(0, 2, 3, 1)  # Final output shape [B, H, W, num_classes]

        return x


""" # Instantiate Model
model = SwinTransformer(
    in_channels=65,
    patch_size=4,
    embed_dim=96,
    num_heads=4,
    mlp_ratio=4,
    num_classes=3
)
 """
""" # Example Input
input_tensor = torch.rand(16, 128, 128, 65)  # Input: [batch_size, height, width, channels]
output_tensor = model(input_tensor)
print(output_tensor.shape)  # Expected: [16, 128, 128, 3]
 """

class DtiNet(nn.Module):
    
    def __init__(self, b_value, N):
        super(DtiNet, self).__init__()
        
        #self.signals=signals
        #self.gradients = gradients  # Expected shape: [1, 15, 3] (assuming 15 gradient directions)
        self.b_value = b_value
        self.N=N

        # Define CNN layers for feature extraction

        self.layers = nn.Sequential(
        nn.Conv2d(self.N, 64, kernel_size=3, padding=1),   # Input channels: 31 (S0 + 30 gradients)
        #nn.BatchNorm2d(64),
        nn.ReLU(),
        #nn.Dropout(p=0.3),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #nn.BatchNorm2d(128),
        nn.ReLU(),
        #nn.Dropout(p=0.3),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #nn.BatchNorm2d(128),
        nn.ReLU(),
        #nn.Dropout(p=0.3),

        nn.Conv2d(128, 128, kernel_size=3, padding=1),  
        #nn.BatchNorm2d(128),
        nn.ReLU(),
        #nn.Dropout(p=0.3),

        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(),
        #nn.Dropout(p=0.3),

        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 6, kernel_size=3, padding=1),  
        #nn.BatchNorm2d(6),
        
        #nn.ReLU()
            )


        #self.fc = nn.Linear(32, 6)
    # Spatial Attention Layer
        """ self.spatial_attention = nn.Sequential(
            nn.Conv2d(3, 1, kernel_size=7, padding=3),  # Compress to single attention map
            nn.Sigmoid()  # Scale attention weights to [0, 1]
        )
     """
    def map_to_range(self, y, min_val, max_val):
                
        scaled_arr = (y - torch.min(y)) / ( torch.max(y) -  torch.min(y))
        
        return min_val + scaled_arr * (max_val - min_val)

    
    def generate_dti_signals(self, D=None, gradients=None, b_values=None, S0=None, device=None):
    
        epsilon=1e-10; 
        #S0 = S0  # Shape: [128, 128, 32, 1]
        #S0=1
        if S0 is not None:
            S0=S0[...,None]
        else:
            S0=1    

        exponent = -b_values * torch.einsum('ij,...jk,ik->...i', gradients, D, gradients)  # Shape: [128, 128, 32, 65]
        S = S0* torch.exp(exponent)
        
        return S + epsilon
    
    def correct_to_pure_rotation(self, evecs_est):
        """
        Ensure that evecs_est is a pure rotation matrix (det = +1).
        
        Args:
            evecs_est: Tensor of shape (..., 3, 3) representing a batch of 3x3 matrices.
        
        Returns:
            A corrected version of evecs_est with det = +1.
        """
        # Calculate determinant along the last two dimensions
        det = torch.det(evecs_est)  # Shape: (...,)

        # Find matrices with det = -1
        mask = (det < 0)  # Boolean mask for improper rotations, Shape: (...)

        # Expand mask to match the last column of evecs_est
        mask = mask.unsqueeze(-1).unsqueeze(-1)  # Shape: (..., 1, 1)

        # Correct the matrices by flipping the sign of the last column
        corrected_evecs = evecs_est.clone()
        corrected_evecs[mask.expand_as(evecs_est)] *= -1  # Flip the last column if det = -1

        return corrected_evecs


    def ensure_rotation_matrices(self, eigvecs):
        """
        Ensures that the eigenvectors form rotation matrices with unit determinant.
        
        Args:
            eigvecs (torch.Tensor): Eigenvectors tensor of shape [..., 3, 3].
            
        Returns:
            torch.Tensor: Modified eigenvectors ensuring a unit determinant (rotation matrices).
        """
        # Check the determinant of the eigenvectors matrix
        dets = torch.det(eigvecs)  # Shape: [...]
        
        # Ensure that the determinant is positive by flipping signs of columns if necessary
        signs = torch.sign(dets)  # 1 if positive, -1 if negative
        eigvecs = eigvecs * signs.unsqueeze(-1).unsqueeze(-1)  # Adjust sign of columns if needed
        
        return eigvecs 
    
    def generate_random_rots_uniform(self, w=None, h=None, batch_size=None, device="cuda"):
        """
        Generate random 3x3 rotation matrices uniformly sampled from SO(3) for a 128x128x16 grid.

        Parameters:
            w (int): First grid dimension (default: 128).
            h (int): Second grid dimension (default: 128).
            batch_size (int): Third grid dimension (default: 16).
            device (str): Device to perform computation on ("cuda" or "cpu").

        Returns:
            torch.Tensor: Tensor of shape (w, h, batch_size, 3, 3) containing rotation matrices.
        """
        total_batches = w * h * batch_size  # Flattened total number of matrices to generate

        # Step 1: Generate random vectors
        x1 = torch.randn((total_batches, 3), device=device)
        x2 = torch.randn((total_batches, 3), device=device)

        # Step 2: Normalize the first vector
        v1 = x1 / torch.norm(x1, dim=-1, keepdim=True)

        # Step 3: Make the second vector orthogonal to the first
        proj = torch.sum(x2 * v1, dim=-1, keepdim=True) * v1  # Projection of x2 onto v1
        v2 = x2 - proj
        v2 = v2 / torch.norm(v2, dim=-1, keepdim=True)

        # Step 4: Compute the third vector using the cross product
        v3 = torch.cross(v1, v2)

        # Step 5: Stack the vectors to form the rotation matrix
        Q = torch.stack((v1, v2, v3), dim=-1)  # Shape: (total_batches, 3, 3)

        # Step 6: Reshape to the desired grid shape
        rotation_matrices = Q.view(w, h, batch_size, 3, 3)
        return rotation_matrices
    
    

    def ensure_spd_tensor(self, D_components=None):
        
        epsilon = 1e-10
        batch_size, _, height, width = D_components.shape
        
        D =torch.zeros(batch_size, height, width, 3, 3, device=D_components.device) +epsilon
    
        if torch.isnan(D_components).any():
            print("Warning: D_components contains NaN values")
            D_components = torch.nan_to_num(D_components, nan=epsilon, posinf=epsilon, neginf=epsilon)  # Replace NaNs with 0
    
        # Assign the components to the appropriate entries in the diffusion tensor
        D[..., 0, 0] = torch.abs(D_components[:, 0, :, :])  # D11
        D[..., 1, 1] = torch.abs(D_components[:, 1, :, :])  # D22
        D[..., 2, 2] = torch.abs(D_components[:, 2, :, :] ) # D33
        D[..., 0, 1] = D[..., 1, 0] = D_components[:, 3, :, :] # D12
        D[..., 0, 2] = D[..., 2, 0] = D_components[:, 4, :, :]  # D13
        D[..., 1, 2] = D[..., 2, 1] = D_components[:, 5, :, :]  # D23
        
            
        D=torch.nan_to_num(D, nan=epsilon, posinf=epsilon, neginf=epsilon)
        D=D@D.transpose(-1,-2)
        #evals, evecs = torch.linalg.eigh(D) 
        #evecs=self.correct_to_pure_rotation(evecs)
        #D=evecs.transpose(-1,-2)@torch.diag_embed(evals)@evecs        
        #print(torch.det(evecs))
        
        return D + epsilon
    
    
    def forward(self, S, gradients=None):
                      
        epsilon=1e-10
        D_components = self.layers(S) + epsilon # shape [batch_size, 6, height, width]
        #print(D_components)
        D_components=torch.nan_to_num(D_components, nan=epsilon, posinf=epsilon, neginf=epsilon)
        #print(S.shape)
    
        if gradients is not None:
         grads=torch.zeros(gradients.shape[0]+1,3, device=gradients.device)
         grads[1:,:]=gradients
        
        #D=D_components.permute(2,3,0, 1)
        #rotations=self.generate_random_rots_uniform(w=D.shape[0], h=D.shape[1], batch_size=D.shape[2], device=D.device)
        
        Dt = self.ensure_spd_tensor(D_components)  # Shape: [batch_size, height, width, 3, 3]
        
        return Dt, S


class DtiEigsTrans(nn.Module):
    def __init__(self, patch_size=16, d_model=128, num_heads=8, dim_feedforward=256, num_layers=2, input_channels=65, num_eigenvalues=3):
        """
        Initialize the DtiEigsTrans model.

        Parameters:
        - patch_size: Size of the patches to divide the spatial dimensions into.
        - d_model: Dimensionality of the transformer model.
        - num_heads: Number of attention heads in the transformer.
        - dim_feedforward: Hidden layer size in the transformer feedforward network.
        - num_layers: Number of transformer encoder layers.
        - input_channels: Number of input channels per spatial location.
        - num_eigenvalues: Number of eigenvalues to output per patch.
        """
        super(DtiEigsTrans, self).__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        self.num_eigenvalues = num_eigenvalues
        
        # Input projection layer to map input to d_model
        self.input_proj = nn.Linear(patch_size * patch_size * input_channels, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=num_heads, 
            dim_feedforward=dim_feedforward, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection layer to map d_model to the number of eigenvalues
        self.output_proj = nn.Linear(d_model, patch_size * patch_size * num_eigenvalues)
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Parameters:
        - x: Input tensor of shape (batch_size, 128, 128, 16, input_channels)
        
        Returns:
        - Tensor of shape (batch_size, 128, 128, 16, num_eigenvalues)
        """
        # Input dimensions
        batch_size, height, width, depth, channels = x.shape
        
        # Reshape into patches
        assert height % self.patch_size == 0 and width % self.patch_size == 0, "Height and width must be divisible by patch_size"
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = x.view(
            batch_size, 
            num_patches_h, self.patch_size, 
            num_patches_w, self.patch_size, 
            depth, channels
        )
        patches = patches.permute(0, 1, 3, 5, 2, 4, 6).reshape(
            batch_size, num_patches_h * num_patches_w * depth, -1
        )  # Shape: [batch_size, num_patches, patch_size * patch_size * channels]
        
        # Input projection
        patches = self.input_proj(patches)  # Shape: [batch_size, num_patches, d_model]
        
        # Transformer encoding
        transformer_out = self.transformer_encoder(patches)  # Shape: [batch_size, num_patches, d_model]
        
        # Output projection
        eigenvalues = self.output_proj(transformer_out)  # Shape: [batch_size, num_patches, patch_size * patch_size * num_eigenvalues]
        
        # Reshape back to spatial dimensions
        eigenvalues = eigenvalues.view(
            batch_size, 
            num_patches_h, num_patches_w, depth, self.patch_size, self.patch_size, self.num_eigenvalues
        )
        eigenvalues = eigenvalues.permute(0, 1, 4, 2, 5, 3, 6).reshape(
            batch_size, height, width, depth, self.num_eigenvalues
        )
        
        return eigenvalues, x


def extract_image_patches(x, kernel, stride=1, dilation=1):
    b,c,h,w = x.shape
    h2 = math.ceil(h / stride)
    w2 = math.ceil(w / stride)
    pad_row = (h2 - 1) * stride + (kernel - 1) * dilation + 1 - h
    pad_col = (w2 - 1) * stride + (kernel - 1) * dilation + 1 - w
    x = F.pad(x, (pad_row//2, pad_row - pad_row//2, pad_col//2, pad_col - pad_col//2))
   
    # Extract patches
    patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
    patches = patches.permute(0,4,5,1,2,3).contiguous()
    return patches


""" # Example usage
if __name__ == "__main__":
    # Input tensor: height x width x batch_size x channels
    height, width, batch_size, channels = 128, 128, 16, 64
    input_tensor = torch.randn(height, width, batch_size, channels)

    # Create the model
    model = DtiEigsTrans(patch_size=32, d_model=64,n_heads=4, num_eigenvalues=3)
    patches=extract_image_patches(input_tensor, kernel=3)
    # Forward pass
    output = model(patches)
    print("Output shape:", output.shape)  # Should be (128, 128, 16, 3)
 """
class DtiEigs(nn.Module):
    
    def __init__(self, b_value, N):
        super(DtiEigs, self).__init__()
        
        #self.signals=signals
        #self.gradients = gradients  # Expected shape: [1, 15, 3] (assuming 15 gradient directions)
        self.b_value = b_value
        self.N = N

        # Define CNN layers for feature extraction
        self.layers = nn.Sequential(
        nn.Conv2d(self.N, 64, kernel_size=3, padding=1),   # Input channels: 31 (S0 + 30 gradients)
        #nn.BatchNorm2d(64),
        nn.ReLU(),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #nn.BatchNorm2d(128),
        #nn.ReLU(),
               
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #nn.BatchNorm2d(128),
        nn.ReLU(),
                
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #nn.BatchNorm2d(128),
        nn.ReLU(),
               
        
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #nn.BatchNorm2d(128),
        nn.ReLU(),
        
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(),
         
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #nn.BatchNorm2d(64),
        nn.ReLU(),
        #nn.Linear(64, 3)
        nn.Conv2d(64, 3, kernel_size=3, padding=1),  
        #nn.ReLU()
            )

    def forward(self, S):
        #print(S.shape)
        #exit()
        epsilon=1e-10
        eigvals = self.layers(S)  
        eigvals=torch.abs(eigvals)
        eigvals=torch.nan_to_num(eigvals, nan=epsilon, posinf=epsilon, neginf=epsilon)
        S=torch.nan_to_num(S, nan=epsilon, posinf=epsilon, neginf=epsilon)
        
        eigvals, S=eigvals.permute(2,3,0,1), S.permute(2,3,0,1) + epsilon
        #print(eigvals.shape, S.shape)
        #eigvals=map_to_range(eigvals, 0.0, 1.0)
        
        return eigvals
    

class DtiEigs_31(nn.Module):
    
    def __init__(self, b_value):
        super(DtiEigs_31, self).__init__()
        
        #self.signals=signals
        #self.gradients = gradients  # Expected shape: [1, 15, 3] (assuming 15 gradient directions)
        self.b_value = b_value

        # Define CNN layers for feature extraction
        self.layers = nn.Sequential(
        nn.Conv2d(31, 64, kernel_size=3, padding=1),   # Input channels: 31 (S0 + 30 gradients)
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),  
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),  
        nn.ReLU()
            )

    def forward(self, S):
        #print(S.shape)
        #exit()
        epsilon=1e-10
        eigvals = (self.layers(S)) + 1e-10 
        eigvals=torch.nan_to_num(eigvals, nan=epsilon, posinf=epsilon, neginf=epsilon)
        eigvals=eigvals.real
        eigvals, S=eigvals.permute(2,3,0,1), S.permute(2,3,0,1)
        #print(eigvals.shape, S.shape)
        return eigvals, S + epsilon



import numpy as np

class le_mse(nn.Module):
    
    def __init__(self):
        
        super(le_mse, self).__init__()

    def forward(self, D1, D2):
        device=D1.device
        #log_e_dist = le_metric.dist(D1.detach().cpu().numpy(), D2.detach().cpu().numpy())
        D1, D2= D1.detach().cpu().numpy(), D2.detach().cpu().numpy()
    
        return torch.tensor(np.mean(le_metric.dist(D1, D2)**2), device=device, requires_grad=True)  

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example usage:
""" 
batch_size = 16  # Example batch size
height = 240
width = 240
b_value = 1000  # Example b-value

# Simulated signals (replace with your actual input signals)
s = torch.randn(16, 1, height, width).to(device)  
grads=torch.rand(30, 3).to(device)
# Instantiate the model and pass the input signal and b-value
model = DtiNet(grads, b_value).to(device)
s1=S.permute(2,3, 0,1)

s2=torch.cat( (s[:,0:1,:,:], s1), 1)

a, b=model(s2)
torch.det(b[1,10,10,:,:])
b[1,20,10,:,:]
 """

