import torch
from math import sqrt
from modula.abstract import *
from modula.bond import *

def svd_orthogonalize(M):
    U, _, Vh = torch.linalg.svd(M, full_matrices=False)
    return U @ Vh

class LinearSVD(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # [fanout, fanin]
        return x @ weights.T

    def initialize(self, key=None):
        print("No need init")
        return None

    def project(self, w):
        weight = w[0]
        weight = svd_orthogonalize(weight) * sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        
        # 1. Calculate the scalar factor
        scalar_factor = sqrt(self.fanout / self.fanin) * target_norm
        
        # 2. Compute the dual weight
        # svd_orthogonalize returns the "direction", scalar_factor applies the "magnitude"
        d_weight = svd_orthogonalize(grad) * scalar_factor
        
    
        
        return [d_weight]


class Conv2DSVD(Atom):
    def __init__(self, fanout, fanin, kernel_size):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.kernel_size = kernel_size
        self.smooth = True
        self.mass = 0.5
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # [fanout, fanin, k, k]
        return torch.nn.functional.conv2d(x, weights, padding='same')

    def initialize(self, key=None):
        weight = torch.randn(self.fanout, self.fanin, self.kernel_size, self.kernel_size)
        weight = self._ortho_spatial(weight)
        scale = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin)
        return [weight * scale]

    def project(self, w):
        weight = w[0]
        weight = self._ortho_spatial(weight)
        scale = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin)
        return [weight * scale]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        
        # 1. Calculate the scalar factor
        # The paper defines this specifically for Conv2D to normalize spatial dimensions
        scalar_factor = (1.0 / self.kernel_size ** 2) * sqrt(self.fanout / self.fanin) * target_norm
        
        # 2. Compute the dual weight
        d_weight_ortho = self._ortho_spatial(grad)
        d_weight = d_weight_ortho * scalar_factor
        
        # 3. Print Debug Info
        
        return [d_weight]

    def _ortho_spatial(self, weight):
        """SVD orthogonalize each [fanout, fanin] slice over spatial dims."""
        k = self.kernel_size
        # weight shape: [fanout, fanin, k, k]
        result = torch.zeros_like(weight)
        for i in range(k):
            for j in range(k):
                result[:, :, i, j] = svd_orthogonalize(weight[:, :, i, j])
        return result


def ViT_B_16(num_classes=512, num_blocks=12, d_embed=768, num_heads=12, patch_size=16, input_channels=3):
    mlp_width = 4 * d_embed
    patch_dim = input_channels * (patch_size ** 2)

    # 1. Patch Embed (conv1 in checkpoint)
    # Note: Checkpoint shows [768, 3, 16, 16] which is a Conv layer
    
    conv1 = Conv2DSVD(fanin=input_channels, fanout=d_embed,kernel_size=patch_size)
    # 2. Positional & Class Embedding
    visual_pos_embed = LinearSVD(197, d_embed)
    # Pre-transformer norm (ln_pre)

    # 3. Transformer Blocks
    a1 = LinearSVD(d_embed, d_embed) 
    a2 = LinearSVD(3*d_embed, d_embed) 
    att = a1@ a2

    m1 = LinearSVD(d_embed, mlp_width)
    m2 = LinearSVD(mlp_width, d_embed)
    mlp = m1@ GeLU() @ m2
    
    # Residual paths
    
    transformer = (mlp @ att) ** num_blocks

    # 4. Final Head (ln_post and proj)
    proj = LinearSVD(d_embed, num_classes)
    # Correct Flow: Input -> Patch -> Pos -> ln_pre -> Transformer -> ln_post -> Head
    return proj @ transformer  @ visual_pos_embed @ conv1
###

def build_duality_map(layer_names, grads, device):
    """
    Build modular duality map assuming layers are in execution order.
    Applies composition sequentially: layer_N ∘ ... ∘ layer_1 ∘ layer_0
    """
    print("\n" + "="*80)
    print("STEP 1: Creating Atomic Modules with Dualized Gradients")
    print("="*80)
    m = ViT_B_16()

    to_consider_name = []
    to_consider_grad = []
    for name in layer_names:
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
            continue
        if 'visual.conv1.weight' in name or ('visual.proj' in name and 'out_proj' not in name) or 'visual.positional_embedding' in name or ('visual.transformer.resblocks' in name and 'weight' in name and ('attn.in_proj_weight' in name or 'attn.out_proj.weight' in name or 'mlp.c_fc.weight' in name or 'mlp.c_proj.weight' in name)):
            to_consider_name.append(name)
            to_consider_grad.append(grads[name].to(device))
        else:
            print(f"⚠ {name}: Ignored")
            continue
    # Print first 100 values of the gradient just appended
    print(f"Total Atomic Modules: {m.atoms} {m.mass}, To Consider: {len(to_consider_grad)}, {len(to_consider_name)}")
    # Dualize directly in PyTorch — no JAX conversion needed
    to_consider_dualized_grad = m.dualize(to_consider_grad)
    print(f"Dualized: {len(to_consider_dualized_grad)}")
    # Return the dictionary of all dualized gradients
    return dict(zip(to_consider_name, to_consider_dualized_grad))
