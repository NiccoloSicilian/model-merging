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
        expected_shape = (self.fanout, self.fanin)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Dimension mismatch in dualize: "
                f"Expected {expected_shape}, but got {grad.shape}"
            )
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
        expected_shape = (self.fanout, self.fanin, self.kernel_size, self.kernel_size)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Dimension mismatch in dualize: "
                f"Expected {expected_shape}, but got {grad.shape}"
            )
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
def linear_mass_schedule(current_l, tot_layer):
    return 0.1 + current_l/tot_layer * (0.5-0.1)    
def uniform_mass_schedule(current_l, tot_layer):
    return 0.5

def ViT_B_16(num_classes=512, num_blocks=12, d_embed=768, num_heads=12, patch_size=16, input_channels=3, mass_schedule='uniform'):
    mlp_width = 4 * d_embed
    patch_dim = input_channels * (patch_size ** 2)
    tot_layers= 4*num_blocks+3
    # 1. Patch Embed (conv1 in checkpoint)
    # Note: Checkpoint shows [768, 3, 16, 16] which is a Conv layer
    
    conv1 = Conv2DSVD(fanin=input_channels, fanout=d_embed,kernel_size=patch_size)
    conv1.mass = uniform_mass_schedule(0,tot_layers)
    # 2. Positional & Class Embedding
    visual_pos_embed = LinearSVD(197, d_embed)
    visual_pos_embed.mass = uniform_mass_schedule(1,tot_layers)
    
    transformer = None
    # Pre-transformer norm (ln_pre)
    for b in range(num_blocks):
        # 3. Transformer Blocks
        a1 = LinearSVD(3*d_embed, d_embed) 
        a1.mass = uniform_mass_schedule(b*4+2,tot_layers)
        
        a2 = LinearSVD(d_embed, d_embed) 
        a2.mass = uniform_mass_schedule(b*4+3,tot_layers)
        
        att = a2@ a1
    
        m1 = LinearSVD(mlp_width,d_embed)
        m1.mass = uniform_mass_schedule(b*4+4,tot_layers)
        
        m2 = LinearSVD( d_embed,mlp_width)
        m2.mass = uniform_mass_schedule(b*4+5,tot_layers)
        
        mlp = m2 @ m1
        
        # Residual paths
        if transformer:
            transformer = (mlp @ att) @ transformer
        else:
            transformer = (mlp @ att)

    # 4. Final Head (ln_post and proj)
    proj = LinearSVD(d_embed, num_classes)
    proj.mass = uniform_mass_schedule(tot_layers,tot_layers)
    
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
