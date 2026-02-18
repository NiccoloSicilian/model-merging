import copy
import logging
import math
from math import sqrt
import numpy as np
import copy
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    print_memory,
)
from model_merging.merging.structured import (
    get_svd_dict,
    isotropic_sum,
    avg_layers,
)
import re
import torch
import gc

pylogger = logging.getLogger(__name__)
import torch
import copy
from pathlib import Path
import torch
from math import sqrt
###NEW
# Assuming svd_orthogonalize is defined elsewhere or imported
# If strictly using PyTorch SVD as discussed:

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


def get_vit_topological_order(keys):
    """
    Robustly sorts CLIP ViT keys in topological order:
    conv1 -> pos_embed -> blocks (0..N) -> proj
    """
    def sort_key(k):
        # 1. Conv1 (Input)
        if 'visual.conv1' in k: 
            return (0, 0, 0)
        
        # 2. Positional Embeddings
        if 'positional_embedding' in k: 
            return (1, 0, 0)
            
        # 3. Class Embedding (if present)
        if 'class_embedding' in k:
            return (1, 1, 0)
            
        # 4. Transformer Blocks
        if 'resblocks' in k:
            # Extract block number
            match = re.search(r'resblocks\.(\d+)', k)
            block_idx = int(match.group(1)) if match else 999
            
            # Sub-layer order within block
            sub_order = 0
            if 'attn.in_proj' in k: sub_order = 1
            elif 'attn.out_proj' in k: sub_order = 2
            elif 'ln_1' in k: sub_order = 3
            elif 'mlp.c_fc' in k: sub_order = 4
            elif 'mlp.c_proj' in k: sub_order = 5
            elif 'ln_2' in k: sub_order = 6
            
            return (2, block_idx, sub_order)
            
        # 5. Final LayerNorm (ln_post)
        if 'ln_post' in k:
            return (3, 0, 0)
            
        # 6. Final Projection (Output)
        if 'visual.proj' in k: 
            return (4, 0, 0)
            
        return (5, 0, 0) # Unknown keys last

    return sorted(keys, key=sort_key)

def print_first_singular_values(module_vec_flat, tol=1e-6):
    """
    For each layer in module_vec_flat (dict: layer_name -> weight tensor),
    compute SVD and print the largest singular value.
    Skip tensors with dimension > 2.
    Also checks if all singular values are (approximately) equal.
    
    tol: numerical tolerance for equality check
    """
    for layer_name, weight in module_vec_flat.items():

        # Ensure tensor
        if not isinstance(weight, torch.Tensor):
            continue

        # Skip tensors with dimension > 2
        if weight.ndim > 2:
            print(f"{layer_name}: skipped (ndim={weight.ndim})")
            continue

        # Only process 2D matrices
        if weight.ndim != 2:
            print(f"{layer_name}: skipped (shape={weight.shape})")
            continue

        try:
            W = weight
            singular_values = torch.linalg.svdvals(W)

            first_sv = singular_values[0].item()

            # Check if all singular values are approximately equal
            all_equal = torch.allclose(
                singular_values,
                singular_values[0].expand_as(singular_values),
                atol=tol,
                rtol=0
            )

            print(
                f"{layer_name}: first singular value = {first_sv:.6f} | "
                f"all singular values equal? {all_equal}"
            )

        except RuntimeError as e:
            print(f"{layer_name}: SVD failed ({e})")


class DualMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alphas, svd_path, svd_compress_factor, model_name, device=None):
        super().__init__()
        self.optimal_alphas = optimal_alphas
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 DualMerger initialized on device: {self.device}")
    
    @torch.no_grad()
    def merge(self, base_model, finetuned_models):
        base_model = base_model.to(self.device)

        task_dicts = {}
        datasets = list(finetuned_models.keys())
        num_tasks = len(datasets) 

        for dataset in datasets:
            ft_state_dict = {
                k: v.to(self.device) for k, v in finetuned_models[dataset].items()
            }

            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), ft_state_dict
            )
            
            del ft_state_dict 
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        ref_state_dict = {k: v.to(self.device) for k, v in base_model.state_dict().items()}

        multi_task_vector = avg_layers(
            ref_state_dict=ref_state_dict,
            svd_dict=svd_dict,
        )
        
        # Move to CPU before building network
        multi_task_vector_cpu = {k: v.cpu() for k, v in multi_task_vector.items()}
        del multi_task_vector
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
            
        raw_keys = list(multi_task_vector_cpu.keys())
        ordered_keys = get_vit_topological_order(raw_keys)
        
        print(ordered_keys)
        
        module_net = build_duality_map(ordered_keys, multi_task_vector_cpu, self.device)  # ← add self.device
        module_vec_flat = module_net
        print_first_singular_values(module_vec_flat)
        #compute_average_SAR(module_vec_flat, finetuned_models, datasets)

        # Update dualized keys (come back as GPU tensors from build_duality_map)
        for key in module_vec_flat:
            multi_task_vector_cpu[key] = module_vec_flat[key]

        # Move everything to device in one clean pass (.to() ensures contiguous)
        multi_task_vector_cpu = {
            k: v.to(self.device) for k, v in multi_task_vector_cpu.items()
        }

        del module_vec_flat
        gc.collect()
            
        model_name = self.model_name
        coefficient = 1.0 

        if (
            model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[model_name]
        ):
            coefficient = self.optimal_alphas[model_name][num_tasks]

        merged_encoder = copy.deepcopy(base_model)
        print("USING ALPHA:", coefficient)
        merged_encoder = apply_dict_to_model(
            multi_task_vector_cpu,
            merged_encoder,
            coefficient=coefficient,
        )
        
        return merged_encoder

