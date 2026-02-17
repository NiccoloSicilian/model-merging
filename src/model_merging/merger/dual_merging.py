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
import jax
import jax.numpy as jnp
from modula.abstract import *
from modula.bond import *

pylogger = logging.getLogger(__name__)
import torch
import copy
from pathlib import Path
import torch
from math import sqrt
###NEW

def svd_orthogonalize(M):
    """
    Computes the Polar Factor (U @ Vh) of matrix M using SVD.
    This is the exact projection onto the Stiefel manifold (closest orthogonal matrix).
    """
    # full_matrices=False is crucial for non-square matrices
    U, _, Vh = jnp.linalg.svd(M, full_matrices=False)
    return U @ Vh

class Linear(Atom):
    def __init__(self, fanout, fanin):
        super().__init__()
        self.fanin  = fanin
        self.fanout = fanout
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        # x shape is [..., fanin]
        weights = w[0]  # shape is [fanout, fanin]
        return jnp.einsum("...ij,...j->...i", weights, x)

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin))
        # Exact SVD orthogonalization
        weight = svd_orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def project(self, w):
        weight = w[0]
        # Project weights back to the scaled orthogonal manifold
        weight = svd_orthogonalize(weight) * jnp.sqrt(self.fanout / self.fanin)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        # Map dual (gradient/task vector) to primal (weight space)
        # Using SVD ensures we find the best orthogonal direction
        d_weight = svd_orthogonalize(grad) * jnp.sqrt(self.fanout / self.fanin) * target_norm
        return [d_weight]


class Conv2D(Atom):
    def __init__(self, fanout, fanin, kernel_size):
        super().__init__()
        self.fanin = fanin
        self.fanout = fanout
        self.kernel_size = kernel_size
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]  # shape [fanout, fanin, k, k]
        # Using explicit dimension numbers for clarity: NHWC input, OIHW weights
        return jax.lax.conv_general_dilated(
            lhs=x,
            rhs=weights,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC')
        )

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.fanout, self.fanin, self.kernel_size, self.kernel_size))
        
        # Apply SVD orthogonalization to each k*k slice independently
        # vmap over axes 2 and 3 (the kernel spatial dimensions)
        ortho_map = jax.vmap(jax.vmap(svd_orthogonalize, in_axes=2, out_axes=2), in_axes=2, out_axes=2)
        weight = ortho_map(weight)
        
        # Scale factor from the paper: 1/k^2 * sqrt(dout/din)
        scale = (1.0 / (self.kernel_size ** 2)) * jnp.sqrt(self.fanout / self.fanin)
        return [weight * scale]

    def project(self, w):
        weight = w[0]
        ortho_map = jax.vmap(jax.vmap(svd_orthogonalize, in_axes=2, out_axes=2), in_axes=2, out_axes=2)
        weight = ortho_map(weight)
        scale = (1.0 / (self.kernel_size ** 2)) * jnp.sqrt(self.fanout / self.fanin)
        return [weight * scale]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        
        # SVD on spatial slices
        ortho_map = jax.vmap(jax.vmap(svd_orthogonalize, in_axes=2, out_axes=2), in_axes=2, out_axes=2)
        d_weight = ortho_map(grad)
        
        scale = (1.0 / (self.kernel_size ** 2)) * jnp.sqrt(self.fanout / self.fanin)
        return [d_weight * scale * target_norm]


class Embed(Atom):
    # Note: Embed usually defines orthogonality row-wise (spherical), 
    # not as a matrix SVD. The original implementation was mathematically correct
    # for spherical embeddings. I kept it stable here.
    def __init__(self, d_embed, num_embed):
        super().__init__()
        self.num_embed = num_embed
        self.d_embed = d_embed
        self.smooth = True
        self.mass = 1
        self.sensitivity = 1

    def forward(self, x, w):
        weights = w[0]
        return weights[x]

    def initialize(self, key):
        weight = jax.random.normal(key, shape=(self.num_embed, self.d_embed))
        # Project rows to sphere
        weight = weight / jnp.linalg.norm(weight, axis=1, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def project(self, w):
        weight = w[0]
        # Project rows to sphere
        weight = weight / jnp.linalg.norm(weight, axis=1, keepdims=True) * jnp.sqrt(self.d_embed)
        return [weight]

    def dualize(self, grad_w, target_norm=1.0):
        grad = grad_w[0]
        # The dual of the sphere constraint is normalizing the gradient direction
        d_weight = grad / jnp.linalg.norm(grad, axis=1, keepdims=True) * jnp.sqrt(self.d_embed) * target_norm
        d_weight = jnp.nan_to_num(d_weight)
        return [d_weight]
def ViT_B_16(num_classes=512, num_blocks=12, d_embed=768, num_heads=12, patch_size=16, input_channels=3):
    mlp_width = 4 * d_embed
    patch_dim = input_channels * (patch_size ** 2)

    # 1. Patch Embed (conv1 in checkpoint)
    # Note: Checkpoint shows [768, 3, 16, 16] which is a Conv layer
    
    conv1 = Conv2d(fanin=input_channels, fanout=d_embed,kernel_size=patch_size)
    conv1.tare(0.5)
    # 2. Positional & Class Embedding
    visual_pos_embed = Linear(197, d_embed)
    visual_pos_embed.tare(0.5)
    
    # Pre-transformer norm (ln_pre)

    # 3. Transformer Blocks
    a1 = Linear(d_embed, d_embed) 
    a1.tare(0.5)
    a2 = Linear(3*d_embed, d_embed) 
    a2.tare(0.5)
    att = a1@ a2

    m1 = Linear(d_embed, mlp_width)
    m1.tare(0.5)
    m2 = Linear(mlp_width, d_embed)
    m2.tare(0.5)
    mlp = m1@ GeLU() @ m2
    
    # Residual paths
    att_block = (1 - 1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * att
    mlp_block = (1 - 1/(2*num_blocks)) * Identity() + 1/(2*num_blocks) * mlp
    transformer = (mlp_block @ att_block) ** num_blocks

    # 4. Final Head (ln_post and proj)
    proj = Linear(d_embed, num_classes)
    proj.tare(0.5)
    # Correct Flow: Input -> Patch -> Pos -> ln_pre -> Transformer -> ln_post -> Head
    return proj @ transformer  @ visual_pos_embed @ conv1
###
def linear_mod(g, name):
    """Apply Linear layer duality map (RMS→RMS operator norm)"""
    g_cpu = g.cpu()
    U, S, Vt = torch.linalg.svd(g_cpu, full_matrices=False)
    result = U @ Vt * sqrt(g_cpu.shape[0] / g_cpu.shape[1])
    return {name: result}


def conv2d_mod(g, name):
    """Apply Conv2D layer duality map (max RMS→RMS over kernel indices)"""
    g_cpu = g.cpu()
    dout, din, k, _ = g_cpu.shape
    
    scaling_factor = (1.0 / k**2) * sqrt(dout / din)
    transformed = torch.zeros_like(g_cpu)
    
    for i in range(k):
        for j in range(k):
            slice_matrix = g_cpu[:, :, i, j]
            U, S, Vt = torch.linalg.svd(slice_matrix, full_matrices=False)
            transformed[:, :, i, j] = scaling_factor * (U @ Vt)
    
    return {name: transformed}


class Module:
    def __init__(self, mass, sensitivity, grads_dict, name):
        """
        grads_dict: dictionary {layer_name: dualized_gradient_tensor}
        """
        self.mass = mass
        self.sensitivity = sensitivity
        self.grads = grads_dict  # Store as dict, not list
        self.name = name
    
    def update_gradients(self, grads_dict):
        self.grads = grads_dict
    
    def set_mass(self, mass):
        self.mass = mass
    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
    
    def get_mass(self):
        return self.mass
    
    def get_sensitivity(self):
        return self.sensitivity
    
    def get_gradients(self):
        return self.grads
    
    def get_name(self):
        return self.name


def scale_grad_dict(grad_dict, scalar):
    """Multiply all tensors in gradient dictionary by scalar"""
    return {name: tensor * scalar for name, tensor in grad_dict.items()}


def merge_grad_dicts(dict1, dict2):
    """Merge two gradient dictionaries"""
    result = dict1.copy()
    result.update(dict2)
    return result


def compose(M_later, M_earlier):
    """
    Compose M_later ∘ M_earlier (execution order: earlier → later)
    
    According to Equation (11) in the paper:
    - M_earlier gradients scaled by: (1/M_later.sensitivity) * (M_earlier.mass / total_mass)
    - M_later gradients scaled by: M_later.mass / total_mass
    """
    total_mass = M_earlier.get_mass() + M_later.get_mass()
    composed_sensitivity = M_earlier.get_sensitivity() * M_later.get_sensitivity()
    composed_name = f"{M_later.get_name()}∘{M_earlier.get_name()}"
    print("Args: Meralier", M_earlier.get_mass(), "M_later: ", M_later.get_mass(), "later sens:",M_later.get_sensitivity())
    # Compute scaling factors (Equation 11)
    sensitivity_factor = 1.0/M_later.get_sensitivity()
    ratio_earlier = M_earlier.get_mass() / total_mass
    ratio_later = M_later.get_mass() / total_mass
    
    scalar_earlier = sensitivity_factor * ratio_earlier
    scalar_later = ratio_later
    
    # Scale and merge gradients
    scaled_grads_earlier = scale_grad_dict(M_earlier.get_gradients(), scalar_earlier)
    scaled_grads_later = scale_grad_dict(M_later.get_gradients(), scalar_later)
    
    composed_grads = merge_grad_dicts(scaled_grads_earlier, scaled_grads_later)
    
    # Create composed module
    M = Module(total_mass, composed_sensitivity, composed_grads, composed_name)
    
    print(f"  Composed: {M.get_name()}")
    print(f"    Mass: {total_mass:.2f}, Sensitivity: {composed_sensitivity:.2f}")
    print(f"    Scalars: earlier={scalar_earlier:.4f}, later={scalar_later:.4f}")
    
    return M
def uniform_mass(tot_layers, current_l):
    return 0.5
def quad_mass(tot_layers, current_l):
    mass = 0.5*(current_l / tot_layers)**2 +0.01
    return mass
def cubic_mass(tot_layers, current_l):
    mass = (current_l / tot_layers)**3 * 0.5
    return mass
def linear_mass(tot_layers, current_l):
    mass = 0.01 + current_l*((0.5-0.01)/tot_layers)
    return mass
def log_mass(tot_layers, current_l):
    end_val = 0.5
    start_val  = 0.002
    b = (end_val - start_val) / np.log(tot_layers)
    
    # Calculate value: a + b * ln(x)
    mass = start_val + b * np.log(current_l)
    return mass
def OneMOne_schedule(tot_layers, current_l):
    if current_l == 1 or current_l == tot_layers:
        mass = 1.0
    else:
        mass = linear_mass(tot_layers, current_l)
    return mass
def different_schedule_mlp_attn(layer_names):
    block_id = 'n'
    masses = {}
    linear_tot_layers = 0
    attn_tot_layers = 0
    ll = 1
    al = 1
    mass_value = 0.5
    for name in layer_names:
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
                continue
        if 'visual.conv1.weight' in name or( 'visual.proj' in name and 'out_proj' not in name) or 'visual.positional_embedding' in name:
            linear_tot_layers += 1
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            if 'attn.in_proj_weight' in name or 'attn.out_proj.weight' in name: 
                attn_tot_layers += 1
            elif 'mlp.c_fc.weight' in name or 'mlp.c_proj.weight' in name:
                linear_tot_layers += 1
    print("-----TOT Linear LAYERS: ",linear_tot_layers, "ATTN:", attn_tot_layers)
    for name in layer_names:
        # Skip non-trainable parameters
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
            continue
        if 'visual.conv1.weight' in name or( 'visual.proj' in name and 'out_proj' not in name) or 'visual.positional_embedding' in name:
            masses[name] =quad_mass(linear_tot_layers,ll)
            print("Linear index:", ll, masses[name])
            
            ll += 1
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            if 'attn.in_proj_weight' in name or 'attn.out_proj.weight' in name: 
                masses[name] = linear_mass(attn_tot_layers, al)
                print("Attn index:", al, masses[name])
                
                al += 1
            elif 'mlp.c_fc.weight' in name or 'mlp.c_proj.weight' in name:
                masses[name] = quad_mass(linear_tot_layers, ll)
                print("Linear index:", ll, masses[name])
                
                ll += 1
    return masses
def linear_mass_scheduler_per_transfblock(layer_names): #Asuming layers list ordered by execution
    block_id = 'n'
    masses = {}
    tot_layers = 0
    l = 1
    mass_value = 0.5
    for name in layer_names:
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
                continue
        if 'visual.conv1.weight' in name or( 'visual.proj' in name and 'out_proj' not in name) or 'visual.positional_embedding' in name:
            tot_layers += 1
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            if 'attn.in_proj_weight' in name or 'attn.out_proj.weight' in name or 'mlp.c_fc.weight' in name or 'mlp.c_proj.weight' in name:
                tot_layers += 1
    print("TOT LAYERS: ", l)
    for name in layer_names:
        # Skip non-trainable parameters
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
            continue
        if 'visual.conv1.weight' in name or( 'visual.proj' in name and 'out_proj' not in name) or 'visual.positional_embedding' in name:
            masses[name] =OneMOne_schedule(tot_layers,l)
            print("MASS:",masses[name], tot_layers, l )
            l += 1
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            if 'attn.in_proj_weight' in name or 'attn.out_proj.weight' in name or 'mlp.c_fc.weight' in name or 'mlp.c_proj.weight' in name:
                if name.split('resblocks.')[1].split('.')[0] == block_id: 
                    masses[name] = OneMOne_schedule(tot_layers, l)
                    print("MASS:",masses[name], tot_layers, l )
                    
                    l += 1 
                else:
                    block_id = name.split('resblocks.')[1].split('.')[0]
                    masses[name] = OneMOne_schedule(tot_layers, l)
                    print("MASS:",masses[name], tot_layers, l )
                    
                    l += 1
    return masses
    
def build_duality_map(layer_names, grads):
    """
    Build modular duality map assuming layers are in execution order.
    Applies composition sequentially: layer_N ∘ ... ∘ layer_1 ∘ layer_0
    """
    print("\n" + "="*80)
    print("STEP 1: Creating Atomic Modules with Dualized Gradients")
    print("="*80)
    m = ViT_B_16()
    #modules = []
    #masses = linear_mass_scheduler_per_transfblock(layer_names)

    to_consider_name = []
    to_consider_grad = []
    for name in layer_names:
        # Skip non-trainable parameters
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
            continue

        # Determine layer type and apply corresponding duality map
        if 'visual.conv1.weight' in name:
            to_consider_name.append(name)
            to_consider_grad.append(grads[name])
        elif 'visual.proj' in name and 'out_proj' not in name:
            to_consider_name.append(name)
            to_consider_grad.append(grads[name])
        elif 'visual.positional_embedding' in name:
            to_consider_name.append(name)
            to_consider_grad.append(grads[name])
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            if 'attn.in_proj_weight' in name:
                to_consider_name.append(name)
                to_consider_grad.append(grads[name])
            elif 'attn.out_proj.weight' in name:
                to_consider_name.append(name)
                to_consider_grad.append(grads[name])
            elif 'mlp.c_fc.weight' in name:
                to_consider_name.append(name)
                to_consider_grad.append(grads[name])
            elif 'mlp.c_proj.weight' in name:
                to_consider_name.append(name)
                to_consider_grad.append(grads[name])
        else:
            print(f"⚠ {name}: Ignored")
    print(f"Total Atomic Modules: {m.atoms} {m.mass}, To Consider: {len(to_consider_grad)}, {len(to_consider_name)}")

    # Convert PyTorch tensors → JAX arrays before calling modula's dualize
    to_consider_grad_jax = [
        jnp.array(g.detach().cpu().numpy()) for g in to_consider_grad
    ]

    to_consider_dualized_grad_jax = m.dualize(to_consider_grad_jax)
    print(f"Dualized: {len(to_consider_dualized_grad_jax)}")

    # Convert JAX arrays → PyTorch tensors for the rest of the pipeline
    to_consider_dualized_grad = []
    for g in to_consider_dualized_grad_jax:
        jax.block_until_ready(g)                          # force JAX to finish
        arr = np.asarray(g)                               # zero-copy if possible
        to_consider_dualized_grad.append(
            torch.from_numpy(arr.copy()).contiguous()     # .copy() detaches from JAX memory
        )

    # Return the dictionary of all dualized gradients
    return dict(zip(to_consider_name, to_consider_dualized_grad))

def compute_average_SAR(module_vec_flat, finetuned_models, datasets):
    """
    Computes the Spectral Analysis of Regularization (SAR) or similar projection metric.
    Formula implemented: || U @ U.T @ W_ft ||_F / || W_ft ||_F
    """
    merged_svd_dict = {}

    # 1. Pre-compute SVD (U component) for the merged/reference model
    for k, v in module_vec_flat.items():
        if v.dim() == 2:
            # torch.linalg.svd returns U, S, Vh
            # full_matrices=False is usually preferred for efficiency
            U, S, Vh = torch.linalg.svd(v, full_matrices=False)
            merged_svd_dict[k] = U

    SAR = {k: 0.0 for k in module_vec_flat}
    count = {k: 0 for k in module_vec_flat}

    # 2. Iterate through datasets and models
    for dataset in datasets:
        if dataset not in finetuned_models:
            continue
            
        for k, weight_ft in finetuned_models[dataset].items():
            # Only process if we computed SVD for this layer (implies dim == 2)
            if k in merged_svd_dict:
                count[k] += 1
                
                U = merged_svd_dict[k]
                
                # --- CALCULATION FIXES ---
                # 1. Use torch.linalg.norm with ord='fro' for Frobenius norm
                # 2. Use @ for Matrix Multiplication (not *)
                # 3. Optimization: U @ (U.T @ W) is faster than (U @ U.T) @ W
                
                # Project fine-tuned weight onto the subspace of the merged weight
                # equivalent to: (U * U.T) * weight_ft
                projected_weight = U @ (U.T @ weight_ft)
                
                numerator = torch.linalg.norm(projected_weight, ord='fro')
                denominator = torch.linalg.norm(weight_ft, ord='fro')
                
                if denominator > 1e-9: # Avoid division by zero
                    SAR[k] += (numerator / denominator).item()

    # 3. Average the results
    avg_SAR_per_layer = {}
    for k in count:
        if count[k] > 0:
            avg_SAR_per_layer[k] = SAR[k] / count[k]

    print("AVERAGE SAR PER LAYER:\n", avg_SAR_per_layer)
    return avg_SAR_per_layer


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
import torch


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
        module_net = build_duality_map(ordered_keys, multi_task_vector_cpu)
        module_vec_flat = module_net

        compute_average_SAR(module_vec_flat, finetuned_models, datasets)

        # Update dualized keys
        for key in module_vec_flat:
            multi_task_vector_cpu[key] = module_vec_flat[key].contiguous()

        # Move everything to device in one clean pass
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


class DualCommonTaskSpecificMerger(TaskVectorBasedMerger):
    def __init__(
        self,
        common_space_fraction,
        optimal_alphas,
        model_name,
        device,
        svd_path, 
        svd_compress_factor,
    ):
        super().__init__()

        self.common_space_fraction = common_space_fraction
        self.optimal_alphas = optimal_alphas
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        
    @torch.no_grad()
    def merge(self, base_model, finetuned_models) -> ImageEncoder | None:
        multi_task_vector = {}
        datasets = list(finetuned_models.keys())
        
        task_dicts = {}
        list_layer = [key for key in finetuned_models[datasets[0]]]
        masses = {key: 0.5 for key in finetuned_models[datasets[0]]}
        num_tasks = len(datasets)

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                gc.collect()

        pylogger.info("Computing SVD...")
        ref_task_dict = task_dicts[datasets[0]]
        
        for key in ref_task_dict:
            shape_ = ref_task_dict[key].shape

            is_2d_matrix = (len(shape_) == 2) and ("text_projection" not in key)
            if not is_2d_matrix:
                pylogger.info(f"Combining by avg {key}...")

                for i, (dataset, task_dict) in enumerate(task_dicts.items()):
                    vec = task_dict[key].to(self.device)
                    if i == 0:
                        multi_task_vector[key] = vec.clone()
                    else:
                        multi_task_vector[key] += (vec - multi_task_vector[key]) / (i + 1)
                continue

            pylogger.info(f"Computing common space using sum for {key}...")
            combined_w = sum(
                [task_dict[key].to(self.device) for task_dict in task_dicts.values()]
            )

            common_space_index_s = int(min(shape_) * self.common_space_fraction)
            _task_specific_total_space_index_s = (
                round((min(shape_) - common_space_index_s) / num_tasks) * num_tasks
            )
            common_space_index_s = min(shape_) - _task_specific_total_space_index_s

            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            common_space_u = u[:, :common_space_index_s]
            common_space_s = s[:common_space_index_s]
            common_space_v = v[:common_space_index_s, :]
            
            del combined_w, u, s, v
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

            n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
            
            for i, task_dict in enumerate(task_dicts.values()):
                w = task_dict[key].to(self.device)
                w_ts = w - common_space_u @ common_space_u.T @ w
                u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)

                if i == 0:
                    combined_space_u = torch.zeros_like(u_ts, device=self.device)
                    combined_space_s = torch.zeros_like(s_ts, device=self.device)
                    combined_space_v = torch.zeros_like(v_ts, device=self.device)

                combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = (
                    u_ts[:, :n_dims_per_task]
                )
                combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = (
                    s_ts[:n_dims_per_task]
                )
                combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = (
                    v_ts[:n_dims_per_task, :]
                )
                
                del w, w_ts, u_ts, s_ts, v_ts
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

            combined_space_u[
                :,
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s,
            ] = common_space_u
            combined_space_s[
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s
            ] = common_space_s
            combined_space_v[
                num_tasks * n_dims_per_task : num_tasks * n_dims_per_task
                + common_space_index_s,
                :,
            ] = common_space_v

            u_combined_space_u, s_combined_space_u, v_combined_space_u = (
                torch.linalg.svd(combined_space_u, full_matrices=False)
            )
            u_combined_space_v, s_combined_space_v, v_combined_space_v = (
                torch.linalg.svd(combined_space_v, full_matrices=False)
            )
            combined_space_u = u_combined_space_u @ v_combined_space_u
            combined_space_v = u_combined_space_v @ v_combined_space_v

            combined_space_s = (
                torch.ones_like(combined_space_s) * combined_space_s.mean()
            )

            multi_task_vector[key] = torch.linalg.multi_dot(
                (
                    combined_space_u,
                    torch.diag(combined_space_s),
                    combined_space_v,
                )
            )
            
            del combined_space_u, combined_space_s, combined_space_v
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # Move to CPU for module building
        multi_task_vector_cpu = {k: v.cpu() for k, v in multi_task_vector.items()}
        del multi_task_vector
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()
        
        module_net = build_clip_vit_network_module(list_layer, multi_task_vector_cpu, masses)
        module_vec_cpu = module_net['network'].get_dualitymap()()
        module_vec_flat = flatten_and_move_to_device(module_vec_cpu, device='cpu', clone=False)
        
        del module_net, module_vec_cpu
        gc.collect()
        
        # Move back to GPU
        for key in module_vec_flat:
            multi_task_vector_cpu[key] = module_vec_flat[key].to(self.device)
        
        del module_vec_flat
        gc.collect()
        
        coefficient = self.optimal_alphas[self.model_name][num_tasks]

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector_cpu,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
