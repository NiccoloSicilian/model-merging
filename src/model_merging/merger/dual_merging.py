import copy
import logging
import math
from math import sqrt
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

import torch
import gc

pylogger = logging.getLogger(__name__)
import torch
import copy
from pathlib import Path

def scale_nested(scalar, data):
    """
    Recursively multiply a scalar with a tensor or nested tuple of tensors.
    """
    if isinstance(data, dict):
        result = {}
        for k in data:
            result[k] = data[k] * scalar
        return result
    elif isinstance(data, tuple):
        return tuple(scale_nested(scalar, x) for x in data)
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

def flatten_and_move_to_device(nested, device='cuda:0', clone=False):
    """
    Recursively flatten nested tuples of dictionaries into a single dictionary,
    move all tensors to the specified device, optionally clone them.
    """
    flat_dict = {}
    
    if isinstance(nested, dict):
        for k, v in nested.items():
            if clone:
                flat_dict[k] = v.clone().to(device)
            else:
                flat_dict[k] = v.to(device) if v.device != torch.device(device) else v
    elif isinstance(nested, tuple):
        for item in nested:
            flat_dict.update(flatten_and_move_to_device(item, device, clone))
    else:
        raise TypeError(f"Unexpected type: {type(nested)}")
    
    return flat_dict


class Module:
    def __init__(self, mass, sensitivity, dualize=None):
        self.mass = mass
        self.sensitivity = sensitivity
        self.dualize = dualize
    
    def set_mass(self, mass):
        self.mass = mass
        
    def set_dualize(self, dualize):
        self.dualize = dualize
        
    def get_mass(self):
        return self.mass
    
    def get_sensitivity(self):
        return self.sensitivity
        
    def get_dualitymap(self):
        return self.dualize


def create_linear_mod(g, name, mass):
    # Move to CPU immediately for SVD
    g_cpu = g.cpu()
    
    def linear_dualize():
        U, S, Vt = torch.linalg.svd(g_cpu, full_matrices=False)
        result = U @ Vt * sqrt(g_cpu.shape[0] / g_cpu.shape[1])
        # Keep on CPU
        return {name: result}
    
    M = Module(mass, 1, linear_dualize)
    return M


def create_conv2d_mod(g, name, mass):
    # Move to CPU immediately
    g_cpu = g.cpu()
    
    def conv_dualize():
        matrix = g_cpu
        dout, din, k, _ = matrix.shape
        
        scaling_factor = (1.0 / k**2) * math.sqrt(dout / din)
        transformed = torch.zeros_like(matrix)
        
        for i in range(k):
            for j in range(k):
                slice_matrix = matrix[:, :, i, j]
                U, S, Vt = torch.linalg.svd(slice_matrix, full_matrices=False)
                reconstructed = U @ Vt
                transformed[:, :, i, j] = scaling_factor * reconstructed
        
        return {name: transformed}
    
    M = Module(mass, 1, conv_dualize)
    return M


def create_embedding_mod(g, name, mass):
    # Move to CPU
    g_cpu = g.cpu()
    
    def embedding_dualize():
        rms_norm = torch.sqrt(torch.mean(g_cpu ** 2, dim=0, keepdim=True) + 1e-8)
        return {name: g_cpu / rms_norm}
    
    M = Module(mass, 1, embedding_dualize)
    return M


def concatenate(M1, M2):
    M = Module(M1.get_mass() + M2.get_mass(), 
               M1.get_sensitivity() + M2.get_sensitivity())
    
    def concat_dualize():
        ratio1 = M1.get_mass() / M.get_mass()
        ratio2 = M2.get_mass() / M.get_mass()
        g1 = M1.get_dualitymap()()
        g2 = M2.get_dualitymap()()
        result = (scale_nested(ratio1, g1), scale_nested(ratio2, g2))
        return result
    
    M.set_dualize(concat_dualize)
    return M


def compose(M2, M1):
    M = Module(M1.get_mass() + M2.get_mass(), 
               M1.get_sensitivity() * M2.get_sensitivity())
    
    def compose_dualize():
        sensitivity_factor = 1.0 / M2.get_sensitivity()
        ratio1 = M1.get_mass() / M.get_mass()
        ratio2 = M2.get_mass() / M.get_mass()
        g1 = M1.get_dualitymap()()
        g2 = M2.get_dualitymap()()
        result = (scale_nested(sensitivity_factor * ratio1, g1),
                scale_nested(ratio2, g2))
        return result
    
    M.set_dualize(compose_dualize)
    return M


def build_clip_vit_network_module(layer_names, grads, masses):
    """
    Build a modular duality network for CLIP ViT - MEMORY OPTIMIZED VERSION
    """
    module_map = {}
    
    print("\n" + "="*80)
    print("Step 1: Creating Atomic Layer Modules")
    print("="*80)
    
    for name in layer_names:
        # Skip biases, layer norms, and non-trainable parameters
        if any(skip in name for skip in ['bias', 'ln_', 'class_embedding', 'logit_scale']):
            continue
        
        mass = masses[name]
        
        # Visual conv1
        if 'visual.conv1.weight' in name:
            module_map['visual_conv1'] = create_conv2d_mod(grads[name], name, mass)
            print(f"✓ visual_conv1: Conv2D module")
        
        # Visual projection
        elif 'visual.proj' in name and 'out_proj' not in name:
            module_map['visual_proj'] = create_linear_mod(grads[name], name, mass)
            print(f"✓ visual_proj: Linear module")
        
        # Visual positional embedding
        elif 'visual.positional_embedding' in name:
            module_map['visual_positional_embedding'] = create_linear_mod(grads[name], name, mass)
            print(f"✓ visual.positional_embedding: Linear module")
        
        # Visual transformer blocks
        elif 'visual.transformer.resblocks' in name and 'weight' in name:
            parts = name.split('.')
            try:
                resblocks_idx = parts.index('resblocks')
                block_idx = int(parts[resblocks_idx + 1])
            except (ValueError, IndexError):
                print(f"⚠ Skipping malformed name: {name}")
                continue
            
            block_name = f"visual_block_{block_idx}"
            
            if 'attn.in_proj_weight' in name:
                key = f'{block_name}_attn_in'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name], name, mass)
                    print(f"✓ {key}: Linear module")
            elif 'attn.out_proj.weight' in name:
                key = f'{block_name}_attn_out'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name], name, mass)
                    print(f"✓ {key}: Linear module")
            elif 'mlp.c_fc.weight' in name:
                key = f'{block_name}_mlp_fc'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name], name, mass)
                    print(f"✓ {key}: Linear module")
            elif 'mlp.c_proj.weight' in name:
                key = f'{block_name}_mlp_proj'
                if key not in module_map:
                    module_map[key] = create_linear_mod(grads[name], name, mass)
                    print(f"✓ {key}: Linear module")
    
    # ========================================================================
    # Step 2: Build visual transformer blocks
    # ========================================================================
    print("\n" + "="*80)
    print("Step 2: Building Visual Transformer Blocks")
    print("="*80)
    
    block_indices = set()
    for key in module_map.keys():
        if key.startswith('visual_block_'):
            parts = key.split('_')
            if len(parts) >= 3 and parts[2].isdigit():
                block_indices.add(int(parts[2]))
    
    num_blocks = len(block_indices)
    print(f"\nFound {num_blocks} transformer blocks")
    
    visual_blocks = []
    for i in sorted(block_indices):
        block_name = f"visual_block_{i}"
        
        required_keys = [
            f'{block_name}_attn_in',
            f'{block_name}_attn_out',
            f'{block_name}_mlp_fc',
            f'{block_name}_mlp_proj'
        ]
        
        if not all(key in module_map for key in required_keys):
            print(f"⚠ Skipping incomplete block {i}")
            continue
        
        # Compose attention: out_proj ∘ in_proj
        attn_block = compose(
            module_map[f'{block_name}_attn_out'],
            module_map[f'{block_name}_attn_in']
        )
        
        # Compose MLP: c_proj ∘ c_fc
        mlp_block = compose(
            module_map[f'{block_name}_mlp_proj'],
            module_map[f'{block_name}_mlp_fc']
        )
        
        # Compose full block: mlp ∘ attn
        full_block = compose(mlp_block, attn_block)
        visual_blocks.append(full_block)
        
        # Clean up intermediate modules
        del module_map[f'{block_name}_attn_in']
        del module_map[f'{block_name}_attn_out']
        del module_map[f'{block_name}_mlp_fc']
        del module_map[f'{block_name}_mlp_proj']
        
        print(f"✓ {block_name} = mlp ∘ attn  [Mass: {full_block.get_mass():.2f}]")
    
    # ========================================================================
    # Step 3: Compose all visual transformer blocks sequentially
    # ========================================================================
    print("\n" + "="*80)
    print("Step 3: Composing Visual Transformer Blocks Sequentially")
    print("="*80)
    
    if len(visual_blocks) == 0:
        print("⚠ ERROR: No visual blocks found!")
        return module_map
    
    visual_transformer = visual_blocks[0]
    for i in range(1, len(visual_blocks)):
        visual_transformer = compose(visual_blocks[i], visual_transformer)
        print(f"✓ Composed blocks 0-{i}  [Mass: {visual_transformer.get_mass():.2f}]")
    
    module_map['visual_transformer'] = visual_transformer
    print(f"\n✓ visual_transformer complete [Mass: {visual_transformer.get_mass():.2f}]")
    
    # ========================================================================
    # Step 4: Build visual encoder
    # ========================================================================
    print("\n" + "="*80)
    print("Step 4: Building Visual Encoder")
    print("="*80)
    
    if 'visual_conv1' not in module_map:
        print("⚠ ERROR: visual_conv1 not found!")
        return module_map
    
    visual_pre_transf = module_map['visual_conv1']
    
    if 'visual_positional_embedding' in module_map:
        visual_pre_transf = compose(
            module_map['visual_positional_embedding'],
            visual_pre_transf
        )
    
    visual_backbone = compose(visual_transformer, visual_pre_transf)
    
    print(f"✓ visual_backbone = visual_transformer ∘ conv1")
    print(f"  Mass: {visual_backbone.get_mass():.2f}")
    
    if 'visual_proj' in module_map:
        visual_encoder = compose(module_map['visual_proj'], visual_backbone)
        print(f"✓ visual_encoder = visual_proj ∘ visual_backbone")
        print(f"  Mass: {visual_encoder.get_mass():.2f}")
    else:
        visual_encoder = visual_backbone
    
    module_map['network'] = visual_encoder
    
    print(f"\n{'='*80}")
    print(f"✓ NETWORK = visual_encoder")
    print(f"  Total Mass:        {module_map['network'].get_mass():.2f}")
    print(f"  Total Sensitivity: {module_map['network'].get_sensitivity():.2f}")
    print(f"{'='*80}")
    
    return module_map


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
            
            del finetuned_models[dataset] 
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
        
        list_layer = [key for key in multi_task_vector_cpu]
        masses = {key: 0.5 for key in multi_task_vector_cpu}
        
        # Build network on CPU
        module_net = build_clip_vit_network_module(list_layer, multi_task_vector_cpu, masses)
        
        # Get dualized vectors (already on CPU)
        module_vec_cpu = module_net['network'].get_dualitymap()()
        module_vec_flat = flatten_and_move_to_device(
            module_vec_cpu,
            device='cpu',
            clone=False
        )
        
        del module_net
        del module_vec_cpu
        gc.collect()
        
        # Move back to GPU only what we need
        for key in module_vec_flat:
            multi_task_vector_cpu[key] = module_vec_flat[key].to(self.device)
        
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
