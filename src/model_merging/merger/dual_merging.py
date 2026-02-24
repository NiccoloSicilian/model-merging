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
    save_module_vec_fast,
)

from model_merging.merging.structured import (
    get_svd_dict,
    isotropic_sum,
    avg_layers,
)
import re
import torch
import gc
from model_merging.merging.dual_arithmetic import *
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
        #save_module_vec_fast(module_net,"matrixesDual_"+self.model_name.replace("/", "-")+"task"+str(len(datasets))+".txt", path="/kaggle/working")
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
