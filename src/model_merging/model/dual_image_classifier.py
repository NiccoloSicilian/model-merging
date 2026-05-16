import logging

import hydra
import torch

from model_merging.model.image_classifier import ImageClassifier
from model_merging.merging.dual_arithmetic import build_duality_map, ViT_B_16, ViT_B_32, ViT_L_14
import re
pylogger = logging.getLogger(__name__)


def build_duality_map_with_module(layer_names, grads,  device,mass_schedule, model_name, m):
    """
    Build modular duality map assuming layers are in execution order.
    Applies composition sequentially: layer_N ∘ ... ∘ layer_1 ∘ layer_0
    """
    
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
        if 'positional_embedding' in k and 'visual' in k:
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

class DualImageClassifier(ImageClassifier):
    """ImageClassifier that applies the duality map on gradients before the optimizer step."""

    def __init__(self, encoder, classifier, mass_schedule="uniform", **kwargs):
        super().__init__(encoder=encoder, classifier=classifier, **kwargs)
        self.mass_schedule = mass_schedule
        self._duality_module = ViT_B_16(mass_schedule=mass_schedule)

    def on_before_optimizer_step(self, optimizer):
        # Collect gradients from encoder parameters
        layer_names = []
        grads = {}
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                layer_names.append(name)
                grads[name] = param.grad.clone()

        if not layer_names:
            return

        layer_names = get_vit_topological_order(layer_names)
        # Apply duality map
        dualized_grads = build_duality_map_with_module(
            layer_names=layer_names,
            grads=grads,
            device=self.device,
            mass_schedule=self.mass_schedule,
            model_name='B-16',
            m=self._duality_module
        )

        # Replace gradients with dualized versions
        for name, param in self.encoder.named_parameters():
            if name in dualized_grads:
                param.grad.copy_(dualized_grads[name])
