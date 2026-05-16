import logging

import hydra
import torch

from model_merging.model.image_classifier import ImageClassifier
from model_merging.merging.dual_arithmetic import build_duality_map
import re
pylogger = logging.getLogger(__name__)



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

        # Extract model name from encoder
        model_name = self.encoder.model_name if hasattr(self.encoder, "model_name") else ""
        layer_names = get_vit_topological_order(layer_names)
        print(layer_names)
        # Apply duality map
        dualized_grads = build_duality_map(
            layer_names=layer_names,
            grads=grads,
            device=self.device,
            mass_schedule=self.mass_schedule,
            model_name='B-16',
        )

        # Replace gradients with dualized versions
        for name, param in self.encoder.named_parameters():
            if name in dualized_grads:
                param.grad.copy_(dualized_grads[name])
