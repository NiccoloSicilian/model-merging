


import copy
import logging
from typing import Dict
import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    save_module_vec_fast,
    sum_task_dict,
)
pylogger = logging.getLogger(__name__)


class TaskArithmeticMerger(TaskVectorBasedMerger):
    def __init__(self, optimal_alpha, svd_path, svd_compress_factor, model_name, device=None):
        super().__init__()
        self.optimal_alpha = optimal_alpha
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 TaskArithmeticMerger initialized on device: {self.device}")

    @torch.no_grad()
    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, object]
    ) -> ImageEncoder:
        base_model = base_model.to(self.device)
        base_state_dict = base_model.state_dict()
        pretrained_model = copy.deepcopy(base_model)

        task_dicts = {}
        datasets = list(finetuned_models.keys())

        # 1. Compute task vectors
        for dataset in datasets:
            model = finetuned_models[dataset]
            if isinstance(model, dict):
                ft_state_dict = {k: v.to(self.device) for k, v in model.items()}
            else:
                model.to(self.device)
                ft_state_dict = model.state_dict()

            task_dicts[dataset] = compute_task_dict(base_state_dict, ft_state_dict)

            del finetuned_models[dataset]
            del ft_state_dict
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # 2. Sum task vectors directly
        cumulative_dict = {}
        for task_dict in task_dicts.values():
            cumulative_dict = sum_task_dict(cumulative_dict, task_dict)

        # 3. Apply summed task vector to base model
        merged_encoder = apply_dict_to_model(
            cumulative_dict, pretrained_model, coefficient=self.optimal_alpha
        )
        return merged_encoder
