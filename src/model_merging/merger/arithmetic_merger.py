import copy
import logging
from typing import Dict
import torch
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    sum_task_dict,
    save_module_vec_fast,
)

pylogger = logging.getLogger(__name__)

class TaskArithmeticMerger(TaskVectorBasedMerger):
    def __init__(self, optimal_alpha, device="cuda"):
        super().__init__()
        self.optimal_alpha = optimal_alpha

    def merge(
        self, base_model: ImageEncoder, finetuned_models: Dict[str, object]
    ) -> ImageEncoder:
        cumulative_dict = {}
        datasets = list(finetuned_models.keys())

        base_model.cuda()
        base_state_dict = base_model.state_dict()
        pretrained_model = copy.deepcopy(base_model)
        
        for dataset in datasets:
            model = finetuned_models[dataset]

            # Handle finetuned model as either a model or state_dict
            if isinstance(model, dict):
                finetuned_state_dict = {k: v.cuda() for k, v in model.items()}
            else:
                model.cuda()
                finetuned_state_dict = model.state_dict()

            cumulative_dict = sum_task_dict(
                cumulative_dict,
                compute_task_dict(base_state_dict, finetuned_state_dict),
            )
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        save_module_vec_fast(
            cumulative_dict,
            "matrixesTA_VITB16" + "task" + str(len(datasets)) + ".txt",
            path="/kaggle/working"
        )

        merged_encoder = apply_dict_to_model(
            cumulative_dict, pretrained_model, coefficient=self.optimal_alpha
        )
        return merged_encoder
