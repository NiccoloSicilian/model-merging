import copy
import logging
import os
from typing import Tuple
import torch
from tqdm import tqdm
import numpy as np
from pathlib import Path
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.merging.structured import aggregate_decomposed_task_vectors, get_svd_dict
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    is_matrix,
    print_memory,
    save_module_vec_fast,
)

pylogger = logging.getLogger(__name__)


class TaskSingularVectorsMerger(TaskVectorBasedMerger):
    def __init__(self, svd_path, svd_compress_factor, non_matrix_params_aggregation, low_rank_factor=1.0):
        super().__init__()
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.non_matrix_params_aggregation = non_matrix_params_aggregation
        self.alpha
        self.low_rank_factor = low_rank_factor  # fraction of singular values to keep, e.g. 0.5 = top 50%
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 TaskSingularVectorsMerger initialized on device: {self.device}")

    def merge(self, base_model, finetuned_models):
        task_dicts = {}
        datasets = list(finetuned_models.keys())

        # 1. Compute task vectors
        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]
            torch.cuda.empty_cache()

        print_memory("after computing task dicts")

        # 2. SVD-compress all task vectors
        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        # 3. Aggregate decomposed task vectors
        multi_task_vector = aggregate_decomposed_task_vectors(
            ref_state_dict=copy.deepcopy(base_model.state_dict()),
            decomposed_task_vectors=svd_dict,
            non_matrix_params_aggregation=self.non_matrix_params_aggregation,
        )

        # 4. Low-rank compression of the aggregated task vector
        if self.low_rank_factor < 1.0:
            low_rank_vector = {}
            for layer_name in tqdm(multi_task_vector.keys(), desc="Low-rank compression"):
                tensor = multi_task_vector[layer_name]
                if tensor.dim() == 2 and "text_projection" not in layer_name:
                    u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
                    k = max(1, int(len(s) * self.low_rank_factor))
                    low_rank_vector[layer_name] = u[:, :k] @ torch.diag(s[:k]) @ vh[:k, :]
                else:
                    low_rank_vector[layer_name] = tensor
            multi_task_vector = low_rank_vector

        # 5. Apply to base model
        coefficient = 0.8
        print(f"USING coefficient: {coefficient}, low_rank_factor: {self.low_rank_factor}")

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)
        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient
        )
        return merged_encoder
