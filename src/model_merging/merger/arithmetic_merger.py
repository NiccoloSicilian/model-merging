import copy
import logging
from typing import Dict
import torch
from tqdm import tqdm
from model_merging.merger.merger import TaskVectorBasedMerger
from model_merging.model.encoder import ImageEncoder
from model_merging.utils.utils import (
    apply_dict_to_model,
    compute_task_dict,
    save_module_vec_fast,
)
from model_merging.merging.structured import (
    get_svd_dict,
    isotropic_sum,
)

pylogger = logging.getLogger(__name__)


class TaskArithmeticMerger(TaskVectorBasedMerger):
    def __init__(self, optimal_alpha, svd_path, svd_compress_factor, low_rank_factor, model_name, device=None):
        super().__init__()
        self.optimal_alpha = optimal_alpha
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        self.low_rank_factor = low_rank_factor  # fraction of singular values to keep, e.g. 0.5 keeps top 50%
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

        # 2. SVD-compress all task vectors
        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        # 3. Arithmetic sum over SVD-reconstructed task vectors
        layer_names = list(base_state_dict.keys())
        cumulative_dict = {}

        for layer_name in tqdm(layer_names, desc="Summing SVD"):
            if "text_projection" in layer_name:
                continue

            is_matrix = base_state_dict[layer_name].dim() == 2

            for i, dataset in enumerate(datasets):
                if is_matrix:
                    delta_layer_svd = svd_dict[dataset][layer_name]
                    u = delta_layer_svd["u"].to(self.device)
                    s = delta_layer_svd["s"].to(self.device)
                    v = delta_layer_svd["v"].to(self.device)
                    delta = u @ torch.diag_embed(s) @ v
                    if i == 0:
                        cumulative_dict[layer_name] = torch.zeros_like(delta)
                    cumulative_dict[layer_name] += delta
                else:
                    delta_layer = svd_dict[dataset][layer_name]["dim1"].to(self.device)
                    if i == 0:
                        cumulative_dict[layer_name] = delta_layer
                    else:
                        cumulative_dict[layer_name] += delta_layer

        # 4. Low-rank decomposition of the aggregated task vector
        low_rank_dict = {}
        for layer_name in tqdm(cumulative_dict.keys(), desc="Low-rank compression"):
            tensor = cumulative_dict[layer_name]

            if tensor.dim() == 2 and "text_projection" not in layer_name:
                u, s, vh = torch.linalg.svd(tensor, full_matrices=False)
                k = max(1, int(len(s) * self.low_rank_factor))
                u_k  = u[:, :k]
                s_k  = s[:k]
                vh_k = vh[:k, :]
                low_rank_dict[layer_name] = u_k @ torch.diag(s_k) @ vh_k
            else:
                # 1D layers (biases etc.) are kept as-is
                low_rank_dict[layer_name] = tensor

        # 5. Apply low-rank aggregated task vector to base model
        merged_encoder = apply_dict_to_model(
            low_rank_dict, pretrained_model, coefficient=self.optimal_alpha
        )
        return merged_encoder
