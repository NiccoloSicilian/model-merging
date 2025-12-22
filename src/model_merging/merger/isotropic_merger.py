import copy
import logging
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
)

import torch

pylogger = logging.getLogger(__name__)

class IsotropicMerger(TaskVectorBasedMerger):

    def __init__(self, optimal_alphas, svd_path, svd_compress_factor, device=None):
        super().__init__()
        self.optimal_alphas = optimal_alphas
        self.svd_path = svd_path
        self.svd_compress_factor = svd_compress_factor
        
        # 1. Store the device (default to CPU if not provided)
        self.device = device if device is not None else torch.device("cpu")

    @torch.no_grad()
    def merge(self, base_model, finetuned_models):
        # 2. Move base model to the correct device immediately
        base_model = base_model.to(self.device)

        task_dicts = {}
        datasets = list(finetuned_models.keys())
        
        # BUG FIX: Calculate num_tasks BEFORE deleting items from the dictionary
        num_tasks = len(datasets) 

        for dataset in datasets:
            # 3. Move the specific finetuned model to device just before use
            # We assume finetuned_models[dataset] is a model object
            ft_model = finetuned_models[dataset].to(self.device)

            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), ft_model
            )
            
            # Cleanup to save VRAM
            del finetuned_models[dataset] 
            del ft_model # Explicitly delete the reference
            
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        # print_memory("after computing task dicts")

        svd_dict = get_svd_dict(
            task_dicts, datasets, self.svd_path, self.svd_compress_factor
        )

        # Ensure the reference state dict is on the right device
        ref_state_dict = {k: v.to(self.device) for k, v in base_model.state_dict().items()}

        multi_task_vector = isotropic_sum(
            ref_state_dict=ref_state_dict,
            svd_dict=svd_dict,
        )

        model_name = self.cfg.nn.module.encoder.model_name

        # Default coefficient in case the condition below is not met
        coefficient = 1.0 

        if (
            model_name in self.optimal_alphas
            and num_tasks in self.optimal_alphas[model_name]
        ):
            coefficient = self.optimal_alphas[model_name][num_tasks]

        # base_model is already on self.device, so the copy will be too
        merged_encoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder

class IsotropicCommonTaskSpecificMerger(TaskVectorBasedMerger):
    def __init__(
        self,
        common_space_fraction,
        optimal_alphas,
        model_name,
        device,
    ):
        super().__init__()

        self.common_space_fraction = common_space_fraction
        self.optimal_alphas = optimal_alphas
        self.model_name = model_name
        self.device = device

    @torch.no_grad()
    def merge(self, base_model, finetuned_models) -> ImageEncoder | None:

        multi_task_vector = {}

        task_dicts = {}

        datasets = list(finetuned_models.keys())
        num_tasks = len(datasets)

        for dataset in datasets:
            task_dicts[dataset] = compute_task_dict(
                base_model.state_dict(), finetuned_models[dataset]
            )
            del finetuned_models[dataset]  # Delete one model at a time
            torch.cuda.empty_cache()

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
                        multi_task_vector[key] += (vec - multi_task_vector[key]) / (
                            i + 1
                        )
                continue

            pylogger.info(f"Computing common space using sum for {key}...")
            combined_w = sum(
                [task_dict[key].to(self.device) for task_dict in task_dicts.values()]
            )

            ### Calculate the common space size (making sure that task specific space is equally divisible) ###
            common_space_index_s = int(min(shape_) * self.common_space_fraction)
            _task_specific_total_space_index_s = (
                round((min(shape_) - common_space_index_s) / num_tasks) * num_tasks
            )
            common_space_index_s = min(shape_) - _task_specific_total_space_index_s

            u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
            common_space_u = u[:, :common_space_index_s]
            common_space_s = s[:common_space_index_s]
            common_space_v = v[:common_space_index_s, :]
            ###################################################################

            ### Calculate task specific space ###
            n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
            for i, task_dict in enumerate(task_dicts.values()):
                w = task_dict[key].to(self.device)

                # calculate the projection onto task specific space to remove the common space
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
            ###################################################################

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

            ### Orthogonalize combined_space_u and combined_space_v ###
            u_combined_space_u, s_combined_space_u, v_combined_space_u = (
                torch.linalg.svd(combined_space_u, full_matrices=False)
            )
            u_combined_space_v, s_combined_space_v, v_combined_space_v = (
                torch.linalg.svd(combined_space_v, full_matrices=False)
            )
            combined_space_u = u_combined_space_u @ v_combined_space_u
            combined_space_v = u_combined_space_v @ v_combined_space_v
            ###################################################################

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

        coefficient = self.optimal_alphas[self.model_name][num_tasks]

        merged_encoder: ImageEncoder = copy.deepcopy(base_model)

        merged_encoder = apply_dict_to_model(
            multi_task_vector,
            merged_encoder,
            coefficient=coefficient,
        )

        return merged_encoder
