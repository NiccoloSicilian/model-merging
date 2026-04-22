"""
Evaluate language-model merging (TA and TSV) on the GLUE benchmark.

Backbone   : google/flan-t5-base
Checkpoints: tanganke/google/flan-t5-base_glue-{task}   (HuggingFace)
Datasets   : GLUE  (cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb)

Merging methods
---------------
  ta  – Task Arithmetic: θ_merged = θ_pre + α · Σ_i τ_i
  tsv – Task Singular Vectors: SVD-compressed task vectors, then aggregate

Usage
-----
    python scripts/evaluate_language_merging.py
    python scripts/evaluate_language_merging.py merger.method=tsv
    python scripts/evaluate_language_merging.py merger.method=ta merger.ta_alpha=0.8
    python scripts/evaluate_language_merging.py device=cpu
    python scripts/evaluate_language_merging.py benchmark.datasets=[cola,rte,sst2]
"""

import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

import model_merging  # noqa – triggers package __init__
from model_merging.data.language.glue_load_dataset import load_glue_dataset
from model_merging.model.language_classifier import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    SentenceClassification,
    Regression,
)
from model_merging.merging.structured import decompose_task_vectors
from model_merging.utils.utils import get_finetuning_accuracies, compute_avg_accuracy

pylogger = logging.getLogger(__name__)

# Keys to skip when computing task vectors for encoder-decoder T5 models
_TV_SKIP_KEYS = ("embed_tokens", "lm_head", "shared")


# ── Task-vector helpers ───────────────────────────────────────────────────────

def _skip_key(key: str) -> bool:
    return any(s in key for s in _TV_SKIP_KEYS)


def compute_language_task_vector(
    pretrained_sd: dict,
    finetuned_sd: dict,
) -> dict:
    """τ = finetuned − pretrained, skipping embedding / head layers."""
    tv = {}
    for k, bw in pretrained_sd.items():
        if _skip_key(k):
            continue
        if k in finetuned_sd and finetuned_sd[k].shape == bw.shape:
            tv[k] = finetuned_sd[k].float() - bw.float()
    return tv


# ── Merging methods ───────────────────────────────────────────────────────────

def merge_ta(
    pretrained_sd: dict,
    task_vectors: Dict[str, dict],
    alpha: float = 1.0,
) -> dict:
    """Task Arithmetic: θ_merged = θ_pre + α · Σ_i τ_i"""
    merged = {k: v.clone().float() for k, v in pretrained_sd.items()}
    summed: dict = {}
    for tv in task_vectors.values():
        for k, delta in tv.items():
            if k not in summed:
                summed[k] = torch.zeros_like(delta)
            summed[k] = summed[k] + delta
    for k, delta in summed.items():
        if k in merged:
            merged[k] = merged[k] + alpha * delta
    return merged


@torch.no_grad()
def _aggregate_svd_language(
    pretrained_sd: dict,
    svd_dict: dict,
    device: str = "cpu",
    non_matrix_params_agg: str = "mean",
) -> dict:
    """
    TSV aggregation for language models.

    Concatenates low-rank factors across tasks and computes a double SVD,
    then adds the result onto the pretrained weights.

    Operates only on layers present in svd_dict (i.e. task-vector layers),
    leaving embedding/head layers at their pretrained values.
    """
    merged = {k: v.clone().float() for k, v in pretrained_sd.items()}
    datasets = list(svd_dict.keys())
    layer_keys = list(svd_dict[datasets[0]].keys())

    for layer_name in tqdm(layer_keys, desc="TSV aggregation"):
        is_matrix = "u" in svd_dict[datasets[0]][layer_name]
        offset = 0

        if is_matrix:
            # Collect total rank across all tasks
            total_rank = sum(svd_dict[d][layer_name]["s"].shape[0] for d in datasets)
            ref_u = svd_dict[datasets[0]][layer_name]["u"]
            ref_v = svd_dict[datasets[0]][layer_name]["v"]
            sum_u = torch.zeros(ref_u.shape[0], total_rank, device=device)
            sum_s = torch.zeros(total_rank, device=device)
            sum_v = torch.zeros(total_rank, ref_v.shape[1], device=device)

            for dataset in datasets:
                d = svd_dict[dataset][layer_name]
                u = d["u"].to(device)
                s = d["s"].to(device)
                v = d["v"].to(device)
                rank_i = s.shape[0]
                sum_u[:, offset: offset + rank_i] = u
                sum_s[offset: offset + rank_i] = s
                sum_v[offset: offset + rank_i, :] = v
                offset += rank_i

            u_u, _, v_u = torch.linalg.svd(sum_u, full_matrices=False)
            u_v, _, v_v = torch.linalg.svd(sum_v, full_matrices=False)
            task_delta = torch.linalg.multi_dot((u_u, v_u, torch.diag(sum_s), u_v, v_v))
            if layer_name in merged:
                merged[layer_name] = (merged[layer_name].to(device) + task_delta).cpu()

        else:
            # 1-D parameter (bias, etc.)
            if non_matrix_params_agg == "mean":
                vals = [svd_dict[d][layer_name]["dim1"].float() for d in datasets]
                delta = torch.stack(vals).mean(0)
                if layer_name in merged:
                    merged[layer_name] = merged[layer_name].float() + delta.cpu()
            # else: keep pretrained (do nothing)

    return merged


def merge_tsv(
    pretrained_sd: dict,
    task_vectors: Dict[str, dict],
    compress_factor: Optional[float] = None,
    device: str = "cpu",
) -> dict:
    """Task Singular Vectors merge.

    Args:
        pretrained_sd: Pretrained model state dict.
        task_vectors: {task_name: {layer: tensor}}.
        compress_factor: SVD rank = original_rank / compress_factor.
                         Defaults to num_tasks (standard TSV setting).
        device: Computation device.
    """
    num_tasks = len(task_vectors)
    compress_factor = compress_factor or num_tasks
    compress_ratio = 1.0 / compress_factor
    pylogger.info(f"TSV compress_ratio = 1/{compress_factor:.0f} = {compress_ratio:.4f}")
    svd_dict = decompose_task_vectors(task_vectors, compress_ratio)
    return _aggregate_svd_language(pretrained_sd, svd_dict, device=device)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _get_validation_split(task: str) -> str:
    """Return the correct HuggingFace split name for each GLUE task."""
    if task == "mnli":
        return "validation_matched"
    return "validation"


def evaluate_on_glue(
    merged_sd: dict,
    tasks: List[str],
    tokenizer,
    finetuned_accuracies: dict,
    device: str = "cuda",
    batch_size: int = 16,
    cache_dir: Optional[str] = None,
) -> dict:
    """Apply merged_sd to a fresh T5 and evaluate on all GLUE tasks.

    Returns:
        dict mapping task_name → list[metric_dict] (same format as trainer.test()).
    """
    # Load merged model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-base", torch_dtype=torch.float32
    )
    # Build a state dict with the same dtypes as the original model
    orig_sd = model.state_dict()
    cast_merged = {}
    for k, orig_v in orig_sd.items():
        if k in merged_sd:
            cast_merged[k] = merged_sd[k].to(orig_v.dtype)
        else:
            cast_merged[k] = orig_v
    model.load_state_dict(cast_merged, strict=True)
    model = model.to(device)
    model.eval()

    accelerator = "gpu" if device.startswith("cuda") else "cpu"
    results = {}

    for task in tasks:
        pylogger.info(f"  Evaluating {task}…")
        split = _get_validation_split(task)

        glue_ds = load_glue_dataset(
            name=task,
            tokenizer=tokenizer,
            batch_size=batch_size,
            cache_dir=cache_dir,
            split=split,
        )

        if task in REGRESSION_TASKS:
            tester = Regression(moe_model=model, tokenizer=tokenizer)
        else:
            tester = SentenceClassification(moe_model=model, tokenizer=tokenizer)

        tester.set_task(task)
        tester.set_metrics()
        tester.set_finetuning_accuracy(finetuned_accuracies.get(task, 1.0))

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        test_out = trainer.test(
            model=tester,
            dataloaders=glue_ds.data_loader,
            verbose=False,
        )
        results[task] = test_out

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="language_merging.yaml")
def main(cfg: DictConfig) -> None:
    seed_index_everything(cfg)

    tasks: List[str] = list(cfg.benchmark.datasets)
    num_tasks = len(tasks)
    device: str = cfg.device

    pylogger.info(f"Tasks ({num_tasks}): {tasks}")
    pylogger.info(f"Merger: {cfg.merger.method}  |  device: {device}")

    # ── 1. Tokenizer ─────────────────────────────────────────────────────────
    pylogger.info("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(cfg.nn.encoder.tokenizer_id)

    # ── 2. Pretrained state dict ──────────────────────────────────────────────
    pylogger.info(f"Loading pretrained {cfg.nn.encoder.pretrained_model_id}…")
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.nn.encoder.pretrained_model_id
    )
    pretrained_sd = {k: v.float().cpu() for k, v in pretrained_model.state_dict().items()}
    del pretrained_model
    torch.cuda.empty_cache()

    # ── 3. Task vectors ───────────────────────────────────────────────────────
    pylogger.info("Loading finetuned models and computing task vectors…")
    task_vectors: Dict[str, dict] = {}
    for task in tasks:
        hf_id = cfg.nn.encoder.finetuned_model_id.format(task=task)
        pylogger.info(f"  {task}: {hf_id}")
        ft_model = AutoModelForSeq2SeqLM.from_pretrained(hf_id)
        ft_sd = {k: v.float().cpu() for k, v in ft_model.state_dict().items()}
        del ft_model
        torch.cuda.empty_cache()
        task_vectors[task] = compute_language_task_vector(pretrained_sd, ft_sd)
        pylogger.info(f"    → {len(task_vectors[task])} task-vector layers")

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    method = cfg.merger.method.lower()
    pylogger.info(f"Merging with {method.upper()}…")

    if method == "ta":
        alpha = float(omegaconf.OmegaConf.select(cfg, "merger.ta_alpha", default=1.0))
        pylogger.info(f"  α = {alpha}")
        merged_sd = merge_ta(pretrained_sd, task_vectors, alpha=alpha)

    elif method == "tsv":
        compress_factor = omegaconf.OmegaConf.select(cfg, "merger.svd_compress_factor", default=None)
        merged_sd = merge_tsv(
            pretrained_sd,
            task_vectors,
            compress_factor=compress_factor,
            device=device,
        )
    else:
        raise ValueError(f"Unknown merger method: {method!r}. Choose 'ta' or 'tsv'.")

    # Ensure all pretrained keys are present (embeddings / head kept from pretrained)
    for k, v in pretrained_sd.items():
        if k not in merged_sd:
            merged_sd[k] = v.clone()

    # ── 5. Finetuned accuracies (upper-bound for normalisation) ───────────────
    finetuned_accuracies = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    ).get(cfg.nn.encoder.model_name, {})

    # ── 6. Evaluate on GLUE ───────────────────────────────────────────────────
    pylogger.info("Running GLUE evaluation…")
    cache_dir = str(PROJECT_ROOT / ".cache" / "glue_datasets")
    results = evaluate_on_glue(
        merged_sd=merged_sd,
        tasks=tasks,
        tokenizer=tokenizer,
        finetuned_accuracies=finetuned_accuracies,
        device=device,
        batch_size=int(omegaconf.OmegaConf.select(cfg, "batch_size", default=16)),
        cache_dir=cache_dir,
    )

    avg = compute_avg_accuracy(results)
    results["avg"] = [avg]

    pylogger.info("=" * 60)
    pylogger.info(f"Method: {method.upper()}  |  Tasks: {num_tasks}")
    for task, metrics in results.items():
        if task == "avg":
            pylogger.info(f"  avg: {metrics[0]}")
        else:
            for m in metrics:
                key = f"acc/test/{task}"
                if key in m:
                    pylogger.info(f"  {task}: {m[key]:.4f}")
    pylogger.info("=" * 60)

    # ── 7. Save results ───────────────────────────────────────────────────────
    results_path = Path(cfg.misc.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    out_file = results_path / f"{method}_{num_tasks}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    pylogger.info(f"Results saved → {out_file}")


if __name__ == "__main__":
    main()
