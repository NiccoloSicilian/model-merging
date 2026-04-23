"""
Evaluate language model merging on the GLUE benchmark.

Backbone   : google/flan-t5-base
Checkpoints: tanganke/flan-t5-base_glue-{task}
Datasets   : GLUE (cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb)

Usage
-----
    python scripts/evaluate_language_merging.py
    python scripts/evaluate_language_merging.py merger=tsv
    python scripts/evaluate_language_merging.py merger=ta merger.optimal_alpha=0.3
    python scripts/evaluate_language_merging.py device=cpu
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

import model_merging  # noqa
from model_merging.data.language.glue_load_dataset import load_glue_dataset
from model_merging.model.language_classifier import (
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
    SentenceClassification,
    Regression,
)
from model_merging.utils.utils import get_finetuning_accuracies, compute_avg_accuracy

pylogger = logging.getLogger(__name__)


def _get_validation_split(task: str) -> str:
    return "validation_matched" if task == "mnli" else "validation"


def run(cfg: DictConfig) -> None:
    seed_index_everything(cfg)

    tasks: List[str] = list(cfg.benchmark.datasets)
    num_tasks = len(tasks)

    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = num_tasks
    omegaconf.OmegaConf.set_struct(cfg, True)

    # ── 1. Finetuned accuracies (upper-bound for normalisation) ───────────────
    finetuned_accuracies: Dict[str, float] = get_finetuning_accuracies(
        cfg.misc.finetuned_accuracy_path
    )[cfg.nn.encoder.model_name]

    # ── 2. Load pretrained model ──────────────────────────────────────────────
    pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg.nn.encoder.pretrained_model_id
    )

    # ── 3. Load finetuned models ──────────────────────────────────────────────
    finetuned_models = {
        task: AutoModelForSeq2SeqLM.from_pretrained(
            cfg.nn.encoder.finetuned_model_id.format(task=task)
        ).state_dict()
        for task in tasks
    }

    # ── 4. Merge ──────────────────────────────────────────────────────────────
    merger = instantiate(cfg.merger)
    merged_model = merger.merge(pretrained_model, finetuned_models)

    # ── 5. Tokenizer ──────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg.nn.encoder.tokenizer_id)

    # ── 6. Evaluate on GLUE ───────────────────────────────────────────────────
    accelerator = "gpu" if cfg.device.startswith("cuda") else "cpu"
    merged_model = merged_model.to(cfg.device)
    merged_model.eval()

    cache_dir = str(PROJECT_ROOT / ".cache" / "glue_datasets")
    results = {}

    for task in tasks:
        print(f"\n[{task}] evaluating…")
        glue_ds = load_glue_dataset(
            name=task,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            cache_dir=cache_dir,
            split=_get_validation_split(task),
        )

        if task in REGRESSION_TASKS:
            tester = Regression(moe_model=merged_model, tokenizer=tokenizer)
        else:
            tester = SentenceClassification(moe_model=merged_model, tokenizer=tokenizer)

        tester.set_task(task)
        tester.set_metrics()
        tester.set_finetuning_accuracy(finetuned_accuracies[task])

        trainer = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
        )
        test_out = trainer.test(model=tester, dataloaders=glue_ds.data_loader, verbose=False)
        results[task] = test_out

        ft_acc = finetuned_accuracies[task]
        for m in test_out:
            acc  = float(m.get(f"acc/test/{task}", float("nan")))
            norm = float(m.get(f"normalized_acc/test/{task}", acc / ft_acc))
            print(f"  acc = {acc:.4f}  |  normalized_acc = {norm:.4f}  (finetuned = {ft_acc:.4f})")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    avg = compute_avg_accuracy(results)
    results["avg"] = [avg]

    col = 22
    print("\n" + "=" * 60)
    print(f"  Merger: {cfg.merger._target_.split('.')[-1]}   Tasks: {num_tasks}")
    print("=" * 60)
    print(f"  {'Task':<{col}} {'Acc':>8}   {'Norm Acc':>10}   {'Finetuned':>10}")
    print("  " + "-" * 56)
    for task in tasks:
        ft_acc = finetuned_accuracies[task]
        for m in results[task]:
            acc  = float(m.get(f"acc/test/{task}",  float("nan")))
            norm = float(m.get(f"normalized_acc/test/{task}", acc / ft_acc))
            print(f"  {task:<{col}} {acc:>8.4f}   {norm:>10.4f}   {ft_acc:>10.4f}")
    avg_acc  = float(avg.get("acc/test/avg",  float("nan")))
    avg_norm = float(avg.get("normalized_acc/test/avg", float("nan")))
    print("  " + "-" * 56)
    print(f"  {'avg':<{col}} {avg_acc:>8.4f}   {avg_norm:>10.4f}")
    print("=" * 60)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    results_path = Path(cfg.misc.results_path)
    results_path.mkdir(parents=True, exist_ok=True)
    merger_name = cfg.merger._target_.split(".")[-1]
    out_file = results_path / f"{merger_name}_{num_tasks}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved → {out_file}")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="language_merging.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
