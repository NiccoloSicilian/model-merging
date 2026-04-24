"""
Pre-download and cache all models and datasets needed for language merging evaluation.
Uses the same functions and config as evaluate_language_merging.py.

Usage
-----
    python scripts/cache_language_assets.py
"""

import logging
from typing import List

import hydra
import omegaconf
from omegaconf import DictConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything

import model_merging  # noqa
from model_merging.data.language.glue_load_dataset import load_glue_dataset

pylogger = logging.getLogger(__name__)


def _get_validation_split(task: str) -> str:
    return "validation_matched" if task == "mnli" else "validation"


def run(cfg: DictConfig) -> None:
    seed_index_everything(cfg)

    tasks: List[str] = list(cfg.benchmark.datasets)

    omegaconf.OmegaConf.set_struct(cfg, False)
    cfg.num_tasks = len(tasks)
    omegaconf.OmegaConf.set_struct(cfg, True)

    # ── 1. Pretrained model ───────────────────────────────────────────────────
    print(f"Downloading pretrained model: {cfg.nn.encoder.pretrained_model_id} ...")
    AutoModelForSeq2SeqLM.from_pretrained(cfg.nn.encoder.pretrained_model_id)
    print("  Done.")

    # ── 2. Finetuned models ───────────────────────────────────────────────────
    for task in tasks:
        model_id = cfg.nn.encoder.finetuned_model_id.format(task=task)
        print(f"Downloading finetuned model: {model_id} ...")
        AutoModelForSeq2SeqLM.from_pretrained(model_id)
        print("  Done.")

    # ── 3. Tokenizer ──────────────────────────────────────────────────────────
    print(f"Downloading tokenizer: {cfg.nn.encoder.tokenizer_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.nn.encoder.tokenizer_id)
    print("  Done.")

    # ── 4. GLUE datasets ──────────────────────────────────────────────────────
    cache_dir = str(PROJECT_ROOT / ".cache" / "glue_datasets")
    for task in tasks:
        print(f"Caching GLUE dataset: {task} ...")
        load_glue_dataset(
            name=task,
            tokenizer=tokenizer,
            batch_size=cfg.batch_size,
            cache_dir=cache_dir,
            split=_get_validation_split(task),
        )
        print("  Done.")

    print("\nAll assets cached.")


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="language_merging.yaml")
def main(cfg: omegaconf.DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
