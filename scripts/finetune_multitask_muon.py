import logging
import os
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset, WeightedRandomSampler
import torch.distributed as dist
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from model_merging.model.encoder import ImageEncoder
from model_merging.model.heads import get_classification_head
from model_merging.model.multitask_classifier import MultiTaskImageClassifier
from model_merging.utils.io_utils import load_model_from_hf
from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")


class TaskLabeledDataset(Dataset):
    """Wraps a dataset and appends the task name to each sample: (x, y) → (x, y, task_name)."""
    def __init__(self, dataset, task_name: str):
        self.dataset = dataset
        self.task_name = task_name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, y, self.task_name


def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    # Load pretrained encoder — keep original weights, just unfreeze
    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )
    for param in zeroshot_encoder.parameters():
        param.requires_grad = True

    classification_heads = nn.ModuleDict()
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for task_config in cfg.benchmark.datasets:
        task_name = task_config.name

        # Load pretrained zeroshot head — freeze it (only encoder is finetuned)
        head = get_classification_head(
            cfg.nn.encoder.model_name,
            task_name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )
        for param in head.parameters():
            param.requires_grad = False
        classification_heads[task_name] = head

        task_dataset_train = instantiate(
            task_config,
            preprocess_fn=zeroshot_encoder.train_preprocess,
            batch_size=cfg.train.batch_size,
        )
        task_dataset_val = instantiate(
            task_config,
            preprocess_fn=zeroshot_encoder.val_preprocess,
            batch_size=cfg.train.batch_size,
        )
        train_datasets.append(TaskLabeledDataset(task_dataset_train.train_dataset, task_name))
        val_datasets.append(TaskLabeledDataset(task_dataset_val.train_dataset, task_name))
        test_datasets.append(TaskLabeledDataset(task_dataset_val.test_dataset, task_name))

    num_workers = task_dataset_train.train_loader.num_workers

    # Merge all tasks — train and val are parallel (same indices, different transforms)
    full_train = ConcatDataset(train_datasets)
    full_val   = ConcatDataset(val_datasets)
    full_test  = ConcatDataset(test_datasets)

    val_size = int(0.1 * len(full_train))
    train_indices = list(range(val_size, len(full_train)))
    val_indices   = list(range(0, val_size))

    train_split = Subset(full_train, train_indices)
    val_split   = Subset(full_val,   val_indices)

    # Balanced sampler: weight each sample by 1/task_size so all tasks contribute equally
    task_sizes = [len(d) for d in train_datasets]
    all_weights = []
    for size in task_sizes:
        all_weights.extend([1.0 / size] * size)
    train_weights = [all_weights[i] for i in train_indices]
    sampler = WeightedRandomSampler(train_weights, num_samples=len(train_indices), replacement=True)

    train_loader = DataLoader(train_split, batch_size=cfg.train.batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_split,   batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(full_test,   batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Muon requires torch.distributed even on a single GPU
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")

    model: MultiTaskImageClassifier = hydra.utils.instantiate(
        cfg.nn.module,
        encoder=zeroshot_encoder,
        classifiers=classification_heads,
        _recursive_=False,
    )

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        logger=logger,
        enable_checkpointing=False,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_loader)

    # Save — naming: muon_mlt_ft_{epochs}ep
    epochs = cfg.train.trainer.max_epochs
    save_dir = cfg.misc.output_dir
    os.makedirs(save_dir, exist_ok=True)

    trainer.save_checkpoint(os.path.join(save_dir, f"muon_mlt_ft_{epochs}ep.ckpt"))

    torch.save(
        model.encoder.state_dict(),
        os.path.join(save_dir, f"muon_mlt_ft_encoder_{epochs}ep.pt"),
    )

    for task_name, head in model.classifiers.items():
        torch.save(
            head.state_dict(),
            os.path.join(save_dir, f"muon_mlt_ft_head_{task_name}_{epochs}ep.pt"),
        )

    pylogger.info(f"Model saved to {save_dir} (prefix: muon_mlt_ft_*_{epochs}ep)")

    logger.log_configuration(model, cfg)

    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="finetune_multitask_muon.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
