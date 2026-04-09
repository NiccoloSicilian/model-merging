import logging
import os
from torch.utils.data import DataLoader, ConcatDataset, Dataset, Subset
import torch.distributed as dist
from rich.console import Console
from rich.table import Table
import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from model_merging.model.encoder import ImageEncoder
from model_merging.model.heads import get_classification_head
from model_merging.model.multitask_classifier import MultiTaskImageClassifier 
from model_merging.utils.io_utils import load_model_from_hf, upload_model_to_hf
from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")
def reset_all_weights(model: nn.Module) -> None:
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # 1. Use built-in reset if available
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()
            return

        # 2. Handle OpenCLIP's MultiheadAttention manually
        # (it wraps nn.MultiheadAttention internals directly as raw Parameters)
        if type(m).__name__ == "MultiheadAttention":
            if hasattr(m, "in_proj_weight") and m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if hasattr(m, "in_proj_bias") and m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            if hasattr(m, "out_proj"):
                nn.init.xavier_uniform_(m.out_proj.weight)
                if m.out_proj.bias is not None:
                    nn.init.zeros_(m.out_proj.bias)

        # 3. Handle ResidualAttentionBlock (has learned scale params)
        if type(m).__name__ == "ResidualAttentionBlock":
            if hasattr(m, "ln_1"):
                nn.init.ones_(m.ln_1.weight)
                nn.init.zeros_(m.ln_1.bias)
            if hasattr(m, "ln_2"):
                nn.init.ones_(m.ln_2.weight)
                nn.init.zeros_(m.ln_2.bias)

    model.apply(weight_reset)
def verify_weights_are_random(model: nn.Module, model_name: str = "model"):
    stats = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            mean = param.data.mean().item()
            std = param.data.std().item()
            abs_max = param.data.abs().max().item()
            stats.append((name, mean, std, abs_max))

    console = Console()
    table = Table(title=f"Weight Stats: {model_name}", header_style="bold magenta")
    table.add_column("Layer", style="cyan", width=50)
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("AbsMax", justify="right")
    table.add_column("OK?", justify="center")

    all_ok = True
    for name, mean, std, abs_max in stats:
        is_bias = name.endswith(".bias")
        is_norm = any(x in name for x in ["ln_", "ln_post", "ln_pre", "ln_final", "norm"])
        is_scalar = std != std  # nan check for single-value params

        if is_norm:
            # LayerNorm: weight=1, bias=0 is correct
            ok = True
        elif is_bias:
            # Biases initialized to 0 is correct
            ok = True
        elif is_scalar:
            # Scalar params like logit_scale — just check it exists
            ok = True
        else:
            # Actual weight matrices: mean~0, std>0
            ok = abs(mean) < 0.1 and std > 1e-6

        if not ok:
            all_ok = False

        table.add_row(
            name,
            f"{mean:.4f}",
            f"{std:.4f}" if std == std else "nan",
            f"{abs_max:.4f}",
            "✅" if ok else "❌"
        )

    console.print(table)

    if all_ok:
        pylogger.info("✅ All weights properly initialized!")
    else:
        pylogger.warning("❌ Some weight matrices may not have been reset!")
        
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

    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )
    
    # IMPORTANT: Unfreeze the encoder
    for param in zeroshot_encoder.parameters():
        param.requires_grad = True

    # ---- NEW: SCRAMBLE THE ENCODER TO START FROM ZERO ----
    pylogger.info("Resetting all encoder weights for from-scratch training!")
    
    reset_all_weights(zeroshot_encoder)
    verify_weights_are_random(zeroshot_encoder, "ViT-B/16 Encoder")
    # ------------------------------------------------------

    classification_heads = nn.ModuleDict()
    train_datasets = []
    val_datasets = []
    test_datasets = []

    for task_config in cfg.benchmark.datasets:
        task_name = task_config.name

        head = get_classification_head(
            cfg.nn.encoder.model_name,
            task_name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )
        for param in head.parameters():
            param.requires_grad = True
        reset_all_weights(head)
        verify_weights_are_random(head, f"Head_{task_name}")
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

    train_loader = DataLoader(train_split, batch_size=cfg.train.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_split,   batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(full_test,   batch_size=cfg.train.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Muon optimizer requires torch.distributed even on single GPU
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="nccl")

    # 3. Instantiate our custom MultiTask model
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

    # Save the trained encoder and full model
    pylogger.info("Saving trained model...")
    save_dir = PROJECT_ROOT
    os.makedirs(save_dir, exist_ok=True)

    # Save the full Lightning model
    trainer.save_checkpoint(os.path.join(save_dir, "multitask_model.ckpt"))

    # Save just the encoder separately (useful for merging experiments)
    torch.save(
        model.encoder.state_dict(),
        os.path.join(save_dir, "encoder.pt")
    )

    # Save each classification head separately
    for task_name, head in model.classifiers.items():
        torch.save(
            head.state_dict(),
            os.path.join(save_dir, f"head_{task_name}.pt")
        )

    pylogger.info(f"✅ Model saved to {save_dir}")
    logger.log_configuration(model, cfg)

    if logger is not None:
        logger.experiment.finish()
        
@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="multitask_muon_train.yaml")
def main(cfg: omegaconf.DictConfig):
    run(cfg)

if __name__ == "__main__":
    main()
    
