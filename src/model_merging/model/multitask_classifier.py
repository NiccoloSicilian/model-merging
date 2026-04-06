import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union
from rich.table import Table
from rich.console import Console
import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer
from muon import MuonWithAuxAdam

from nn_core.model_logging import NNLogger

from model_merging.data.datamodule import MetaData
from model_merging.data.dataset import maybe_dictionarize
from model_merging.utils.utils import torch_load, torch_save

pylogger = logging.getLogger(__name__)


class MultiTaskImageClassifier(pl.LightningModule):
    logger: NNLogger

    def __init__(
        self, encoder, classifiers: Dict[str, nn.Module], metadata: Optional[MetaData] = None, *args, **kwargs
    ) -> None:
        super().__init__()

        # Populate self.hparams with args and kwargs automagically!
        self.save_hyperparameters(logger=False, ignore=("metadata", "classifiers"))

        self.metadata = metadata
        self.encoder = encoder
        
        # 1. Store classifiers in a ModuleDict so PyTorch registers their parameters correctly
        self.classifiers = nn.ModuleDict(classifiers)
        self.task_names = list(self.classifiers.keys())
        # 2. Setup separate metrics for each task
        self.metrics = nn.ModuleDict()
        for task_name, classifier in self.classifiers.items():
            num_classes = classifier.out_features
            metric = torchmetrics.Accuracy(
                task="multiclass", num_classes=num_classes, top_k=1
            )
            # Registering metrics as sub-modules ensures they are placed on the correct device
            self.metrics[f"train_acc_{task_name}"] = metric.clone()
            self.metrics[f"val_acc_{task_name}"] = metric.clone()
            self.metrics[f"test_acc_{task_name}"] = metric.clone()

        self.log_fn = lambda metric, val: self.log(
            metric, val, on_step=False, on_epoch=True, sync_dist=True
        )

        self.finetuning_accuracy = None

    def set_encoder(self, encoder: torch.nn.Module):
        self.encoder = encoder

    def forward(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        """ Forward pass now requires knowing which task head to use. """
        embeddings = self.encoder(x)
        logits = self.classifiers[task_name](embeddings)
        return logits

    def _step(self, batch_dict: Dict[str, Any], split: str) -> Mapping[str, Any]:
        total_loss = 0.0
        all_logits = {}
        task_accuracies = []
        current_bs = 0
    
        for task_name, task_batch in batch_dict.items():
            task_batch = maybe_dictionarize(task_batch, self.hparams.x_key, self.hparams.y_key)
            x = task_batch[self.hparams.x_key]
            if isinstance(x, list):
                x = torch.stack(x)
            gt_y = task_batch[self.hparams.y_key]
            current_bs = x.shape[0]
    
            logits = self(x, task_name)
            all_logits[task_name] = logits.detach()
    
            loss = F.cross_entropy(logits, gt_y, label_smoothing=0.1)
            total_loss += loss / len(batch_dict)
    
            # Per-task accuracy — epoch level only, not progress bar
            metric = self.metrics[f"{split}_acc_{task_name}"]
            acc = metric(logits, gt_y)
            task_accuracies.append(acc)
    
            self.log(f"{split}/acc/{task_name}", metric,
                     prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=current_bs)
    
            # Per-task loss — useful to detect if one task is dominating
            self.log(f"{split}/loss/{task_name}", loss,
                     prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=current_bs)
    
        # Mean accuracy — only meaningful at epoch level since each step has 1 task
        mean_acc = torch.stack(task_accuracies).mean()
    
        # Progress bar: only total loss and mean acc, step-level for train
        self.log(f"{split}/loss", total_loss,
                 prog_bar=True, on_step=(split == "train"), on_epoch=True,
                 batch_size=current_bs)
    
        self.log(f"{split}/acc_mean", mean_acc,
                 prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=current_bs)
    
        return {"logits": all_logits, "loss": total_loss}
    def on_train_epoch_end(self):
        # 1. Initialize Rich Console for a beautiful table
        console = Console()
        table = Table(title=f"📊 Epoch {self.current_epoch} Summary", show_header=True, header_style="bold magenta")
        
        table.add_column("Dataset", style="cyan", width=20)
        table.add_column("Accuracy", justify="right", style="green")

        total_acc = 0.0
        count = 0

        # 2. Collect results for each task
        for task_name in self.task_names:
            # Retrieve the metric value for this epoch
            metric = self.metrics[f"train_acc_{task_name}"]
            acc = metric.compute()
            
            table.add_row(task_name, f"{acc:.4f}")
            
            total_acc += acc
            count += 1
            
            # Reset metric for the next epoch so it doesn't accumulate 
            # (Lightning usually does this, but manual compute() needs a reset)
            metric.reset()

        # 3. Calculate Average
        avg_acc = total_acc / count if count > 0 else 0
        
        table.add_section()
        table.add_row("OVERALL AVG", f"{avg_acc:.4f}", style="bold yellow")

        # 4. Print to terminal
        console.print(table)
        
        # 5. Log the average to your logger (WandB/Tensorboard)
        self.log("train/acc_epoch_mean", avg_acc, on_epoch=True, prog_bar=True)
    # Add this to get a clean summary after each validation run
    def on_validation_epoch_end(self):
        # This will print a nice table in your console/logs every epoch
        pylogger.info(f"----- Epoch {self.current_epoch} Validation Summary -----")
        for task_name in self.task_names:
            acc = self.metrics[f"val_acc_{task_name}"].compute()
            pylogger.info(f"{task_name:10}: {acc:.4f}")

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        # SAFETY CHECK: PyTorch Lightning's CombinedLoader sometimes wraps the output 
        # into a tuple formatted as (batch_data, batch_idx, dataloader_idx). 
        # This unpacks it if necessary.
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[2], int):
            batch, _, dataloader_idx = batch

        # 1. Use the dataloader index to find the name of the current task
        task_name = self.task_names[dataloader_idx]
        
        # 2. Wrap the single batch in a dictionary so _step() knows how to read it
        wrapped_batch = {task_name: batch}
        
        # 3. Pass it to your existing logic
        return self._step(batch_dict=wrapped_batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        # During val, 'batch' is a single batch. We use dataloader_idx to find the name, 
        # and wrap it in a dict so _step knows how to read it.
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[2], int):
            batch, _, dataloader_idx = batch
        task_name = self.task_names[dataloader_idx]
        wrapped_batch = {task_name: batch}
        return self._step(batch_dict=wrapped_batch, split="val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        if isinstance(batch, tuple) and len(batch) == 3 and isinstance(batch[2], int):
            batch, _, dataloader_idx = batch
        task_name = self.task_names[dataloader_idx]
        wrapped_batch = {task_name: batch}
        return self._step(batch_dict=wrapped_batch, split="test")
        
    def freeze_heads(self):
        """ Freezes all task heads. """
        for head in self.classifiers.values():
            head.weight.requires_grad_(False)
            head.bias.requires_grad_(False)



    def configure_optimizers(self):
        # Muon: internal 2D encoder weights only
        hidden_weights = [
            p for name, p in self.named_parameters()
            if p.requires_grad
            and p.ndim >= 2
            and not any(x in name for x in [
                "embedding", "patch_embed", "cls_token", "pos_embed"
            ])
        ]
    
        # AdamW: encoder biases/norms/embeddings + all classification heads
        muon_param_ids = {id(p) for p in hidden_weights}
        adamw_params = [
            p for p in self.parameters()
            if p.requires_grad and id(p) not in muon_param_ids
        ]
    
        param_groups = [
            dict(params=hidden_weights, use_muon=True,
                 lr=self.hparams.optimizer.lr,
                 weight_decay=self.hparams.optimizer.adamw_wd),
            dict(params=adamw_params, use_muon=False,
                 lr=self.hparams.optimizer.adamw_lr,
                 betas=(0.9, 0.95),
                 weight_decay=self.hparams.optimizer.adamw_wd),
        ]
    
        opt = MuonWithAuxAdam(param_groups)
    
        if "lr_scheduler" not in self.hparams:
            return [opt]
    
        scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]

    def save(self, filename):
        print(f"Saving multi-task image classifier to {filename}")
        torch_save(self, filename)

    @classmethod
    def load(cls, filename):
        print(f"Loading multi-task image classifier from {filename}")
        return torch_load(filename)

    def set_finetuning_accuracy(self, finetuning_accuracy: Dict[str, float]):
        """ Now expects a dictionary of baseline accuracies for each task """
        self.finetuning_accuracy = finetuning_accuracy

    def on_test_epoch_end(self):
        console = Console()
        table = Table(title="🧪 Test Results", show_header=True, header_style="bold cyan")
        
        table.add_column("Dataset", style="cyan", width=20)
        table.add_column("Accuracy", justify="right", style="green")
    
        total_acc = 0.0
        count = 0
    
        for task_name in self.task_names:
            metric = self.metrics[f"test_acc_{task_name}"]
            acc = metric.compute()
            
            table.add_row(task_name, f"{acc:.4f}")
            total_acc += acc
            count += 1
            metric.reset()
    
        avg_acc = total_acc / count if count > 0 else 0
        table.add_section()
        table.add_row("OVERALL AVG", f"{avg_acc:.4f}", style="bold yellow")
        console.print(table)
    
        self.log("test/acc_epoch_mean", avg_acc, on_epoch=True, prog_bar=True)
    
        # Keep existing normalized accuracy logic
        if self.finetuning_accuracy is not None:
            for task_name, baseline_acc in self.finetuning_accuracy.items():
                accuracy = self.trainer.callback_metrics[f"test/acc/{task_name}"].cpu().item()
                normalized_acc = accuracy / baseline_acc
                self.log_fn(f"normalized_acc/test/{task_name}", normalized_acc)
