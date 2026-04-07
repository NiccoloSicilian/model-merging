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

    def on_train_epoch_end(self):
        # 1. Initialize Rich Console for a beautiful table
        console = Console()
        table = Table(title=f"📊 Epoch {self.current_epoch} Summary", show_header=True, header_style="bold magenta")
        
        table.add_column("Dataset", style="cyan", width=20)
        table.add_column("Accuracy", justify="right", style="green")

        total_acc = 0.0
        count = 0

        # 2. Collect results for each task — read values Lightning already computed
        for task_name in self.task_names:
            acc = self.trainer.callback_metrics.get(f"train/acc/{task_name}", torch.tensor(0.0))
            table.add_row(task_name, f"{acc:.4f}")
            total_acc += acc
            count += 1

        # 3. Calculate Average
        avg_acc = total_acc / count if count > 0 else 0
        
        table.add_section()
        table.add_row("OVERALL AVG", f"{avg_acc:.4f}", style="bold yellow")

        # 4. Print to terminal
        console.print(table)
        
        # 5. Log the average to your logger (WandB/Tensorboard)
        self.log("train/acc_epoch_mean", avg_acc, on_epoch=True, prog_bar=True)
    # Add this to get a clean summary after each validation run
    def on_validation_batch_end(
        self, outputs: Mapping[str, Any], batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """
        Useful step-level logging for validation.
        outputs contains the dictionary returned by validation_step: {"logits": ..., "loss": ...}
        """
        batch_loss = outputs.get("loss")
        
        if batch_loss is not None:
            # 1. Log to the progress bar for live updates during the validation loop
            # We set on_step=True and on_epoch=False so it doesn't interfere with your epoch-level averages
            self.log(
                "val/batch_loss", 
                batch_loss, 
                prog_bar=True, 
                on_step=True, 
                on_epoch=False, 
                sync_dist=True
            )
            
            # 2. Debugging: Log a warning if the loss spikes unexpectedly on a specific batch
            # (Adjust the 5.0 threshold based on your specific dataset and normal loss scale)
            if batch_loss > 5.0: 
                task_name = self.task_names[dataloader_idx]
                pylogger.warning(
                    f"⚠️ High validation loss ({batch_loss:.4f}) detected on task '{task_name}' "
                    f"at batch_idx {batch_idx}."
                )

    def training_step(self, batch, batch_idx):
        # Batch from ConcatDataset: (x, y, task_names) where task_names is a list of strings
        x, gt_y, task_names = batch

        # Group sample indices by task
        task_indices: Dict[str, list] = {}
        for i, task_name in enumerate(task_names):
            task_indices.setdefault(task_name, []).append(i)

        total_loss = 0.0
        for task_name, indices in task_indices.items():
            idx = torch.tensor(indices, device=x.device)
            x_task = x[idx]
            y_task = gt_y[idx]

            logits = self(x_task, task_name)
            loss = F.cross_entropy(logits, y_task, label_smoothing=0.1)
            total_loss += loss / len(task_indices)

            metric = self.metrics[f"train_acc_{task_name}"]
            acc = metric(logits, y_task)
            self.log(f"train/acc/{task_name}", acc, on_step=False, on_epoch=True, batch_size=x_task.shape[0], sync_dist=True)
            self.log(f"train/loss/{task_name}", loss, on_step=False, on_epoch=True, batch_size=x_task.shape[0], sync_dist=True)

        self.log("train/loss", total_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        return total_loss

    def _labeled_step(self, batch: Any, split: str) -> Mapping[str, Any]:
        """Shared logic for val/test steps. Batch format: (x, y, task_names)."""
        x, gt_y, task_names = batch

        task_indices: Dict[str, list] = {}
        for i, task_name in enumerate(task_names):
            task_indices.setdefault(task_name, []).append(i)

        total_loss = 0.0
        for task_name, indices in task_indices.items():
            idx = torch.tensor(indices, device=x.device)
            x_task = x[idx]
            y_task = gt_y[idx]

            logits = self(x_task, task_name)
            loss = F.cross_entropy(logits, y_task)
            total_loss += loss / len(task_indices)

            metric = self.metrics[f"{split}_acc_{task_name}"]
            acc = metric(logits, y_task)
            self.log(f"{split}/acc/{task_name}", acc, on_step=False, on_epoch=True, batch_size=x_task.shape[0], sync_dist=True)
            self.log(f"{split}/loss/{task_name}", loss, on_step=False, on_epoch=True, batch_size=x_task.shape[0], sync_dist=True)

        self.log(f"{split}/loss", total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        return {"loss": total_loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._labeled_step(batch, split="val")

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        return self._labeled_step(batch, split="test")
        
    def freeze_heads(self):
        """ Freezes all task heads. """
        for head in self.classifiers.values():
            head.weight.requires_grad_(False)
            head.bias.requires_grad_(False)



    def configure_optimizers(self):
        # Muon: internal 2D encoder weights only
        hidden_weights = [
            p for name, p in self.encoder.named_parameters()
            if p.requires_grad
            and p.ndim == 2                  # Muon expects 2D matrices only
            and not any(x in name for x in [
                "embedding",                 # positional_embedding, class_embedding
                "conv1",                     # patch projection (OpenCLIP name)
                "cls_token",
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
    
        muon_ids = {id(p) for p in hidden_weights}
        console = Console()
        table = Table(title="Optimizer Assignment", header_style="bold magenta")
        table.add_column("Parameter", style="cyan", width=60)
        table.add_column("Shape", justify="right")
        table.add_column("Optimizer", justify="center")
        for name, p in self.named_parameters():
            if p.requires_grad:
                opt_name = "[green]Muon[/green]" if id(p) in muon_ids else "[yellow]AdamW[/yellow]"
                table.add_row(name, str(list(p.shape)), opt_name)
        console.print(table)

        opt = MuonWithAuxAdam(param_groups)
    
        if "lr_scheduler" not in self.hparams:
            return [opt]
    
        scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=opt
        )
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]

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
            acc = self.trainer.callback_metrics.get(f"test/acc/{task_name}", torch.tensor(0.0))
            table.add_row(task_name, f"{acc:.4f}")
            total_acc += acc
            count += 1
    
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
