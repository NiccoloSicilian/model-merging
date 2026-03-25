import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch.optim import Optimizer

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
        """ 
        PyTorch Lightning will pass a dict of batches if given a dict of dataloaders.
        Format: {"task1": batch_for_task1, "task2": batch_for_task2}
        """
        total_loss = 0.0
        all_logits = {}

        # 3. Iterate through all tasks present in the current batch
        for task_name, task_batch in batch_dict.items():
            task_batch = maybe_dictionarize(task_batch, self.hparams.x_key, self.hparams.y_key)

            x = task_batch[self.hparams.x_key]
            gt_y = task_batch[self.hparams.y_key]

            logits = self(x, task_name)
            all_logits[task_name] = logits.detach()

            loss = F.cross_entropy(logits, gt_y)
            preds = torch.softmax(logits, dim=-1)

            # Update and log task-specific metrics
            metric = self.metrics[f"{split}_acc_{task_name}"]
            metric.update(preds, gt_y)

            self.log_fn(f"acc/{split}/{task_name}", metric)
            self.log_fn(f"loss/{split}/{task_name}", loss)

            # 4. Accumulate the loss
            total_loss += loss

        # Log the aggregated loss
        self.log_fn(f"loss/{split}/total", total_loss)

        return {"logits": all_logits, "loss": total_loss}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        # During training, 'batch' is a dictionary of tasks. We can pass it directly.
        return self._step(batch_dict=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        # During val, 'batch' is a single batch. We use dataloader_idx to find the name, 
        # and wrap it in a dict so _step knows how to read it.
        task_name = self.task_names[dataloader_idx]
        wrapped_batch = {task_name: batch}
        return self._step(batch_dict=wrapped_batch, split="val")

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Mapping[str, Any]:
        # Same as validation!
        task_name = self.task_names[dataloader_idx]
        wrapped_batch = {task_name: batch}
        return self._step(batch_dict=wrapped_batch, split="test")
        
    def freeze_heads(self):
        """ Freezes all task heads. """
        for head in self.classifiers.values():
            head.weight.requires_grad_(False)
            head.bias.requires_grad_(False)

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters())
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
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
        if self.finetuning_accuracy is not None:
            for task_name, baseline_acc in self.finetuning_accuracy.items():
                accuracy = (
                    self.trainer.callback_metrics[f"acc/test/{task_name}"].cpu().item()
                )
                normalized_acc = accuracy / baseline_acc
                self.log_fn(f"normalized_acc/test/{task_name}", normalized_acc)
