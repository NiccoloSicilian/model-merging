import logging

import hydra
import torch
import torch.nn as nn
from muon import MuonWithAuxAdam
from rich.console import Console
from rich.table import Table

from model_merging.model.image_classifier import ImageClassifier

pylogger = logging.getLogger(__name__)


class MuonImageClassifier(ImageClassifier):
    """ImageClassifier with Muon optimizer for the encoder + AdamW for everything else."""

    def configure_optimizers(self):
        # Muon: encoder 2D weight matrices only
        hidden_weights = [
            p for name, p in self.encoder.named_parameters()
            if p.requires_grad
            and p.ndim == 2
            and not any(x in name for x in [
                "embedding",   # positional_embedding, class_embedding
                "conv1",       # patch projection
                "cls_token",
            ])
        ]

        muon_param_ids = {id(p) for p in hidden_weights}

        # AdamW: encoder biases/norms/embeddings + classification head
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

        # Print optimizer assignment table
        console = Console()
        table = Table(title="Optimizer Assignment", header_style="bold magenta")
        table.add_column("Parameter", style="cyan", width=60)
        table.add_column("Shape", justify="right")
        table.add_column("Optimizer", justify="center")
        for name, p in self.named_parameters():
            if p.requires_grad:
                opt_name = "[green]Muon[/green]" if id(p) in muon_param_ids else "[yellow]AdamW[/yellow]"
                table.add_row(name, str(list(p.shape)), opt_name)
        console.print(table)

        opt = MuonWithAuxAdam(param_groups)

        if "lr_scheduler" not in self.hparams:
            return [opt]

        scheduler = hydra.utils.instantiate(
            self.hparams.lr_scheduler, optimizer=opt
        )
        return [opt], [{"scheduler": scheduler, "interval": "epoch"}]
