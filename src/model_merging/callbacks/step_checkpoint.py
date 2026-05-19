import json
import os
import logging

import pytorch_lightning as pl
import torch

pylogger = logging.getLogger(__name__)


class StepCheckpointCallback(pl.Callback):
    """Saves the full encoder weights after each optimizer step in float16.

    Saves every `save_every_n_steps` steps, plus always the first step.
    Also logs training and validation losses to a single JSON file.
    """

    def __init__(self, save_dir: str, save_every_n_steps: int = 5):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        self.loss_log = []
        self.val_log = []
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        # Log training loss at every step
        loss = outputs["loss"].detach().item() if isinstance(outputs, dict) and "loss" in outputs else None
        if loss is not None:
            self.loss_log.append({"step": global_step, "train_loss": loss})

        if global_step == 0 or global_step % self.save_every_n_steps == 0:
            weights = {
                name: param.detach().to(torch.float16).cpu()
                for name, param in pl_module.encoder.named_parameters()
            }

            save_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
            torch.save(weights, save_path)

            pylogger.info(f"Saved full weights at step {global_step}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip sanity check validation
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        task_name = getattr(pl_module, "task_name", "")

        entry = {
            "epoch": trainer.current_epoch,
            "global_step": trainer.global_step,
        }

        val_loss_key = f"loss/val/{task_name}"
        val_acc_key = f"acc/val/{task_name}"

        if val_loss_key in metrics:
            entry["val_loss"] = metrics[val_loss_key].item()
        if val_acc_key in metrics:
            entry["val_acc"] = metrics[val_acc_key].item()

        self.val_log.append(entry)
        pylogger.info(f"Validation epoch {trainer.current_epoch}: {entry}")

    def on_train_end(self, trainer, pl_module):
        log = {
            "train": self.loss_log,
            "val": self.val_log,
        }
        log_path = os.path.join(self.save_dir, "loss_log.json")
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)
        pylogger.info(f"Saved loss log ({len(self.loss_log)} train steps, {len(self.val_log)} val epochs) to {log_path}")
