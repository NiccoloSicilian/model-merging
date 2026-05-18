import os
import copy
import logging

import pytorch_lightning as pl
import torch

pylogger = logging.getLogger(__name__)


class StepCheckpointCallback(pl.Callback):
    """Saves the full encoder weights after each optimizer step in float16.

    Saves every `save_every_n_steps` steps, plus always the first step.
    """

    def __init__(self, save_dir: str, save_every_n_steps: int = 5):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        if global_step == 0 or global_step % self.save_every_n_steps == 0:
            weights = {
                name: param.detach().to(torch.float16).cpu()
                for name, param in pl_module.encoder.named_parameters()
                if param.requires_grad
            }

            save_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
            torch.save(weights, save_path)

            pylogger.info(f"Saved full weights at step {global_step}")
