import os
import copy
import logging

import pytorch_lightning as pl
import torch

pylogger = logging.getLogger(__name__)


class StepCheckpointCallback(pl.Callback):
    """Saves the weight update delta (W_before - W_after) at each step in float16.

    Saves every `save_every_n_steps` steps, plus always the first step.
    """

    def __init__(self, save_dir: str, save_every_n_steps: int = 5):
        super().__init__()
        self.save_dir = save_dir
        self.save_every_n_steps = save_every_n_steps
        self.weights_before = None
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        global_step = trainer.global_step
        if global_step == 0 or global_step % self.save_every_n_steps == 0:
            self.weights_before = {
                name: param.detach().clone()
                for name, param in pl_module.encoder.named_parameters()
                if param.requires_grad
            }

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.weights_before is None:
            return

        global_step = trainer.global_step

        # Compute delta = W_before - W_after (the update applied by the optimizer)
        delta = {}
        for name, param in pl_module.encoder.named_parameters():
            if name in self.weights_before:
                delta[name] = (self.weights_before[name] - param.detach()).to(torch.float16).cpu()

        save_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
        torch.save(delta, save_path)

        pylogger.info(f"Saved update delta at step {global_step}")
        self.weights_before = None
