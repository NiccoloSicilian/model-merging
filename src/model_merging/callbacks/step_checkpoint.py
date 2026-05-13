import os
import logging

import pytorch_lightning as pl
import torch

pylogger = logging.getLogger(__name__)


class StepCheckpointCallback(pl.Callback):
    """Saves gradients and learning rate after every gradient update step.

    Saves the two components of the update rule W = W - η * g:
        - g: the gradient for each encoder parameter
        - η: the learning rate at that step
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step

        # Collect gradients
        grads = {}
        for name, param in pl_module.encoder.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone().cpu()

        # Get the current learning rate from the optimizer
        optimizer = trainer.optimizers[0]
        lr = optimizer.param_groups[0]["lr"]

        save_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
        torch.save({"gradients": grads, "lr": lr}, save_path)

        pylogger.info(f"Saved gradients and lr={lr} at step {global_step}")
