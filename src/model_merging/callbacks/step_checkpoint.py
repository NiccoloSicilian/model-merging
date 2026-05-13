import os
import copy
import logging

import pytorch_lightning as pl
import torch

pylogger = logging.getLogger(__name__)


class StepCheckpointCallback(pl.Callback):
    """Saves all components needed to reconstruct the AdamW update at each step.

    For AdamW the update is:
        m = β1 * m + (1 - β1) * g
        v = β2 * v + (1 - β2) * g²
        m̂ = m / (1 - β1^t)
        v̂ = v / (1 - β2^t)
        W = W * (1 - lr * weight_decay) - lr * m̂ / (√v̂ + ε)

    Saved per step:
        - gradients: raw gradient g for each param
        - lr: current learning rate
        - weight_decay: weight decay coefficient
        - betas: (β1, β2)
        - eps: ε
        - optimizer_state: exp_avg (m) and exp_avg_sq (v) for each param
        - step: optimizer step count t (needed for bias correction)
    """

    def __init__(self, save_dir: str):
        super().__init__()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step
        optimizer = trainer.optimizers[0]
        print("Saving checkpoints")
        # Collect gradients and optimizer state per encoder parameter
        gradients = {}
        exp_avg = {}
        exp_avg_sq = {}
        opt_step = None

        encoder_params = dict(pl_module.encoder.named_parameters())

        for group in optimizer.param_groups:
            for param in group["params"]:
                # Find the name of this param in the encoder
                for name, enc_param in encoder_params.items():
                    if param is enc_param and param.grad is not None:
                        state = optimizer.state[param]
                        gradients[name] = param.grad.clone().cpu()
                        exp_avg[name] = state["exp_avg"].clone().cpu()
                        exp_avg_sq[name] = state["exp_avg_sq"].clone().cpu()
                        opt_step = state["step"]

        # Get hyperparams from the first param group
        group = optimizer.param_groups[0]

        save_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
        torch.save(
            {
                "gradients": gradients,
                "exp_avg": exp_avg,
                "exp_avg_sq": exp_avg_sq,
                "lr": group["lr"],
                "weight_decay": group["weight_decay"],
                "betas": group["betas"],
                "eps": group["eps"],
                "opt_step": opt_step,
            },
            save_path,
        )

        pylogger.info(f"Saved AdamW components at step {global_step}")
