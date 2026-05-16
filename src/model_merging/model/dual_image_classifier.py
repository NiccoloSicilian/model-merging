import logging

import hydra
import torch

from model_merging.model.image_classifier import ImageClassifier
from model_merging.merging.dual_arithmetic import build_duality_map

pylogger = logging.getLogger(__name__)


class DualImageClassifier(ImageClassifier):
    """ImageClassifier that applies the duality map on gradients before the optimizer step."""

    def __init__(self, encoder, classifier, mass_schedule="uniform", **kwargs):
        super().__init__(encoder=encoder, classifier=classifier, **kwargs)
        self.mass_schedule = mass_schedule

    def on_before_optimizer_step(self, optimizer):
        # Collect gradients from encoder parameters
        layer_names = []
        grads = {}
        for name, param in self.encoder.named_parameters():
            if param.requires_grad and param.grad is not None:
                layer_names.append(name)
                grads[name] = param.grad.clone()

        if not layer_names:
            return

        # Extract model name from encoder
        model_name = self.encoder.model_name if hasattr(self.encoder, "model_name") else ""

        # Apply duality map
        dualized_grads = build_duality_map(
            layer_names=layer_names,
            grads=grads,
            device=self.device,
            mass_schedule=self.mass_schedule,
            model_name=model_name,
        )

        # Replace gradients with dualized versions
        for name, param in self.encoder.named_parameters():
            if name in dualized_grads:
                param.grad.copy_(dualized_grads[name])
