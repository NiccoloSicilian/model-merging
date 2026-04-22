from typing import Any, List, Optional

import logging
import pytorch_lightning as pl
import torchmetrics

from model_merging.data.language.glue_evaluation import evaluate_accuracy, evaluate_spearman_rho

pylogger = logging.getLogger(__name__)

CLASSIFICATION_TASKS = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2"]
REGRESSION_TASKS = ["stsb"]


class LanguageTester(pl.LightningModule):

    def __init__(self, moe_model, tokenizer, custom_logger: Optional[Any] = None):
        super().__init__()
        self.moe_model = moe_model
        self.tokenizer = tokenizer
        self.custom_logger = custom_logger
        self.log_fn = lambda metric, val: self.log(metric, val, on_step=False, on_epoch=True)

    def set_task(self, task_name: str):
        self.task_name = task_name

    def set_finetuning_accuracy(self, finetuning_accuracy: float):
        self.finetuning_accuracy = finetuning_accuracy

    def set_metrics(self, num_classes: int = None):
        self.output_classes = num_classes
        self.train_acc = torchmetrics.MeanMetric()
        self.val_acc = torchmetrics.MeanMetric()
        self.test_acc = torchmetrics.MeanMetric()

    def _step(self, batch, split: str):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="train")

    def validation_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="val")

    def test_step(self, batch: Any, batch_idx: int):
        return self._step(batch=batch, split="test")

    def on_test_epoch_end(self):
        accuracy = (
            self.trainer.callback_metrics[f"acc/test/{self.task_name}"].cpu().item()
        )
        normalized_acc = accuracy / self.finetuning_accuracy
        self.log_fn(f"normalized_acc/test/{self.task_name}", normalized_acc)

    def configure_optimizers(self):
        return None


class SentenceClassification(LanguageTester):
    def _step(self, batch, split: str):
        logits, acc = evaluate_accuracy(self.moe_model, batch, self.tokenizer)
        metrics = getattr(self, f"{split}_acc")
        metrics.update(acc)
        running_acc = metrics.compute()
        self.log_fn(f"acc/{split}/{self.task_name}", running_acc)
        return {"logits": logits.detach()}


class Regression(LanguageTester):
    def _step(self, batch, split: str):
        logits, acc = evaluate_spearman_rho(self.moe_model, batch, self.tokenizer)
        metrics = getattr(self, f"{split}_acc")
        metrics.update(acc)
        running_acc = metrics.compute()
        self.log_fn(f"acc/{split}/{self.task_name}", running_acc)
        return {"logits": logits.detach()}
