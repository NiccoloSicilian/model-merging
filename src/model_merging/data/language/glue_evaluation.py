import logging
from typing import List

import numpy as np
import torch

pylogger = logging.getLogger(__name__)


def remove_special_tokens(tokenizer, token_list: list):
    """Remove special tokens and stop at -100 sentinel."""
    ret = []
    for token in token_list:
        if token not in tokenizer.all_special_ids and token > 0:
            ret.append(token)
        if token == -100:
            break
    return ret


def evaluate_accuracy(model, batch, tokenizer):
    """Run generation and return (outputs, batch_accuracy)."""
    correct = 0
    total = 0

    with torch.no_grad():
        outputs = model.generate(batch["input_ids"], max_length=10)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels = [
            remove_special_tokens(tokenizer, label_token)
            for label_token in batch["labels"]
        ]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for pred, gold in zip(output_text, labels):
            if pred == gold:
                correct += 1
            total += 1

    return outputs, correct / total


def evaluate_spearman_rho(model, batch, tokenizer):
    """Run generation and return (outputs, spearman_rho) for regression tasks."""
    model = model.eval()
    all_preds: List[str] = []
    all_labels: List[str] = []

    with torch.no_grad():
        outputs = model.generate(batch["input_ids"], max_length=10)
        output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        labels = [
            remove_special_tokens(tokenizer, label_token)
            for label_token in batch["labels"]
        ]
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        all_preds.extend(output_text)
        all_labels.extend(labels)

    from scipy.stats import spearmanr

    def parse_float(s: str):
        try:
            return float(s)
        except Exception:
            return 0.0

    all_preds_f = np.array([parse_float(p) for p in all_preds])
    all_labels_f = np.array([parse_float(l) for l in all_labels])
    rho = spearmanr(all_preds_f, all_labels_f)[0]
    return outputs, rho
