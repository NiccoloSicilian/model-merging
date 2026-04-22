"""
Preprocesses any NLP dataset into a text-to-text format for T5-style models.
"""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Union  # noqa: F401

from transformers import AutoTokenizer


def preprocess(
    tokenizer: AutoTokenizer,
    input_text: str,
    target_text: str,
    tokenizer_kwawgs: Dict[str, Any] = None,
):
    """
    Preprocesses input and target text using a tokenizer and returns model inputs.

    Args:
        tokenizer: Tokenizer instance.
        input_text: Input text to tokenize.
        target_text: Target text to tokenize. Pad tokens are replaced with -100.

    Returns:
        dict of model inputs.
    """
    if tokenizer_kwawgs is None:
        tokenizer_kwawgs = {}
    model_inputs = tokenizer(input_text, **tokenizer_kwawgs)
    if target_text is not None:
        labels = tokenizer(target_text, **tokenizer_kwawgs)
        labels = labels["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
    return model_inputs


class DatasetPreprocessor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        tokenizer_kwargs: Dict[str, Any] = None,
        template: Union[str, Path, Dict] = None,
    ):
        """
        Initializes the preprocessor with a tokenizer and optional prompt template.

        Args:
            tokenizer: Tokenizer instance.
            tokenizer_kwargs: Extra kwargs forwarded to the tokenizer.
            template: Path to a JSON template file, or a dict.
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        if template is not None:
            if isinstance(template, str):
                assert os.path.exists(template), f"Template file not found at {template}"
                with open(template, "r") as f:
                    self.template = json.load(f)
            elif isinstance(template, dict):
                self.template = template
            else:
                raise ValueError("Template must be a path to a json file or a dictionary")
