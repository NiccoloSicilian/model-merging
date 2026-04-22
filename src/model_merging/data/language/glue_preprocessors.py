from typing import Any, Dict

from .datasets_preprocess import DatasetPreprocessor, preprocess


class CoLA_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/cola"""

    def preprocess(self, sentence: str, label: int):
        input_text = self.template["input_text"].format(sentence=sentence)
        target_text = self.template["target_text"][str(label)] if label in [0, 1] else ""
        return input_text, target_text

    def __call__(self, example: Dict[str, Any]):
        if isinstance(example["sentence"], str):
            input_text, target_text = self.preprocess(example["sentence"], example["label"])
        else:
            input_text, target_text = [], []
            for sentence, label in zip(example["sentence"], example["label"]):
                i, t = self.preprocess(sentence, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class RTE_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/rte"""

    def preprocess(self, sentence1, sentence2, label):
        input_text = self.template["input_text"].format(sentence1=sentence1, sentence2=sentence2)
        target_text = self.template["target_text"][str(label)] if label in [0, 1] else ""
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["sentence1"], str):
            input_text, target_text = self.preprocess(
                example["sentence1"], example["sentence2"], example["label"]
            )
        else:
            input_text, target_text = [], []
            for s1, s2, label in zip(example["sentence1"], example["sentence2"], example["label"]):
                i, t = self.preprocess(s1, s2, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class MNLI_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/mnli"""

    def preprocess(self, hypothesis, premise, label):
        input_text = self.template["input_text"].format(hypothesis=hypothesis, premise=premise)
        target_text = self.template["target_text"][str(label)] if label in [0, 1, 2] else ""
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["hypothesis"], str):
            input_text, target_text = self.preprocess(
                example["hypothesis"], example["premise"], example["label"]
            )
        else:
            input_text, target_text = [], []
            for hyp, prem, label in zip(example["hypothesis"], example["premise"], example["label"]):
                i, t = self.preprocess(hyp, prem, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class MRPC_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/mrpc"""

    def preprocess(self, sentence1: str, sentence2: str, label: int):
        input_text = self.template["input_text"].format(sentence1=sentence1, sentence2=sentence2)
        target_text = self.template["target_text"][str(label)] if label in [0, 1] else ""
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["sentence1"], str):
            input_text, target_text = self.preprocess(
                example["sentence1"], example["sentence2"], example["label"]
            )
        else:
            input_text, target_text = [], []
            for s1, s2, label in zip(example["sentence1"], example["sentence2"], example["label"]):
                i, t = self.preprocess(s1, s2, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class QNLI_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/qnli"""

    def preprocess(self, question: str, sentence: str, label: int):
        input_text = self.template["input_text"].format(question=question, sentence=sentence)
        target_text = self.template["target_text"][str(label)] if label in [0, 1] else ""
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["question"], str):
            input_text, target_text = self.preprocess(
                example["question"], example["sentence"], example["label"]
            )
        else:
            input_text, target_text = [], []
            for q, s, label in zip(example["question"], example["sentence"], example["label"]):
                i, t = self.preprocess(q, s, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class QQP_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/qqp"""

    def preprocess(self, question1, question2, label):
        input_text = self.template["input_text"].format(question1=question1, question2=question2)
        target_text = self.template["target_text"][str(label)] if label in [0, 1] else ""
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["question1"], str):
            input_text, target_text = self.preprocess(
                example["question1"], example["question2"], example["label"]
            )
        else:
            input_text, target_text = [], []
            for q1, q2, label in zip(example["question1"], example["question2"], example["label"]):
                i, t = self.preprocess(q1, q2, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class SST2_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/sst2"""

    def preprocess(self, sentence: str, label: int):
        input_text = self.template["input_text"].format(sentence=sentence)
        target_text = self.template["target_text"][str(label)] if label in [0, 1] else ""
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["sentence"], str):
            input_text, target_text = self.preprocess(example["sentence"], example["label"])
        else:
            input_text, target_text = [], []
            for sentence, label in zip(example["sentence"], example["label"]):
                i, t = self.preprocess(sentence, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


class STSB_Preprocessor(DatasetPreprocessor):
    """https://huggingface.co/datasets/glue/viewer/stsb"""

    def preprocess(self, sentence1, sentence2, label):
        input_text = self.template["input_text"].format(sentence1=sentence1, sentence2=sentence2)
        target_text = self.template["target_text"].format(label)
        return input_text, target_text

    def __call__(self, example):
        if isinstance(example["sentence1"], str):
            input_text, target_text = self.preprocess(
                example["sentence1"], example["sentence2"], example["label"]
            )
        else:
            input_text, target_text = [], []
            for s1, s2, label in zip(example["sentence1"], example["sentence2"], example["label"]):
                i, t = self.preprocess(s1, s2, label)
                input_text.append(i)
                target_text.append(t)
        return preprocess(self.tokenizer, input_text, target_text, self.tokenizer_kwargs)


glue_processors = {
    "cola": CoLA_Preprocessor,
    "mnli": MNLI_Preprocessor,
    "mrpc": MRPC_Preprocessor,
    "qnli": QNLI_Preprocessor,
    "qqp": QQP_Preprocessor,
    "rte": RTE_Preprocessor,
    "sst2": SST2_Preprocessor,
    "stsb": STSB_Preprocessor,
}
