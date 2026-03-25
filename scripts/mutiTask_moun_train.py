import logging
import os
from typing import Dict, List, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import seed_index_everything
from nn_core.model_logging import NNLogger

from model_merging.model.encoder import ImageEncoder
from model_merging.model.heads import get_classification_head
# Immagina di aver rinominato/creato un nuovo modulo MultiTask
from model_merging.model.image_classifier import MultiTaskImageClassifier 
from model_merging.utils.io_utils import load_model_from_hf, upload_model_to_hf
from hydra.utils import instantiate

pylogger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("high")

def run(cfg: DictConfig):
    seed_index_everything(cfg)

    template_core: NNTemplateCore = NNTemplateCore(
        restore_cfg=cfg.train.get("restore", None),
    )

    logger: NNLogger = NNLogger(
        logging_cfg=cfg.train.logging, cfg=cfg, resume_id=template_core.resume_id
    )

    zeroshot_encoder: ImageEncoder = load_model_from_hf(
        model_name=cfg.nn.encoder.model_name
    )

    # 1. Crea un dizionario per le teste di classificazione e per i dataset
    classification_heads = nn.ModuleDict()
    train_dataloaders = {}
    test_dataloaders = {}

    # Assumiamo che cfg.dataset.tasks sia una lista, es: ["cifar10", "cars"]
    for task_name in cfg.dataset.tasks:
        # Istanzia la head per la task specifica
        classification_heads[task_name] = get_classification_head(
            cfg.nn.encoder.model_name,
            task_name,
            ckpt_path=cfg.misc.ckpt_path,
            openclip_cachedir=cfg.misc.openclip_cachedir,
            device=cfg.device,
        )

        # Istanzia il dataset per la task specifica
        # Nota: in Hydra dovrai strutturare cfg.dataset in modo che possa gestire config multiple
        task_dataset = instantiate(
            cfg.dataset.configs[task_name], 
            preprocess_fn=zeroshot_encoder.val_preprocess,
            batch_size=cfg.train.batch_size,
        )
        train_dataloaders[task_name] = task_dataset.train_loader
        test_dataloaders[task_name] = task_dataset.test_loader

    # 2. Istanzia il modello MultiTask
    model: MultiTaskImageClassifier = hydra.utils.instantiate(
        cfg.nn.module,
        encoder=zeroshot_encoder,
        classifiers=classification_heads, # Passiamo il dizionario di heads
        _recursive_=False,
    )

    # model.task_name = cfg.dataset.name <-- Questo potrebbe non servire più, o diventare una lista
    model.freeze_heads() # Aggiorna questo metodo per freezare tutte le head nel ModuleDict

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=cfg.core.storage_dir,
        logger=logger,
        enable_checkpointing=False,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    # PyTorch Lightning accetta dizionari di dataloader nativamente
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloaders,
    )

    pylogger.info("Starting testing!")
    trainer.test(model=model, dataloaders=test_dataloaders)

    # 3. Salva l'encoder (che ora è stato fine-tunato su più task)
    upload_model_to_hf(model.encoder, cfg.nn.encoder.model_name, "multitask_finetuned")

    logger.log_configuration(model, cfg)

    if logger is not None:
        logger.experiment.finish()

# ... resto del file (main e if __name__ == "__main__")
